#!/usr/bin/env python3
"""
ingest_llamaindex.py

CLI utility for building and persisting a LlamaIndex vector store using Qdrant.

This script handles the ingestion of documents from various sources (local directories, URLs),
processes them into nodes, generates embeddings using OpenAI, and stores them in a Qdrant
vector database. It also persists LlamaIndex's document and index store metadata locally.

Key functionalities include:
- Loading documents from local directories or URLs.
- Configuring OpenAI models for LLM and embeddings.
- Setting up Qdrant as the vector store.
- Processing documents into text nodes using configurable chunking.
- Optional adaptive parallel processing for document chunking to handle API rate limits.
- Persisting the vector index to Qdrant and local metadata for LlamaIndex.
- Configuration management via CLI options, environment variables, .env file, and config.py.
"""

import os
import sys
import logging # Added
import click
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional, Callable, Tuple
import platform
from concurrent.futures import ProcessPoolExecutor
from itertools import chain # repeat will be re-added if needed by final logic
import time
import dataclasses
from collections import deque
import openai # For openai.error.RateLimitError

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Document,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# Import web reader directly
try:
    from llama_index.readers.web import SimpleWebPageReader
    WebReader = SimpleWebPageReader
    # print("[info] Successfully imported SimpleWebPageReader") # Replaced by logger
except ImportError:
    try:
        from llama_index.core.readers.web import SimpleWebPageReader
        WebReader = SimpleWebPageReader
        # print("[info] Successfully imported SimpleWebPageReader from core.readers.web") # Replaced by logger
    except ImportError:
        try:
            from llama_index_readers_web.simple_web import SimpleWebPageReader
            WebReader = SimpleWebPageReader
            # print("[info] Successfully imported SimpleWebPageReader from llama_index_readers_web") # Replaced by logger
        except ImportError:
            WebReader = None
            # print("[warning] Could not import any web reader. URL ingestion may not work.") # Replaced by logger

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models as qdrant_models

# Import from config.py
from config import (
    OPENAI_API_KEY as CFG_OPENAI_API_KEY,
    OPENAI_MODEL_EMBEDDING as CFG_OPENAI_MODEL_EMBEDDING,
    OPENAI_MODEL_LLM as CFG_OPENAI_MODEL_LLM,
    QDRANT_URL as CFG_QDRANT_URL,
    QDRANT_API_KEY as CFG_QDRANT_API_KEY,
    DEFAULT_PERSIST_DIR as CFG_DEFAULT_PERSIST_DIR,
    INGEST_DEFAULT_COLLECTION_NAME as CFG_INGEST_DEFAULT_COLLECTION_NAME,
    DEFAULT_CHUNK_SIZE as CFG_DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP as CFG_DEFAULT_CHUNK_OVERLAP,
    get_embedding_dim as cfg_get_embedding_dim,
)

# Basic logging configuration at module level for early messages
# This will be potentially reconfigured in cli()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Log initial import status for WebReader
if WebReader:
    logger.info(f"Successfully imported a web reader: {WebReader.__name__}")
else:
    logger.warning("Could not import any web reader. URL ingestion may not work.")


@dataclasses.dataclass
class WorkerResp:
    """
    Dataclass for storing the result of processing a single document in a worker.
    
    Attributes:
        doc_id: The ID of the document processed.
        nodes: A list of Document objects (nodes) generated from the document.
        success: Boolean indicating if processing was successful.
        rate_limited: Boolean indicating if processing failed due to a rate limit.
        exception_info: String containing exception information if an error occurred.
        original_doc_dict: The original document dictionary, for re-queueing if needed.
    """
    doc_id: Optional[str]
    nodes: Optional[List[Document]] = None # LlamaIndex uses Document objects as nodes here
    success: bool = False
    rate_limited: bool = False
    exception_info: Optional[str] = None
    original_doc_dict: Optional[dict] = None # For re-queueing

# Helper function for ProcessPoolExecutor, must be defined at the top level for pickling
def _process_single_document(config_tuple: Tuple) -> WorkerResp:
    """
    Processes a single document (serialized as a dictionary) using a LlamaIndex ingestion pipeline.

    This function is designed to be run in a separate process by `ProcessPoolExecutor`.
    It re-initializes LlamaIndex `Settings` (LLM and Embeddings) for the current process
    to ensure thread-safety and proper context for OpenAI API calls.

    Args:
        config_tuple: A tuple containing:
            - doc_dict (dict): The document to process, serialized as a dictionary.
            - chunk_size (int): The target size for text chunks.
            - chunk_overlap (int): The overlap between text chunks.
            - openai_api_key (str): The OpenAI API key.
            - openai_model_llm (str): The OpenAI LLM model name.
            - openai_model_embedding (str): The OpenAI embedding model name.

    Returns:
        WorkerResp: An object containing the processing result, including generated nodes,
                    status (success, rate_limited), and any error information.
    """
    worker_logger = logging.getLogger(__name__ + ".worker") 
    doc_dict, chunk_size, chunk_overlap, openai_api_key, openai_model_llm, openai_model_embedding = config_tuple
    doc_id_for_error = doc_dict.get("id_", doc_dict.get("doc_id", "unknown_id"))

    try:
        # Re-initialize settings per process
        Settings.llm = OpenAI(model=openai_model_llm, api_key=openai_api_key)
        Settings.embed_model = OpenAIEmbedding(model=openai_model_embedding, api_key=openai_api_key)
    except Exception as e:
        err_msg = f"Failed to initialize OpenAI models: {e}"
        worker_logger.error(f"DocID {doc_id_for_error}: {err_msg}")
        return WorkerResp(doc_id=doc_id_for_error, success=False, exception_info=err_msg, original_doc_dict=doc_dict)

    doc = Document(**doc_dict)
    # Use TokenTextSplitter instead of SentenceSplitter to avoid NLTK dependency
    from llama_index.core.node_parser import TokenTextSplitter
    node_parser = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    pipeline = IngestionPipeline(transformations=[node_parser])
    
    try:
        processed_nodes = pipeline.run(documents=[doc])
        # worker_logger.debug(f"DocID {doc.doc_id}: Processed, generated {len(processed_nodes)} nodes.") # Example debug
        return WorkerResp(doc_id=doc.doc_id, nodes=processed_nodes, success=True, original_doc_dict=doc_dict)
    except openai.RateLimitError as rle:
        worker_logger.warning(f"DocID {doc.doc_id}: Rate limit error encountered.")
        return WorkerResp(doc_id=doc.doc_id, success=False, rate_limited=True, exception_info=str(rle), original_doc_dict=doc_dict)
    except Exception as e:
        err_msg = f"Failed to process document: {e}"
        worker_logger.error(f"DocID {doc.doc_id}: {err_msg}", exc_info=True) # Added exc_info=True
        return WorkerResp(doc_id=doc.doc_id, success=False, exception_info=err_msg, original_doc_dict=doc_dict)

# _log helper function removed

@dataclasses.dataclass
class PipelineConfig:
    """
    Dataclass to hold resolved configuration values used throughout the ingestion pipeline.

    Attributes:
        final_openai_api_key: The resolved OpenAI API key.
        final_qdrant_api_key: The resolved Qdrant API key (if any).
        final_openai_model_embedding: The OpenAI embedding model name to use.
        final_openai_model_llm: The OpenAI LLM model name to use.
        final_qdrant_url: The URL for the Qdrant instance.
        final_persist_dir: The base directory for persisting local LlamaIndex storage.
        effective_collection_name: The name of the Qdrant collection to use.
        local_docstore_persist_path: The specific path for local docstore/index store persistence.
        chunk_size: The chunk size for document processing.
        chunk_overlap: The chunk overlap for document processing.
    """
    final_openai_api_key: Optional[str]
    final_qdrant_api_key: Optional[str]
    final_openai_model_embedding: str
    final_openai_model_llm: str
    final_qdrant_url: str
    final_persist_dir: str
    effective_collection_name: str
    local_docstore_persist_path: Path
    chunk_size: int
    chunk_overlap: int


def _setup_pipeline_config(
    openai_api_key_arg: Optional[str],
    qdrant_api_key_arg: Optional[str],
    collection_name_arg: Optional[str],
    persist_dir_arg: str,
    openai_model_embedding_arg: str,
    openai_model_llm_arg: str,
    qdrant_url_arg: str,
    chunk_size: int,
    chunk_overlap: int
) -> Optional[PipelineConfig]:
    """
    Resolves and validates all necessary configurations for the ingestion pipeline.

    This function determines the final values for settings like API keys, model names,
    and paths by considering CLI arguments, environment variables, and values from
    `config.py`, adhering to a defined precedence (CLI > ENV > config.py > code default).
    It also initializes global LlamaIndex `Settings` for LLM and embedding models.

    Args:
        openai_api_key_arg: OpenAI API key from CLI/env.
        qdrant_api_key_arg: Qdrant API key from CLI/env.
        collection_name_arg: Qdrant collection name from CLI/env.
        persist_dir_arg: Base persistence directory from CLI/env.
        openai_model_embedding_arg: OpenAI embedding model from CLI/env.
        openai_model_llm_arg: OpenAI LLM model from CLI/env.
        qdrant_url_arg: Qdrant URL from CLI/env.
        chunk_size: Chunk size from CLI/env.
        chunk_overlap: Chunk overlap from CLI/env.

    Returns:
        A PipelineConfig object if all critical configurations are resolved successfully.
        None if a critical configuration (like OpenAI API key) is missing or
        if LlamaIndex Settings initialization fails.
    """
    logger = logging.getLogger(__name__ + "._setup_pipeline_config")

    # Determine source and final value for OpenAI API Key
    if openai_api_key_arg and openai_api_key_arg != CFG_OPENAI_API_KEY: # CLI/ENV takes precedence
        final_openai_api_key = openai_api_key_arg
        logger.info(f"Using OpenAI API Key from CLI/environment variable.")
    else: # Fallback to config.py or default if CLI/ENV not set or same as config default
        final_openai_api_key = CFG_OPENAI_API_KEY
        if final_openai_api_key:
            logger.info(f"Using OpenAI API Key from config.py.")
        else: # Should only happen if CFG_OPENAI_API_KEY is also None
            logger.info(f"OpenAI API Key not found in CLI, environment, or config.py.")
            # The check below will handle the error if it's still None

    # Determine source and final value for Qdrant API Key
    if qdrant_api_key_arg and qdrant_api_key_arg != CFG_QDRANT_API_KEY:
        final_qdrant_api_key = qdrant_api_key_arg
        logger.info(f"Using Qdrant API Key from CLI/environment variable.")
    else:
        final_qdrant_api_key = CFG_QDRANT_API_KEY
        if final_qdrant_api_key: # Log only if it's actually set in config.py
            logger.info(f"Using Qdrant API Key from config.py (or relying on client's default behavior if None).")
        # No specific log if it's None from all sources, Qdrant client handles it

    # OpenAI Model Embedding
    if openai_model_embedding_arg != CFG_OPENAI_MODEL_EMBEDDING:
        final_openai_model_embedding = openai_model_embedding_arg
        logger.info(f"Using OpenAI Embedding Model from CLI/environment: {final_openai_model_embedding}")
    else:
        final_openai_model_embedding = CFG_OPENAI_MODEL_EMBEDDING
        logger.info(f"Using OpenAI Embedding Model from config.py: {final_openai_model_embedding}")

    # OpenAI LLM Model
    if openai_model_llm_arg != CFG_OPENAI_MODEL_LLM:
        final_openai_model_llm = openai_model_llm_arg
        logger.info(f"Using OpenAI LLM Model from CLI/environment: {final_openai_model_llm}")
    else:
        final_openai_model_llm = CFG_OPENAI_MODEL_LLM
        logger.info(f"Using OpenAI LLM Model from config.py: {final_openai_model_llm}")

    # Qdrant URL
    if qdrant_url_arg != CFG_QDRANT_URL:
        final_qdrant_url = qdrant_url_arg
        logger.info(f"Using Qdrant URL from CLI/environment: {final_qdrant_url}")
    else:
        final_qdrant_url = CFG_QDRANT_URL
        logger.info(f"Using Qdrant URL from config.py: {final_qdrant_url}")
    
    final_persist_dir = persist_dir_arg # Default is from CFG_DEFAULT_PERSIST_DIR via Click
    logger.info(f"Using persist directory: {final_persist_dir}")

    # Effective Collection Name
    if collection_name_arg and collection_name_arg != CFG_INGEST_DEFAULT_COLLECTION_NAME:
        effective_collection_name = collection_name_arg
        logger.info(f"Using collection name from CLI/environment: '{effective_collection_name}'")
    else:
        effective_collection_name = CFG_INGEST_DEFAULT_COLLECTION_NAME
        logger.info(f"Using default collection name from config.py: '{effective_collection_name}'")
        
    if final_openai_api_key:
        masked_key = final_openai_api_key[:4] + "..." + final_openai_api_key[-4:] if len(final_openai_api_key) > 8 else "***"
        logger.debug(f"Resolved OpenAI API key: {masked_key}")
    if final_qdrant_api_key:
        masked_q_key = final_qdrant_api_key[:4] + "..." + final_qdrant_api_key[-4:] if len(final_qdrant_api_key) > 8 else "***"
        logger.debug(f"Resolved Qdrant API key: {masked_q_key}")


    if not final_openai_api_key:
        logger.error("CRITICAL: OpenAI API Key is not set. This is required for embeddings and LLM operations.")
        logger.debug("Environment variables for OpenAI key check:")
        for key, value in os.environ.items():
            if key.startswith("OPENAI") or key == "OPENAI_API_KEY":
                masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                logger.debug(f"  {key}={masked_value}")
        return None

    logger.info("Configuring LlamaIndex Settings...")
    logger.info(f"  LLM Model: {final_openai_model_llm}")
    logger.info(f"  Embedding Model: {final_openai_model_embedding}")

    try:
        Settings.llm = OpenAI(model=final_openai_model_llm, api_key=final_openai_api_key)
        Settings.embed_model = OpenAIEmbedding(model=final_openai_model_embedding, api_key=final_openai_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI models for LlamaIndex: {e}")
        return None

    final_persist_path = Path(final_persist_dir)
    local_docstore_persist_path = final_persist_path / effective_collection_name.strip()

    return PipelineConfig(
        final_openai_api_key=final_openai_api_key,
        final_qdrant_api_key=final_qdrant_api_key,
        final_openai_model_embedding=final_openai_model_embedding,
        final_openai_model_llm=final_openai_model_llm,
        final_qdrant_url=final_qdrant_url,
        final_persist_dir=final_persist_dir,
        effective_collection_name=effective_collection_name,
        local_docstore_persist_path=local_docstore_persist_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

def _load_source_documents(
    data_dir: Optional[str],
    source_url: Optional[str],
    verbose_mode: bool
) -> Optional[List[Document]]:
    """
    Loads documents from the specified source (local directory or URL).

    It handles loading from either `data_dir` (for local files) or `source_url`.
    If `source_url` is provided, it requires `SimpleWebPageReader` (from
    `llama-index-readers-web`) to be installed.

    Args:
        data_dir: Path to a local directory containing documents.
        source_url: URL of a web page to load as a document.
        verbose_mode: If True, enables progress display for `SimpleDirectoryReader`.

    Returns:
        A list of loaded `Document` objects.
        Returns an empty list if no documents are found at the source but no error occurred.
        Returns `None` if there's an error in arguments (e.g., both sources provided),
        if a required reader is unavailable, or if a loading error occurs.
    """
    logger = logging.getLogger(__name__ + "._load_source_documents")
    documents: List[Document] = []

    if data_dir and source_url:
        logger.error("--data-dir and --source-url are mutually exclusive. Please provide only one.")
        return None # Indicates an error in argument parsing, handled by perform_ingestion
    if not data_dir and not source_url:
        logger.error("Either --data-dir or --source-url must be provided.")
        return None # Indicates an error in argument parsing

    if source_url:
        logger.info(f"Loading documents directly from URL: {source_url}")
        try:
            if WebReader is None:
                logger.error("No web reader module available. URL ingestion is not available.")
                logger.error("Please install required packages: pip install llama-index-readers-web")
                return None
            
            logger.info(f"Using {WebReader.__name__} to load URL")
            loader = WebReader()
            
            loaded_docs = loader.load_data(urls=[source_url])
            if not loaded_docs:
                logger.warning(f"No documents loaded from URL '{source_url}'.")
            else:
                documents.extend(loaded_docs)
                logger.info(f"Loaded {len(loaded_docs)} document(s) from URL '{source_url}'.")
        except Exception as e:
            logger.error(f"Failed to load documents from URL '{source_url}': {e}")
            logger.error("Suggestion: Try installing required packages: pip install llama-index-readers-web beautifulsoup4 requests")
            return None
    elif data_dir:
        logger.info(f"Loading documents from local directory: {data_dir}")
        try:
            reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
            loaded_docs = reader.load_data(show_progress=verbose_mode) 
            if not loaded_docs:
                logger.warning(f"No documents found or loaded from '{data_dir}'.")
            else:
                documents.extend(loaded_docs)
                logger.info(f"Loaded {len(loaded_docs)} document(s) from directory '{data_dir}'.")
        except Exception as e:
            logger.error(f"Failed to load documents from '{data_dir}': {e}")
            return None
    
    if not documents:
        logger.warning("No documents were loaded.")
        return [] # Return empty list, perform_ingestion will handle exit

    return documents

def _create_document_nodes(
    documents: List[Document],
    config: PipelineConfig,
    num_workers_opt: int,
    max_workers_opt: int,
    min_workers_opt: int,
    adaptive_backoff_seconds_opt: int,
    adaptive_success_streak_opt: int,
    verbose_mode: bool
) -> List[Document]: 
    """
    Processes a list of LlamaIndex `Document` objects into a list of text nodes.

    This function handles both sequential processing (for small numbers of documents or
    when `num_workers_opt <= 1`) and adaptive parallel processing using a
    `ProcessPoolExecutor`. The parallel processing adapts the number of workers
    based on success rates and API rate limit errors.

    Args:
        documents: The list of LlamaIndex `Document` objects to process.
        config: The `PipelineConfig` object containing resolved configurations.
        num_workers_opt: Initial number of worker processes for parallel processing.
        max_workers_opt: Maximum number of worker processes.
        min_workers_opt: Minimum number of worker processes.
        adaptive_backoff_seconds_opt: Seconds to wait after a rate limit error.
        adaptive_success_streak_opt: Number of successful batches before increasing workers.
        verbose_mode: If True, enables progress display for sequential pipeline.

    Returns:
        A list of processed `Document` objects (nodes). This list can be empty if
        no nodes are generated (e.g., input documents were empty or filtered out entirely).
    """
    logger = logging.getLogger(__name__ + "._create_document_nodes")
    all_processed_nodes: List[Document] = []
    
    logger.info("Preparing for document ingestion pipeline (chunking)...")

    if num_workers_opt <= 1:
        logger.info("Processing documents sequentially.")
        from llama_index.core.node_parser import TokenTextSplitter
        node_parser = TokenTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        logger.info("Using TokenTextSplitter instead of SentenceSplitter to avoid NLTK dependency")
        pipeline = IngestionPipeline(transformations=[node_parser])
        all_processed_nodes = pipeline.run(documents=documents, show_progress=verbose_mode)
    else:
        logger.info(f"Starting adaptive parallel processing: initial_workers={num_workers_opt}, max_workers={max_workers_opt}, min_workers={min_workers_opt}")

        docs_to_process_deque = deque(documents)
        min_workers = max(1, min_workers_opt)
        max_workers = max(min_workers, max_workers_opt if max_workers_opt > 0 else os.cpu_count() or 1)
        current_num_workers = max(min_workers, min(num_workers_opt, max_workers))
        logger.info(f"Adjusted worker range: initial={current_num_workers}, min={min_workers}, max={max_workers}")

        successful_streak = 0
        cooldown_until_timestamp = 0
        total_docs_processed_count = 0
        total_rate_limit_hits = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            while docs_to_process_deque:
                if time.time() < cooldown_until_timestamp:
                    wait_time = cooldown_until_timestamp - time.time()
                    logger.info(f"Cooling down due to rate limit. Waiting for {wait_time:.1f}s...")
                    time.sleep(wait_time)

                batch_size = min(len(docs_to_process_deque), current_num_workers)
                current_batch_docs_dicts = [docs_to_process_deque.popleft().dict() for _ in range(batch_size)]
                logger.info(f"Processing batch of {len(current_batch_docs_dicts)} documents with {current_num_workers} target workers.")

                tasks_for_executor = [
                    (doc_dict, config.chunk_size, config.chunk_overlap, config.final_openai_api_key, config.final_openai_model_llm, config.final_openai_model_embedding)
                    for doc_dict in current_batch_docs_dicts
                ]
                batch_results: List[WorkerResp] = []
                try:
                    batch_results = list(executor.map(_process_single_document, tasks_for_executor, chunksize=1))
                except Exception as e:
                    logger.error(f"Critical error during executor.map: {e}")
                    for doc_dict_failed in current_batch_docs_dicts:
                         docs_to_process_deque.append(Document(**doc_dict_failed))
                    logger.error(f"Re-queued {len(current_batch_docs_dicts)} documents from failed batch. Reducing workers and backing off.")
                    current_num_workers = max(min_workers, current_num_workers // 2)
                    cooldown_until_timestamp = time.time() + adaptive_backoff_seconds_opt
                    successful_streak = 0
                    continue

                num_successful_in_batch = 0
                num_rate_limited_in_batch = 0
                for resp in batch_results:
                    if resp.success and resp.nodes is not None:
                        all_processed_nodes.extend(resp.nodes)
                        num_successful_in_batch += 1
                        total_docs_processed_count +=1
                    elif resp.rate_limited:
                        num_rate_limited_in_batch += 1
                        total_rate_limit_hits += 1
                        if resp.original_doc_dict:
                            docs_to_process_deque.append(Document(**resp.original_doc_dict))
                    else:
                        logger.error(f"Failed to process doc {resp.doc_id}: {resp.exception_info}")

                logger.info(f"Batch summary: {num_successful_in_batch} succeeded, {num_rate_limited_in_batch} rate-limited.")
                if num_rate_limited_in_batch > 0:
                    logger.warning(f"Rate limit hit. Reducing workers from {current_num_workers} to {max(min_workers, current_num_workers // 2)} and backing off for {adaptive_backoff_seconds_opt}s.")
                    current_num_workers = max(min_workers, current_num_workers // 2)
                    cooldown_until_timestamp = time.time() + adaptive_backoff_seconds_opt
                    successful_streak = 0
                elif num_successful_in_batch == len(current_batch_docs_dicts):
                    successful_streak += 1
                    if successful_streak >= adaptive_success_streak_opt and current_num_workers < max_workers:
                        logger.info(f"Success streak of {successful_streak} reached. Increasing workers from {current_num_workers} to {current_num_workers + 1}.")
                        current_num_workers += 1
                        successful_streak = 0
                else:
                    successful_streak = 0
        logger.info(f"Adaptive parallel processing finished. Total documents processed into nodes: {total_docs_processed_count} (initial: {len(documents)}). Total rate limit hits: {total_rate_limit_hits}.")

    if not all_processed_nodes:
        logger.warning("No nodes were generated by the ingestion pipeline.")
        return [] # Return empty list, perform_ingestion will handle exit
    
    logger.info(f"Generated {len(all_processed_nodes)} nodes from documents.")
    return all_processed_nodes

def _initialize_vector_store(
    config: PipelineConfig
    # embedding_dim_func: Callable[[str], int] # Removed
) -> Optional[QdrantVectorStore]: 
    """
    Initializes and returns the QdrantVectorStore.
    
    This function connects to the Qdrant instance specified in the configuration.
    It determines the required embedding dimension for the configured OpenAI model.
    It then attempts to connect to the specified Qdrant collection.
    If the collection does not exist, it will attempt to create it with the correct
    vector parameters (embedding dimension and COSINE distance).

    Args:
        config: The `PipelineConfig` object containing Qdrant URL, API key,
                collection name, and embedding model information.

    Returns:
        An initialized `QdrantVectorStore` instance if successful.
        `None` if there's an error connecting to Qdrant, checking the collection,
        or creating the collection.
    """
    logger = logging.getLogger(__name__ + "._initialize_vector_store")

    logger.info(f"Initializing Qdrant client for URL: {config.final_qdrant_url}")
    q_client = QdrantClient(url=config.final_qdrant_url, api_key=config.final_qdrant_api_key)

    # Get embedding dimension directly using the model name from config
    embedding_dim = cfg_get_embedding_dim(config.final_openai_model_embedding)
    logger.info(f"Determined embedding dimension: {embedding_dim} for model {config.final_openai_model_embedding}")

    try:
        # Attempt to get the collection. If it fails, it might mean the collection doesn't exist
        # or there's a connection issue. The Qdrant client library might raise different exceptions.
        # qdrant_client.http.exceptions.UnexpectedResponse: Response HTTP Status Code: 404
        # is common for "not found". Other errors for connection issues.
        logger.debug(f"Checking for existing Qdrant collection: {config.effective_collection_name}")
        q_client.get_collection(collection_name=config.effective_collection_name)
        logger.info(f"Using existing Qdrant collection: {config.effective_collection_name}")
        vector_store = QdrantVectorStore(client=q_client, collection_name=config.effective_collection_name)
        logger.info("Successfully connected to Qdrant vector store")
    except Exception as e_qdrant: 
        logger.error(f"Qdrant connection/collection check failed for '{config.effective_collection_name}': {e_qdrant}. Using SimpleVectorStore instead.")
        from llama_index.core.vector_stores import SimpleVectorStore # Keep import local if only for this fallback
        vector_store = SimpleVectorStore() 
        logger.info("Using SimpleVectorStore as fallback (in-memory)")
    
    return vector_store

def _build_and_persist_index_and_docstore(
    nodes: List[Document], # LlamaIndex uses Document for nodes too
    vector_store, # Can be QdrantVectorStore or SimpleVectorStore
    config: PipelineConfig,
    verbose_mode: bool
) -> bool:
    """Builds the VectorStoreIndex and persists components."""
    logger = logging.getLogger(__name__ + "._build_and_persist_index_and_docstore")

    logger.info("Building VectorStoreIndex from processed nodes. This may take a while...")
    logger.info(f"Local docstore/index metadata will be persisted to: {config.local_docstore_persist_path}")
    if isinstance(vector_store, QdrantVectorStore):
        logger.info(f"Vectors will be stored in Qdrant collection: {config.effective_collection_name} at {config.final_qdrant_url}")
    else:
        logger.info("Vectors will be stored in the in-memory SimpleVectorStore.")

    try:
        logger.debug(f"Attempting os.makedirs for: {str(config.local_docstore_persist_path)}")
        os.makedirs(config.local_docstore_persist_path, exist_ok=True)
        logger.debug(f"After os.makedirs: Exists? {config.local_docstore_persist_path.exists()}, IsDir? {config.local_docstore_persist_path.is_dir()}")
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=verbose_mode,
        )
        
        if config.local_docstore_persist_path.exists() and config.local_docstore_persist_path.is_dir():
            logger.info(f"Persisting non-vector components (docstore, index_store) to: {str(config.local_docstore_persist_path)}")
            index.storage_context.persist(persist_dir=str(config.local_docstore_persist_path))
            
            docstore_json_path = config.local_docstore_persist_path / "docstore.json"
            if docstore_json_path.exists() and docstore_json_path.is_file():
                import json
                try:
                    with open(docstore_json_path, 'r') as f:
                        docstore_content = json.load(f)
                    if not docstore_content or not docstore_content.get("docstore", {}).get("docs"):
                        logger.warning(f"docstore.json was created but appears empty or has no documents at {docstore_json_path}")
                        logger.debug("Attempting to repopulate docstore from nodes...")
                        for node_doc in nodes: # Renamed 'node' to 'node_doc' to avoid conflict
                            if not index.storage_context.docstore.document_exists(node_doc.node_id): # Assuming nodes are Document type with node_id
                                index.storage_context.docstore.add_documents([node_doc], allow_update=True)
                        index.storage_context.persist(persist_dir=str(config.local_docstore_persist_path))
                        logger.debug(f"Repopulated docstore with {len(index.storage_context.docstore.docs)} documents and persisted again.")
                    else:
                        logger.debug(f"Verified docstore.json contains {len(docstore_content.get('docstore', {}).get('docs', {}))} documents.")
                except Exception as e_validate:
                    logger.warning(f"Failed to validate docstore.json: {e_validate}. BM25 retrieval may not work properly.")
            else:
                logger.warning(f"docstore.json was not created at {docstore_json_path}. BM25 retrieval may not work properly.")
            
            logger.info(f"Ingestion completed. Vectors stored successfully.")
            logger.info(f"Docstore and index metadata persisted to '{config.local_docstore_persist_path}'.")
        else:
            logger.info(f"Ingestion completed. Vectors stored successfully.")
            logger.warning(f"Local persistence of docstore/index_store skipped (directory issue with '{config.local_docstore_persist_path}').")
        return True

    except Exception as e:
        logger.error(f"An error occurred during index building or persistence: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}", exc_info=e)
        return False

def perform_ingestion(
    data_dir_opt: Optional[str],
    source_url_opt: Optional[str],
    collection_name_opt: Optional[str],
    persist_dir_opt: str,
    openai_api_key_opt: Optional[str],
    openai_model_embedding_opt: str,
    openai_model_llm_opt: str,
    qdrant_url_opt: str,
    qdrant_api_key_opt: Optional[str],
    chunk_size_opt: int,
    chunk_overlap_opt: int,
    # embedding_dim_func: Callable[[str], int], # This argument has been removed.
    num_workers_opt: int,
    max_workers_opt: int,
    min_workers_opt: int,
    adaptive_backoff_seconds_opt: int,
    adaptive_success_streak_opt: int,
    verbose_mode: bool
) -> bool:
    """
    Main coordinator for the document ingestion, processing, and indexing pipeline.

    This function orchestrates the entire ingestion process by calling various helper
    functions to:
    1. Set up pipeline configurations (API keys, models, paths).
    2. Load source documents (from local directory or URL).
    3. Process documents into text nodes (sequentially or in parallel).
    4. Initialize the Qdrant vector store (checking/creating the collection).
    5. Build the LlamaIndex `VectorStoreIndex` and persist components.

    Args:
        data_dir_opt: Optional path to a local data directory.
        source_url_opt: Optional URL for a single document source.
        collection_name_opt: Optional Qdrant collection name.
        persist_dir_opt: Base directory for local persistence.
        openai_api_key_opt: Optional OpenAI API key.
        openai_model_embedding_opt: OpenAI embedding model name.
        openai_model_llm_opt: OpenAI LLM model name.
        qdrant_url_opt: Qdrant server URL.
        qdrant_api_key_opt: Optional Qdrant API key.
        chunk_size_opt: Chunk size for node parsing.
        chunk_overlap_opt: Chunk overlap for node parsing.
        num_workers_opt: Initial number of worker processes for parallel node creation.
        max_workers_opt: Maximum number of worker processes.
        min_workers_opt: Minimum number of worker processes.
        adaptive_backoff_seconds_opt: Backoff seconds after rate limit errors.
        adaptive_success_streak_opt: Success streak to increase workers.
        verbose_mode: Boolean flag for verbose output and progress bars.

    Returns:
        True if the entire ingestion process completes successfully.
        False if any critical step fails.
    """
    logger = logging.getLogger(__name__ + ".perform_ingestion")

    pipeline_config = _setup_pipeline_config(
        openai_api_key_arg=openai_api_key_opt,
        qdrant_api_key_arg=qdrant_api_key_opt,
        collection_name_arg=collection_name_opt,
        persist_dir_arg=persist_dir_opt,
        openai_model_embedding_arg=openai_model_embedding_opt,
        openai_model_llm_arg=openai_model_llm_opt,
        qdrant_url_arg=qdrant_url_opt,
        chunk_size=chunk_size_opt,
        chunk_overlap=chunk_overlap_opt
    )
    if not pipeline_config:
        logger.error("Failed to set up pipeline configuration. Exiting.")
        return False

    documents = _load_source_documents(data_dir_opt, source_url_opt, verbose_mode)
    if documents is None: # Indicates an error or invalid arguments
        logger.error("Failed to load source documents due to invalid arguments or error. Exiting.")
        return False
    if not documents: # Indicates no documents found but arguments were valid
        logger.warning("No documents loaded. Ingestion process will stop.")
        return True # Considered a "successful" run as there was no error, just no data.

    nodes = _create_document_nodes(
        documents=documents,
        config=pipeline_config,
        num_workers_opt=num_workers_opt,
        max_workers_opt=max_workers_opt,
        min_workers_opt=min_workers_opt,
        adaptive_backoff_seconds_opt=adaptive_backoff_seconds_opt,
        adaptive_success_streak_opt=adaptive_success_streak_opt,
        verbose_mode=verbose_mode
    )
    if not nodes: 
        # If documents were loaded but no nodes were created (e.g. all filtered out),
        # this is not necessarily an error, but the process cannot continue to index.
        logger.warning("No nodes were generated from the documents. Ingestion process will stop.")
        return True # Graceful exit, as no error occurred, but nothing to index.

    vector_store = _initialize_vector_store(pipeline_config) # Removed embedding_dim_func
    if not vector_store:
        # Error already logged by _initialize_vector_store
        return False
        
    return _build_and_persist_index_and_docstore(
        nodes=nodes, 
        vector_store=vector_store,
        config=pipeline_config,
        verbose_mode=verbose_mode
    )

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--env-file",
    type=click.Path(dir_okay=False, readable=True),
    default=".env",
    show_default=True,
    help="Path to .env file with environment variables (e.g., OPENAI_API_KEY)."
)
@click.option(
    "--data-dir",
    "data_dir_opt",
    required=False,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help="Directory containing the documents to ingest. Mutually exclusive with --source-url."
)
@click.option(
    "--source-url",
    "source_url_opt",
    type=str,
    default=None,
    help="URL of a single document to download and ingest. Mutually exclusive with --data-dir."
)
@click.option(
    "--collection-name",
    "collection_name_opt",
    type=str,
    default=None,
    help=f"Optional name for the Qdrant collection and local persist subdirectory. Defaults to '{CFG_INGEST_DEFAULT_COLLECTION_NAME}' from config or environment."
)
@click.option(
    "--persist-dir",
    "persist_dir_opt",
    default=CFG_DEFAULT_PERSIST_DIR,
    show_default=True,
    type=click.Path(file_okay=False, writable=True),
    help="Base directory to persist the LlamaIndex storage context (docstore, etc.)."
)
@click.option(
    "--openai-api-key",
    "openai_api_key_opt",
    envvar="OPENAI_API_KEY",
    default=None,
    help="OpenAI API key. Overrides .env and config.py."
)
@click.option(
    "--openai-model-embedding",
    "openai_model_embedding_opt",
    default=CFG_OPENAI_MODEL_EMBEDDING,
    show_default=True,
    help="OpenAI embedding model name."
)
@click.option(
    "--openai-model-llm",
    "openai_model_llm_opt",
    default=CFG_OPENAI_MODEL_LLM,
    show_default=True,
    help="OpenAI LLM model name."
)
@click.option(
    "--qdrant-url",
    "qdrant_url_opt",
    default=CFG_QDRANT_URL,
    show_default=True,
    envvar="QDRANT_URL",
    help="Qdrant server URL."
)
@click.option(
    "--qdrant-api-key",
    "qdrant_api_key_opt",
    envvar="QDRANT_API_KEY",
    default=None,
    help="Qdrant API key (if required). Overrides .env and config.py."
)
@click.option(
    "--chunk-size",
    "chunk_size_opt",
    default=CFG_DEFAULT_CHUNK_SIZE,
    show_default=True,
    type=int,
    help="Chunk size for node parser."
)
@click.option(
    "--chunk-overlap",
    "chunk_overlap_opt",
    default=CFG_DEFAULT_CHUNK_OVERLAP,
    show_default=True,
    type=int,
    help="Chunk overlap for node parser."
)
# Replaced --parallel-workers with --num-workers and adaptive options
@click.option(
    "--num-workers",
    "num_workers_opt",
    default=0,
    show_default=True,
    type=int,
    help="Initial number of worker processes. 0 or 1 for sequential. >1 for adaptive parallel processing."
)
@click.option(
    "--max-workers",
    "max_workers_opt",
    default=0, # Default 0 means it will be derived (e.g. CPU count) or based on num_workers if num_workers is higher
    show_default=True,
    type=int,
    help="Maximum number of workers for adaptive parallel processing. Defaults to CPU count or num_workers if higher."
)
@click.option(
    "--min-workers",
    "min_workers_opt",
    default=1,
    show_default=True,
    type=int,
    help="Minimum number of workers for adaptive parallel processing."
)
@click.option(
    "--adaptive-backoff-seconds",
    "adaptive_backoff_seconds_opt",
    default=30,
    show_default=True,
    type=int,
    help="Seconds to wait after a rate limit error before retrying."
)
@click.option(
    "--adaptive-success-streak",
    "adaptive_success_streak_opt",
    default=3,
    show_default=True,
    type=int,
    help="Number of consecutive successful batches before increasing workers."
)
@click.option(
    "--verbose",
    "verbose_opt",
    is_flag=True,
    default=False,
    help="Enable verbose output."
)
def cli(
    env_file: str,
    data_dir_opt: Optional[str],
    source_url_opt: Optional[str],
    collection_name_opt: Optional[str],
    persist_dir_opt: str,
    openai_api_key_opt: Optional[str],
    openai_model_embedding_opt: str,
    openai_model_llm_opt: str,
    qdrant_url_opt: str,
    qdrant_api_key_opt: Optional[str],
    chunk_size_opt: int,
    chunk_overlap_opt: int,
    num_workers_opt: int,
    max_workers_opt: int,
    min_workers_opt: int,
    adaptive_backoff_seconds_opt: int,
    adaptive_success_streak_opt: int,
    verbose_opt: bool,
) -> None:
    """
    CLI for ingesting documents from DATA_DIR or a SOURCE_URL,
    building a LlamaIndex VectorStoreIndex, and persisting it.
    Supports sequential or adaptive parallel processing for document chunking.
    Uses settings from config.py as defaults, which can be overridden
    by environment variables or command-line options.
    """
    # Configure logging based on verbosity
    # Note: basicConfig can only be called once effectively.
    # If module-level basicConfig was already called, this might show a message or need adjustment.
    # For simplicity, we'll reconfigure here if cli is the main entry.
    # A more robust way is to get the root logger and set its level and handlers.
    log_level = logging.DEBUG if verbose_opt else logging.INFO
    
    # Get root logger and remove existing handlers to avoid duplicate messages if basicConfig was called before
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up new basicConfig
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__) # Get logger for this module

    # cli_print_fn removed

    if os.path.exists(env_file) and os.path.isfile(env_file):
        load_dotenv(env_file, override=True)
        logger.info(f"Environment variables loaded from {env_file}")
        
        # Debug output for environment variables
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
            logger.debug(f"OPENAI_API_KEY loaded from .env: {masked_key}")
        else:
            logger.warning("OPENAI_API_KEY not found in environment after loading .env")
            
        logger.debug("Relevant environment variables after loading .env:")
        for key, value in os.environ.items():
            if key.startswith("OPENAI") or key.startswith("QDRANT"):
                masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                logger.debug(f"  {key}={masked_value}")
    else:
        if env_file != ".env": # User specified a file but it wasn't found
            logger.warning(f"Specified .env file '{env_file}' not found. Proceeding without it.")
        else: # Default .env not found, which is fine, just inform if verbose
            logger.debug("Default .env file not found or not specified. Relying on environment or CLI options for API keys.")

    success = perform_ingestion(
        data_dir=data_dir_opt,
        source_url=source_url_opt,
        collection_name_arg=collection_name_opt,
        persist_dir_arg=persist_dir_opt,
        openai_api_key_arg=openai_api_key_opt, 
        openai_model_embedding_arg=openai_model_embedding_opt,
        openai_model_llm_arg=openai_model_llm_opt,
        qdrant_url_arg=qdrant_url_opt,
        qdrant_api_key_arg=qdrant_api_key_opt, 
        chunk_size=chunk_size_opt,
        chunk_overlap=chunk_overlap_opt,
    # embedding_dim_func no longer passed
        num_workers_opt=num_workers_opt,
        max_workers_opt=max_workers_opt,
        min_workers_opt=min_workers_opt,
        adaptive_backoff_seconds_opt=adaptive_backoff_seconds_opt,
        adaptive_success_streak_opt=adaptive_success_streak_opt,
        verbose_mode=verbose_opt # Pass verbose_mode for LlamaIndex show_progress flags
        # print_fn parameter removed
    )

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    cli()
