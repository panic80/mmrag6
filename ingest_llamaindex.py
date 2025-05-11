#!/usr/bin/env python3
"""
ingest_llamaindex.py

A simplified CLI utility for building a basic LlamaIndex vector store.
It loads documents from a specified directory, uses OpenAI for embeddings
and LLM (if needed by components), builds a VectorStoreIndex, and persists
it to a local directory.
"""

import os
import sys
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
from llama_index.readers.web import BeautifulSoupWebReader
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

_VERBOSE_OUTPUT = False

@dataclasses.dataclass
class WorkerResp:
    """Response from a worker process."""
    doc_id: Optional[str]
    nodes: Optional[List[Document]] = None
    success: bool = False
    rate_limited: bool = False
    exception_info: Optional[str] = None
    original_doc_dict: Optional[dict] = None # For re-queueing

# Helper function for ProcessPoolExecutor, must be defined at the top level for pickling
def _process_single_document(config_tuple: Tuple) -> WorkerResp:
    """
    Processes a single document dictionary through the LlamaIndex ingestion pipeline.
    Designed to be used with ProcessPoolExecutor.
    """
    doc_dict, chunk_size, chunk_overlap, openai_api_key, openai_model_llm, openai_model_embedding = config_tuple
    doc_id_for_error = doc_dict.get("id_", doc_dict.get("doc_id", "unknown_id"))

    try:
        # Re-initialize settings per process
        Settings.llm = OpenAI(model=openai_model_llm, api_key=openai_api_key)
        Settings.embed_model = OpenAIEmbedding(model=openai_model_embedding, api_key=openai_api_key)
    except Exception as e:
        # Cannot use click.echo here as it's not available in the worker process's context easily
        err_msg = f"Failed to initialize OpenAI models: {e}"
        print(f"[worker-error] DocID {doc_id_for_error}: {err_msg}", file=sys.stderr)
        return WorkerResp(doc_id=doc_id_for_error, success=False, exception_info=err_msg, original_doc_dict=doc_dict)

    doc = Document(**doc_dict)
    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    pipeline = IngestionPipeline(transformations=[node_parser])
    
    try:
        processed_nodes = pipeline.run(documents=[doc])
        # print(f"[worker-debug] DocID {doc.doc_id}: Processed, generated {len(processed_nodes)} nodes.", file=sys.stderr)
        return WorkerResp(doc_id=doc.doc_id, nodes=processed_nodes, success=True, original_doc_dict=doc_dict)
    except openai.RateLimitError as rle:
        # print(f"[worker-info] DocID {doc.doc_id}: Rate limit error encountered.", file=sys.stderr)
        return WorkerResp(doc_id=doc.doc_id, success=False, rate_limited=True, exception_info=str(rle), original_doc_dict=doc_dict)
    except Exception as e:
        err_msg = f"Failed to process document: {e}"
        print(f"[worker-error] DocID {doc.doc_id}: {err_msg}", file=sys.stderr)
        # import traceback # Avoid importing traceback in worker unless absolutely necessary for debugging
        # traceback.print_exc(file=sys.stderr)
        return WorkerResp(doc_id=doc.doc_id, success=False, exception_info=err_msg, original_doc_dict=doc_dict)

def _log(message: str, print_fn: Callable[[str, bool, bool], None], error: bool = False, debug: bool = False):
    """Helper function for logging based on verbosity, using a provided print_fn."""
    print_fn(message, error, debug)

def perform_ingestion(
    data_dir: Optional[str],
    source_url: Optional[str],
    collection_name_arg: Optional[str],
    persist_dir_arg: str,
    openai_api_key_arg: Optional[str],
    openai_model_embedding_arg: str,
    openai_model_llm_arg: str,
    qdrant_url_arg: str,
    qdrant_api_key_arg: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    embedding_dim_func: Callable[[str], int],
    num_workers_opt: int,
    max_workers_opt: int,
    min_workers_opt: int,
    adaptive_backoff_seconds_opt: int,
    adaptive_success_streak_opt: int,
    verbose_mode: bool,
    print_fn: Callable[[str, bool, bool], None]
) -> bool:
    """
    Core logic for ingesting documents, building a LlamaIndex VectorStoreIndex,
    and persisting it to Qdrant and local filesystem.

    Returns True if ingestion was successful, False otherwise.
    """
    def current_log(message: str, error: bool = False, debug: bool = False):
        print_fn(message, error, debug)

    if data_dir and source_url:
        current_log("--data-dir and --source-url are mutually exclusive. Please provide only one.", error=True)
        return False
    if not data_dir and not source_url:
        current_log("Either --data-dir or --source-url must be provided.", error=True)
        return False

    final_openai_api_key = openai_api_key_arg or CFG_OPENAI_API_KEY
    final_qdrant_api_key = qdrant_api_key_arg or CFG_QDRANT_API_KEY
    final_openai_model_embedding = openai_model_embedding_arg
    final_openai_model_llm = openai_model_llm_arg
    final_qdrant_url = qdrant_url_arg
    final_persist_dir = persist_dir_arg
    
    effective_collection_name = collection_name_arg or CFG_INGEST_DEFAULT_COLLECTION_NAME
    if not collection_name_arg and verbose_mode:
        current_log(f"No collection name provided, using default: '{effective_collection_name}'")

    if not final_openai_api_key:
        current_log("OPENAI_API_KEY is not set. Please provide it via --openai-api-key option, set the OPENAI_API_KEY environment variable, or in config.py.", error=True)
        return False

    current_log(f"Configuring LlamaIndex Settings...")
    current_log(f"  LLM Model: {final_openai_model_llm}")
    current_log(f"  Embedding Model: {final_openai_model_embedding}")

    try:
        Settings.llm = OpenAI(model=final_openai_model_llm, api_key=final_openai_api_key)
        Settings.embed_model = OpenAIEmbedding(model=final_openai_model_embedding, api_key=final_openai_api_key)
    except Exception as e:
        current_log(f"Failed to initialize OpenAI models for LlamaIndex: {e}", error=True)
        return False

    documents: List[Document] = [] 

    if source_url:
        current_log(f"Loading documents directly from URL using BeautifulSoupWebReader: {source_url}")
        try:
            loader = BeautifulSoupWebReader()
            loaded_docs = loader.load_data(urls=[source_url])
            if not loaded_docs:
                current_log(f"[warning] No documents loaded from URL '{source_url}' using BeautifulSoupWebReader.")
            else:
                documents.extend(loaded_docs)
                current_log(f"Loaded {len(loaded_docs)} document(s) from URL '{source_url}'.")
        except Exception as e:
            current_log(f"Failed to load documents from URL '{source_url}': {e}", error=True)
            return False
    elif data_dir:
        current_log(f"Loading documents from local directory: {data_dir}")
        try:
            reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
            loaded_docs = reader.load_data(show_progress=verbose_mode) 
            if not loaded_docs:
                current_log(f"[warning] No documents found or loaded from '{data_dir}'.")
            else:
                documents.extend(loaded_docs)
                current_log(f"Loaded {len(loaded_docs)} document(s) from directory '{data_dir}'.")
        except Exception as e:
            current_log(f"Failed to load documents from '{data_dir}': {e}", error=True)
            return False
    
    if not documents:
        current_log("[warning] No documents were loaded. Exiting without building an index.")
        return True

    final_persist_path = Path(final_persist_dir)
    local_docstore_persist_path = final_persist_path / effective_collection_name.strip()

    current_log(f"Configuring ingestion pipeline...")

    current_log("Preparing for document ingestion pipeline (chunking)...")
    all_processed_nodes: List[Document] = []

    if num_workers_opt <= 1:
        current_log("Processing documents sequentially.")
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pipeline = IngestionPipeline(transformations=[node_parser])
        all_processed_nodes = pipeline.run(documents=documents, show_progress=verbose_mode)
    else:
        # Adaptive parallel processing
        current_log(f"Starting adaptive parallel processing: initial_workers={num_workers_opt}, max_workers={max_workers_opt}, min_workers={min_workers_opt}")

        docs_to_process_deque = deque(documents)
        
        # Ensure min_workers is at least 1
        min_workers = max(1, min_workers_opt)
        # Ensure max_workers is at least min_workers
        max_workers = max(min_workers, max_workers_opt if max_workers_opt > 0 else os.cpu_count() or 1)
        # Ensure num_workers_opt is within [min_workers, max_workers]
        current_num_workers = max(min_workers, min(num_workers_opt, max_workers))

        current_log(f"Adjusted worker range: initial={current_num_workers}, min={min_workers}, max={max_workers}")

        successful_streak = 0
        cooldown_until_timestamp = 0
        total_docs_processed_count = 0
        total_rate_limit_hits = 0
        
        # Create the executor once
        with ProcessPoolExecutor(max_workers=max_workers) as executor: # Max_workers here defines the pool's capacity
            while docs_to_process_deque:
                if time.time() < cooldown_until_timestamp:
                    wait_time = cooldown_until_timestamp - time.time()
                    current_log(f"Cooling down due to rate limit. Waiting for {wait_time:.1f}s...")
                    time.sleep(wait_time)

                # Determine batch size based on current_num_workers
                batch_size = min(len(docs_to_process_deque), current_num_workers)
                current_batch_docs_dicts = []
                for _ in range(batch_size):
                    current_batch_docs_dicts.append(docs_to_process_deque.popleft().dict())
                
                current_log(f"Processing batch of {len(current_batch_docs_dicts)} documents with {current_num_workers} target workers.")

                tasks_for_executor = [
                    (doc_dict, chunk_size, chunk_overlap, final_openai_api_key, final_openai_model_llm, final_openai_model_embedding)
                    for doc_dict in current_batch_docs_dicts
                ]

                batch_results: List[WorkerResp] = []
                try:
                    # Submit current batch of tasks, actual parallelism depends on current_num_workers and executor's max_workers
                    # We are effectively using the executor to run 'batch_size' tasks, 
                    # and ProcessPoolExecutor will manage running them up to its 'max_workers' limit.
                    # For adaptive sizing, we control 'batch_size' which is 'current_num_workers'.
                    batch_results = list(executor.map(_process_single_document, tasks_for_executor, chunksize=1)) # chunksize=1 for faster feedback
                except Exception as e:
                    current_log(f"Critical error during executor.map: {e}", error=True)
                    # Re-queue documents from this failed batch
                    for doc_dict_failed in current_batch_docs_dicts:
                         docs_to_process_deque.append(Document(**doc_dict_failed)) # Reconstruct and add back
                    current_log(f"Re-queued {len(current_batch_docs_dicts)} documents from failed batch. Reducing workers and backing off.", error=True)
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
                        if resp.original_doc_dict: # Re-queue if rate limited
                            docs_to_process_deque.append(Document(**resp.original_doc_dict))
                    else: # Other errors
                        current_log(f"Failed to process doc {resp.doc_id}: {resp.exception_info}", error=True)
                        # Optionally re-queue for other errors if they might be transient
                        # For now, non-RL errors are not re-queued to avoid infinite loops on persistent errors.

                current_log(f"Batch summary: {num_successful_in_batch} succeeded, {num_rate_limited_in_batch} rate-limited.")

                if num_rate_limited_in_batch > 0:
                    current_log(f"Rate limit hit. Reducing workers from {current_num_workers} to {max(min_workers, current_num_workers // 2)} and backing off for {adaptive_backoff_seconds_opt}s.")
                    current_num_workers = max(min_workers, current_num_workers // 2)
                    cooldown_until_timestamp = time.time() + adaptive_backoff_seconds_opt
                    successful_streak = 0
                elif num_successful_in_batch == len(current_batch_docs_dicts): # All docs in batch succeeded
                    successful_streak += 1
                    if successful_streak >= adaptive_success_streak_opt and current_num_workers < max_workers:
                        current_log(f"Success streak of {successful_streak} reached. Increasing workers from {current_num_workers} to {current_num_workers + 1}.")
                        current_num_workers += 1
                        successful_streak = 0 # Reset streak after increase
                else: # Mixed results or non-RL failures, maintain current workers
                    successful_streak = 0
        
        current_log(f"Adaptive parallel processing finished. Total documents processed into nodes: {total_docs_processed_count} (initial: {len(documents)}). Total rate limit hits: {total_rate_limit_hits}.")

    if not all_processed_nodes:
        current_log("[warning] No nodes were generated by the ingestion pipeline. Exiting.")
        return True

    current_log(f"Generated {len(all_processed_nodes)} nodes from documents.")

    current_log(f"Building VectorStoreIndex from processed nodes. This may take a while...")
    current_log(f"Local docstore/index metadata will be persisted to: {local_docstore_persist_path}")
    current_log(f"Vectors will be stored in Qdrant collection: {effective_collection_name} at {qdrant_url_arg}")

    try:
        current_log(f"Attempting os.makedirs for: {str(local_docstore_persist_path)}", debug=True)
        os.makedirs(local_docstore_persist_path, exist_ok=True)
        current_log(f"After os.makedirs: Exists? {local_docstore_persist_path.exists()}, IsDir? {local_docstore_persist_path.is_dir()}", debug=True)

        current_log(f"Initializing Qdrant client for URL: {qdrant_url_arg}")
        q_client = QdrantClient(url=qdrant_url_arg, api_key=final_qdrant_api_key)

        embedding_dim = embedding_dim_func(final_openai_model_embedding)
        current_log(f"Determined embedding dimension: {embedding_dim} for model {final_openai_model_embedding}")

        try:
            q_client.get_collection(collection_name=effective_collection_name)
            current_log(f"Using existing Qdrant collection: {effective_collection_name}")
        except Exception: 
            current_log(f"Qdrant collection '{effective_collection_name}' not found. Creating new collection...")
            q_client.create_collection(
                collection_name=effective_collection_name,
                vectors_config=qdrant_models.VectorParams(size=embedding_dim, distance=qdrant_models.Distance.COSINE)
            )
            current_log(f"Created Qdrant collection '{effective_collection_name}' with {embedding_dim} dimensions and Cosine distance.")

        vector_store = QdrantVectorStore(client=q_client, collection_name=effective_collection_name)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex(
            nodes=all_processed_nodes,
            storage_context=storage_context,
            show_progress=verbose_mode,
        )
        
        if local_docstore_persist_path.exists() and local_docstore_persist_path.is_dir():
            current_log(f"Persisting non-vector components (docstore, index_store) to: {str(local_docstore_persist_path)}")
            index.storage_context.persist(persist_dir=str(local_docstore_persist_path))
            success_msg_part1 = f"[success] Ingestion completed. Vectors stored in Qdrant collection '{effective_collection_name}'."
            success_msg_part2 = f"Docstore and index metadata persisted to '{local_docstore_persist_path}'."
            current_log(f"\n{success_msg_part1}\n{success_msg_part2}")

        else:
            warn_msg_part1 = f"[success] Ingestion completed. Vectors stored in Qdrant collection '{effective_collection_name}'."
            warn_msg_part2 = f"[warning] Local persistence of docstore/index_store skipped (directory issue with '{local_docstore_persist_path}')."
            current_log(f"\n{warn_msg_part1}\n{warn_msg_part2}")
        return True

    except Exception as e:
        current_log(f"An error occurred during index building or Qdrant interaction: {e}", error=True)
        if verbose_mode:
            import traceback
            current_log(f"Traceback:\n{traceback.format_exc()}", debug=True)
        return False

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
    
    def cli_print_fn(message: str, error: bool = False, debug: bool = False):
        is_success = "[success]" in message
        is_warning = "[warning]" in message
        
        if error:
            click.secho(f"[fatal] {message}", fg="red", err=True)
        elif debug and verbose_opt:
            click.secho(f"[debug] {message}", fg="cyan")
        elif not debug and verbose_opt:
            click.secho(f"[info] {message}", fg="blue")
        elif not debug and not verbose_opt:
            if is_success:
                click.secho(message, fg="green")
            elif is_warning:
                click.secho(message, fg="yellow")
            elif message.startswith("Loaded") or message.startswith("Ingestion completed") or "Qdrant collection" in message:
                 click.echo(message)

    if os.path.exists(env_file) and os.path.isfile(env_file):
        load_dotenv(env_file, override=True) 
        cli_print_fn(f"Environment variables loaded from {env_file}")
    else:
        if env_file != ".env": 
            cli_print_fn(f"[warning] Specified .env file '{env_file}' not found. Proceeding without it.")
        elif verbose_opt:
            cli_print_fn(f"Default .env file not found or not specified. Relying on environment or CLI options for API keys.")

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
        embedding_dim_func=cfg_get_embedding_dim,
        num_workers_opt=num_workers_opt,
        max_workers_opt=max_workers_opt,
        min_workers_opt=min_workers_opt,
        adaptive_backoff_seconds_opt=adaptive_backoff_seconds_opt,
        adaptive_success_streak_opt=adaptive_success_streak_opt,
        verbose_mode=verbose_opt,
        print_fn=cli_print_fn
    )

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    cli()
