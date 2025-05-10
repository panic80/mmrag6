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
import requests
import tempfile # May not be needed for URL handling if Trafilatura is used directly
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.web import BeautifulSoupWebReader # Changed to BeautifulSoupWebReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models as qdrant_models


# Global flag for verbose output
_VERBOSE_OUTPUT = False

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
    "data_dir",
    required=False, # Will be made effectively required by logic below
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help="Directory containing the documents to ingest. Mutually exclusive with --source-url."
)
@click.option(
    "--source-url",
    "source_url",
    type=str,
    default=None,
    help="URL of a single document to download and ingest. Mutually exclusive with --data-dir."
)
@click.option(
    "--collection-name",
    "collection_name",
    type=str,
    default=None,
    help="Optional name for the collection, creates a subdirectory in persist-dir."
)
@click.option(
    "--persist-dir",
    default="./storage_llamaindex_db",
    show_default=True,
    type=click.Path(file_okay=False, writable=True),
    help="Directory to persist the LlamaIndex storage context (index, docstore, etc.)."
)
@click.option(
    "--openai-api-key",
    envvar="OPENAI_API_KEY",
    default=None,
    help="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable."
)
@click.option(
    "--openai-model-embedding",
    default="text-embedding-3-large", # Changed as per user request
    show_default=True,
    help="OpenAI embedding model name."
)
@click.option(
    "--openai-model-llm",
    default="gpt-4.1-mini", # Changed as per user request
    show_default=True,
    help="OpenAI LLM model name (used by some LlamaIndex components if needed)."
)
@click.option(
    "--qdrant-url",
    default="http://qdrant:6333", # Default to service name in docker-compose
    show_default=True,
    envvar="QDRANT_URL", # Allow override from environment
    help="Qdrant server URL."
)
@click.option(
    "--qdrant-api-key",
    envvar="QDRANT_API_KEY",
    default=None,
    help="Qdrant API key (if required). Can also be set via QDRANT_API_KEY environment variable."
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output."
)
def cli(
    env_file: str,
    data_dir: str | None,
    source_url: str | None,
    collection_name: str | None,
    persist_dir: str,
    openai_api_key: str | None,
    openai_model_embedding: str,
    openai_model_llm: str,
    qdrant_url: str,
    qdrant_api_key: str | None,
    verbose: bool,
) -> None:
    """
    Ingests documents from DATA_DIR or a SOURCE_URL,
    builds a LlamaIndex VectorStoreIndex, and persists it.
    """
    global _VERBOSE_OUTPUT
    _VERBOSE_OUTPUT = verbose

    # Validate inputs: --data-dir and --source-url are mutually exclusive and one is required.
    if data_dir and source_url:
        click.echo("[fatal] --data-dir and --source-url are mutually exclusive. Please provide only one.", err=True)
        sys.exit(1)
    if not data_dir and not source_url:
        click.echo("[fatal] Either --data-dir or --source-url must be provided.", err=True)
        sys.exit(1)

    if os.path.exists(env_file) and os.path.isfile(env_file):
        load_dotenv(env_file, override=True) # Override allows CLI key to take precedence if also in .env
        if _VERBOSE_OUTPUT:
            click.echo(f"[info] Environment variables loaded from {env_file}")
    else:
        if env_file != ".env": # Only warn if a specific non-default file was given and not found
            click.echo(f"[warning] Specified .env file '{env_file}' not found. Proceeding without it.", err=True)
        elif _VERBOSE_OUTPUT:
            click.echo(f"[info] Default .env file not found or not specified. Relying on environment or CLI options for API keys.")


    # Resolve API key: CLI option > environment variable
    final_openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    final_qdrant_api_key = qdrant_api_key # Already resolved by Click from option or envvar

    if not final_openai_api_key:
        click.echo("[fatal] OPENAI_API_KEY is not set. Please provide it via --openai-api-key option or set the OPENAI_API_KEY environment variable.", err=True)
        sys.exit(1)

    # 1. Configure LlamaIndex Settings
    if _VERBOSE_OUTPUT:
        click.echo(f"[info] Configuring LlamaIndex Settings...")
        click.echo(f"  LLM Model: {openai_model_llm}")
        click.echo(f"  Embedding Model: {openai_model_embedding}")

    try:
        Settings.llm = OpenAI(model=openai_model_llm, api_key=final_openai_api_key)
        Settings.embed_model = OpenAIEmbedding(model=openai_model_embedding, api_key=final_openai_api_key)
    except Exception as e:
        click.echo(f"[fatal] Failed to initialize OpenAI models for LlamaIndex: {e}", err=True)
        sys.exit(1)

    # 2. Determine data source and load documents
    documents = [] # Initialize documents list

    if source_url:
        if _VERBOSE_OUTPUT:
            click.echo(f"[info] Loading documents directly from URL using BeautifulSoupWebReader: {source_url}")
        try:
            # BeautifulSoupWebReader will parse HTML and extract text content.
            # We can customize parsing if needed, but default often works well.
            loader = BeautifulSoupWebReader() 
            loaded_docs = loader.load_data(urls=[source_url])
            if not loaded_docs:
                click.echo(f"[warning] No documents loaded from URL '{source_url}' using BeautifulSoupWebReader.", err=True)
            else:
                documents.extend(loaded_docs)
                click.echo(f"[info] Loaded {len(loaded_docs)} document(s) from URL '{source_url}'.")
        except Exception as e:
            click.echo(f"[fatal] Failed to load documents from URL '{source_url}': {e}", err=True)
            sys.exit(1)
    elif data_dir:
        if _VERBOSE_OUTPUT:
            click.echo(f"[info] Loading documents from local directory: {data_dir}")
        try:
            # SimpleDirectoryReader is generally good for diverse types.
            # For specific needs with CSV/XLSX, dedicated readers might be used in a more complex setup.
            reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
            loaded_docs = reader.load_data(show_progress=_VERBOSE_OUTPUT)
            if not loaded_docs:
                click.echo(f"[warning] No documents found or loaded from '{data_dir}'.", err=True)
            else:
                documents.extend(loaded_docs)
                click.echo(f"[info] Loaded {len(loaded_docs)} document(s) from directory '{data_dir}'.")
        except Exception as e:
            click.echo(f"[fatal] Failed to load documents from '{data_dir}': {e}", err=True)
            sys.exit(1)
    else:
        click.echo("[fatal] No data source specified (this should not happen).", err=True)
        sys.exit(1)

    if not documents:
        click.echo("[warning] No documents were loaded. Exiting without building an index.", err=True)
        sys.exit(0)

    # 3. Configure and run Ingestion Pipeline
    final_persist_path = Path(persist_dir)
    if collection_name:
        final_persist_path = final_persist_path / collection_name.strip()

    if _VERBOSE_OUTPUT:
        click.echo(f"[info] Configuring ingestion pipeline...")

    # 3.1 Configure Node Parser (Chunking)
    node_parser = SentenceSplitter(
        chunk_size=2048,  # Increased chunk_size
        chunk_overlap=50, # Reduced chunk_overlap
    )

    # 3.2 Configure Metadata Extractors (ensure Settings.llm is already configured)
    # title_extractor = TitleExtractor(nodes=5, llm=Settings.llm) 
    # keyword_extractor = KeywordExtractor(keywords=10, llm=Settings.llm)

    # metadata_extractors_list = [title_extractor, keyword_extractor]
    # For this specific content, let's try without metadata extractors to focus on raw text.
    metadata_extractors_list = []
    
    # To add SummaryExtractor (generates a summary for each node using the LLM):
    # from llama_index.core.extractors import SummaryExtractor
    # summary_extractor = SummaryExtractor(llm=Settings.llm, summaries=["self"])
    # metadata_extractors_list.append(summary_extractor)

    # 3.3 Create Ingestion Pipeline
    transformations = [node_parser]
    if metadata_extractors_list: # Only add if not empty
        transformations.extend(metadata_extractors_list)
        
    pipeline = IngestionPipeline(
        transformations=transformations
    )

    # 3.4 Run documents through the pipeline to get nodes
    if _VERBOSE_OUTPUT:
        click.echo("[info] Running documents through ingestion pipeline (chunking, metadata extraction)...")
    nodes = pipeline.run(documents=documents, show_progress=_VERBOSE_OUTPUT)

    if not nodes:
        click.echo("[warning] No nodes were generated by the ingestion pipeline. Exiting.", err=True)
        sys.exit(0)
        
    if _VERBOSE_OUTPUT:
        click.echo(f"[info] Generated {len(nodes)} nodes from documents.")


    # 4. Build and persist the index from nodes
    if _VERBOSE_OUTPUT:
        click.echo(f"[info] Building VectorStoreIndex from processed nodes. This may take a while...")
        click.echo(f"[info] Index will be persisted to: {final_persist_path}")

    if _VERBOSE_OUTPUT:
        try:
            abs_final_persist_path = final_persist_path.resolve() # Get absolute path
            click.echo(f"[debug] final_persist_path: {str(final_persist_path)}")
            click.echo(f"[debug] Absolute final_persist_path: {str(abs_final_persist_path)}")
            click.echo(f"[debug] Parent of final_persist_path: {str(final_persist_path.parent)}")
            click.echo(f"[debug] Does parent exist? {final_persist_path.parent.exists()}")
            click.echo(f"[debug] Is parent a directory? {final_persist_path.parent.is_dir()}")
        except Exception as e_debug_path:
            click.echo(f"[debug] Error during path debugging: {e_debug_path}")

    try:
        if _VERBOSE_OUTPUT:
            click.echo(f"[debug] Attempting os.makedirs for: {str(final_persist_path)}")
        os.makedirs(final_persist_path, exist_ok=True)
        
        if _VERBOSE_OUTPUT:
            click.echo(f"[debug] After os.makedirs:")
            click.echo(f"[debug] Does final_persist_path exist? {final_persist_path.exists()}")
            click.echo(f"[debug] Is final_persist_path a directory? {final_persist_path.is_dir()}")
            if final_persist_path.exists() and final_persist_path.is_dir():
                # Check writability by trying to create a temporary file
                temp_file_test = final_persist_path / ".writable_test"
                try:
                    with open(temp_file_test, "w") as f_test:
                        f_test.write("test")
                    os.remove(temp_file_test)
                    click.echo(f"[debug] Is final_persist_path writable? Yes")
                except Exception as e_write:
                    click.echo(f"[debug] Is final_persist_path writable? No - {e_write}")
            else:
                click.echo(f"[debug] final_persist_path is not a directory or does not exist after makedirs.")

        # Initialize Qdrant client
        if _VERBOSE_OUTPUT:
            click.echo(f"[info] Initializing Qdrant client for URL: {qdrant_url}")
        q_client = QdrantClient(url=qdrant_url, api_key=final_qdrant_api_key) # Renamed to q_client

        # Determine embedding dimension based on model name
        # OpenAI text-embedding-3-small is 1536, text-embedding-3-large is 3072, ada-002 is 1536
        embedding_dim = 1536 # Default, common for ada-002 or small
        if "text-embedding-3-large" in openai_model_embedding:
            embedding_dim = 3072
        elif "text-embedding-3-small" in openai_model_embedding:
            embedding_dim = 1536
        
        if _VERBOSE_OUTPUT:
            click.echo(f"[info] Determined embedding dimension: {embedding_dim} for model {openai_model_embedding}")

        # Ensure collection_name is set (it's an option with no default in CLI, but Qdrant needs it)
        effective_collection_name = collection_name
        if not effective_collection_name:
            # Fallback to a default if not provided, though CLI option should be made required or have a default for Qdrant
            effective_collection_name = "llamaindex_default_collection" 
            click.echo(f"[warning] --collection-name not provided, using default Qdrant collection name: '{effective_collection_name}'. It's recommended to specify one.", err=True)
        
        # Create Qdrant collection if it doesn't exist
        try:
            q_client.get_collection(collection_name=effective_collection_name)
            if _VERBOSE_OUTPUT:
                click.echo(f"[info] Using existing Qdrant collection: {effective_collection_name}")
        except Exception: # Catches specific Qdrant exception for non-existent collection
            if _VERBOSE_OUTPUT:
                click.echo(f"[info] Qdrant collection '{effective_collection_name}' not found. Creating new collection...")
            q_client.create_collection(
                collection_name=effective_collection_name,
                vectors_config=qdrant_models.VectorParams(size=embedding_dim, distance=qdrant_models.Distance.COSINE)
            )
            if _VERBOSE_OUTPUT:
                click.echo(f"[info] Created Qdrant collection '{effective_collection_name}' with {embedding_dim} dimensions and Cosine distance.")

        vector_store = QdrantVectorStore(client=q_client, collection_name=effective_collection_name)
        
        # StorageContext will use Qdrant for vector store, and simple file store for docstore/indexstore by default
        # if persist_dir is provided for those.
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context, 
            show_progress=_VERBOSE_OUTPUT,
        )
        
        # Persist the non-vector store components (docstore, index_store) to the local filesystem
        # if a persist_dir is specified. This is useful for BM25 or other retrievers
        # that might need access to the raw text or node structure locally.
        # The vector embeddings themselves are already in Qdrant.
        if persist_dir and final_persist_path.exists() and final_persist_path.is_dir():
            if _VERBOSE_OUTPUT:
                click.echo(f"[info] Persisting non-vector components (docstore, index_store) to: {str(final_persist_path)}")
            # This persist call will save the docstore, index_store, etc., to final_persist_path.
            # The vector_store (Qdrant) part of the storage_context doesn't get "persisted" here in the same way;
            # it's already live in the Qdrant service.
            index.storage_context.persist(persist_dir=str(final_persist_path))
            click.secho(f"\n[success] Ingestion completed. Vectors stored in Qdrant collection '{effective_collection_name}'.\n"
                        f"Docstore and index metadata persisted to '{final_persist_path}'.", fg="green")
        else:
            click.secho(f"\n[success] Ingestion completed. Vectors stored in Qdrant collection '{effective_collection_name}'.\n"
                        f"Local persistence of docstore/index_store skipped (no valid persist_dir provided or directory issue).", fg="green")

    except Exception as e:
        click.echo(f"[fatal] An error occurred during index building or Qdrant interaction: {e}", err=True)
        if _VERBOSE_OUTPUT: # Print traceback if verbose
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    cli()
