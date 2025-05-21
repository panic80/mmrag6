# MMRag

## Introduction

`MMRag` is a Flask-based server that provides Retrieval Augmented Generation (RAG) capabilities. It integrates seamlessly with Mattermost via slash commands, allowing users to ingest data into a vector store and query it using natural language. The system leverages Qdrant for vector storage, LlamaIndex for RAG pipeline orchestration, and OpenAI models for language understanding and generation.

## Core Features

*   `/inject` (or `/injest`) Command:**
    *   Ingests data from various sources:
        *   Current Mattermost channel history (if no other source is specified).
        *   URLs.
        *   Local files or directories.
    *   Stores processed data into a specified Qdrant vector collection.
    *   Supports dynamic collection naming (e.g., `/inject --collection-name my_data ...`).
    *   Allows purging and recreating collections before ingestion using the `--purge` flag.
    *   Provides real-time, threaded feedback to the Mattermost channel during the ingestion process.

*   `/ask` Command:**
    *   Queries the RAG system using a Qdrant collection (typically configured via the `QDRANT_COLLECTION_NAME` environment variable).
    *   Utilizes LlamaIndex and OpenAI models (e.g., `gpt-4.1-mini` for generation, `text-embedding-3-large` for embeddings) to process queries and generate relevant answers.
    *   Employs a hybrid retrieval mechanism by default (combining dense vector search with BM25 sparse search) for improved relevance, which has been verified to function as intended when `RETRIEVAL_USE_HYBRID` is enabled.
    *   Supports advanced retrieval options, such as disabling Maximal Marginal Relevance (MMR) with `--no-mmr` (MMR is enabled by default).
    *   Posts formatted answers back to the originating Mattermost channel.

*   **Health Check:**
    *   `/health` endpoint returns `{"status": "ok"}`.
    *   A GET request to the root `/` path also serves as a basic health check.

## Technologies Used

*   **Backend:** Python, Flask
*   **RAG & Vector Store:** LlamaIndex, Qdrant
*   **LLM & Embeddings:** OpenAI (GPT series models, text embedding models)
*   **Containerization:** Docker, Docker Compose
*   **Mattermost Integration:** Slash commands, Mattermost REST API

## Prerequisites

*   Python (3.9+ recommended)
*   Docker and Docker Compose
*   An active OpenAI API key
*   A **mandatory and accessible** Qdrant instance (version 1.7.x or compatible). Qdrant is essential for storing and retrieving document embeddings. The system will not operate with an in-memory fallback.
*   A Mattermost instance configured for slash command integration.

## Setup and Configuration

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd MMRag
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Create a `.env` file in the `MMRag` directory or set these variables in your environment:

    *   `OPENAI_API_KEY`: **Required.** Your OpenAI API key.
    *   `QDRANT_URL`: URL of your Qdrant instance (e.g., `http://localhost:6333` for local, `http://qdrant:6333` when using the provided Docker Compose setup).
    *   `QDRANT_API_KEY`: (Optional) API key for Qdrant if authentication is enabled on your instance.
    *   `QDRANT_COLLECTION_NAME`: Default Qdrant collection name for `/ask` and `/inject` if not specified (e.g., `rag_llamaindex_data`).
    *   `SLASH_TOKEN`: **Required.** Generic Mattermost slash command token.
    *   `SLASH_TOKEN_INJECT`: (Optional) Specific token for the `/inject` command. Overrides `SLASH_TOKEN` if set.
    *   `SLASH_TOKEN_ASK`: (Optional) Specific token for the `/ask` command. Overrides `SLASH_TOKEN` if set.
    *   `MATTERMOST_URL`: **Required.** Your Mattermost server URL (e.g., `https://your.mattermost.server.com`).
    *   `MATTERMOST_TOKEN`: **Required.** A Mattermost Personal Access Token with permissions to post messages. This token is crucial for all asynchronous message delivery from `/ask` and `/inject` via the Mattermost REST API. If API calls using this token fail, messages will not be delivered, and errors will be logged on the server. The system does not use `response_url` as a fallback.
    *   `OPENAI_MODEL_LLM`: (Optional) OpenAI model for language generation (defaults to `gpt-4.1-mini`).
    *   `OPENAI_MODEL_EMBEDDING`: (Optional) OpenAI model for text embeddings (defaults to `text-embedding-3-large`). Ensure `config.py`'s `get_embedding_dim` function supports your chosen model, or ingestion will fail.
    *   `DEFAULT_VECTOR_SIZE`: (Optional) Vector size for new Qdrant collections if created by the system (defaults to `3072`, matching `text-embedding-3-large`). This is determined by `get_embedding_dim` in `config.py` based on the embedding model.
    *   `RETRIEVAL_SIMILARITY_TOP_K`: (Optional) Number of top similar documents to retrieve from dense (vector) search. Defaults to 25.
    *   `RETRIEVAL_USE_HYBRID`: (Optional) Set to `True` or `False` to enable/disable hybrid search (default is `True`).
    *   `RETRIEVAL_SPARSE_TOP_K`: (Optional) Number of top documents for sparse retrieval (BM25) in hybrid search. Defaults to 7.
    *   `RETRIEVAL_RERANK_TOP_N`: (Optional) Number of documents to rerank using a cross-encoder after initial retrieval. Defaults to 20 (0 disables reranking).
    *   `RETRIEVAL_MMR_LAMBDA`: (Optional) Lambda value for Maximal Marginal Relevance (MMR) re-ranking (default 0.7).
    *   `RETRIEVAL_COMPRESS_CONTEXT`: (Optional) Set to `True` to enable context compression (requires Cohere API key for CohereRerank).
    *   `COHERE_API_KEY`: (Optional) Cohere API key, required if using CohereRerank for context compression.
    *   *Note on Retrieval Parameters*: These `RETRIEVAL_*` values have been tuned for a general balance of performance and relevance. You may need to adjust them based on your specific dataset and query patterns.

4.  **Interactive Environment Setup Script:**
    The [`configure_env.py`](configure_env.py:1) script is an interactive command-line tool designed to help users set up the necessary environment variables for the `MMRag` application by generating a `.env` file. It simplifies the configuration process, reducing errors by guiding users through prompts for all required settings.

    **How to Use:**

    1.  Navigate to the project's root directory in your terminal.
    2.  Run the script using the command:
        ```bash
        python configure_env.py
        ```
    3.  The script will interactively ask for each configuration value. It will display prompts and default values where applicable.
    4.  Enter your desired values or press Enter to accept defaults.
    5.  Upon completion, the script will create or overwrite a `.env` file in the current working directory. This file will be used by the `MMRag` application to load its configuration.

## Running the Application

*   **Directly (for development):**
    ```bash
    python server.py
    ```
    Ensure Qdrant is accessible and environment variables are set.

*   **With Docker Compose (recommended for production-like setup):**
    The `docker-compose.yml` file typically defines the `MMRag` application service and a Qdrant service.
    ```bash
    docker-compose up --build -d
    ```
    To view logs:
    ```bash
    docker-compose logs -f
    ```

## Usage - Mattermost Slash Commands

### `/inject [options] [source1 source2 ...]`

*   **Purpose:** Ingests data into the RAG system. If no source is provided, it attempts to ingest the history of the current Mattermost channel.
*   **Options:**
    *   `--collection-name <name>` or `-c <name>`: Specifies the Qdrant collection to use or create.
    *   `--purge`: Deletes and recreates the specified collection before ingesting new data.
    *   `--no-rich-metadata`: Disables rich metadata extraction during ingestion.
    *   Other arguments supported by `ingest_llamaindex.py` may be passed through.
*   **Sources:** Can be URLs, local file paths, or directory paths accessible by the server.
*   **Examples:**
    *   `/inject`
        *(Ingests current Mattermost channel history into the default collection)*
    *   `/inject --collection-name project_docs https://example.com/api_docs ./project_notes/`
        *(Ingests from a URL and a local directory into the `project_docs` collection)*
    *   `/inject --collection-name old_data --purge`
        *(Clears and recreates the `old_data` collection. No new data is ingested unless sources follow)*

### `/ask [options] <your query text>`

*   **Purpose:** Asks a question to the RAG system. The query is processed against the default collection (set by `QDRANT_COLLECTION_NAME`) or the collection specified during the last relevant `/inject` for that channel (behavior might vary based on exact server logic for context).
*   **Options:**
    *   `--no-mmr`: Disables Maximal Marginal Relevance for the retrieval step.
    *   `--mmr`: Explicitly enables MMR (this is the default behavior).
*   **Examples:**
    *   `/ask What are the main features of MMRag?`
    *   `/ask --no-mmr Summarize the discussion about the Q3 roadmap.`

## Project Structure Overview

*   `server.py`: The main Flask application; handles incoming Mattermost slash command requests and orchestrates ingestion/querying.
*   `ingest_llamaindex.py`: Python script containing the core logic for data ingestion, processing, and storage into Qdrant. It requires a functional Qdrant instance; it will not fall back to an in-memory store.
*   `query_llamaindex.py`: Python script that handles the RAG querying process, including retrieval, synthesis, and interfacing with OpenAI. This script also strictly requires Qdrant for querying.
*   `requirements.txt`: A list of Python package dependencies for the project. Some features depend on packages not included in the base `requirements.txt` (e.g., specific rerankers, web readers).
*   `Dockerfile`: Instructions for building the Docker image for the `MMRag` application.
*   `docker-compose.yml`: Docker Compose configuration to run the `MMRag` application and dependent services (like Qdrant) together.
*   `storage_llamaindex_db/`: Default local directory where LlamaIndex stores document metadata (docstore) and index store information. Qdrant is used exclusively for vector storage.

## Important Considerations & Error Handling

*   **Qdrant is Mandatory**: Both ingestion (`/inject`) and querying (`/ask`) operations strictly require a running and accessible Qdrant instance configured via `QDRANT_URL`. If Qdrant is unavailable or the specified collection cannot be accessed/created, operations will fail with logged errors.
*   **Missing Python Packages for Features**:
    *   **URL Ingestion**: To use `/inject <URL>`, you must have `llama-index-readers-web` (which provides `SimpleWebPageReader`) and its dependencies (like `beautifulsoup4`) installed. If these are missing, attempting URL ingestion will result in a Python error (e.g., `ImportError` or `NameError`) and the request will fail. The `server.py` script's `check_url_dependencies()` function only prints warnings at startup for some related packages (`langchain-community`, etc.) but does not prevent runtime errors if `SimpleWebPageReader` itself is unavailable.
    *   **Advanced Retrieval Components**:
        *   **BM25/Hybrid Search**: For hybrid search functionality using BM25 (enabled by default via `RETRIEVAL_USE_HYBRID=True`), the `llama-index-retrievers-bm25` package must be installed. You can typically install this via `pip install llama-index-retrievers-bm25` or as part of a broader optional group like `pip install llama-index[bm25]`. If this package is missing, `query_llamaindex.py` will fail with an `ImportError` when attempting to initialize `BM25Retriever`.
        *   **Rerankers**: The querying script (`query_llamaindex.py`) may be configured to use components like `CohereRerank` or `SentenceTransformerRerank`. If the Python packages for these components (e.g., `llama-index-postprocessor-cohere`, `llama-index-postprocessor-sbert-rerank`) are not installed, any attempt to use them (e.g., by setting a non-zero `RETRIEVAL_RERANK_TOP_N` or enabling context compression with Cohere) will lead to an `ImportError`, causing the `/ask` request to fail.
*   **Mattermost API Communication**: All asynchronous messages and updates to Mattermost (e.g., from `/ask` responses, `/inject` progress) are sent via the Mattermost REST API using the `MATTERMOST_TOKEN`. If these API calls fail (e.g., invalid token, network issues), errors will be logged on the server, and messages will not be delivered. There is no fallback to using `response_url` for these posts.
*   **OpenAI Model Configuration**: Ensure that the embedding model specified (e.g., `OPENAI_MODEL_EMBEDDING`) is recognized by the `get_embedding_dim` function in `config.py`. If an unknown model is used, `ingest_llamaindex.py` will fail during vector store initialization due to a `ValueError`.

## Development Notes

*   Logging is configured within the Flask application. Check server logs for detailed information and troubleshooting, especially for issues related to Qdrant connectivity, missing packages, or Mattermost API errors.
*   When running with Docker Compose, ensure the Qdrant service is healthy and accessible to the application service as per its configuration in `docker-compose.yml` and the application's environment variables.
