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
*   Access to a Qdrant instance (can be run locally via Docker Compose or a remote instance)
*   A Mattermost instance configured for slash command integration

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
    *   `MATTERMOST_TOKEN`: **Required.** A Mattermost Personal Access Token with permissions to post messages. Used for asynchronous replies and threaded updates.
    *   `OPENAI_MODEL_LLM`: (Optional) OpenAI model for language generation (defaults to `gpt-4.1-mini`).
    *   `OPENAI_MODEL_EMBEDDING`: (Optional) OpenAI model for text embeddings (defaults to `text-embedding-3-large`).
    *   `DEFAULT_VECTOR_SIZE`: (Optional) Vector size for new Qdrant collections if created by the system (defaults to `3072`, matching `text-embedding-3-large`).
    *   `RETRIEVAL_SIMILARITY_TOP_K`: (Optional) Number of top similar documents to retrieve initially.
    *   `RETRIEVAL_USE_HYBRID`: (Optional) Set to `True` or `False` to enable/disable hybrid search (default is `True`).
    *   `RETRIEVAL_SPARSE_TOP_K`: (Optional) Number of top documents for sparse retrieval in hybrid search.
    *   `RETRIEVAL_RERANK_TOP_N`: (Optional) Number of documents to rerank (0 disables reranking).
    *   `RETRIEVAL_MMR_LAMBDA`: (Optional) Lambda value for MMR (default 0.5).
    *   `RETRIEVAL_COMPRESS_CONTEXT`: (Optional) Set to `True` to enable context compression.
    *   `COHERE_API_KEY`: (Optional) Cohere API key, required if using CohereRerank for context compression.

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
*   `ingest_llamaindex.py`: Python script containing the core logic for data ingestion, processing, and storage into Qdrant.
*   `query_llamaindex.py`: Python script that handles the RAG querying process, including retrieval, synthesis, and interfacing with OpenAI.
*   `requirements.txt`: A list of Python package dependencies for the project.
*   `Dockerfile`: Instructions for building the Docker image for the `MMRag` application.
*   `docker-compose.yml`: Docker Compose configuration to run the `MMRag` application and dependent services (like Qdrant) together.
*   `storage_llamaindex_db/`: Default local directory where LlamaIndex might store document metadata or local components of the index if not solely relying on Qdrant for all parts.

## Development Notes

*   The `server.py` script includes a `check_url_dependencies()` function that verifies if optional packages for URL handling (`langchain-community`, `langchain`, `unstructured`, `bs4`) are installed. These are needed for the `/inject <URL>` functionality.
*   Logging is configured within the Flask application. Check server logs for detailed information and troubleshooting.
*   When running with Docker Compose, ensure the Qdrant service is healthy and accessible to the application service.
