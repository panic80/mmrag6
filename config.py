import os

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL_EMBEDDING = os.environ.get("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large")
OPENAI_MODEL_LLM = os.environ.get("OPENAI_MODEL_LLM", "gpt-4.1-mini")

# --- Qdrant Configuration ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333") # Default for Docker Compose
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
DEFAULT_QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "rag_llamaindex_data")
DEFAULT_VECTOR_SIZE_EMBEDDING_LARGE = 3072 # For text-embedding-3-large
DEFAULT_VECTOR_SIZE_EMBEDDING_SMALL = 1536 # For text-embedding-3-small or ada-002
DEFAULT_VECTOR_SIZE = int(os.environ.get("DEFAULT_VECTOR_SIZE", DEFAULT_VECTOR_SIZE_EMBEDDING_LARGE))


# --- LlamaIndex Configuration ---
DEFAULT_PERSIST_DIR = "./storage_llamaindex_db"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128

# --- Server Specific Configuration (can be expanded) ---
MATTERMOST_URL = os.environ.get("MATTERMOST_URL")
MATTERMOST_TOKEN = os.environ.get("MATTERMOST_TOKEN")
SLASH_TOKEN = os.environ.get("SLASH_TOKEN")
SLASH_TOKEN_INJECT = os.environ.get("SLASH_TOKEN_INJECT", SLASH_TOKEN)
SLASH_TOKEN_ASK = os.environ.get("SLASH_TOKEN_ASK", SLASH_TOKEN)
SERVER_HOST = os.environ.get("HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("PORT", "5000"))

# --- Retrieval Configuration (defaults from query_llamaindex.py) ---
RETRIEVAL_SIMILARITY_TOP_K = int(os.environ.get("RETRIEVAL_SIMILARITY_TOP_K", 10))
RETRIEVAL_SPARSE_TOP_K = int(os.environ.get("RETRIEVAL_SPARSE_TOP_K", 10))
RETRIEVAL_RERANK_TOP_N = int(os.environ.get("RETRIEVAL_RERANK_TOP_N", 0)) # 0 to disable
RETRIEVAL_RERANKER_MODEL = os.environ.get("RETRIEVAL_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RETRIEVAL_USE_HYBRID = os.environ.get("RETRIEVAL_USE_HYBRID", "True").lower() == "true"
RETRIEVAL_USE_MMR = os.environ.get("RETRIEVAL_USE_MMR", "False").lower() == "false"
RETRIEVAL_MMR_LAMBDA = float(os.environ.get("RETRIEVAL_MMR_LAMBDA", 0.5))
RETRIEVAL_USE_QUERY_EXPANSION = os.environ.get("RETRIEVAL_USE_QUERY_EXPANSION", "True").lower() == "true"
RETRIEVAL_MAX_EXPANSIONS = int(os.environ.get("RETRIEVAL_MAX_EXPANSIONS", 3))
RETRIEVAL_COMPRESS_CONTEXT = os.environ.get("RETRIEVAL_COMPRESS_CONTEXT", "False").lower() == "true"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY") # For CohereRerank compression

# --- Ingestion Configuration ---
INGEST_DEFAULT_COLLECTION_NAME = "llamaindex_default_collection" # Fallback if not specified

# --- Logging ---
# Basic logging configuration can be added here if needed globally
# For now, individual scripts handle their logging.

def get_embedding_dim(model_name: str) -> int:
    """
    Returns the embedding dimension for known OpenAI models.
    """
    if "text-embedding-3-large" in model_name:
        return DEFAULT_VECTOR_SIZE_EMBEDDING_LARGE
    elif "text-embedding-3-small" in model_name:
        return DEFAULT_VECTOR_SIZE_EMBEDDING_SMALL
    elif "ada-002" in model_name: # Older model, but common
        return DEFAULT_VECTOR_SIZE_EMBEDDING_SMALL
    # Fallback to a general default or raise an error
    return DEFAULT_VECTOR_SIZE

