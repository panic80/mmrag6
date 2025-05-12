import os
import shlex

def get_input(prompt_message: str, default_value: str = None) -> str:
    """
    Prompts the user for input with an optional default value.
    If a default is provided, it's shown in the prompt.
    If the user enters nothing and a default exists, the default is returned.
    """
    if default_value is not None:
        prompt_with_default = f"{prompt_message} (default: {default_value}): "
    else:
        prompt_with_default = f"{prompt_message}: "

    user_input = input(prompt_with_default).strip()
    if not user_input and default_value is not None:
        return default_value
    return user_input

def get_bool_input(prompt_message: str, default_value: bool) -> bool:
    """
    Prompts the user for a boolean (True/False) input.
    """
    default_str = "True" if default_value else "False"
    while True:
        user_input = get_input(f"{prompt_message} (True/False)", default_str).lower()
        if user_input in ["true", "t", "yes", "y"]:
            return True
        elif user_input in ["false", "f", "no", "n"]:
            return False
        print("Invalid input. Please enter True or False.")

def quote_if_needed(value: str) -> str:
    """
    Adds double quotes around a value if it contains spaces or special characters
    that might cause issues in a .env file, or if it's empty.
    """
    if not value or any(c in value for c in [' ', '#', "'", '"', '=', '`', '$', '{', '}']):
        # Escape existing double quotes and backslashes
        escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped_value}"'
    return value

def get_vector_size_suggestion(embedding_model_name: str) -> str:
    """Suggests a vector size based on common OpenAI embedding models."""
    if "text-embedding-3-large" in embedding_model_name:
        return "3072"
    elif "text-embedding-3-small" in embedding_model_name:
        return "1536"
    elif "ada-002" in embedding_model_name: # Older model, but common
        return "1536"
    return "3072" # Default fallback

def main():
    """
    Main function to collect environment variables and write to .env file.
    """
    print("--- Environment Configuration Script ---")
    print("This script will help you configure the necessary environment variables.")
    print("Press Enter to use the default value if one is provided.")
    print("-" * 30)

    env_vars = {}

    # --- OpenAI Configuration ---
    print("\n--- OpenAI Configuration ---")
    env_vars["OPENAI_API_KEY"] = get_input(
        "Enter your OpenAI API Key. You can find this on your OpenAI account page (https://platform.openai.com/api-keys)"
    )
    env_vars["OPENAI_MODEL_EMBEDDING"] = get_input(
        "Enter the OpenAI Embedding Model name", "text-embedding-3-large"
    )
    env_vars["OPENAI_MODEL_LLM"] = get_input(
        "Enter the OpenAI LLM Model name for responses", "gpt-4.1-mini"
    )

    # --- Qdrant Configuration ---
    print("\n--- Qdrant Vector Store Configuration ---")
    env_vars["QDRANT_URL"] = get_input(
        "Enter the Qdrant instance URL (e.g., http://localhost:6333 if local, http://qdrant:6333 if using Docker)",
        "http://qdrant:6333"
    )
    env_vars["QDRANT_API_KEY"] = get_input(
        "Enter the Qdrant API Key (if Qdrant is secured, otherwise leave blank)"
    )
    env_vars["QDRANT_COLLECTION_NAME"] = get_input(
        "Enter the default Qdrant collection name for storing data", "rag_llamaindex_data"
    )
    default_vector_size_suggestion = get_vector_size_suggestion(env_vars.get("OPENAI_MODEL_EMBEDDING", ""))
    env_vars["DEFAULT_VECTOR_SIZE"] = get_input(
        f"Enter the default vector size for embeddings (depends on the embedding model. Suggested for '{env_vars.get('OPENAI_MODEL_EMBEDDING', 'N/A')}': {default_vector_size_suggestion})",
        default_vector_size_suggestion
    )


    # --- Mattermost Integration Configuration ---
    print("\n--- Mattermost Integration Configuration ---")
    env_vars["MATTERMOST_URL"] = get_input(
        "Enter your Mattermost server URL (e.g., https://your-mattermost-server.com)"
    )
    env_vars["MATTERMOST_TOKEN"] = get_input(
        "Enter your Mattermost Personal Access Token (for the bot/user posting messages)"
    )
    env_vars["SLASH_TOKEN"] = get_input(
        "Enter the primary Mattermost Slash Command Token (create one in Mattermost Integrations)"
    )
    env_vars["SLASH_TOKEN_INJECT"] = get_input(
        "Enter the Slash Command Token for /inject (if different from primary, otherwise leave blank to use primary)",
        env_vars.get("SLASH_TOKEN", "")
    )
    env_vars["SLASH_TOKEN_ASK"] = get_input(
        "Enter the Slash Command Token for /ask (if different from primary, otherwise leave blank to use primary)",
        env_vars.get("SLASH_TOKEN", "")
    )

    # --- Server Configuration ---
    print("\n--- Application Server Configuration ---")
    env_vars["HOST"] = get_input(
        "Enter the host address for the application server to listen on", "0.0.0.0"
    )
    env_vars["PORT"] = get_input(
        "Enter the port for the application server to listen on", "5000"
    )

    # --- Retrieval Configuration ---
    print("\n--- RAG Retrieval Configuration ---")
    env_vars["RETRIEVAL_SIMILARITY_TOP_K"] = get_input(
        "Number of similar documents to retrieve initially (dense search)", "10"
    )
    env_vars["RETRIEVAL_USE_HYBRID"] = str(get_bool_input(
        "Use hybrid search (dense + sparse BM25)?", True
    ))
    if env_vars["RETRIEVAL_USE_HYBRID"].lower() == "true":
        env_vars["RETRIEVAL_SPARSE_TOP_K"] = get_input(
            "Number of documents to retrieve from sparse search (BM25, if hybrid is True)", "10"
        )
    else: # Ensure it's set, even if not used, to avoid missing key if config.py expects it
        env_vars["RETRIEVAL_SPARSE_TOP_K"] = "10"


    env_vars["RETRIEVAL_RERANK_TOP_N"] = get_input(
        "Number of documents to pass to reranker (0 to disable reranking)", "0"
    )
    if env_vars["RETRIEVAL_RERANK_TOP_N"] != "0":
        env_vars["RETRIEVAL_RERANKER_MODEL"] = get_input(
            "SBERT cross-encoder model for reranking (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2)",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    else:
        env_vars["RETRIEVAL_RERANKER_MODEL"] = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Set default even if not used

    env_vars["RETRIEVAL_USE_MMR"] = str(get_bool_input(
        "Use Maximal Marginal Relevance (MMR) for diverse results?", False
    ))
    if env_vars["RETRIEVAL_USE_MMR"].lower() == "true":
        env_vars["RETRIEVAL_MMR_LAMBDA"] = get_input(
            "MMR lambda parameter (0.0 to 1.0, higher means more diversity)", "0.5"
        )
    else:
        env_vars["RETRIEVAL_MMR_LAMBDA"] = "0.5" # Set default even if not used

    env_vars["RETRIEVAL_USE_QUERY_EXPANSION"] = str(get_bool_input(
        "Use query expansion (generate sub-questions)?", True
    ))
    if env_vars["RETRIEVAL_USE_QUERY_EXPANSION"].lower() == "true":
        env_vars["RETRIEVAL_MAX_EXPANSIONS"] = get_input(
            "Maximum number of query expansions", "3"
        )
    else:
        env_vars["RETRIEVAL_MAX_EXPANSIONS"] = "3" # Set default

    env_vars["RETRIEVAL_COMPRESS_CONTEXT"] = str(get_bool_input(
        "Compress context before sending to LLM (requires Cohere API key if using CohereRerank for this)?", False
    ))
    if env_vars["RETRIEVAL_COMPRESS_CONTEXT"].lower() == "true":
        env_vars["COHERE_API_KEY"] = get_input(
            "Enter your Cohere API Key (for CohereRerank context compression). Find at https://dashboard.cohere.com/api-keys"
        )
    else:
        env_vars["COHERE_API_KEY"] = "" # Set empty if not used

    env_vars["RETRIEVAL_RAW_OUTPUT"] = str(get_bool_input(
        "Output raw response from LLM for /ask command (True/False)?", False
    ))

    # --- Write to .env file ---
    env_file_path = ".env"
    try:
        with open(env_file_path, "w", encoding="utf-8") as f:
            f.write("# Environment variables for MMRag application\n")
            f.write("# Generated by configure_env.py\n")
            f.write("-" * 30 + "\\n\\n")

            f.write("# --- OpenAI Configuration ---\\n")
            f.write(f"OPENAI_API_KEY={quote_if_needed(env_vars.get('OPENAI_API_KEY', ''))}\\n")
            f.write(f"OPENAI_MODEL_EMBEDDING={quote_if_needed(env_vars.get('OPENAI_MODEL_EMBEDDING', 'text-embedding-3-large'))}\\n")
            f.write(f"OPENAI_MODEL_LLM={quote_if_needed(env_vars.get('OPENAI_MODEL_LLM', 'gpt-4.1-mini'))}\\n\\n")

            f.write("# --- Qdrant Vector Store Configuration ---\\n")
            f.write(f"QDRANT_URL={quote_if_needed(env_vars.get('QDRANT_URL', 'http://qdrant:6333'))}\\n")
            f.write(f"QDRANT_API_KEY={quote_if_needed(env_vars.get('QDRANT_API_KEY', ''))}\\n")
            f.write(f"QDRANT_COLLECTION_NAME={quote_if_needed(env_vars.get('QDRANT_COLLECTION_NAME', 'rag_llamaindex_data'))}\\n")
            f.write(f"DEFAULT_VECTOR_SIZE={quote_if_needed(env_vars.get('DEFAULT_VECTOR_SIZE', '3072'))}\\n\\n")

            f.write("# --- Mattermost Integration Configuration ---\\n")
            f.write(f"MATTERMOST_URL={quote_if_needed(env_vars.get('MATTERMOST_URL', ''))}\\n")
            f.write(f"MATTERMOST_TOKEN={quote_if_needed(env_vars.get('MATTERMOST_TOKEN', ''))}\\n")
            f.write(f"SLASH_TOKEN={quote_if_needed(env_vars.get('SLASH_TOKEN', ''))}\\n")
            f.write(f"SLASH_TOKEN_INJECT={quote_if_needed(env_vars.get('SLASH_TOKEN_INJECT', env_vars.get('SLASH_TOKEN', '')))}\\n")
            f.write(f"SLASH_TOKEN_ASK={quote_if_needed(env_vars.get('SLASH_TOKEN_ASK', env_vars.get('SLASH_TOKEN', '')))}\\n\\n")

            f.write("# --- Application Server Configuration ---\\n")
            f.write(f"HOST={quote_if_needed(env_vars.get('HOST', '0.0.0.0'))}\\n")
            f.write(f"PORT={quote_if_needed(env_vars.get('PORT', '5000'))}\\n\\n")

            f.write("# --- RAG Retrieval Configuration ---\\n")
            f.write(f"RETRIEVAL_SIMILARITY_TOP_K={quote_if_needed(env_vars.get('RETRIEVAL_SIMILARITY_TOP_K', '10'))}\\n")
            f.write(f"RETRIEVAL_USE_HYBRID={quote_if_needed(env_vars.get('RETRIEVAL_USE_HYBRID', 'True'))}\\n")
            f.write(f"RETRIEVAL_SPARSE_TOP_K={quote_if_needed(env_vars.get('RETRIEVAL_SPARSE_TOP_K', '10'))}\\n")
            f.write(f"RETRIEVAL_RERANK_TOP_N={quote_if_needed(env_vars.get('RETRIEVAL_RERANK_TOP_N', '0'))}\\n")
            f.write(f"RETRIEVAL_RERANKER_MODEL={quote_if_needed(env_vars.get('RETRIEVAL_RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2'))}\\n")
            f.write(f"RETRIEVAL_USE_MMR={quote_if_needed(env_vars.get('RETRIEVAL_USE_MMR', 'False'))}\\n")
            f.write(f"RETRIEVAL_MMR_LAMBDA={quote_if_needed(env_vars.get('RETRIEVAL_MMR_LAMBDA', '0.5'))}\\n")
            f.write(f"RETRIEVAL_USE_QUERY_EXPANSION={quote_if_needed(env_vars.get('RETRIEVAL_USE_QUERY_EXPANSION', 'True'))}\\n")
            f.write(f"RETRIEVAL_MAX_EXPANSIONS={quote_if_needed(env_vars.get('RETRIEVAL_MAX_EXPANSIONS', '3'))}\\n")
            f.write(f"RETRIEVAL_COMPRESS_CONTEXT={quote_if_needed(env_vars.get('RETRIEVAL_COMPRESS_CONTEXT', 'False'))}\\n")
            f.write(f"COHERE_API_KEY={quote_if_needed(env_vars.get('COHERE_API_KEY', ''))}\\n")
            f.write(f"RETRIEVAL_RAW_OUTPUT={quote_if_needed(env_vars.get('RETRIEVAL_RAW_OUTPUT', 'False'))}\\n")

        print(f"\nConfiguration saved to {os.path.abspath(env_file_path)}")
        print("You can now run the application. If using Docker, it will pick up this .env file.")
    except IOError as e:
        print(f"\nError: Could not write to .env file: {e}")
        print("Please check permissions and try again.")

if __name__ == "__main__":
    main()
