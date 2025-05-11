import os
import subprocess
import threading
from flask import Flask, request, jsonify
import requests
from pathlib import Path # Added import
import openai # Added for OpenAI API
import shlex # Already present, but good to note
import sys # Added for print to stderr

# LlamaIndex imports for /ask
from llama_index.core import (
    StorageContext,
    load_index_from_storage, # Will be replaced by VectorStoreIndex.from_vector_store for Qdrant
    VectorStoreIndex,      # Added for from_vector_store
    Settings,
)
from llama_index.llms.openai import OpenAI as LlamaOpenAI 
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore # Added for Qdrant
from qdrant_client import QdrantClient # Added for Qdrant

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/inject",  methods=["GET", "POST"])
@app.route("/inject/", methods=["GET", "POST"])
@app.route("/injest",  methods=["GET", "POST"])
@app.route("/injest/", methods=["GET", "POST"])
@app.route("/ask",     methods=["GET", "POST"])
@app.route("/ask/",   methods=["GET", "POST"])
@app.route("/",        methods=["GET", "POST"])
def handle_slash():
    # Bare GET to root path can be used as a health check without slash payloads
    if request.path == "/" and request.method == "GET":
        return jsonify({"status": "ok"}), 200
    try:
        # Parse incoming payload (JSON or form)
        payload = request.get_json(silent=True) or request.values
        # Extract text parameter
        text = payload.get("text", "")
        # DEBUG: log incoming request values
        try:
            app.logger.info(f"Slash command hit: path={request.path}, payload={payload}")
        except Exception:
            pass
        # Context for asynchronous replies
        response_url = payload.get("response_url")
        channel_id = payload.get("channel_id")
        mattermost_url = os.environ.get("MATTERMOST_URL")  # e.g. https://your-mattermost-server
        mattermost_token = os.environ.get("MATTERMOST_TOKEN")
        # (Using Personal Access Token for Mattermost REST API)

        # Determine which slash command was invoked and validate its token
        trigger = payload.get("command") or payload.get("trigger_word") or ""
        trigger = trigger.strip()
        cmd_name = trigger.lstrip("/").lower()

        # Validate slash-command token (supporting per-command overrides)
        req_token = payload.get("token")
        generic_token = os.environ.get("SLASH_TOKEN")
        inject_token = os.environ.get("SLASH_TOKEN_INJECT", generic_token)
        ask_token = os.environ.get("SLASH_TOKEN_ASK", generic_token)
        # Pick the expected token; alias 'injest' to same as 'inject'.
        if cmd_name in ("inject", "injest"):
            expected_token = inject_token
        elif cmd_name == "ask":
            expected_token = ask_token
        else:
            expected_token = generic_token
        # ------------------------------------------------------------------
        # Harden token validation
        # ------------------------------------------------------------------

        # 1. Refuse the request outright if *no* token is configured for the
        #    invoked command – this prevents accidentally exposing the
        #    endpoint when the admin forgot to set the environment variable.
        if not expected_token:
            # Service unavailable until the administrator configures a token.
            return jsonify({"text": "slash token not set"}), 503

        # 2. Explicitly reject mismatching tokens.
        if req_token != expected_token:
            return jsonify({"text": "Invalid token."}), 403

        # 'ask' triggers a RAG query using OpenAI gpt-4.1-mini
        if cmd_name == "ask":
            if not text:
                return jsonify({"text": "No text provided."}), 400

            # Determine collection_name for the immediate acknowledgement message
            collection_name_for_ack_env = os.environ.get("QDRANT_COLLECTION_NAME")
            final_collection_name_for_ack = collection_name_for_ack_env if collection_name_for_ack_env else "rag_llamaindex_data"

            # Asynchronous execution
            def run_and_post(query_text_param, collection_name_param): # Added parameters
                # Track whether REST calls are still worth trying for this request
                rest_usable: dict[str, bool] = {"ok": bool(mattermost_url and mattermost_token and channel_id)}
                answer_text = "An error occurred while processing your query." # Default error message

                def _post_message(txt: str):
                    """Post *txt* to the originating channel.
                    (Reuses existing _post_message logic)
                    """
                    # Ensure app.logger is available if used within this function outside of request context
                    # For simplicity, assuming direct calls or already configured logger
                    if rest_usable["ok"]:
                        try:
                            hdrs = {"Authorization": f"Bearer {mattermost_token}"}
                            resp = requests.post(
                                f"{mattermost_url}/api/v4/posts",
                                headers=hdrs,
                                json={"channel_id": channel_id, "message": txt},
                                timeout=10,
                            )
                            if resp.status_code in (200, 201):
                                return
                            if resp.status_code in (401, 403):
                                rest_usable["ok"] = False
                            app.logger.warning(
                                "Posting via REST API failed with %s – falling back to response_url",
                                resp.status_code,
                            )
                        except Exception:
                            rest_usable["ok"] = False
                            app.logger.exception("REST API post failed – falling back to response_url")

                    if response_url:
                        try:
                            requests.post(
                                response_url,
                                json={"response_type": "in_channel", "text": txt},
                                timeout=10,
                            )
                            return
                        except Exception:
                            app.logger.exception("Failed to post via response_url")
                
                try:
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    if not openai_api_key:
                        _post_message("Error: OPENAI_API_KEY environment variable not set.")
                        return

                    client = openai.OpenAI(api_key=openai_api_key) # This client init seems redundant if only LlamaIndex is used now.
                                                                # However, LlamaOpenAI itself will use the API key.
                                                                # Let's keep it for now, or remove if truly unused.
                                                                # For now, the LlamaIndex settings will handle OpenAI interactions.

                    # query_text is now query_text_param
                    if not query_text_param:
                        _post_message("Error: No query text provided.")
                        return

                    # openai_api_key is already fetched and checked once.
                    # This second check is redundant if the first one passes.
                    # openai_api_key = os.environ.get("OPENAI_API_KEY") 
                    if not openai_api_key:
                        _post_message("Error: OPENAI_API_KEY environment variable not set.")
                        return

                    # Configure LlamaIndex Settings
                    # Ensure these model names match what ingest_llamaindex.py uses or expects
                    # Using "gpt-4.1-mini" for LLM as per previous change, and a common embedding model.
                    llm_model_name = os.environ.get("OPENAI_MODEL_LLM", "gpt-4.1-mini")
                    embedding_model_name = os.environ.get("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large") # Ensure consistency
                    
                    try:
                        Settings.llm = LlamaOpenAI(model=llm_model_name, api_key=openai_api_key)
                        Settings.embed_model = LlamaOpenAIEmbedding(model=embedding_model_name, api_key=openai_api_key)
                    except Exception as e_settings:
                        app.logger.error(f"Failed to initialize LlamaIndex OpenAI models: {e_settings}")
                        _post_message(f"Error initializing LlamaIndex models: {e_settings}")
                        return

                    # Determine collection name (already passed as collection_name_param)
                    # Determine persist path for the local docstore
                    persist_dir_base = Path("./storage_llamaindex_db")
                    local_docstore_persist_path = persist_dir_base / collection_name_param

                    # Initialize Qdrant client
                    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
                    qdrant_api_key_env = os.environ.get("QDRANT_API_KEY") # Use a different var to avoid conflict
                    
                    app.logger.info(f"Connecting to Qdrant at {qdrant_url} for collection {collection_name_param}")

                    try:
                        q_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key_env)
                        # Check if collection exists in Qdrant
                        q_client.get_collection(collection_name=collection_name_param)
                        app.logger.info(f"Successfully connected to Qdrant collection '{collection_name_param}'.")
                    except Exception as e_qdrant_check:
                        app.logger.error(f"Failed to connect to or find Qdrant collection '{collection_name_param}': {e_qdrant_check}")
                        _post_message(f"Error: Qdrant collection '{collection_name_param}' not found or Qdrant is inaccessible. Please ensure ingestion was successful. Details: {e_qdrant_check}")
                        return

                    vector_store = QdrantVectorStore(
                        client=q_client,
                        collection_name=collection_name_param
                    )
                    
                    # Attempt to load local docstore if it exists
                    app.logger.info(f"Checking for local docstore at: {str(local_docstore_persist_path)}")
                    loaded_storage_context: StorageContext
                    if local_docstore_persist_path.exists() and local_docstore_persist_path.is_dir():
                        docstore_json_file = local_docstore_persist_path / "docstore.json"
                        if docstore_json_file.exists():
                            app.logger.info(f"Local docstore found at {str(local_docstore_persist_path)}. Loading StorageContext.")
                            loaded_storage_context = StorageContext.from_defaults(
                                vector_store=vector_store, # Provide Qdrant vector store
                                persist_dir=str(local_docstore_persist_path)
                            )
                        else:
                            app.logger.warning(f"Local docstore.json not found in {str(local_docstore_persist_path)}. Proceeding with Qdrant vector store only for StorageContext.")
                            loaded_storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    else:
                        app.logger.warning(f"Local docstore path {str(local_docstore_persist_path)} not found. Proceeding with Qdrant vector store only for StorageContext.")
                        loaded_storage_context = StorageContext.from_defaults(vector_store=vector_store)

                    # Load the index using the Qdrant vector store
                    index = VectorStoreIndex.from_vector_store(
                        vector_store=vector_store,
                        storage_context=loaded_storage_context # This context now includes Qdrant VS and potentially local docstore
                    )
                    
                    # Instead of building the query engine here, we will call the main_async
                    # function from query_llamaindex.py.
                    # This requires setting up arguments for main_async based on environment variables
                    # and the incoming query.

                    import asyncio
                    import shlex # Ensure shlex is imported in this scope if run_and_post needs it.
                                 # This was the likely cause of the "free variable" error.
                    from query_llamaindex import main_async as query_main_async # Import the async function

                    # Prepare arguments for query_main_async
                    # Most of these will come from environment variables or have defaults
                    # matching query_llamaindex.py's Click options.
                    
                    # Convert text (string) to a sequence of strings for query_text param
                    # query_args_list = shlex.split(query_text_param) # Will be defined after flag parsing


                    # Default values from query_llamaindex.py CLI options
                    # These can be overridden by environment variables if desired.
                    # For simplicity, we'll use the defaults or simple env var mappings here.
                    # A more robust solution might involve a config object.
                    
                    # Fetching from environment or using defaults from query_llamaindex.py
                    # similarity_top_k_val = int(os.environ.get("RETRIEVAL_SIMILARITY_TOP_K", 10)) # Replaced by dynamic kwargs
                    qdrant_url_val = os.environ.get("QDRANT_URL", "http://qdrant:6333") # Match Docker Compose
                    qdrant_api_key_val = os.environ.get("QDRANT_API_KEY")
                    # openai_api_key is already fetched
                    openai_model_embedding_val = os.environ.get("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large")
                    openai_model_llm_val = os.environ.get("OPENAI_MODEL_LLM", "gpt-4.1-mini")
                    
                    # Flags and other parameters (using defaults from query_llamaindex.py for now)
                    # These could also be exposed via environment variables if more control is needed from server.py
                    raw_output_flag_val = os.environ.get("RETRIEVAL_RAW_OUTPUT", "False").lower() == "true"
                    use_hybrid_search_val = os.environ.get("RETRIEVAL_USE_HYBRID", "True").lower() == "true"
                    # sparse_top_k_val = int(os.environ.get("RETRIEVAL_SPARSE_TOP_K", 10)) # Replaced by dynamic kwargs
                    rerank_top_n_val = int(os.environ.get("RETRIEVAL_RERANK_TOP_N", 0)) # Default 0 (disabled)
                    reranker_model_val = os.environ.get("RETRIEVAL_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                    
                    # Determine use_mmr_val: Default to True, allow --no-mmr to override.
                    use_mmr_val = True # Default to ON
                    original_query_text_param_for_logging = query_text_param # For logging

                    temp_query_parts = shlex.split(query_text_param)
                    cleaned_query_parts = []
                    explicit_mmr_setting_found = False

                    for part in temp_query_parts:
                        if part == "--no-mmr":
                            use_mmr_val = False
                            explicit_mmr_setting_found = True
                        elif part == "--mmr": # Explicitly asking for MMR
                            use_mmr_val = True # Reinforce, though it's already default
                            explicit_mmr_setting_found = True
                        else:
                            cleaned_query_parts.append(part)

                    if explicit_mmr_setting_found:
                        query_text_param = " ".join(cleaned_query_parts)
                        app.logger.info(f"Explicit MMR flag found. Setting use_mmr to {use_mmr_val}. Cleaned query: '{query_text_param}'")
                    else:
                        # No explicit flag, use_mmr_val remains True (our hardcoded default)
                        app.logger.info(f"No explicit MMR flag. Defaulting use_mmr to {use_mmr_val}.")
                    
                    # query_args_list should now be based on the potentially cleaned query_text_param
                    query_args_list = shlex.split(query_text_param)
                    
                    mmr_lambda_val = float(os.environ.get("RETRIEVAL_MMR_LAMBDA", 0.5))
                    
                    # Filters: For now, not supporting dynamic filters from slash command in this integration.
                    # This could be added by parsing 'text' for filter arguments.
                    filters_kv_val = [] 
                    
                    evaluate_rag_flag_val = False # Evaluation not typically run in server context
                    compress_context_flag_val = os.environ.get("RETRIEVAL_COMPRESS_CONTEXT", "False").lower() == "true"
                    cohere_api_key_val = os.environ.get("COHERE_API_KEY") # For CohereRerank compression
                    
                    verbose_val = app.debug # Use Flask app's debug status for verbosity

                    # Call query_main_async
                    # Note: query_main_async returns None but prints the output.
                    # We need to capture stdout or modify query_main_async to return the response.
                    # For now, let's assume we modify query_main_async to return the answer string.
                    # This is a significant change to query_llamaindex.py that would be needed.
                    #
                    # Now that query_main_async returns the answer, we can use it directly.
                    app.logger.info(f"Calling advanced query pipeline from query_llamaindex.main_async for query: '{original_query_text_param_for_logging}' (effective query for retrieval: '{query_text_param}')")
                    
                    # Prepare dynamic keyword arguments for query_main_async
                    query_kwargs = {} # DEFINED EARLIER

                    env_similarity_top_k = os.environ.get("RETRIEVAL_SIMILARITY_TOP_K")
                    if env_similarity_top_k is not None:
                        try:
                            query_kwargs['similarity_top_k'] = int(env_similarity_top_k)
                        except ValueError:
                            print(f"[WARNING] Invalid RETRIEVAL_SIMILARITY_TOP_K: {env_similarity_top_k}. query_llamaindex.py default will be used.", file=sys.stderr)

                    env_sparse_top_k = os.environ.get("RETRIEVAL_SPARSE_TOP_K")
                    if env_sparse_top_k is not None:
                        try:
                            query_kwargs['sparse_top_k'] = int(env_sparse_top_k)
                        except ValueError:
                            print(f"[WARNING] Invalid RETRIEVAL_SPARSE_TOP_K: {env_sparse_top_k}. query_llamaindex.py default will be used.", file=sys.stderr)
                    
                    # Logging block should come AFTER query_kwargs is populated
                    print("----------------------------------------------------", file=sys.stderr)
                    print(f"Calling query_main_async with effective parameters:", file=sys.stderr)
                    print(f"  query_text_param (cleaned for query_args_list): '{query_text_param}'", file=sys.stderr)
                    print(f"  query_args_list: {query_args_list}", file=sys.stderr)
                    print(f"  collection_name: {collection_name_param}", file=sys.stderr)
                    # Updated logging for dynamic kwargs
                    print(f"  similarity_top_k: {query_kwargs.get('similarity_top_k', 'Default in query_llamaindex.py')}", file=sys.stderr)
                    print(f"  use_hybrid_search: {use_hybrid_search_val}", file=sys.stderr)
                    print(f"  sparse_top_k: {query_kwargs.get('sparse_top_k', 'Default in query_llamaindex.py')}", file=sys.stderr)
                    print(f"  rerank_top_n: {rerank_top_n_val}", file=sys.stderr)
                    print(f"  use_mmr: {use_mmr_val}", file=sys.stderr) # CRITICAL VALUE TO CHECK
                    print(f"  mmr_lambda: {mmr_lambda_val}", file=sys.stderr)
                    print(f"  compress_context_flag: {compress_context_flag_val}", file=sys.stderr)
                    print(f"  verbose: {verbose_val}", file=sys.stderr)
                    print("----------------------------------------------------", file=sys.stderr)
                    
                    # Ensure an event loop is running if not already (e.g., if Flask runs in a non-async context)
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:  # No running event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    answer_text = loop.run_until_complete(query_main_async(
                        collection_name=collection_name_param, # Use the collection name determined by server
                        qdrant_url=qdrant_url_val,
                        qdrant_api_key=qdrant_api_key_val,
                        openai_api_key=openai_api_key, # Already fetched
                        openai_model_embedding=openai_model_embedding_val,
                        openai_model_llm=openai_model_llm_val,
                        raw_output_flag=raw_output_flag_val,
                        use_hybrid_search=use_hybrid_search_val,
                        # sparse_top_k is now in query_kwargs
                        rerank_top_n=rerank_top_n_val,
                        reranker_model=reranker_model_val,
                        use_mmr=use_mmr_val,
                        mmr_lambda=mmr_lambda_val,
                        filters_kv=filters_kv_val,
                        evaluate_rag_flag=evaluate_rag_flag_val,
                        compress_context_flag=compress_context_flag_val,
                        cohere_api_key=cohere_api_key_val,
                        verbose=verbose_val,
                        query_text=query_args_list, # This is the shlex.split list of query words
                        **query_kwargs
                    ))
                    
                    if not answer_text: # Handle case where query_main_async might return None or empty on error
                        answer_text = "Failed to get a response from the advanced query pipeline."
                        app.logger.error("query_main_async returned no answer_text.")

                except Exception as e:
                    app.logger.error(f"An error occurred calling query_llamaindex.main_async: {e}")
                    import traceback
                    app.logger.error(traceback.format_exc())
                    answer_text = f"An error occurred while processing your query with LlamaIndex. Details: {e}"

                # Compose final message
                final_msg = f"**Q:** {original_query_text_param_for_logging}\n\n**A:**\n{answer_text}" # Use original for Q display
                _post_message(final_msg)

            # Always run asynchronously
            threading.Thread(target=run_and_post, args=(text, final_collection_name_for_ack), daemon=True).start()
            # Immediate acknowledgement
            return jsonify({"text": f"Processing your query with LlamaIndex (collection: {final_collection_name_for_ack})..."}), 200

        # 'inject' ingests the current channel into the RAG collection
        elif cmd_name in ("inject", "injest"):
            import shlex
            def run_inject():
                """Handle the /inject (or /injest) command in a background thread.
                Sends real-time, concise updates to the Mattermost channel.
                """
                import requests
                import os
                import sys
                import tempfile
                import traceback # For logging full exceptions
                from qdrant_client import QdrantClient
                # shlex is imported in the outer scope
                # Path is imported globally

                # Mattermost connection details from outer scope:
                # response_url, channel_id, mattermost_url, mattermost_token, text (for args)
                # app.logger is also available.

                rest_usable: dict[str, bool] = {"ok": bool(mattermost_url and mattermost_token and channel_id)}

                def _send_message_to_mattermost(message_text: str):
                    """Sends a message immediately to the Mattermost channel."""
                    if not message_text: # Avoid sending empty messages
                        return

                    if rest_usable["ok"]:
                        try:
                            hdrs = {"Authorization": f"Bearer {mattermost_token}"}
                            resp = requests.post(
                                f"{mattermost_url}/api/v4/posts",
                                headers=hdrs,
                                json={"channel_id": channel_id, "message": message_text},
                                timeout=10,
                            )
                            if resp.status_code in (200, 201):
                                return
                            if resp.status_code in (401, 403): # Token issue
                                rest_usable["ok"] = False
                                app.logger.warning("Mattermost REST API auth failed (401/403). Falling back to response_url for subsequent messages.")
                            else:
                                app.logger.warning(
                                    "Mattermost REST API post failed with status %s: %s. Falling back to response_url.",
                                    resp.status_code, resp.text
                                )
                        except requests.exceptions.RequestException as e_req:
                            rest_usable["ok"] = False
                            app.logger.exception(f"Mattermost REST API request failed: {e_req}. Falling back to response_url.")
                    
                    if response_url:
                        try:
                            requests.post(
                                response_url,
                                json={"response_type": "in_channel", "text": message_text},
                                timeout=10,
                            )
                            return
                        except requests.exceptions.RequestException as e_resp_url:
                            app.logger.exception(f"Failed to post to response_url: {e_resp_url}")
                    
                    app.logger.error(f"Failed to send message to Mattermost by any method: {message_text}")

                # Main ingestion logic
                try:
                    raw_args_from_text = shlex.split(text or "")
                    
                    collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "rag_llamaindex_data")
                    temp_collection_args = []

                    if "--collection-name" in raw_args_from_text:
                        try:
                            idx = raw_args_from_text.index("--collection-name")
                            collection_name = raw_args_from_text[idx + 1]
                            temp_collection_args.extend([raw_args_from_text[idx], raw_args_from_text[idx+1]])
                        except (ValueError, IndexError):
                            _send_message_to_mattermost("⚠️ Invalid --collection-name usage. Using default or environment variable.")
                    elif "-c" in raw_args_from_text:
                        try:
                            idx = raw_args_from_text.index("-c")
                            collection_name = raw_args_from_text[idx + 1]
                            temp_collection_args.extend([raw_args_from_text[idx], raw_args_from_text[idx+1]])
                        except (ValueError, IndexError):
                            _send_message_to_mattermost("⚠️ Invalid -c usage for collection name. Using default or environment variable.")
                    
                    is_purge_command = "--purge" in raw_args_from_text
                    
                    if is_purge_command:
                        _send_message_to_mattermost(f"ℹ️ Purge command received for collection '{collection_name}'.")
                        qdrant_url_env = os.environ.get("QDRANT_URL", "http://qdrant:6333") # Match Docker Compose
                        qdrant_api_key_env = os.environ.get("QDRANT_API_KEY")
                        
                        try:
                            q_client = QdrantClient(url=qdrant_url_env, api_key=qdrant_api_key_env)
                            q_client.delete_collection(collection_name=collection_name)
                            _send_message_to_mattermost(f"✅ Successfully deleted collection '{collection_name}'.")
                        except Exception as e_del:
                            app.logger.error(f"Failed to delete Qdrant collection '{collection_name}': {e_del}\n{traceback.format_exc()}")
                            _send_message_to_mattermost(f"⚠️ Could not delete collection '{collection_name}'. It might not exist or an error occurred. Check server logs.")
                        
                        try: # Recreate collection
                            from qdrant_client.http import models as qdrant_models
                            vector_size = int(os.environ.get("DEFAULT_VECTOR_SIZE", 3072))
                            # Ensure q_client is initialized if previous block failed partially
                            if 'q_client' not in locals():
                                q_client = QdrantClient(url=qdrant_url_env, api_key=qdrant_api_key_env)

                            q_client.recreate_collection(
                                collection_name=collection_name,
                                vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE)
                            )
                            _send_message_to_mattermost(f"✅ Successfully recreated empty collection '{collection_name}' (vector size: {vector_size}).")
                        except Exception as e_recreate:
                            app.logger.error(f"Failed to recreate Qdrant collection '{collection_name}': {e_recreate}\n{traceback.format_exc()}")
                            _send_message_to_mattermost(f"❌ Failed to recreate collection '{collection_name}'. Check server logs.")
                            return # Stop if recreate fails

                        args_for_source_check = [arg for arg in raw_args_from_text if arg != "--purge" and arg not in temp_collection_args]
                        is_purge_only = not any(not arg.startswith("-") for arg in args_for_source_check)

                        if is_purge_only:
                            _send_message_to_mattermost(f"ℹ️ Purge operation for '{collection_name}' complete. No sources specified for further ingestion.")
                            return

                    ingest_cmd_base = [sys.executable, "-u", "-m", "ingest_llamaindex", "--collection-name", collection_name]
                    ingest_passthrough_args = []
                    potential_sources_from_args = []
                    args_to_filter_for_ingest = [arg for arg in raw_args_from_text if arg != "--purge" and arg not in temp_collection_args]

                    for arg in args_to_filter_for_ingest:
                        if arg == "--rich-metadata": continue
                        elif arg == "--no-rich-metadata": ingest_passthrough_args.append(arg)
                        elif arg in ("--generate-summaries", "--no-generate-summaries", "--quality-checks", "--no-quality-checks", "--crawl-depth", "--depth-crawl", "--parallel"):
                            _send_message_to_mattermost(f"ℹ️ Flag '{arg}' is no longer supported and will be ignored.")
                        elif not arg.startswith("-"): potential_sources_from_args.append(arg)
                        else: ingest_passthrough_args.append(arg)
                    
                    ingest_cmd_base.extend(ingest_passthrough_args)
                    final_sources_to_process = []
                    temp_files_to_clean = []

                    if potential_sources_from_args:
                        final_sources_to_process = potential_sources_from_args
                    else:
                        if not mattermost_url or not channel_id:
                            _send_message_to_mattermost("❌ MATTERMOST_URL or channel_id not configured – unable to fetch channel messages for ingestion.")
                            return
                        _send_message_to_mattermost("ℹ️ No source explicitly provided. Fetching current channel transcript for ingestion...")
                        msgs = []
                        hdrs = {"Authorization": f"Bearer {mattermost_token}"} if mattermost_token else {}
                        per_page = 200
                        page = 0
                        channel_fetch_ok = True
                        while True:
                            try:
                                resp_ct = requests.get(
                                    f"{mattermost_url}/api/v4/channels/{channel_id}/posts",
                                    params={"page": page, "per_page": per_page}, headers=hdrs, timeout=15
                                )
                                resp_ct.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                            except requests.exceptions.RequestException as e_fetch:
                                app.logger.error(f"Error fetching posts from Mattermost: {e_fetch}\n{traceback.format_exc()}")
                                _send_message_to_mattermost(f"❌ Error fetching channel history: {e_fetch}. Check server logs.")
                                channel_fetch_ok = False
                                break
                            
                            data_ct = resp_ct.json()
                            posts_data = data_ct.get("posts", {})
                            order = data_ct.get("order", [])
                            if not order: break
                            for pid in order:
                                p = posts_data.get(pid)
                                if p and p.get("message"): msgs.append(p["message"])
                            if len(order) < per_page: break
                            page += 1
                        
                        if not channel_fetch_ok: return
                        if not msgs:
                            _send_message_to_mattermost("ℹ️ No messages found in the current channel to ingest.")
                            return
                        
                        try:
                            tmp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
                            tmp_file.write("\n".join(m.rstrip("\n") for m in msgs))
                            tmp_file.close()
                            final_sources_to_process = [tmp_file.name]
                            temp_files_to_clean.append(tmp_file.name)
                            _send_message_to_mattermost(f"ℹ️ Fetched {len(msgs)} messages from channel for ingestion.")
                        except IOError as e_io:
                            app.logger.error(f"Failed to write channel transcript to temp file: {e_io}\n{traceback.format_exc()}")
                            _send_message_to_mattermost("❌ Error preparing channel transcript for ingestion. Check server logs.")
                            return


                    if not final_sources_to_process:
                        _send_message_to_mattermost("ℹ️ No sources to process for ingestion.")
                        return

                    all_sources_successful = True
                    for src_idx, source_item in enumerate(final_sources_to_process):
                        _send_message_to_mattermost(f"🚀 Starting ingestion for source ({src_idx+1}/{len(final_sources_to_process)}): `{source_item}`")
                        
                        current_ingest_cmd_for_source = list(ingest_cmd_base)
                        is_url = source_item.startswith("http://") or source_item.startswith("https://")
                        
                        if is_url:
                            current_ingest_cmd_for_source.extend(["--source-url", source_item])
                        else:
                            source_file_path = Path(source_item)
                            if source_file_path.is_file():
                                current_ingest_cmd_for_source.extend(["--data-dir", str(source_file_path.parent)])
                            else:
                                current_ingest_cmd_for_source.extend(["--data-dir", source_item])
                        
                        if "--verbose" not in current_ingest_cmd_for_source and app.debug:
                             current_ingest_cmd_for_source.append("--verbose") # Only add verbose if server is in debug

                        app.logger.info(f"Executing for source '{source_item}': {' '.join(shlex.quote(str(s)) for s in current_ingest_cmd_for_source)}")
                        # _send_message_to_mattermost(f"⚙️ Processing source: `{source_item}`...") # Redundant with "Starting ingestion..."

                        try:
                            env = dict(os.environ, PYTHONUNBUFFERED="1")
                            # Use Popen to allow non-blocking logging if ingest_llamaindex is very verbose in debug
                            proc = subprocess.Popen(current_ingest_cmd_for_source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                            
                            # Log subprocess output to server logs, not directly to Mattermost for conciseness
                            if proc.stdout:
                                for line in proc.stdout:
                                    app.logger.info(f"[ingest_llamaindex:{source_item}] {line.rstrip()}")
                            
                            ret = proc.wait()
                            if ret != 0:
                                _send_message_to_mattermost(f"❌ Error processing source `{source_item}` (exit code: {ret}). Check server logs for details.")
                                all_sources_successful = False
                            else:
                                _send_message_to_mattermost(f"✅ Successfully processed source: `{source_item}`")
                        except Exception as e_proc:
                            app.logger.error(f"Failed to start/run ingestion subprocess for '{source_item}': {e_proc}\n{traceback.format_exc()}")
                            _send_message_to_mattermost(f"❌ Critical error during processing of source `{source_item}`. Check server logs.")
                            all_sources_successful = False
                            continue # Try next source if possible
                    
                    for tf_path in temp_files_to_clean:
                        try:
                            os.remove(tf_path)
                            app.logger.info(f"Cleaned up temporary file: {tf_path}")
                        except OSError as e_clean:
                            app.logger.warning(f"Failed to clean up temporary file '{tf_path}': {e_clean}")
                    
                    if all_sources_successful and final_sources_to_process:
                        _send_message_to_mattermost("✅ Ingestion process completed successfully for all sources.")
                    elif final_sources_to_process: # Some sources might have failed
                        _send_message_to_mattermost("⚠️ Ingestion process finished, but some sources may have encountered errors. Check server logs and previous messages for details.")
                    # If no sources were processed (e.g. purge only and it returned early), no final message here.

                except Exception as e_outer: # Catch any unexpected errors in the main try block
                    app.logger.error(f"An unexpected error occurred during the inject process: {e_outer}\n{traceback.format_exc()}")
                    _send_message_to_mattermost("❌ An critical unexpected error occurred during the inject process. Please check server logs for details.")
                # No 'finally' block needed as messages are sent immediately.

            # This was the old synchronous purge handling block, now integrated into run_inject
            # threading.Thread(target=run_inject, daemon=True).start()
            # return jsonify({"text": "Ingestion with LlamaIndex started... progress will be posted shortly."}), 200

            # Start the run_inject thread
            threading.Thread(target=run_inject, daemon=True).start()
            # Return immediate acknowledgement
            return jsonify({"text": "Ingestion command received. Processing... progress will be posted."}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return exception message to Mattermost
        return jsonify({"text": f"Error: {e}"}), 200

def check_url_dependencies():
    """Check if URL handling dependencies are installed."""
    missing_deps = []
    try:
        import langchain_community
    except ImportError:
        missing_deps.append("langchain-community")
    
    try:
        import langchain
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import unstructured
    except ImportError:
        missing_deps.append("unstructured")
    
    try:
        import bs4
    except ImportError:
        missing_deps.append("bs4")
    
    if missing_deps:
        print("WARNING: Missing packages for URL handling:", ", ".join(missing_deps))
        print("To install: pip install " + " ".join(missing_deps))
        print("Without these packages, /inject <URL> may not work properly.")
    else:
        print("URL handling dependencies: OK")

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    check_url_dependencies()
    app.run(host=host, port=port)
