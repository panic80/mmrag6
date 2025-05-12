import os
import subprocess
import threading
from flask import Flask, request, jsonify
import requests
from pathlib import Path # Added import
import openai # Added for OpenAI API
import shlex # Already present, but good to note
import sys # Added for print to stderr
import datetime # Added for timestamps in Mattermost messages

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

                def _post_message(txt: str, attachments: list = None):
                    """Post *txt* to the originating channel, optionally with attachments."""
                    # nonlocal rest_usable # rest_usable is in the scope of run_and_post

                    post_data = {
                        "channel_id": channel_id,
                        "message": txt
                    }
                    if attachments:
                        post_data["props"] = {"attachments": attachments}

                    if rest_usable["ok"]:
                        try:
                            hdrs = {"Authorization": f"Bearer {mattermost_token}"}
                            resp = requests.post(
                                f"{mattermost_url}/api/v4/posts",
                                headers=hdrs,
                                json=post_data, # Use the constructed post_data
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
                        response_payload = {"response_type": "in_channel", "text": txt}
                        if attachments: # Mattermost supports attachments in the initial response_url post
                            response_payload["props"] = {"attachments": attachments}
                        try:
                            requests.post(
                                response_url,
                                json=response_payload,
                                timeout=10,
                            )
                            return
                        except Exception:
                            app.logger.exception("Failed to post via response_url")
                
                try:
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    if not openai_api_key:
                        error_attachment = {
                            "color": "#DC3545", # Red for error
                            "title": "Query Processing Error",
                            "text": "OPENAI_API_KEY environment variable not set.",
                            "fields": [{"short": False, "title": "Suggestion", "value": "Please contact an administrator to configure the OpenAI API key."}]
                        }
                        _post_message(txt="⚠️ Configuration Error", attachments=[error_attachment])
                        return

                    client = openai.OpenAI(api_key=openai_api_key) # This client init seems redundant if only LlamaIndex is used now.
                                                                # However, LlamaOpenAI itself will use the API key.
                                                                # Let's keep it for now, or remove if truly unused.
                                                                # For now, the LlamaIndex settings will handle OpenAI interactions.

                    # query_text is now query_text_param
                    if not query_text_param:
                        error_attachment = {
                            "color": "#DC3545", # Red for error
                            "title": "Query Processing Error",
                            "text": "No query text provided.",
                            "fields": [{"short": False, "title": "Usage", "value": "Please provide a query after the `/ask` command."}]
                        }
                        _post_message(txt="⚠️ Invalid Usage", attachments=[error_attachment])
                        return

                    # openai_api_key is already fetched and checked once.
                    # This second check is redundant if the first one passes.
                    # openai_api_key = os.environ.get("OPENAI_API_KEY") 
                    # Redundant check removed, initial check with attachment is sufficient.

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
                        error_attachment = {
                            "color": "#DC3545",
                            "title": "Query Initialization Error",
                            "text": f"Failed to initialize LlamaIndex models: {e_settings}",
                            "fields": [{"short": False, "title": "Suggestion", "value": "Check server logs for details."}]
                        }
                        _post_message(txt="⚠️ Initialization Error", attachments=[error_attachment])
                        return

                    # Determine collection name (already passed as collection_name_param)
                    # Determine persist path for the local docstore
                    persist_dir_base = Path("./storage_llamaindex_db")
                    local_docstore_persist_path = persist_dir_base / collection_name_param
                    
                    # Log storage paths for debugging
                    app.logger.info(f"[storage] Base persist directory: {persist_dir_base}")
                    app.logger.info(f"[storage] Full docstore path: {local_docstore_persist_path}")
                    app.logger.info(f"[storage] Directory exists: {local_docstore_persist_path.exists()}")
                    app.logger.info(f"[storage] Is directory: {local_docstore_persist_path.is_dir() if local_docstore_persist_path.exists() else 'N/A'}")
                    if local_docstore_persist_path.exists():
                        app.logger.info(f"[storage] Contents: {list(local_docstore_persist_path.iterdir())}")

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
                        error_attachment = {
                            "color": "#DC3545",
                            "title": "Data Store Error",
                            "text": f"Qdrant collection '{collection_name_param}' not found or Qdrant is inaccessible.",
                            "fields": [
                                {"short": False, "title": "Suggestion", "value": f"Ensure ingestion for collection '{collection_name_param}' was successful. Check Qdrant server status."},
                                {"short": False, "title": "Details", "value": str(e_qdrant_check)}
                            ]
                        }
                        _post_message(txt="⚠️ Data Access Error", attachments=[error_attachment])
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
                        docstore_valid = False
                        
                        if docstore_json_file.exists():
                            # Validate docstore.json content
                            import json
                            try:
                                with open(docstore_json_file) as f:
                                    docstore_content = json.load(f)
                                if not docstore_content or not docstore_content.get("docstore", {}).get("docs"):
                                    app.logger.warning(f"[warning] docstore.json exists but appears empty or has no documents at {docstore_json_file}")
                                    app.logger.warning(f"[debug] docstore content: {docstore_content}")
                                else:
                                    app.logger.info(f"[success] Valid docstore found at {docstore_json_file} with {len(docstore_content.get('docstore', {}).get('docs', {}))} documents")
                                    docstore_valid = True
                            except json.JSONDecodeError as e:
                                app.logger.error(f"[error] Invalid JSON in docstore at {docstore_json_file}: {e}")
                            except Exception as e:
                                app.logger.error(f"[error] Failed to read docstore at {docstore_json_file}: {e}")
                        else:
                            app.logger.warning(f"Local docstore.json not found in {str(local_docstore_persist_path)}.")
                        
                        # Load storage context with vector store and possibly docstore
                        app.logger.info(f"Loading StorageContext from {str(local_docstore_persist_path)}.")
                        loaded_storage_context = StorageContext.from_defaults(
                            vector_store=vector_store,
                            persist_dir=str(local_docstore_persist_path)
                        )
                        
                        # Check if docstore is empty after loading
                        if not docstore_valid and hasattr(loaded_storage_context, 'docstore') and not loaded_storage_context.docstore.docs:
                            app.logger.warning(f"[warning] Loaded StorageContext but docstore is empty. Will attempt to populate from vector store.")
                    else:
                        app.logger.warning(f"Local docstore path {str(local_docstore_persist_path)} not found. Creating directory.")
                        try:
                            os.makedirs(local_docstore_persist_path, exist_ok=True)
                            app.logger.info(f"Created directory: {local_docstore_persist_path}")
                        except Exception as e_mkdir:
                            app.logger.error(f"[error] Failed to create directory {local_docstore_persist_path}: {e_mkdir}")
                        
                        loaded_storage_context = StorageContext.from_defaults(vector_store=vector_store)

                    # Load the index using the Qdrant vector store
                    index = VectorStoreIndex.from_vector_store(
                        vector_store=vector_store,
                        storage_context=loaded_storage_context # This context now includes Qdrant VS and potentially local docstore
                    )
                    
                    # Check if docstore is empty and try to populate it from the vector store
                    if hasattr(loaded_storage_context, 'docstore') and not loaded_storage_context.docstore.docs:
                        app.logger.warning("[warning] Docstore is empty. Attempting to populate from vector store nodes...")
                        
                        try:
                            # Try to get nodes from the index
                            if hasattr(index, '_all_nodes_dict') and index._all_nodes_dict:
                                app.logger.info(f"[info] Found {len(index._all_nodes_dict)} nodes in index. Populating docstore...")
                                for node_id, node in index._all_nodes_dict.items():
                                    if not loaded_storage_context.docstore.document_exists(node_id):
                                        loaded_storage_context.docstore.add_documents([node], allow_update=True)
                                
                                # Persist the populated docstore
                                if loaded_storage_context.docstore.docs:
                                    app.logger.info(f"[success] Populated docstore with {len(loaded_storage_context.docstore.docs)} nodes from index.")
                                    try:
                                        loaded_storage_context.persist(persist_dir=str(local_docstore_persist_path))
                                        app.logger.info(f"[success] Persisted populated docstore to {local_docstore_persist_path}")
                                    except Exception as e_persist:
                                        app.logger.error(f"[error] Failed to persist populated docstore: {e_persist}")
                                else:
                                    app.logger.warning("[warning] Failed to populate docstore from index nodes.")
                            else:
                                app.logger.warning("[warning] No nodes found in index to populate docstore.")
                        except Exception as e_populate:
                            app.logger.error(f"[error] Error while attempting to populate docstore: {e_populate}")
                    
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
                    
                    mmr_lambda_val = float(os.environ.get("RETRIEVAL_MMR_LAMBDA", 0.8))
                    
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
                            query_kwargs['similarity_top_k'] = 20 # Default value
                        except ValueError:
                            print(f"[WARNING] Invalid RETRIEVAL_SIMILARITY_TOP_K: {env_similarity_top_k}. query_llamaindex.py default will be used.", file=sys.stderr)

                    env_sparse_top_k = os.environ.get("RETRIEVAL_SPARSE_TOP_K")
                    if env_sparse_top_k is not None:
                        try:
                            query_kwargs['sparse_top_k'] = 10
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
                    # answer_text = f"An error occurred while processing your query with LlamaIndex. Details: {e}" # Replaced by attachment
                    error_attachment = {
                        "color": "#DC3545",
                        "title": "Query Execution Error",
                        "text": f"An error occurred while processing your query with the LlamaIndex pipeline.",
                        "fields": [
                            {"short": False, "title": "Details", "value": str(e)},
                            {"short": False, "title": "Suggestion", "value": "Check server logs for a full traceback."}
                        ]
                    }
                    _post_message(txt=f"⚠️ Error during query for: {original_query_text_param_for_logging}", attachments=[error_attachment])
                    return # Exit after posting error

                # Compose final message with attachments
                answer_attachment = {
                    "color": "#28A745", # Green for success
                    "author_name": "RAG System",
                    "title": f"Re: {original_query_text_param_for_logging}",
                    "text": answer_text if answer_text else "No answer could be generated.",
                    "footer": f"Powered by LlamaIndex & OpenAI ({openai_model_llm_val})", # Use llm_model_name from earlier
                    "ts": datetime.datetime.now().timestamp()
                }
                _post_message(txt=f"**Q:** {original_query_text_param_for_logging}", attachments=[answer_attachment])


            # Always run asynchronously
            threading.Thread(target=run_and_post, args=(text, final_collection_name_for_ack), daemon=True).start()
            
            # Immediate acknowledgement using attachments
            ack_payload = {
                # "response_type": "in_channel", # This is for response_url, direct post is always in_channel
                "text": f"⏳ Processing your query with LlamaIndex (collection: {final_collection_name_for_ack})...",
                "props": {
                    "attachments": [{
                        "color": "#007BFF", # Blue for informational
                        "text": "Your query is being processed. An answer will appear shortly."
                    }]
                }
            }
            # For immediate response, we can use jsonify with props for attachments if Mattermost supports it directly
            # in the slash command immediate response. Otherwise, a simple text ack is safer.
            # Let's stick to the simpler text ack for the immediate response to avoid potential issues,
            # as the main answer will have rich formatting.
            return jsonify({"response_type": "in_channel", "text": f"⏳ Processing your query with LlamaIndex (collection: {final_collection_name_for_ack}). Your answer will appear shortly."}), 200


        # 'inject' ingests the current channel into the RAG collection
        elif cmd_name in ("inject", "injest"):
            import shlex
            def run_inject():
                """Handle the /inject (or /injest) command in a background thread.
                Sends real-time, concise updates to the Mattermost channel with threading and attachments.
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
                current_thread_root_id: str | None = None # For threading replies

                def _send_message_to_mattermost(message_text: str, attachments: list = None, is_first_message: bool = False):
                    """Sends a message immediately to the Mattermost channel.
                    Can post as a new message or a reply, and can include attachments.
                    Updates current_thread_root_id if it's the first message and successful.
                    """
                    nonlocal rest_usable, current_thread_root_id

                    if not message_text and not attachments:
                        return

                    post_data = {
                        "channel_id": channel_id,
                        "message": message_text
                    }
                    if attachments:
                        post_data["props"] = {"attachments": attachments}
                    
                    if not is_first_message and current_thread_root_id:
                        post_data["root_id"] = current_thread_root_id

                    if rest_usable["ok"]:
                        try:
                            hdrs = {"Authorization": f"Bearer {mattermost_token}"}
                            resp = requests.post(
                                f"{mattermost_url}/api/v4/posts",
                                headers=hdrs,
                                json=post_data,
                                timeout=10,
                            )
                            if resp.status_code in (200, 201):
                                if is_first_message and not current_thread_root_id: # Store root_id only once
                                    try:
                                        current_thread_root_id = resp.json().get("id")
                                        app.logger.info(f"Captured root_id for inject thread: {current_thread_root_id}")
                                    except Exception as e_json:
                                        app.logger.error(f"Could not parse post ID from response for threading: {e_json}")
                                return
                            if resp.status_code in (401, 403):
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
                        # Fallback for response_url: threading and complex follow-up attachments are less reliable.
                        # Send as new message if REST failed.
                        fallback_payload = {"response_type": "in_channel", "text": message_text}
                        if is_first_message and attachments: # Try to send attachments for the very first message
                            fallback_payload["props"] = {"attachments": attachments}
                        
                        # If it's a follow-up and REST failed, we can't reliably thread.
                        # Prepend to indicate it's an update.
                        if not is_first_message and not rest_usable["ok"] and current_thread_root_id:
                             # Indicate it's part of an ongoing operation if threading failed
                            fallback_payload["text"] = f"(Update for /inject) {message_text}"

                        try:
                            requests.post(response_url, json=fallback_payload, timeout=10)
                            if is_first_message: # If REST failed for the first message, response_url was used.
                                app.logger.warning("First inject message sent via response_url; threading may not be available for follow-ups if REST remains down.")
                            return
                        except requests.exceptions.RequestException as e_resp_url:
                            app.logger.exception(f"Failed to post to response_url: {e_resp_url}")
                    
                    app.logger.error(f"Failed to send message to Mattermost by any method: {message_text}")

                # Main ingestion logic
                try:
                    _send_message_to_mattermost(
                        message_text="⚙️ **`/inject` command received.** Processing...",
                        attachments=[{
                            "color": "#007BFF", # Blue for informational
                            "fields": [{"short": True, "title": "Status", "value": "Initializing"}]
                        }],
                        is_first_message=True
                    )
                    raw_args_from_text = shlex.split(text or "")
                    
                    collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "rag_llamaindex_data")
                    temp_collection_args = []
                    # Helper for sending attachment-based messages for arg parsing errors
                    def _send_arg_error(title, text_detail):
                        _send_message_to_mattermost(
                            message_text="⚠️ Argument Error",
                            attachments=[{"color": "#FFC107", "title": title, "text": text_detail}]
                        )

                    if "--collection-name" in raw_args_from_text:
                        try:
                            idx = raw_args_from_text.index("--collection-name")
                            collection_name = raw_args_from_text[idx + 1]
                            temp_collection_args.extend([raw_args_from_text[idx], raw_args_from_text[idx+1]])
                        except (ValueError, IndexError):
                            _send_arg_error("Invalid Collection Name Usage", "Invalid --collection-name usage. Using default or environment variable.")
                    elif "-c" in raw_args_from_text:
                        try:
                            idx = raw_args_from_text.index("-c")
                            collection_name = raw_args_from_text[idx + 1]
                            temp_collection_args.extend([raw_args_from_text[idx], raw_args_from_text[idx+1]])
                        except (ValueError, IndexError):
                             _send_arg_error("Invalid Collection Name Usage", "Invalid -c usage for collection name. Using default or environment variable.")
                    
                    is_purge_command = "--purge" in raw_args_from_text
                    
                    if is_purge_command:
                        _send_message_to_mattermost(
                            message_text=f"ℹ️ Purge command for '{collection_name}'",
                            attachments=[{
                                "color": "#007BFF", "title": "Purge Operation Started",
                                "text": f"Attempting to delete and recreate collection: `{collection_name}`."
                            }]
                        )
                        qdrant_url_env = os.environ.get("QDRANT_URL", "http://qdrant:6333")
                        qdrant_api_key_env = os.environ.get("QDRANT_API_KEY")
                        
                        try:
                            q_client = QdrantClient(url=qdrant_url_env, api_key=qdrant_api_key_env)
                            q_client.delete_collection(collection_name=collection_name)
                            _send_message_to_mattermost(message_text="", attachments=[{
                                "color": "#28A745", "title": "Collection Deleted",
                                "text": f"Successfully deleted collection `{collection_name}`."
                            }])
                        except Exception as e_del:
                            app.logger.error(f"Failed to delete Qdrant collection '{collection_name}': {e_del}\n{traceback.format_exc()}")
                            _send_message_to_mattermost(message_text="", attachments=[{
                                "color": "#DC3545", "title": "Deletion Failed",
                                "text": f"Could not delete collection `{collection_name}`. It might not exist or an error occurred. Details: {e_del}",
                                "fields": [{"short": False, "title": "Action", "value": "Check server logs."}]
                            }])
                        
                        try:
                            from qdrant_client.http import models as qdrant_models
                            vector_size = int(os.environ.get("DEFAULT_VECTOR_SIZE", 3072))
                            if 'q_client' not in locals(): # Ensure client is initialized
                                q_client = QdrantClient(url=qdrant_url_env, api_key=qdrant_api_key_env)
                            q_client.recreate_collection(
                                collection_name=collection_name,
                                vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE)
                            )
                            _send_message_to_mattermost(message_text="", attachments=[{
                                "color": "#28A745", "title": "Collection Recreated",
                                "text": f"Successfully recreated empty collection `{collection_name}` (vector size: {vector_size})."
                            }])
                        except Exception as e_recreate:
                            app.logger.error(f"Failed to recreate Qdrant collection '{collection_name}': {e_recreate}\n{traceback.format_exc()}")
                            _send_message_to_mattermost(message_text="", attachments=[{
                                "color": "#DC3545", "title": "Recreation Failed",
                                "text": f"Failed to recreate collection `{collection_name}`. Details: {e_recreate}",
                                "fields": [{"short": False, "title": "Action", "value": "Check server logs."}]
                            }])
                            return # Stop if recreate fails

                        args_for_source_check = [arg for arg in raw_args_from_text if arg != "--purge" and arg not in temp_collection_args]
                        is_purge_only = not any(not arg.startswith("-") for arg in args_for_source_check)

                        if is_purge_only:
                            _send_message_to_mattermost(message_text="", attachments=[{
                                "color": "#007BFF", "title": "Purge Complete",
                                "text": f"Purge operation for `{collection_name}` finished. No sources specified for further ingestion."
                            }])
                            return

                    ingest_cmd_base = [sys.executable, "-u", "-m", "ingest_llamaindex", "--collection-name", collection_name]
                    ingest_passthrough_args = []
                    potential_sources_from_args = []
                    args_to_filter_for_ingest = [arg for arg in raw_args_from_text if arg != "--purge" and arg not in temp_collection_args]

                    for arg in args_to_filter_for_ingest:
                        if arg == "--rich-metadata": continue # Default, no need to pass
                        elif arg == "--no-rich-metadata": ingest_passthrough_args.append(arg)
                        elif arg in ("--generate-summaries", "--no-generate-summaries", "--quality-checks", "--no-quality-checks", "--crawl-depth", "--depth-crawl", "--parallel"):
                            _send_message_to_mattermost(message_text="", attachments=[{
                                "color": "#FFC107", "title": "Unsupported Flag",
                                "text": f"Flag `{arg}` is no longer supported and will be ignored."
                            }])
                        elif not arg.startswith("-"): potential_sources_from_args.append(arg)
                        else: ingest_passthrough_args.append(arg) # Pass other flags through
                    
                    ingest_cmd_base.extend(ingest_passthrough_args)
                    final_sources_to_process = []
                    temp_files_to_clean = []

                    if potential_sources_from_args:
                        final_sources_to_process = potential_sources_from_args
                    else: # No explicit sources, try to fetch from current channel
                        if not mattermost_url or not channel_id or not mattermost_token: # Added mattermost_token check
                            _send_message_to_mattermost(message_text="❌ Configuration Error", attachments=[{
                                "color": "#DC3545", "title": "Cannot Fetch Channel History",
                                "text": "MATTERMOST_URL, MATTERMOST_TOKEN, or channel_id not configured. Unable to fetch channel messages for ingestion."
                            }])
                            return
                        
                        _send_message_to_mattermost(message_text="", attachments=[{
                            "color": "#007BFF", "title": "Fetching Channel Transcript",
                            "text": "No source explicitly provided. Attempting to fetch current channel transcript for ingestion..."
                        }])
                        msgs = []
                        hdrs = {"Authorization": f"Bearer {mattermost_token}"} # Already checked mattermost_token
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
                                _send_message_to_mattermost(message_text="❌ Fetch Error", attachments=[{
                                    "color": "#DC3545", "title": "Channel History Fetch Failed",
                                    "text": f"Error fetching channel history: {e_fetch}. Check server logs."
                                }])
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
                        
                        if not channel_fetch_ok: return # Error message already sent
                        if not msgs:
                            _send_message_to_mattermost(message_text="", attachments=[{
                                "color": "#007BFF", "title": "No Messages Found",
                                "text": "No messages found in the current channel to ingest."
                            }])
                            return
                        
                        try:
                            tmp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
                            tmp_file.write("\n".join(m.rstrip("\n") for m in msgs)) # Ensure messages are strings
                            tmp_file.close()
                            final_sources_to_process = [tmp_file.name]
                            temp_files_to_clean.append(tmp_file.name)
                            _send_message_to_mattermost(message_text="", attachments=[{
                                "color": "#28A745", "title": "Channel Transcript Fetched",
                                "text": f"Fetched {len(msgs)} messages from channel. Prepared for ingestion as `{tmp_file.name}`."
                            }])
                        except IOError as e_io:
                            app.logger.error(f"Failed to write channel transcript to temp file: {e_io}\n{traceback.format_exc()}")
                            _send_message_to_mattermost(message_text="❌ File Error", attachments=[{
                                "color": "#DC3545", "title": "Transcript Preparation Failed",
                                "text": "Error preparing channel transcript for ingestion. Check server logs."
                            }])
                            return

                    if not final_sources_to_process:
                        _send_message_to_mattermost(message_text="", attachments=[{
                            "color": "#007BFF", "title": "No Sources",
                            "text": "No sources to process for ingestion."
                        }])
                        return

                    all_sources_successful = True
                    num_sources = len(final_sources_to_process)
                    for src_idx, source_item in enumerate(final_sources_to_process):
                        _send_message_to_mattermost(
                            message_text="", 
                            attachments=[{
                                "color": "#FFA500", # Orange for in-progress
                                "title": f"🚀 Processing Source ({src_idx+1}/{num_sources}): `{source_item}`",
                                "text": "Starting ingestion for this source..."
                            }]
                        )
                        
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
                                _send_message_to_mattermost(message_text="", attachments=[{
                                    "color": "#DC3545", "title": f"❌ Error: Source `{source_item}`",
                                    "text": f"Processing failed (exit code: {ret}). Check server logs for details."
                                }])
                                all_sources_successful = False
                            else:
                                _send_message_to_mattermost(message_text="", attachments=[{
                                    "color": "#28A745", "title": f"✅ Success: Source `{source_item}`",
                                    "text": "Successfully processed and ingested."
                                }])
                        except Exception as e_proc:
                            app.logger.error(f"Failed to start/run ingestion subprocess for '{source_item}': {e_proc}\n{traceback.format_exc()}")
                            _send_message_to_mattermost(message_text="", attachments=[{
                                "color": "#DC3545", "title": f"❌ Critical Error: Source `{source_item}`",
                                "text": f"Critical error during processing. Details: {e_proc}. Check server logs."
                            }])
                            all_sources_successful = False
                            continue 
                    
                    for tf_path in temp_files_to_clean:
                        try:
                            os.remove(tf_path)
                            app.logger.info(f"Cleaned up temporary file: {tf_path}")
                        except OSError as e_clean:
                            app.logger.warning(f"Failed to clean up temporary file '{tf_path}': {e_clean}")
                    
                    if final_sources_to_process: # Only send summary if sources were attempted
                        if all_sources_successful:
                            _send_message_to_mattermost(
                                message_text="**`/inject` process finished.**",
                                attachments=[{
                                    "color": "#28A745", "title": "🎉 Ingestion Complete",
                                    "text": f"All {num_sources} specified source(s) were processed successfully into collection `{collection_name}`.",
                                    "fields": [{"short": True, "title": "Collection", "value": collection_name}]
                                }]
                            )
                        else:
                            _send_message_to_mattermost(
                                message_text="**`/inject` process finished.**",
                                attachments=[{
                                    "color": "#FFC107", "title": "⚠️ Ingestion Finished with Issues",
                                    "text": f"Some sources encountered errors during processing for collection `{collection_name}`. Please check previous messages in this thread and server logs for details.",
                                    "fields": [{"short": True, "title": "Collection", "value": collection_name}]
                                }]
                            )
                except Exception as e_outer:
                    app.logger.error(f"An unexpected error occurred during the inject process: {e_outer}\n{traceback.format_exc()}")
                    _send_message_to_mattermost(
                        message_text="❌ Critical Error in `/inject`",
                        attachments=[{
                            "color": "#DC3545", "title": "Critical Inject Process Error",
                            "text": f"An unexpected critical error occurred. Details: {e_outer}. Please check server logs."
                        }]
                    )

            threading.Thread(target=run_inject, daemon=True).start()
            # Immediate acknowledgement for /inject
            return jsonify({
                "response_type": "in_channel", # This ensures it's visible
                "text": "🚀 **`/inject` command received.** An initial status message will appear shortly in a new thread (if supported by your Mattermost version and REST API is working).",
                 # Props for attachments in immediate response might not always work as expected,
                 # but some versions of MM might pick it up. The first _send_message_to_mattermost will be more reliable.
                "props": {
                    "attachments": [{
                        "color": "#007BFF",
                        "text": "Ingestion process is starting. Progress will be posted in a thread."
                    }]
                }
            }), 200
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
