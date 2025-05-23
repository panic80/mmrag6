#!/usr/bin/env python3
"""query_llamaindex.py

CLI to query a Qdrant RAG collection using LlamaIndex.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path # Added missing import
from typing import Sequence, Any, List, Dict, Optional

import click
from dotenv import load_dotenv

# LlamaIndex Core
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    QueryBundle,
)
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
# Try different import paths for BM25Retriever - Fallbacks removed
from llama_index.retrievers.bm25 import BM25Retriever


# Custom HybridRetriever implementation since the official package isn't available
class HybridRetriever(BaseRetriever):
    """Custom implementation of a Hybrid Retriever that combines dense and sparse retrieval.
    
    This is used when the official llama-index-retrievers-hybrid package isn't available.
    """
    
    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        mode: str = "OR",
    ):
        """Initialize with dense and sparse retrievers.
        
        Args:
            dense_retriever: Vector retriever for dense embeddings
            sparse_retriever: BM25 or similar retriever for sparse retrieval
            mode: How to combine results - "OR" (union) or "AND" (intersection)
        """
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._mode = mode
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        # Get results from both retrievers
        dense_nodes = self._dense_retriever.retrieve(query_bundle)
        if _VERBOSE_OUTPUT: click.echo(f"[debug][HybridRetriever] Dense retriever found {len(dense_nodes)} nodes.")
        sparse_nodes = self._sparse_retriever.retrieve(query_bundle)
        if _VERBOSE_OUTPUT: click.echo(f"[debug][HybridRetriever] Sparse retriever found {len(sparse_nodes)} nodes.")
        
        # Combine results based on mode
        if self._mode == "AND":
            # Only keep nodes that appear in both results
            dense_ids = {n.node.node_id: n for n in dense_nodes}
            sparse_ids = {n.node.node_id: n for n in sparse_nodes}
            common_ids = set(dense_ids.keys()).intersection(set(sparse_ids.keys()))
            
            # Use the better score for each node
            combined_nodes = []
            for node_id in common_ids:
                dense_node = dense_ids[node_id]
                sparse_node = sparse_ids[node_id]
                # Use the higher score of the two
                if (dense_node.score or 0.0) > (sparse_node.score or 0.0):
                    combined_nodes.append(dense_node)
                else:
                    combined_nodes.append(sparse_node)
            
            return combined_nodes
        else:  # Default mode: "OR"
            # Combine and deduplicate
            node_dict = {}
            
            # Add dense nodes
            for node in dense_nodes:
                node_dict[node.node.node_id] = node
            
            # Add sparse nodes, keeping the higher score if already exists
            for node in sparse_nodes:
                node_id = node.node.node_id
                if node_id in node_dict:
                    existing_score = node_dict[node_id].score or 0.0
                    new_score = node.score or 0.0
                    if new_score > existing_score:
                        node_dict[node_id] = node
                else:
                    node_dict[node_id] = node
            
            # Convert back to list and sort by score
            combined_nodes = list(node_dict.values())
            combined_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            
            return combined_nodes
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes given query."""
        # Get results from both retrievers asynchronously
        dense_nodes = await self._dense_retriever.aretrieve(query_bundle)
        if _VERBOSE_OUTPUT: click.echo(f"[debug][HybridRetriever] Dense retriever (async) found {len(dense_nodes)} nodes.")
        sparse_nodes = await self._sparse_retriever.aretrieve(query_bundle)
        if _VERBOSE_OUTPUT: click.echo(f"[debug][HybridRetriever] Sparse retriever (async) found {len(sparse_nodes)} nodes.")
        
        # Combine results based on mode
        if self._mode == "AND":
            # Only keep nodes that appear in both results
            dense_ids = {n.node.node_id: n for n in dense_nodes}
            sparse_ids = {n.node.node_id: n for n in sparse_nodes}
            common_ids = set(dense_ids.keys()).intersection(set(sparse_ids.keys()))
            
            # Use the better score for each node
            combined_nodes = []
            for node_id in common_ids:
                dense_node = dense_ids[node_id]
                sparse_node = sparse_ids[node_id]
                # Use the higher score of the two
                if (dense_node.score or 0.0) > (sparse_node.score or 0.0):
                    combined_nodes.append(dense_node)
                else:
                    combined_nodes.append(sparse_node)
            
            return combined_nodes
        else:  # Default mode: "OR"
            # Combine and deduplicate
            node_dict = {}
            
            # Add dense nodes
            for node in dense_nodes:
                node_dict[node.node.node_id] = node
            
            # Add sparse nodes, keeping the higher score if already exists
            for node in sparse_nodes:
                node_id = node.node.node_id
                if node_id in node_dict:
                    existing_score = node_dict[node_id].score or 0.0
                    new_score = node.score or 0.0
                    if new_score > existing_score:
                        node_dict[node_id] = node
                else:
                    node_dict[node_id] = node
            
            # Convert back to list and sort by score
            combined_nodes = list(node_dict.values())
            combined_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            
            return combined_nodes

# from llama_index.core.query_engine import RetrieverQueryEngine # Replaced by QueryPipeline
from llama_index.core.query_pipeline import QueryPipeline, Link, InputComponent
from llama_index.core.response_synthesizers import get_response_synthesizer, BaseSynthesizer

from llama_index.core.postprocessor import (
    SimilarityPostprocessor, # Basic relevance filtering
    LongContextReorder, # For reordering long contexts
    MetadataReplacementPostProcessor, # For replacing content with metadata values
)
# Implement custom MMRNodePostprocessor since the package isn't available
from typing import List, Optional, Sequence
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor

class MMRNodePostprocessor(BaseNodePostprocessor):
    """MaximalMarginalRelevance-based node postprocessor.
    
    Filters nodes by computing Maximal Marginal Relevance 
    with respect to the query embedding.
    """
    
    def __init__(
        self,
        embed_model: Optional[BaseEmbedding] = None,
        top_n: int = 10,
        lambda_mult: float = 0.5,
    ) -> None:
        """Initialize with parameters.
        
        Args:
            embed_model (BaseEmbedding): Embedding model to use for embedding query.
            top_n (int): Number of nodes to return (default: 10).
            lambda_mult (float): Diversity weight parameter (0 to 1).
                0 means maximal diversity, 1 means maximal relevance.
        """
        self._embed_model = embed_model
        self._top_n = top_n
        self._lambda_mult = lambda_mult
    
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if not nodes:
            return []
        
        if len(nodes) <= self._top_n:
            return nodes
        
        if self._embed_model is None:
            # Let's try to get the embed model from Settings
            from llama_index.core import Settings
            self._embed_model = Settings.embed_model
            if self._embed_model is None:
                # No embedding model available, just return the top_n nodes by score
                return nodes[:self._top_n]
        
        # Get query embedding
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        
        # Get node embeddings
        node_embeddings = []
        for node in nodes:
            if hasattr(node.node, "embedding") and node.node.embedding is not None:
                node_embeddings.append(node.node.embedding)
            else:
                # If node doesn't have embedding, get it from the text
                node_text = node.node.get_content(metadata_mode="all")
                node_embedding = self._embed_model.get_text_embedding(node_text)
                node_embeddings.append(node_embedding)
        
        # First, add the highest-scoring node
        selected_indices = [0]  # Start with the first node
        selected_embeddings = [node_embeddings[0]]
        
        # Then, iteratively add the node with the highest MMR score
        remaining_indices = list(range(1, len(nodes)))
        
        while len(selected_indices) < self._top_n and remaining_indices:
            highest_mmr_score = float("-inf")
            highest_mmr_idx = -1
            highest_mmr_remaining_idx = -1
            
            for i, remaining_idx in enumerate(remaining_indices):
                # Calculate relevance score (similarity to query)
                relevance_score = self._cosine_similarity(
                    query_embedding, node_embeddings[remaining_idx]
                )
                
                # Calculate diversity score (inverse of max similarity to already selected)
                max_similarity = float("-inf")
                for selected_embedding in selected_embeddings:
                    similarity = self._cosine_similarity(
                        node_embeddings[remaining_idx], selected_embedding
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # If no selected nodes yet, set max_similarity to 0
                if max_similarity == float("-inf"):
                    max_similarity = 0
                
                # Calculate MMR score
                diversity_score = -max_similarity  # Negative because we want dissimilarity
                mmr_score = (
                    self._lambda_mult * relevance_score 
                    + (1 - self._lambda_mult) * diversity_score
                )
                
                if mmr_score > highest_mmr_score:
                    highest_mmr_score = mmr_score
                    highest_mmr_idx = remaining_idx
                    highest_mmr_remaining_idx = i
            
            if highest_mmr_idx != -1:
                selected_indices.append(highest_mmr_idx)
                selected_embeddings.append(node_embeddings[highest_mmr_idx])
                remaining_indices.pop(highest_mmr_remaining_idx)
            else:
                break
        
        # Return the selected nodes in their original order
        selected_indices.sort()
        result_nodes = [nodes[i] for i in selected_indices]
        
        return result_nodes
    
    def _cosine_similarity(self, embedding1: Sequence[float], embedding2: Sequence[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        import numpy as np
        
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same length")
            
        embedding1_np = np.array(embedding1)
        embedding2_np = np.array(embedding2)
        
        dot_product = np.dot(embedding1_np, embedding2_np)
        norm1 = np.linalg.norm(embedding1_np)
        norm2 = np.linalg.norm(embedding2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

# Try different import paths for rerankers - Fallbacks removed
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from llama_index.core.node_parser import SentenceSplitter # For compression fallback
from llama_index.core.schema import NodeWithScore, BaseNode

# LlamaIndex Integrations
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
# Import evaluation tools conditionally
try:
    from llama_index.core.evaluation import ( # For RAG evaluation
        FaithfulnessEvaluator,
        RelevancyEvaluator,
        CorrectnessEvaluator, 
        SemanticSimilarityEvaluator,
    )
    evaluation_available = True
except ImportError:
    evaluation_available = False
    click.echo("[warning] llama-index-evaluation module not available. Evaluation features will be disabled.")


# Qdrant client
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as qdrant_models # For filter creation

# Project-specific modules
# advanced_rag.py functionalities will be replaced or adapted using LlamaIndex components.
# For example, query expansion via OpenAIQuestionGenerator or similar.
# Contextual compression via rerankers or custom postprocessors.
# Evaluation via llama-index-evaluation.

# Global flag for verbose output
_VERBOSE_OUTPUT = False

# Helper for MMR (adapted from original query_rag.py, may need LlamaIndex equivalent or custom postprocessor)
# LlamaIndex has an MMRNodePostprocessor, but it's not directly a retriever.
# For now, this is a placeholder if we need custom MMR logic.
# However, Qdrant itself supports diversity via `limit` and `offset` in search, or diverse vector search.
# LlamaIndex's `VectorIndexRetriever` can also take `similarity_top_k` and then apply postprocessors.
# Custom MMR function _mmr_rerank_nodes is now removed in favor of MMRNodePostprocessor.

async def main_async(
    query_text: Sequence[str], # Moved to front, as it's the primary required argument
    *, # Make subsequent arguments keyword-only
    collection_name: str = "rag_llamaindex_data",
    similarity_top_k: int = 100, # Changed default
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: str | None = None,
    openai_api_key: str | None = None,
    openai_model_embedding: str = "text-embedding-3-large",
    openai_model_llm: str = "gpt-4.1-mini",
    llm_temperature: float = 0.0, # Changed from llm_temperature to 0.0 for deterministic output
    raw_output_flag: bool = False,
    use_hybrid_search: bool = True,
    sparse_top_k: int = 100, # Changed default
    rerank_top_n: int = 150,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
    filters_kv: Sequence[str] = (), # Default to empty tuple for sequences
    evaluate_rag_flag: bool = False,
    compress_context_flag: bool = False,
    cohere_api_key: str | None = None,
    verbose: bool = False
):
    """Async logic for querying a LlamaIndex RAG setup."""
    global _VERBOSE_OUTPUT
    _VERBOSE_OUTPUT = verbose

    async_qdrant_client: AsyncQdrantClient | None = None
    response_value: str = "" # Initialize with a default empty string

    # Load .env if present
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=False)
        if _VERBOSE_OUTPUT: click.echo(f"[info] Environment variables loaded from {env_path}")

    final_openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    final_qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY")
    final_cohere_api_key = cohere_api_key or os.environ.get("COHERE_API_KEY")

    if not final_openai_api_key:
        click.echo("[fatal] OPENAI_API_KEY is not set.", err=True)
        sys.exit(1)

    try: # Main try block for the function, including client initialization and all operations
        # 1. Configure LlamaIndex Settings
        click.echo("[info] Configuring LlamaIndex Settings...")
        Settings.llm = LlamaOpenAI(
            model=openai_model_llm, 
            api_key=final_openai_api_key,
            temperature=0.0 # Changed from llm_temperature to 0.0 for deterministic output
        )
        Settings.embed_model = OpenAIEmbedding(model=openai_model_embedding, api_key=final_openai_api_key)
        # For SentenceSplitter used in compression fallback or if SemanticSplitter fails
        Settings.chunk_size = 1024 # Default for SentenceSplitter if used standalone
        Settings.chunk_overlap = 20

        # 2. Initialize Qdrant Client and VectorStore
        # Ensure collection_name is set with a consistent default
        effective_collection_name = collection_name
        if not effective_collection_name: # Check if collection_name is None or empty
            effective_collection_name = "llamaindex_default_collection"
            if _VERBOSE_OUTPUT or collection_name is None: # Print warning if it was defaulted
                 click.echo(f"[warning] --collection-name not provided or empty, using default Qdrant collection name: '{effective_collection_name}'. It's recommended to specify one.", err=True)
        
        click.echo(f"[info] Initializing Qdrant vector store for collection: {effective_collection_name}")

        if _VERBOSE_OUTPUT:
            click.echo(f"[info] Qdrant URL: {qdrant_url}")
            click.echo(f"[info] Qdrant API Key provided: {'Yes' if final_qdrant_api_key else 'No'}")

        # Initialize Qdrant Client and VectorStore - Fallback to SimpleVectorStore removed
        try:
            qdrant_native_client = QdrantClient(url=qdrant_url, api_key=final_qdrant_api_key)
            async_qdrant_client = AsyncQdrantClient(url=qdrant_url, api_key=final_qdrant_api_key)
            
            # Check if collection exists
            qdrant_native_client.get_collection(collection_name=effective_collection_name)
            if _VERBOSE_OUTPUT:
                click.echo(f"[info] Successfully connected to existing Qdrant collection: {effective_collection_name}")
            
            vector_store = QdrantVectorStore(
                client=qdrant_native_client,
                aclient=async_qdrant_client,
                collection_name=effective_collection_name
            )
        except Exception as e_qdrant:
            click.echo(f"[fatal] Failed to connect to Qdrant or access collection '{effective_collection_name}': {e_qdrant}", err=True)
            # Ensure async_qdrant_client is None if initialization failed before it was set
            if 'async_qdrant_client' in locals() and async_qdrant_client:
                 # Attempt to close if it was somehow initialized partially
                try:
                    # Create a new event loop for closing if one isn't running
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    loop.run_until_complete(async_qdrant_client.close())
                except Exception as e_close:
                    click.echo(f"[warning] Exception during async_qdrant_client.close() after error: {e_close}", err=True)
            sys.exit(1) # Exit if Qdrant connection fails

        base_persist_dir = Path("./storage_llamaindex_db") 
        collection_specific_persist_path = base_persist_dir / effective_collection_name
        
        if _VERBOSE_OUTPUT:
            click.echo(f"[info] Checking for local docstore at: {str(collection_specific_persist_path)}")
        
        storage_context_args = {"vector_store": vector_store}
        docstore_loaded_successfully = False

        if collection_specific_persist_path.exists() and collection_specific_persist_path.is_dir():
            docstore_json_path = collection_specific_persist_path / "docstore.json"
            if docstore_json_path.exists() and docstore_json_path.is_file():
                click.echo(f"[info] Found docstore.json at {docstore_json_path}. Will attempt to load StorageContext from: {str(collection_specific_persist_path)}")
                storage_context_args["persist_dir"] = str(collection_specific_persist_path)
            else:
                click.echo(f"[warning] Local docstore.json not found in '{str(collection_specific_persist_path)}'. BM25 may be limited if docstore is not loaded.", err=True)
        else:
            click.echo(f"[warning] Local persistence directory '{str(collection_specific_persist_path)}' not found. BM25 will be limited if docstore is not loaded.", err=True)

        try:
            storage_context = StorageContext.from_defaults(**storage_context_args)
            if hasattr(storage_context, 'docstore') and hasattr(storage_context.docstore, 'docs') and storage_context.docstore.docs:
                click.echo(f"[info] Successfully loaded/initialized Docstore with {len(storage_context.docstore.docs)} documents.")
                docstore_loaded_successfully = True
            elif "persist_dir" in storage_context_args:
                 click.echo(f"[warning] Loaded StorageContext from {storage_context_args['persist_dir']}, but Docstore is empty or has no documents. BM25 may be ineffective.")
                 
                 # Validate docstore.json file
                 if "persist_dir" in storage_context_args:
                     docstore_json_path = Path(storage_context_args["persist_dir"]) / "docstore.json"
                     if docstore_json_path.exists():
                         import json
                         try:
                             with open(docstore_json_path, 'r') as f:
                                 docstore_content = json.load(f)
                             if not docstore_content or not docstore_content.get("docstore", {}).get("docs"):
                                 click.echo(f"[warning] docstore.json exists at {docstore_json_path} but appears to be empty or invalid.")
                             else:
                                 click.echo(f"[warning] docstore.json at {docstore_json_path} contains data but wasn't loaded properly.")
                         except Exception as e_json:
                             click.echo(f"[warning] Error validating docstore.json at {docstore_json_path}: {e_json}")
            else:
                 click.echo(f"[info] Initialized new empty Docstore. BM25 will be ineffective unless nodes are added from vector store or other source.")
        except Exception as e_sc:
            click.echo(f"[warning] Failed to load or initialize StorageContext with persist_dir '{collection_specific_persist_path}': {e_sc}. Falling back to default StorageContext. BM25 will be limited.", err=True)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Try to load the index from storage context or create a new one
        try:
            if "persist_dir" in storage_context_args:
                # Try to load the index from storage
                click.echo(f"[info] Attempting to load index from {storage_context_args['persist_dir']}")
                try:
                    from llama_index.core import load_index_from_storage
                    index = load_index_from_storage(storage_context=storage_context)
                    click.echo(f"[info] Successfully loaded index from storage")
                except Exception as e_load:
                    click.echo(f"[warning] Failed to load index from storage: {e_load}. Creating new index from vector store.", err=True)
                    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
            else:
                # Create a new index from vector store
                index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        except ValueError as e_val:
            click.echo(f"[warning] Error creating index: {e_val}. Creating empty index with storage context.", err=True)
            # Create an empty index with the storage context
            index = VectorStoreIndex([], storage_context=storage_context)
        
        if not docstore_loaded_successfully and hasattr(index, '_all_nodes_dict') and index._all_nodes_dict:
            click.echo("[info] Docstore was not loaded from disk or was empty. Attempting to populate docstore from index nodes for BM25.")
            nodes_added = 0
            for node_id, node in index._all_nodes_dict.items():
                if not storage_context.docstore.document_exists(node_id):
                    storage_context.docstore.add_documents([node], allow_update=True)
                    nodes_added += 1
            
            if storage_context.docstore.docs:
                 click.echo(f"[info] Populated docstore with {len(storage_context.docstore.docs)} nodes from the index ({nodes_added} newly added).")
                 
                 # Persist the populated docstore if we have a persist_dir
                 if "persist_dir" in storage_context_args:
                     try:
                         storage_context.persist(persist_dir=storage_context_args["persist_dir"])
                         click.echo(f"[info] Persisted populated docstore to {storage_context_args['persist_dir']}")
                     except Exception as e_persist:
                         click.echo(f"[warning] Failed to persist populated docstore: {e_persist}", err=True)
            else:
                 click.echo("[warning] Failed to populate docstore from index nodes.")

        # 3. Query Expansion
        full_query_text = " ".join(query_text)
        queries_to_run = [QueryBundle(full_query_text)]

        # 4. Build Retriever
        llama_filters = None
        if filters_kv:
            from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
            filter_conditions = []
            for f_kv in filters_kv:
                if "=" not in f_kv:
                    click.echo(f"[warning] Invalid filter '{f_kv}', skipping. Must be key=value.", err=True)
                    continue
                key, value = f_kv.split("=", 1)
                filter_conditions.append(ExactMatchFilter(key=key, value=value))
            if filter_conditions:
                llama_filters = MetadataFilters(filters=filter_conditions)
                click.echo(f"[info] Applying metadata filters: {filters_kv}")

        # Create a retriever that directly uses the documents from the docstore - REMOVED
        # retriever: BaseRetriever - REMOVED
        
        # Get documents from docstore - REMOVED
        # all_documents = [] - REMOVED
        # if hasattr(storage_context, 'docstore') and hasattr(storage_context.docstore, 'docs'): - REMOVED
        #     all_documents = list(storage_context.docstore.docs.values()) - REMOVED
        #     click.echo(f"[info] Found {len(all_documents)} documents in docstore for retrieval") - REMOVED
        
        # if not all_documents and hasattr(index, '_all_nodes_dict'): - REMOVED
        #     all_documents = list(index._all_nodes_dict.values()) - REMOVED
        #     click.echo(f"[info] Found {len(all_documents)} documents in index for retrieval") - REMOVED
        
        # if all_documents: - REMOVED
            # Create a simple retriever that returns all documents - REMOVED
            # from llama_index.core.retrievers import ListIndexRetriever - REMOVED
            
            # Create a simple list index with the documents - REMOVED
            # from llama_index.core import ListIndex - REMOVED
            # list_index = ListIndex(all_documents) - REMOVED
            
            # retriever = ListIndexRetriever( - REMOVED
            #     index=list_index, - REMOVED
            #     similarity_top_k=similarity_top_k - REMOVED
            # ) - REMOVED
            # click.echo("[info] Using ListIndexRetriever with documents from docstore") - REMOVED
        # else: - REMOVED
            # Fallback to vector retriever - REMOVED
            # click.echo("[warning] No documents found in docstore or index. Using empty retriever.") - REMOVED
            # class EmptyRetriever(BaseRetriever): - REMOVED
            #     def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]: - REMOVED
            #         return [] - REMOVED
                
            #     async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]: - REMOVED
            #         return [] - REMOVED
            
            # retriever = EmptyRetriever() - REMOVED

        # Instantiate VectorIndexRetriever
        vector_retriever = VectorIndexRetriever(
            index=index, # The VectorStoreIndex connected to Qdrant
            similarity_top_k=similarity_top_k, # From CLI/config
            filters=llama_filters # Already defined in the script
        )
        click.echo(f"[info] VectorIndexRetriever initialized with similarity_top_k={similarity_top_k}.")

        # Implement Hybrid Search Logic
        retriever: BaseRetriever
        if use_hybrid_search:
            click.echo("[info] Hybrid search is enabled. Initializing BM25Retriever.")
            # Ensure docstore is loaded and has documents
            if hasattr(storage_context, 'docstore') and storage_context.docstore.docs:
                all_docstore_nodes = list(storage_context.docstore.docs.values())
                if all_docstore_nodes:
                    try:
                        # BM25Retriever import is already at the top of the file
                        # Using default tokenizer from LlamaIndex, consider making it configurable if needed
                        bm25_retriever = BM25Retriever.from_defaults(
                            nodes=all_docstore_nodes,
                            similarity_top_k=sparse_top_k # From CLI/config
                        )
                        click.echo(f"[info] BM25Retriever initialized with {len(all_docstore_nodes)} nodes from docstore and sparse_top_k={sparse_top_k}.")
                        
                        # Instantiate the custom HybridRetriever
                        retriever = HybridRetriever(
                            dense_retriever=vector_retriever,
                            sparse_retriever=bm25_retriever
                        )
                        click.echo("[info] Using HybridRetriever (dense + sparse).")
                    except Exception as e_bm25:
                        click.echo(f"[warning] Failed to initialize BM25Retriever: {e_bm25}. Falling back to dense-only search.", err=True)
                        retriever = vector_retriever # Fallback to dense if BM25 fails
                else:
                    click.echo("[warning] Docstore is empty. Cannot initialize BM25Retriever. Using dense-only search.", err=True)
                    retriever = vector_retriever
            else:
                click.echo("[warning] Docstore not available or empty. Cannot initialize BM25Retriever. Using dense-only search.", err=True)
                retriever = vector_retriever
        else:
            click.echo("[info] Hybrid search is disabled. Using dense-only VectorIndexRetriever.")
            retriever = vector_retriever

        # 5. Node Postprocessors
        node_postprocessors_list: List[BaseNodePostprocessor] = []
        if use_mmr:
            mmr_top_k = similarity_top_k if not use_hybrid_search else max(similarity_top_k, sparse_top_k)
            mmr_postprocessor = MMRNodePostprocessor(embed_model=Settings.embed_model, top_n=mmr_top_k, lambda_mult=mmr_lambda)
            node_postprocessors_list.append(mmr_postprocessor)
            click.echo(f"[info] Added MMRNodePostprocessor (top_n={mmr_top_k}).")
        try:
            reorder_postprocessor = LongContextReorder()
            node_postprocessors_list.append(reorder_postprocessor)
            click.echo("[info] Added LongContextReorder postprocessor.")
        except Exception as e_reorder:
            click.echo(f"[warning] Failed to initialize LongContextReorder: {e_reorder}. Skipping.", err=True)

        if compress_context_flag:
            if final_cohere_api_key:
                try:
                    cohere_rerank_top_n = rerank_top_n if rerank_top_n > 0 else similarity_top_k
                    cohere_compressor = CohereRerank(api_key=final_cohere_api_key, top_n=cohere_rerank_top_n)
                    node_postprocessors_list.append(cohere_compressor)
                    click.echo(f"[info] Added CohereRerank for compression/reranking (top_n={cohere_rerank_top_n}).")
                except ImportError:
                     click.echo("[warning] CohereRerank not available. Skipping compression.", err=True)
                except Exception as e_cohere:
                    click.echo(f"[warning] Failed to initialize CohereRerank: {e_cohere}. Skipping compression.", err=True)
            else:
                click.echo("[info] Contextual compression requested, but Cohere API key not found.")

        if rerank_top_n > 0 and not (compress_context_flag and final_cohere_api_key):
            try:
                sbert_reranker = SentenceTransformerRerank(model=reranker_model, top_n=rerank_top_n, device="cpu")
                node_postprocessors_list.append(sbert_reranker)
                click.echo(f"[info] Added SentenceTransformerRerank (model: {reranker_model}, top_n: {rerank_top_n}).")
            except ImportError:
                click.echo("[warning] SentenceTransformerRerank not available. Skipping reranking.", err=True)
            except Exception as e_sbert:
                click.echo(f"[warning] Failed to initialize SentenceTransformerRerank: {e_sbert}. Skipping.", err=True)
        
        # 6. Setup Response Synthesizer
        click.echo("[info] Setting up Response Synthesizer...")
        response_synthesizer: BaseSynthesizer = get_response_synthesizer(llm=Settings.llm)

        # 7. Execute Query/Queries
        click.echo(f"\n[info] Executing query: {full_query_text}")

        all_retrieved_nodes_with_score: Dict[str, NodeWithScore] = {}
        for q_bundle in queries_to_run: # This loop will now run only once
            if _VERBOSE_OUTPUT and q_bundle.query_str != full_query_text: # This condition will likely not be met
                click.echo(f"[info] Retrieving for expanded query: {q_bundle.query_str}")

            if not use_hybrid_search and _VERBOSE_OUTPUT: # Or if retriever is vector_retriever directly
                # This log will be a bit ahead of the actual call, but gives context
                click.echo(f"[debug] Performing dense-only retrieval with top_k={vector_retriever._similarity_top_k}.")
            
            retrieved_nodes_for_query: List[NodeWithScore] = await retriever.aretrieve(q_bundle)
            
            if not use_hybrid_search and _VERBOSE_OUTPUT: # Or if retriever is vector_retriever directly
                 click.echo(f"[debug] Dense-only retrieval found {len(retrieved_nodes_for_query)} nodes.")

            if _VERBOSE_OUTPUT: # This existing log will now act as a summary after retriever (hybrid or dense)
                click.echo(f"[info] Retrieved {len(retrieved_nodes_for_query)} nodes for query '{q_bundle.query_str}' before postprocessing.")
            
            current_processed_nodes = retrieved_nodes_for_query
            if node_postprocessors_list:
                if _VERBOSE_OUTPUT: click.echo(f"[info] Applying {len(node_postprocessors_list)} postprocessor(s)...")
                for post_idx, post_obj in enumerate(node_postprocessors_list):
                    if _VERBOSE_OUTPUT: click.echo(f"[info] Applying {type(post_obj).__name__}...")
                    if hasattr(post_obj, 'async_postprocess_nodes'):
                        current_processed_nodes = await post_obj.async_postprocess_nodes(nodes=current_processed_nodes, query_bundle=q_bundle)
                    else:
                        current_processed_nodes = post_obj.postprocess_nodes(nodes=current_processed_nodes, query_bundle=q_bundle)
                    if _VERBOSE_OUTPUT: click.echo(f"[info] Nodes after {type(post_obj).__name__}: {len(current_processed_nodes)}")
            
            if _VERBOSE_OUTPUT: click.echo(f"[info] Processed {len(current_processed_nodes)} nodes for query '{q_bundle.query_str}'.")
            for node_ws in current_processed_nodes:
                if node_ws.node.node_id not in all_retrieved_nodes_with_score or \
                   (node_ws.score or 0.0) > (all_retrieved_nodes_with_score[node_ws.node.node_id].score or 0.0):
                    all_retrieved_nodes_with_score[node_ws.node.node_id] = node_ws
        
        merged_retrieved_nodes_list = sorted(all_retrieved_nodes_with_score.values(), key=lambda nw: nw.score or 0.0, reverse=True)
        
        click.echo(f"[info] Synthesizing response from {len(merged_retrieved_nodes_list)} merged and processed nodes.")
        final_response = await response_synthesizer.asynthesize(query=QueryBundle(full_query_text), nodes=merged_retrieved_nodes_list)
        final_retrieved_nodes_list = merged_retrieved_nodes_list

        if raw_output_flag:
            click.secho("\n--- Retrieved Nodes ---", fg="yellow")
            if not final_retrieved_nodes_list: click.echo("No nodes retrieved.")
            for idx, node_ws in enumerate(final_retrieved_nodes_list):
                click.echo(f"\n[Node {idx+1}] ID: {node_ws.node.node_id}, Score: {node_ws.score}")
                click.echo(f"Source: {node_ws.metadata.get('source', 'N/A')}")
                snippet_to_show = node_ws.node.get_content(metadata_mode="none")[:500] + "..."
                if compress_context_flag and final_cohere_api_key and "document_with_score" in node_ws.metadata and "text" in node_ws.metadata["document_with_score"]:
                    snippet_to_show = node_ws.metadata["document_with_score"]["text"][:500] + "..."
                click.echo(f"Snippet: {snippet_to_show}")
            click.secho("--- End Retrieved Nodes ---", fg="yellow")

        click.secho("\n[Generated Answer]", fg="green")
        click.echo(str(final_response).strip())
        response_value = str(final_response).strip()

        # 8. Evaluation (Optional)
        if evaluate_rag_flag:
            if not evaluation_available:
                click.echo("\n[warning] Evaluation requested but module not available. Skipping.", err=True)
            else:
                click.echo("\n[info] Evaluating RAG response quality...")
                eval_llm = Settings.llm
                relevancy_eval = RelevancyEvaluator(llm=eval_llm)
                eval_context = "\n\n---\n\n".join([node_ws.node.get_content() for node_ws in final_retrieved_nodes_list])
                eval_result_relevancy = relevancy_eval.evaluate_response(query=full_query_text, response=final_response, contexts=[eval_context])
                click.secho("\n--- RAG Quality Evaluation ---", fg="cyan")
                click.echo(f"Relevancy: {eval_result_relevancy.passing} (Score: {eval_result_relevancy.score:.2f})")
                click.echo(f"Feedback (Relevancy): {eval_result_relevancy.feedback}")
                click.secho("--- End RAG Quality Evaluation ---", fg="cyan")
        
    finally:
        if async_qdrant_client: # This check is important
            if _VERBOSE_OUTPUT:
                click.echo("[info] Closing AsyncQdrantClient in finally block...")
            await async_qdrant_client.close()
            if _VERBOSE_OUTPUT:
                click.echo("[info] AsyncQdrantClient closed in finally block.")
    
    return response_value # Return the captured response value


# Create a Click command that wraps the async function
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--collection-name", default="rag_llamaindex_data", show_default=True, help="Qdrant collection name.")
@click.option("--k", "similarity_top_k", type=int, default=100, show_default=True, help="Number of top similar results to retrieve.") # Changed default
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True, help="Qdrant URL.")
@click.option("--qdrant-api-key", envvar="QDRANT_API_KEY", default=None, help="Qdrant API key.")
@click.option("--openai-api-key", envvar="OPENAI_API_KEY", default=None, help="OpenAI API key.")
@click.option("--openai-model-embedding", default="text-embedding-3-large", show_default=True, help="OpenAI embedding model.")
@click.option("--openai-model-llm", default="gpt-4.1-mini", show_default=True, help="OpenAI LLM for answer generation.")
@click.option("--llm-temperature", type=float, default=0.3, show_default=True, help="Temperature for the LLM (lower for factual, higher for creative). Default is 0.3 for more focused RAG.")
@click.option("--raw-output/--no-raw-output", "raw_output_flag", default=False, help="Show raw retrieval nodes and then the answer.")
@click.option("--hybrid/--no-hybrid", "use_hybrid_search", default=True, show_default=True, help="Enable hybrid (vector + sparse) search.")
@click.option("--sparse-top-k", type=int, default=100, show_default=True, help="Number of sparse results for hybrid search.") # Changed default
@click.option("--rerank-top-n", type=int, default=0, show_default=True, help="Re-rank top N results using a cross-encoder (0 to disable).")
@click.option("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model for reranking.")
@click.option("--mmr/--no-mmr", "use_mmr", default=True, help="Use Maximal Marginal Relevance (MMR) for re-ranking.")
@click.option("--mmr-lambda", type=float, default=0.5, show_default=True, help="Lambda for MMR (0=max diversity, 1=max relevance).")
@click.option("--filter", "-f", "filters_kv", multiple=True, help="Metadata filter key=value (e.g., document_type=academic).")
@click.option("--evaluate-rag/--no-evaluate-rag", "evaluate_rag_flag", default=False, help="Evaluate RAG quality and show feedback.")
@click.option("--compress-context/--no-compress-context", "compress_context_flag", default=False, help="Apply contextual compression (e.g., via CohereRerank if API key available).")
@click.option("--cohere-api-key", envvar="COHERE_API_KEY", default=None, help="Cohere API key (for CohereRerank compression).")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output.")
@click.argument("query_text", nargs=-1, required=True)
# @click.pass_context # No longer needed as ctx is not used directly for param passing
def cli_entrypoint(
    collection_name: str,
    similarity_top_k: int,
    qdrant_url: str,
    qdrant_api_key: str | None,
    openai_api_key: str | None,
    openai_model_embedding: str,
    openai_model_llm: str,
    llm_temperature: float,
    raw_output_flag: bool,
    use_hybrid_search: bool,
    sparse_top_k: int,
    rerank_top_n: int,
    reranker_model: str,
    use_mmr: bool,
    mmr_lambda: float,
    filters_kv: Sequence[str],
    evaluate_rag_flag: bool,
    compress_context_flag: bool,
    cohere_api_key: str | None,
    verbose: bool,
    query_text: Sequence[str],
):
    """Synchronous entry point for Click that runs the async main logic."""
    # Capture all parameters passed to this function into a dictionary
    params_for_main_async = locals()
    
    import asyncio
    asyncio.run(main_async(**params_for_main_async))

if __name__ == "__main__":
    cli_entrypoint()
