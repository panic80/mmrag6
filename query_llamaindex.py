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
from llama_index.retrievers.bm25 import BM25Retriever # If we want to explicitly use it

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
        sparse_nodes = self._sparse_retriever.retrieve(query_bundle)
        
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
        sparse_nodes = await self._sparse_retriever.aretrieve(query_bundle)
        
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
from llama_index.postprocessor.cohere_rerank import CohereRerank # If API key available
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank # Cross-encoder
from llama_index.core.node_parser import SentenceSplitter # For compression fallback
from llama_index.core.schema import NodeWithScore, BaseNode

# LlamaIndex Integrations
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.question_gen.openai import OpenAIQuestionGenerator
# Import evaluation tools conditionally
try:
    from llama_index.evaluation import ( # For RAG evaluation
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
    collection_name: str,
    similarity_top_k: int,
    qdrant_url: str,
    qdrant_api_key: str | None,
    openai_api_key: str | None,
    openai_model_embedding: str,
    openai_model_llm: str,
    raw_output_flag: bool,
    use_hybrid_search: bool,
    sparse_top_k: int,
    rerank_top_n: int,
    reranker_model: str,
    use_mmr: bool,
    mmr_lambda: float,
    filters_kv: Sequence[str],
    use_query_expansion: bool,
    max_expansions: int,
    evaluate_rag_flag: bool,
    compress_context_flag: bool,
    cohere_api_key: str | None,
    verbose: bool,
    query_text: Sequence[str],
):
    """Async logic for querying a LlamaIndex RAG setup."""
    global _VERBOSE_OUTPUT
    _VERBOSE_OUTPUT = verbose

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

    # 1. Configure LlamaIndex Settings
    click.echo("[info] Configuring LlamaIndex Settings...")
    Settings.llm = LlamaOpenAI(model=openai_model_llm, api_key=final_openai_api_key)
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

    try:
        if _VERBOSE_OUTPUT:
            click.echo(f"[info] Qdrant URL: {qdrant_url}")
            click.echo(f"[info] Qdrant API Key provided: {'Yes' if final_qdrant_api_key else 'No'}")

        qdrant_native_client = QdrantClient(url=qdrant_url, api_key=final_qdrant_api_key)
        async_qdrant_client = AsyncQdrantClient(url=qdrant_url, api_key=final_qdrant_api_key)
        
        # Check if collection exists
        try:
            qdrant_native_client.get_collection(collection_name=effective_collection_name)
            if _VERBOSE_OUTPUT:
                click.echo(f"[info] Successfully connected to existing Qdrant collection: {effective_collection_name}")
        except Exception as e:
            click.echo(f"[fatal] Qdrant collection '{effective_collection_name}' not found or error accessing it: {e}", err=True)
            click.echo(f"Please ensure the collection exists and was created with compatible embeddings (e.g., via ingest_llamaindex.py).")
            sys.exit(1)
            
        vector_store = QdrantVectorStore(
            client=qdrant_native_client, 
            aclient=async_qdrant_client,
            collection_name=effective_collection_name
        )

        # For hybrid search (BM25), we need the docstore from the local filesystem.
        # The persist_dir for query should match the one used during ingestion for the specific collection.
        # Default persist_dir for the script. The actual path for a collection is persist_dir / collection_name.
        base_persist_dir = Path("./storage_llamaindex_db") 
        # Construct the specific path for the collection's docstore etc.
        # Use effective_collection_name here to match ingestion logic if collection_name was defaulted.
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
            elif "persist_dir" in storage_context_args: # It attempted to load but docstore is empty
                 click.echo(f"[warning] Loaded StorageContext from {storage_context_args['persist_dir']}, but Docstore is empty or has no documents. BM25 may be ineffective.")
            else: # No persist_dir was provided, so docstore is expected to be empty initially
                 click.echo(f"[info] Initialized new empty Docstore. BM25 will be ineffective unless nodes are added from vector store or other source.")

        except Exception as e_sc:
            click.echo(f"[warning] Failed to load or initialize StorageContext with persist_dir '{collection_specific_persist_path}': {e_sc}. Falling back to default StorageContext. BM25 will be limited.", err=True)
            storage_context = StorageContext.from_defaults(vector_store=vector_store) # Fallback

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        
        # If docstore wasn't loaded successfully from disk, try to populate it from the index's nodes
        # This is a fallback for BM25 if the persisted docstore was problematic
        if not docstore_loaded_successfully and hasattr(index, '_all_nodes_dict') and index._all_nodes_dict:
            click.echo("[info] Docstore was not loaded from disk or was empty. Attempting to populate docstore from index nodes for BM25.")
            for node_id, node in index._all_nodes_dict.items():
                if not storage_context.docstore.document_exists(node_id):
                    storage_context.docstore.add_documents([node], allow_update=True) # Add node to docstore
            if storage_context.docstore.docs:
                 click.echo(f"[info] Populated docstore with {len(storage_context.docstore.docs)} nodes from the index.")
            else:
                 click.echo("[warning] Failed to populate docstore from index nodes.")


    except Exception as e:
        click.echo(f"[fatal] Failed to initialize Qdrant vector store or load index: {e}", err=True)
        if _VERBOSE_OUTPUT:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # 3. Query Expansion
    full_query_text = " ".join(query_text)
    queries_to_run = [QueryBundle(full_query_text)] # LlamaIndex uses QueryBundle

    if use_query_expansion:
        click.echo(f"[info] Expanding query: '{full_query_text}'")
        
        # Define a custom prompt for query expansion, incorporating max_expansions
        # Ensure max_expansions is at least 1 for the prompt to make sense.
        num_queries_to_generate = max(1, max_expansions) 
        
        custom_query_expansion_prompt_template_str = (
            f"Given the following user query, generate up to {num_queries_to_generate} alternative queries "
            "that can help retrieve more comprehensive and relevant information from a knowledge base.\n"
            "These alternative queries should explore different angles, synonyms, related concepts, "
            "or break down the original query into more specific sub-questions.\n"
            "Ensure the generated queries are distinct and aim to cover a broader search space.\n\n"
            "Original User Query:\n"
            "{query_str}\n\n"
            "Generated Alternative Queries (each on a new line, do not number them):\n"
        )

        question_gen = OpenAIQuestionGenerator.from_defaults(
            llm=Settings.llm,
            prompt_template_str=custom_query_expansion_prompt_template_str
        )
        
        # Generate sub-questions or alternative queries
        # The output format of generate needs to be handled. It's usually a list of SubQuestion objects.
        # The 'generate' method of OpenAIQuestionGenerator expects 'num_questions' to be passed via metadata in QueryBundle or similar.
        # However, the default behavior might generate a few questions based on the prompt.
        # Let's try with the default generation first, then adjust if num_questions needs explicit passing.
        # For now, we'll rely on the prompt to guide the number of questions implicitly or the generator's default.
        # The `max_expansions` parameter will limit how many we use.
        try:
            # The generate method takes `tools` and `query` (QueryBundle)
            # It seems `num_questions` is implicitly handled by the prompt or a default in the generator.
            generated_items = question_gen.generate(tools=[], query=QueryBundle(full_query_text))
            
            # Assuming generated_items are SubQuestion objects with 'sub_question' attribute
            # or similar structure that provides the generated question string.
            expanded_q_texts = []
            if isinstance(generated_items, list):
                for item in generated_items:
                    if hasattr(item, 'sub_question') and isinstance(item.sub_question, str):
                        expanded_q_texts.append(item.sub_question)
                    elif isinstance(item, str): # Sometimes it might return strings directly
                        expanded_q_texts.append(item)
            
            # Clean up empty strings that might be generated
            expanded_q_texts = [q.strip() for q in expanded_q_texts if q.strip()]
            
            if expanded_q_texts:
                # Limit expansions and add original query
                unique_expansions = list(dict.fromkeys(expanded_q_texts)) # Deduplicate
                final_expansions = unique_expansions[:max_expansions]
                
                click.echo("\nExpanded queries:")
                for i, exp_q_text in enumerate(final_expansions):
                    click.echo(f"  {i+1}. {exp_q_text}")
                    queries_to_run.append(QueryBundle(exp_q_text))
                # Ensure original query is always included if not generated
                if full_query_text not in final_expansions:
                     queries_to_run.insert(0, QueryBundle(full_query_text)) # Ensure original is first
                
                # Deduplicate QueryBundles based on their string content
                seen_query_strings = set()
                unique_queries_to_run = []
                for qb in queries_to_run:
                    if qb.query_str not in seen_query_strings:
                        unique_queries_to_run.append(qb)
                        seen_query_strings.add(qb.query_str)
                queries_to_run = unique_queries_to_run

        except Exception as e:
            click.echo(f"[warning] Query expansion failed: {e}. Using original query only.", err=True)
            queries_to_run = [QueryBundle(full_query_text)]


    # 4. Build Retriever
    # Metadata filters
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

    retriever: BaseRetriever
    if use_hybrid_search:
        click.echo("[info] Configuring Hybrid Retriever...")
        dense_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            filters=llama_filters,
            # vector_store_query_mode can be "DEFAULT" for dense only
        )
        
        # For BM25Retriever, we need the nodes.
        # This might be memory-intensive for very large indexes if we load all nodes.
        # Alternative: retrieve more with dense, then BM25 on that subset.
        # For now, loading all nodes from the docstore.
        all_nodes: List[BaseNode] = []
        try:
            click.echo("[info] Loading all nodes from docstore for BM25Retriever (may take time for large indexes)...")
            # Note: index.docstore.docs.values() gives Document objects, not BaseNode.
            # We need to get nodes from the vector store or reconstruct them if BM25Retriever needs BaseNode.
            # A simpler way if BM25Retriever can take documents:
            # documents_for_bm25 = list(index.docstore.docs.values())
            # bm25_retriever = BM25Retriever.from_defaults(documents=documents_for_bm25, similarity_top_k=sparse_top_k)
            # However, BM25Retriever typically works with nodes.
            # If index.vector_store has a way to get all nodes, that's better.
            # QdrantVectorStore might not have a direct "get_all_nodes" that reconstructs them perfectly.
            # Let's assume we can retrieve all nodes from the index's node_dict if populated,
            # or iterate through the docstore and reconstruct.
            # This part is crucial and might need adjustment based on how nodes are stored/accessible.
            # For a VectorStoreIndex, nodes are usually in index.storage_context.docstore
            # and index.index_struct contains the mapping.
            # A robust way:
            if hasattr(index, '_nodes_dict') and index._nodes_dict: # If nodes are directly in the index object
                 all_nodes = list(index._nodes_dict.values())
            elif hasattr(index.docstore, 'docs'): # Fallback to docstore, these are Document objects
                 # BM25Retriever can often work with Document objects too by tokenizing their text.
                 # Let's try with Document objects first for simplicity.
                 # If BM25Retriever strictly needs BaseNode, this needs more work.
                 # The from_defaults method of BM25Retriever can take `nodes` or `documents`.
                 all_documents_for_bm25 = list(index.docstore.docs.values())
                 if all_documents_for_bm25:
                    click.echo(f"[info] Using {len(all_documents_for_bm25)} documents for BM25Retriever.")
                 else: # Try to get nodes if documents are empty or not suitable
                    click.echo("[warning] No documents in docstore, attempting to get nodes for BM25. This might be incomplete.")
                    # This is a placeholder, proper node retrieval from vector store might be needed
                    # For now, this will likely lead to an empty BM25 if not handled.
                    # A common pattern is to build BM25 from a corpus loaded separately if not all nodes are easily retrievable.
                    # For this exercise, we'll proceed assuming BM25Retriever can handle it or we'll note the limitation.
                    pass # all_nodes remains empty if no direct way

            if all_documents_for_bm25: # Prefer documents if available
                bm25_retriever = BM25Retriever.from_defaults(
                    documents=all_documents_for_bm25, # Pass Document objects
                    similarity_top_k=sparse_top_k
                )
            elif all_nodes: # Fallback to nodes if documents not suitable/available
                 bm25_retriever = BM25Retriever.from_defaults(
                    nodes=all_nodes,
                    similarity_top_k=sparse_top_k
                )
            else:
                click.echo("[error] Could not load documents/nodes for BM25Retriever. Hybrid search might be ineffective for sparse part.", err=True)
                # Create a dummy BM25 retriever that returns nothing
                class DummySparseRetriever(BaseRetriever):
                    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]: return []
                    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]: return []
                bm25_retriever = DummySparseRetriever()


            retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                sparse_retriever=bm25_retriever,
                # mode="OR" # Default mode, can be configured
            )
            click.echo("[info] Using LlamaIndex HybridRetriever (Dense + BM25).")

        except Exception as e:
            click.echo(f"[error] Failed to initialize BM25Retriever or HybridRetriever: {e}. Falling back to dense search.", err=True)
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=similarity_top_k,
                filters=llama_filters,
            )
            click.echo("[info] Using Vector Retriever (fallback).")
    else:
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            filters=llama_filters,
        )
        click.echo("[info] Using Vector Retriever.")

    # 5. Node Postprocessors (Reranking, Compression, MMR)
    node_postprocessors_list: List[BaseNodePostprocessor] = [] # Renamed to avoid conflict

    if use_mmr:
        mmr_postprocessor = MMRNodePostprocessor(
            embed_model=Settings.embed_model, # Necessary for MMR calculations
            top_n=similarity_top_k, # Number of results to return after MMR
            lambda_mult=mmr_lambda, # Control diversity vs relevance tradeoff
        )
        node_postprocessors_list.append(mmr_postprocessor)
        click.echo(f"[info] Added MMRNodePostprocessor (will select diverse top {similarity_top_k}).")

    # Add LongContextReorder to potentially improve LLM's focus on relevant parts
    # This is generally useful if many documents are passed to the LLM.
    # It should typically be applied after other filtering/reranking but before final synthesis.
    # We can place it after MMR and before compression/final reranking.
    try:
        reorder_postprocessor = LongContextReorder()
        node_postprocessors_list.append(reorder_postprocessor)
        click.echo("[info] Added LongContextReorder postprocessor.")
    except Exception as e_reorder:
        click.echo(f"[warning] Failed to initialize LongContextReorder: {e_reorder}. Skipping.", err=True)

    if compress_context_flag:
        if final_cohere_api_key:
            try:
                # CohereRerank also acts as a compressor by selecting relevant snippets.
                # Ensure top_n for CohereRerank is appropriate.
                cohere_rerank_top_n = rerank_top_n if rerank_top_n > 0 else similarity_top_k
                cohere_compressor = CohereRerank(api_key=final_cohere_api_key, top_n=cohere_rerank_top_n)
                node_postprocessors_list.append(cohere_compressor)
                click.echo(f"[info] Added CohereRerank for compression/reranking (top_n={cohere_rerank_top_n}).")
            except ImportError:
                 click.echo("[warning] CohereRerank not available (llama-index-postprocessor-cohere not installed?). Skipping compression.", err=True)
            except Exception as e:
                click.echo(f"[warning] Failed to initialize CohereRerank: {e}. Skipping compression.", err=True)
        else:
            click.echo("[info] Contextual compression enabled, but Cohere API key not found. True compression might not occur without it or a custom LLM compressor.")
            # Placeholder for LLM-based compression could be added here if desired.

    # Add SentenceTransformerRerank if specified and CohereRerank (which also reranks) is not used for compression.
    if rerank_top_n > 0 and not (compress_context_flag and final_cohere_api_key):
        try:
            sbert_reranker = SentenceTransformerRerank(model=reranker_model, top_n=rerank_top_n, device="cpu") # Explicitly set device
            node_postprocessors_list.append(sbert_reranker)
            click.echo(f"[info] Added SentenceTransformerRerank (model: {reranker_model}, top_n: {rerank_top_n}).")
        except ImportError:
            click.echo("[warning] SentenceTransformerRerank not available (llama-index-postprocessor-sbert-rerank not installed?). Skipping reranking.", err=True)
        except Exception as e:
            click.echo(f"[warning] Failed to initialize SentenceTransformerRerank: {e}. Skipping.", err=True)
            
    # 6. Setup Response Synthesizer (QueryPipeline for retrieval is removed)
    click.echo("[info] Setting up Response Synthesizer...")
    response_synthesizer: BaseSynthesizer = get_response_synthesizer(
        llm=Settings.llm,
        # response_mode="refine" # Example, can be configured
        # streaming=True, # If desired
    )

    # 7. Execute Query/Queries (Direct Retrieval and Manual Postprocessing)
    click.echo(f"\n[info] Executing query: {full_query_text}")
    if len(queries_to_run) > 1:
        click.echo(f"[info] Using {len(queries_to_run)-1} expanded queries as well.")

    all_retrieved_nodes_with_score: Dict[str, NodeWithScore] = {}

    for q_bundle in queries_to_run:
        if _VERBOSE_OUTPUT and q_bundle.query_str != full_query_text:
            click.echo(f"[info] Retrieving for expanded query: {q_bundle.query_str}")

        # Direct retrieval
        retrieved_nodes_for_query: List[NodeWithScore] = await retriever.aretrieve(q_bundle)
        
        if _VERBOSE_OUTPUT:
            click.echo(f"[info] Retrieved {len(retrieved_nodes_for_query)} nodes for query '{q_bundle.query_str}' before postprocessing.")

        # Manual postprocessing
        current_processed_nodes = retrieved_nodes_for_query
        if node_postprocessors_list:
            if _VERBOSE_OUTPUT:
                click.echo(f"[info] Applying {len(node_postprocessors_list)} postprocessor(s) to {len(current_processed_nodes)} nodes...")
            for postprocessor_idx, postprocessor_obj in enumerate(node_postprocessors_list):
                if _VERBOSE_OUTPUT:
                    click.echo(f"[info] Applying postprocessor {postprocessor_idx+1}/{len(node_postprocessors_list)}: {type(postprocessor_obj).__name__}")
                # The postprocess_nodes method is synchronous in BaseNodePostprocessor
                # If an async version is available/needed, this would need adjustment.
                # For now, assuming synchronous postprocessing is acceptable within this async function.
                # Some postprocessors might have async versions, e.g. `async_postprocess_nodes`
                if hasattr(postprocessor_obj, 'async_postprocess_nodes'):
                    current_processed_nodes = await postprocessor_obj.async_postprocess_nodes(
                        nodes=current_processed_nodes, query_bundle=q_bundle
                    )
                else: # Fallback to synchronous
                    current_processed_nodes = postprocessor_obj.postprocess_nodes(
                        nodes=current_processed_nodes, query_bundle=q_bundle
                    )
                if _VERBOSE_OUTPUT:
                    click.echo(f"[info] Nodes after {type(postprocessor_obj).__name__}: {len(current_processed_nodes)}")
        
        if _VERBOSE_OUTPUT:
             click.echo(f"[info] Processed {len(current_processed_nodes)} nodes for query '{q_bundle.query_str}'.")

        for node_ws in current_processed_nodes:
            if node_ws.node.node_id not in all_retrieved_nodes_with_score:
                all_retrieved_nodes_with_score[node_ws.node.node_id] = node_ws
            else: 
                if (node_ws.score or 0.0) > (all_retrieved_nodes_with_score[node_ws.node.node_id].score or 0.0):
                    all_retrieved_nodes_with_score[node_ws.node.node_id] = node_ws
    
    merged_retrieved_nodes_list = sorted(all_retrieved_nodes_with_score.values(), key=lambda nw: nw.score or 0.0, reverse=True)
    
    # Synthesize with the final merged and processed nodes
    click.echo(f"[info] Synthesizing response from {len(merged_retrieved_nodes_list)} merged and processed nodes.")
    final_response = await response_synthesizer.asynthesize(
        query=QueryBundle(full_query_text), 
        nodes=merged_retrieved_nodes_list
    )
    final_retrieved_nodes_list = merged_retrieved_nodes_list # For raw output and evaluation


    if raw_output_flag:
        click.secho("\n--- Retrieved Nodes ---", fg="yellow")
        if not final_retrieved_nodes_list:
            click.echo("No nodes retrieved.")
        for idx, node_ws in enumerate(final_retrieved_nodes_list):
            click.echo(f"\n[Node {idx+1}] ID: {node_ws.node.node_id}, Score: {node_ws.score}") # Show raw score, could be nan
            click.echo(f"Source: {node_ws.metadata.get('source', 'N/A')}")
            
            # Prioritize Cohere compressed snippet if available
            if compress_context_flag and final_cohere_api_key and "document_with_score" in node_ws.metadata and "text" in node_ws.metadata["document_with_score"]:
                snippet = node_ws.metadata["document_with_score"]["text"]
                click.echo(f"Compressed Snippet (Cohere): {snippet[:500]}...") # Show more
            else:
                # Otherwise, show the main text content of the node
                snippet = node_ws.node.get_content(metadata_mode="none") # Get only text content
                click.echo(f"Snippet: {snippet[:500]}...") # Show more
        click.secho("--- End Retrieved Nodes ---", fg="yellow")

    click.secho("\n[Generated Answer]", fg="green")
    click.echo(str(final_response).strip()) # Keep printing for CLI usage

    # 8. Evaluation (Optional)
    if evaluate_rag_flag:
        if not globals().get('evaluation_available', False):
            click.echo("\n[warning] Evaluation was requested but llama-index-evaluation module is not available. Skipping evaluation.", err=True)
        else:
            click.echo("\n[info] Evaluating RAG response quality...")
            # Prepare evaluators
            eval_llm = Settings.llm # Use the same LLM or a dedicated one for evaluation
            
            relevancy_eval = RelevancyEvaluator(llm=eval_llm)

            # Context for evaluation is the text from retrieved nodes
            eval_context = "\n\n---\n\n".join([node_ws.node.get_content() for node_ws in final_retrieved_nodes_list])
            
            eval_result_relevancy = relevancy_eval.evaluate_response(query=full_query_text, response=final_response, contexts=[eval_context])

            click.secho("\n--- RAG Quality Evaluation ---", fg="cyan")
            click.echo(f"Relevancy: {eval_result_relevancy.passing} (Score: {eval_result_relevancy.score:.2f})")
            click.echo(f"Feedback (Relevancy): {eval_result_relevancy.feedback}")
            click.secho("--- End RAG Quality Evaluation ---", fg="cyan")
    
    return str(final_response).strip() # Return the answer string

# Create a Click command that wraps the async function
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--collection-name", default="rag_llamaindex_data", show_default=True, help="Qdrant collection name.")
@click.option("--k", "similarity_top_k", type=int, default=10, show_default=True, help="Number of top similar results to retrieve.")
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True, help="Qdrant URL.")
@click.option("--qdrant-api-key", envvar="QDRANT_API_KEY", default=None, help="Qdrant API key.")
@click.option("--openai-api-key", envvar="OPENAI_API_KEY", default=None, help="OpenAI API key.")
@click.option("--openai-model-embedding", default="text-embedding-3-large", show_default=True, help="OpenAI embedding model.")
@click.option("--openai-model-llm", default="gpt-4.1-mini", show_default=True, help="OpenAI LLM for answer generation.")
@click.option("--raw-output/--no-raw-output", "raw_output_flag", default=False, help="Show raw retrieval nodes and then the answer.")
@click.option("--hybrid/--no-hybrid", "use_hybrid_search", default=True, show_default=True, help="Enable hybrid (vector + sparse) search.")
@click.option("--sparse-top-k", type=int, default=10, show_default=True, help="Number of sparse results for hybrid search.")
@click.option("--rerank-top-n", type=int, default=0, show_default=True, help="Re-rank top N results using a cross-encoder (0 to disable).")
@click.option("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model for reranking.")
@click.option("--mmr/--no-mmr", "use_mmr", default=False, help="Use Maximal Marginal Relevance (MMR) for re-ranking.")
@click.option("--mmr-lambda", type=float, default=0.5, show_default=True, help="Lambda for MMR (0=max diversity, 1=max relevance).")
@click.option("--filter", "-f", "filters_kv", multiple=True, help="Metadata filter key=value (e.g., document_type=academic).")
@click.option("--expand-query/--no-expand-query", "use_query_expansion", default=True, help="Enable query expansion.")
@click.option("--max-expansions", type=int, default=3, show_default=True, help="Max number of expanded queries (excluding original).")
@click.option("--evaluate-rag/--no-evaluate-rag", "evaluate_rag_flag", default=False, help="Evaluate RAG quality and show feedback.")
@click.option("--compress-context/--no-compress-context", "compress_context_flag", default=False, help="Apply contextual compression (e.g., via CohereRerank if API key available).")
@click.option("--cohere-api-key", envvar="COHERE_API_KEY", default=None, help="Cohere API key (for CohereRerank compression).")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output.")
@click.argument("query_text", nargs=-1, required=True)
@click.pass_context # Pass context to get all params
def cli_entrypoint(ctx, **kwargs):
    """Synchronous entry point for Click that runs the async main logic."""
    import asyncio
    asyncio.run(main_async(**kwargs))

if __name__ == "__main__":
    cli_entrypoint()
