"""
table_parser_example.py

Example implementation showing how to use the TableAwareNodeParser
for improved table handling in a RAG system.
"""

from typing import List
from pathlib import Path
import pandas as pd
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    Document,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from table_aware_parser import create_table_aware_parser, TableNode

def create_sample_table_document() -> str:
    """Create a sample document containing both text and tables."""
    return """
# Sales Performance Report

## Overview
This report summarizes our quarterly sales performance across different regions.

## Quarterly Sales Data

| Region    | Q1 Sales | Q2 Sales | Q3 Sales | Q4 Sales |
|-----------|----------|----------|----------|----------|
| North     | 125000   | 145000   | 132000   | 160000   |
| South     | 98000    | 112000   | 108000   | 122000   |
| East      | 156000   | 168000   | 172000   | 182000   |
| West      | 134000   | 142000   | 138000   | 152000   |

## Product Categories

| Category      | Total Units | Avg Price | Revenue   |
|--------------|-------------|-----------|-----------|
| Electronics   | 15200       | 499.99    | 7599848   |
| Appliances   | 8500        | 799.99    | 6799915   |
| Furniture    | 12300       | 299.99    | 3689877   |
| Accessories  | 45600       | 49.99     | 2279544   |

## Analysis
The East region consistently shows the strongest performance,
while the South region may need additional support to improve sales numbers.
"""

def demonstrate_table_aware_rag():
    """
    Demonstrate the usage of TableAwareNodeParser in a RAG system.
    """
    # Initialize LlamaIndex settings
    Settings.llm = OpenAI(model="gpt-4")
    Settings.embed_model = OpenAIEmbedding()
    
    # Create sample document
    doc_text = create_sample_table_document()
    document = Document(text=doc_text)
    
    # Initialize the table-aware parser
    parser = create_table_aware_parser(
        chunk_size=1024,
        chunk_overlap=20
    )
    
    # Parse the document
    nodes = parser.get_nodes_from_documents([document])
    
    # Print node types and content
    print("\nGenerated Nodes:")
    print("===============")
    
    table_nodes = []
    text_nodes = []
    
    for node in nodes:
        if isinstance(node, TableNode):
            table_nodes.append(node)
        else:
            text_nodes.append(node)
    
    print(f"\nFound {len(table_nodes)} table nodes and {len(text_nodes)} text nodes")
    
    # Display table node details
    print("\nTable Node Details:")
    print("==================")
    
    for i, node in enumerate(table_nodes, 1):
        print(f"\nTable {i}:")
        print(f"Headers: {node.header}")
        print(f"Column Types: {node.column_types}")
        print(f"Number of Rows: {len(node.rows)}")
        print(f"Metadata: {node.table_metadata}")
        print("\nFormatted Content:")
        print(node.get_content())
        print("-" * 50)
    
    # Create and query index
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    # Example queries demonstrating table awareness
    example_queries = [
        "What was the highest quarterly sales figure and in which region?",
        "Compare the revenue of Electronics and Furniture categories.",
        "Which region showed consistent growth across quarters?",
        "What is the total revenue from all product categories?"
    ]
    
    print("\nExample Queries and Responses:")
    print("============================")
    
    query_engine = index.as_query_engine(
        response_mode="tree_summarize",
        verbose=True
    )
    
    for query in example_queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Response: {response}")
        print("-" * 50)

def main():
    """Main execution function."""
    print("Table-Aware Parser RAG Example")
    print("=============================")
    
    try:
        demonstrate_table_aware_rag()
    except Exception as e:
        print(f"Error during demonstration: {e}")
        raise

if __name__ == "__main__":
    main()