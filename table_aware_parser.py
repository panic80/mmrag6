"""
table_aware_parser.py

Custom table-aware node parser implementations for LlamaIndex that preserve
table structure and relationships during the ingestion process.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode, TextNode, MetadataMode
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

@dataclass
class TableNode(TextNode):
    """Custom node type for representing table data with structure preservation."""
    header: List[str]
    rows: List[List[str]]
    column_types: Dict[str, str]
    table_metadata: Dict[str, Any]

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get the content of the table node in a structured format."""
        # Create a formatted string representation of the table
        content = []
        
        # Add table metadata if available
        if self.table_metadata.get('title'):
            content.append(f"Table: {self.table_metadata['title']}")
        if self.table_metadata.get('description'):
            content.append(f"Description: {self.table_metadata['description']}")
        
        # Format headers and rows
        header_str = " | ".join(self.header)
        content.append(f"\n{header_str}")
        content.append("-" * len(header_str))
        
        for row in self.rows:
            content.append(" | ".join(str(cell) for cell in row))
            
        return "\n".join(content)

class TableAwareNodeParser(SimpleNodeParser):
    """
    A node parser that intelligently handles tables while preserving their structure
    and relationships during the parsing process.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel

    def _extract_tables_from_html(self, html_content: str) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Extract tables from HTML content using BeautifulSoup."""
        tables = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for table in soup.find_all('table'):
            # Extract table metadata
            metadata = {
                'title': table.get('title', ''),
                'description': table.get('summary', ''),
                'html_id': table.get('id', ''),
                'html_class': table.get('class', [])
            }
            
            # Convert HTML table to DataFrame
            df = pd.read_html(str(table))[0]
            tables.append((df, metadata))
            
        return tables

    def _extract_tables_from_markdown(self, markdown_content: str) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Extract tables from Markdown content."""
        tables = []
        # Match markdown tables using regex
        table_pattern = r'(?P<title>#+\s*([^\n]+)\s*\n)?(?P<table>(?:\|[^\n]+\|\n)+(?:\|[-:| ]+\|\n)(?:\|[^\n]+\|\n)+)'
        
        for match in re.finditer(table_pattern, markdown_content):
            title = match.group('title').strip() if match.group('title') else ''
            table_content = match.group('table')
            
            # Parse markdown table into rows
            rows = [line.strip().strip('|').split('|') for line in table_content.strip().split('\n')]
            headers = [col.strip() for col in rows[0]]
            
            # Create DataFrame
            df = pd.DataFrame(rows[2:], columns=headers)
            metadata = {'title': title, 'format': 'markdown'}
            tables.append((df, metadata))
            
        return tables

    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer the data types of table columns."""
        column_types = {}
        
        for col in df.columns:
            # Try to infer numeric types
            try:
                pd.to_numeric(df[col])
                if df[col].dtype in (np.int32, np.int64):
                    column_types[col] = 'integer'
                else:
                    column_types[col] = 'float'
            except:
                # Check for date/time
                try:
                    pd.to_datetime(df[col])
                    column_types[col] = 'datetime'
                except:
                    # Default to string/categorical
                    if df[col].nunique() / len(df[col]) < 0.5:
                        column_types[col] = 'categorical'
                    else:
                        column_types[col] = 'string'
        
        return column_types

    def _create_table_node(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
        start_idx: Optional[int] = None
    ) -> TableNode:
        """Create a TableNode from a pandas DataFrame with metadata."""
        # Infer column types
        column_types = self._infer_column_types(df)
        
        # Convert DataFrame to lists for storage
        headers = df.columns.tolist()
        rows = df.values.tolist()
        
        # Create node metadata
        node_metadata = {
            "data_types": column_types,
            "row_count": len(rows),
            "column_count": len(headers)
        }
        node_metadata.update(metadata)  # Add any additional metadata
        
        return TableNode(
            text=df.to_string(),  # Fallback plain text representation
            header=headers,
            rows=rows,
            column_types=column_types,
            table_metadata=node_metadata,
            doc_id=doc_id,
            start_char_idx=start_idx if start_idx is not None else 0
        )

    def get_nodes_from_documents(
        self,
        documents: List[Document],
        show_progress: bool = False
    ) -> List[BaseNode]:
        """
        Parse documents into nodes while preserving table structure.
        
        Args:
            documents: List of Document objects to parse
            show_progress: Whether to show a progress bar
            
        Returns:
            List of nodes, including both text and table nodes
        """
        all_nodes = []
        
        for doc in documents:
            text_chunks = []
            table_nodes = []
            
            # Try to extract tables from HTML
            tables_html = self._extract_tables_from_html(doc.text)
            if tables_html:
                for df, metadata in tables_html:
                    table_nodes.append(self._create_table_node(df, metadata, doc.doc_id))
                    
            # Try to extract tables from Markdown
            tables_md = self._extract_tables_from_markdown(doc.text)
            if tables_md:
                for df, metadata in tables_md:
                    table_nodes.append(self._create_table_node(df, metadata, doc.doc_id))
            
            # If no tables found, or there's text between tables, process as regular text
            if not tables_html and not tables_md:
                # Use parent class's text splitting for non-table content
                text_nodes = super().get_nodes_from_documents([doc], show_progress)
                all_nodes.extend(text_nodes)
            
            # Add table nodes
            all_nodes.extend(table_nodes)
            
            # Add relationships between nodes if enabled
            if self.include_prev_next_rel and len(all_nodes) > 1:
                for i in range(len(all_nodes) - 1):
                    all_nodes[i].relationships["next"] = all_nodes[i + 1].node_id
                    all_nodes[i + 1].relationships["previous"] = all_nodes[i].node_id
        
        return all_nodes

# Example usage:
def create_table_aware_parser(
    chunk_size: Optional[int] = 1024,
    chunk_overlap: Optional[int] = 20
) -> TableAwareNodeParser:
    """Extract tables from Markdown content."""

def create_table_aware_parser(
    chunk_size: Optional[int] = 1024,
    chunk_overlap: Optional[int] = 20
) -> TableAwareNodeParser:
    """
    Create a configured instance of the TableAwareNodeParser.
    
    Args:
        chunk_size: Maximum size of text chunks
        chunk_overlap: Overlap between text chunks
        
    Returns:
        Configured TableAwareNodeParser instance
    """
    return TableAwareNodeParser(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_metadata=True,
        include_prev_next_rel=True
    )