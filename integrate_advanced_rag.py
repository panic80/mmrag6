#!/usr/bin/env python3

"""
Integration script for adding advanced RAG features to existing codebase.

This script modifies the necessary files to integrate:
1. Semantic Document Chunking
2. Query Expansion
3. RAG Self-Evaluation

Usage:
    python integrate_advanced_rag.py
"""

import os
import sys
import shutil
import re
from datetime import datetime

# Paths to files we'll modify
INGEST_RAG_PATH = 'ingest_rag.py'
QUERY_RAG_PATH = 'query_rag.py'

# Backup directory for original files
BACKUP_DIR = 'backup_' + datetime.now().strftime('%Y%m%d_%H%M%S')


def backup_files():
    """Create backups of files we'll modify."""
    print(f"Creating backup directory: {BACKUP_DIR}")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    for file_path in [INGEST_RAG_PATH, QUERY_RAG_PATH]:
        if os.path.exists(file_path):
            backup_path = os.path.join(BACKUP_DIR, os.path.basename(file_path))
            print(f"Backing up {file_path} to {backup_path}")
            shutil.copy2(file_path, backup_path)


def check_advanced_rag_module():
    """Check if advanced_rag.py exists; if not, alert the user."""
    if not os.path.exists('advanced_rag.py'):
        print("WARNING: advanced_rag.py not found!")
        print("Please make sure you've created this file with the required functionality")
        return False
    return True


def integrate_semantic_chunking():
    """Integrate semantic chunking into ingest_rag.py."""
    print("\nIntegrating Semantic Document Chunking...")
    
    with open(INGEST_RAG_PATH, 'r') as f:
        content = f.read()
    
    # Rename the existing _smart_chunk_text function to include 'original' in the docstring
    content = content.replace(
        'def _smart_chunk_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:\n    """\n    Chunk text on paragraph and sentence boundaries up to max_chars,', 
        'def _smart_chunk_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:\n    """\n    Chunk text on paragraph and sentence boundaries up to max_chars,\n    \n    This is the original chunking method, kept as fallback if semantic chunking fails.'
    )
    
    # Add the semantic_chunk_text function after _smart_chunk_text
    semantic_chunk_func = '''

def semantic_chunk_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    """
    Chunk text based on semantic topic boundaries.
    
    Args:
        text: The text to chunk
        max_chars: Maximum character length per chunk
        overlap: Overlap between chunks (not used in semantic chunking)
        
    Returns:
        List of semantically chunked text segments
    """
    # Import from advanced_rag if available, otherwise fall back to regular chunking
    try:
        from advanced_rag import semantic_chunk_text as advanced_semantic_chunk
        click.echo("[info] Using semantic chunking from advanced_rag module")
        chunks = advanced_semantic_chunk(text, max_chars=max_chars)
        
        # Verify we got valid chunks
        if chunks and all(isinstance(c, str) for c in chunks):
            click.echo(f"[info] Semantic chunking produced {len(chunks)} chunks")
            return chunks
        else:
            click.echo("[warning] Semantic chunking failed to produce valid chunks, falling back to regular chunking", err=True)
            return _smart_chunk_text(text, max_chars, overlap)
            
    except (ImportError, Exception) as e:
        click.echo(f"[warning] Semantic chunking not available or failed: {e}", err=True)
        click.echo("[info] Falling back to regular chunking", err=True)
        return _smart_chunk_text(text, max_chars, overlap)
'''
    
    # Find the end of _smart_chunk_text function and add semantic_chunk_text after it
    smart_chunk_end = content.find("    return chunks", content.find("def _smart_chunk_text"))
    smart_chunk_end = content.find("\n", smart_chunk_end) + 1  # Move to the next line after the return
    
    modified_content = content[:smart_chunk_end] + semantic_chunk_func + content[smart_chunk_end:]
    
    # Modify _chunk_text_tokenwise to try semantic chunking first
    tokenwise_start = modified_content.find("def _chunk_text_tokenwise")
    tokenwise_try_start = modified_content.find("try:", tokenwise_start)
    tokenwise_try_block = modified_content.find("            from docling.text import TextSplitter", tokenwise_try_start)
    
    # Replace the try block with our modified version
    modified_try_block = '''        # 1. Try semantic chunking first (from advanced_rag)
        try:
            # Use semantic chunking for better context-aware chunks
            chunks = semantic_chunk_text(text, chunk_size, overlap)
            click.echo("[info] Used semantic chunking for text")
        except Exception as e:
            click.echo(f"[warning] Semantic chunking failed, trying fallbacks: {e}", err=True)
            # 2. Try docling's GPT-aware splitter.
            try:'''
    
    modified_content = modified_content[:tokenwise_try_block] + modified_try_block + modified_content[tokenwise_try_block:]
    
    # Write the modified content back
    with open(INGEST_RAG_PATH, 'w') as f:
        f.write(modified_content)
    
    print("Semantic chunking integration complete!")


def integrate_query_expansion():
    """Integrate query expansion into query_rag.py."""
    print("\nIntegrating Query Expansion...")
    
    with open(QUERY_RAG_PATH, 'r') as f:
        content = f.read()
    
    # 1. Add new command-line options before the query argument
    query_arg_match = re.search(r'@click\.argument\("query", nargs=-1, required=True\)', content)
    if not query_arg_match:
        print("Could not find query argument in command definition")
        return
    
    expansion_options = '''@click.option(
    "--use-expansion",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable query expansion for better recall (requires advanced_rag module).",
)
@click.option(
    "--max-expansions",
    type=int,
    default=3,
    show_default=True,
    help="Maximum number of query expansions to use.",
)
@click.option(
    "--evaluate",
    is_flag=True,
    default=False,
    show_default=True,
    help="Evaluate RAG quality and return feedback with results (requires advanced_rag module).",
)
'''
    
    # Insert expansion options before @click.argument("query"...)
    pos = query_arg_match.start()
    modified_content = content[:pos] + expansion_options + content[pos:]
    
    # 2. Add new parameters to main function definition
    main_def_match = re.search(r'def main\([^)]*\)', modified_content)
    if not main_def_match:
        print("Could not find main function definition")
        return
    
    # Find the closing parenthesis of the function definition
    def_end = modified_content.find(")", main_def_match.start()) 
    # Add new parameters before the closing parenthesis
    new_params = ",\n    use_expansion: bool,\n    max_expansions: int,\n    evaluate: bool"
    modified_content = modified_content[:def_end] + new_params + modified_content[def_end:]
    
    # 3. Modify the query embedding and search logic
    vector_match = re.search(r'# Embed the query[\s\S]*?vector = resp\["data"\]\[0\]\["embedding"\]', modified_content)
    if not vector_match:
        print("Could not find query embedding code")
        return
    
    query_expansion_code = '''# Use query expansion if enabled
    expanded_queries = [query_text]
    if use_expansion:
        try:
            from advanced_rag import expand_query
            click.echo(f"[info] Using query expansion from advanced_rag module")
            
            # Expand the query
            expanded_queries = expand_query(query_text, openai_client, max_expansions=max_expansions)
            
            # Show the expanded queries
            if len(expanded_queries) > 1:
                click.echo("\\nExpanded queries:")
                for i, exp_query in enumerate(expanded_queries):
                    click.echo(f"  {i+1}. {exp_query}")
        except ImportError as e:
            click.echo(f"[warning] Query expansion failed (advanced_rag module not found): {e}", err=True)
            click.echo("[info] To use query expansion, install the advanced_rag module")
        except Exception as e:
            click.echo(f"[warning] Query expansion failed: {e}", err=True)
            click.echo("[info] Continuing with original query only")
    
    # Collection for all search results
    all_results = []
    
    # Process each query (original and expanded)
    for q_idx, q_text in enumerate(expanded_queries):
        if q_idx > 0:
            click.echo(f"\\nProcessing expanded query {q_idx+1}: {q_text}")
            
        # Embed this query
'''
    
    pos = vector_match.start()
    end_pos = modified_content.find("# Embed the query", pos) + len("# Embed the query")
    modified_content = modified_content[:end_pos] + query_expansion_code + modified_content[end_pos:]
    
    # Replace the original embedding code to use the current q_text instead of query_text
    modified_content = modified_content.replace(
        'input=[query_text])', 
        'input=[q_text])'
    )
    
    # 4. Modify the search block to handle multiple search results
    search_match = re.search(r'# Search in Qdrant[\s\S]*?raise', modified_content)
    if not search_match:
        print("Could not find search code block")
        return
    
    search_block = search_match.group(0)
    
    # Replace the original search code to handle multiple queries
    modified_search = search_block.replace(
        'scored = getattr(resp, \'points\', [])', 
        'query_results = getattr(resp, \'points\', [])\n        all_results.extend(query_results)'
    ).replace(
        'scored = client.search(',
        'query_results = client.search('
    ).replace(
        ')',
        ')\n        all_results.extend(query_results)'
    )
    
    # Add exception handling for expanded queries
    modified_search = modified_search.replace(
        'sys.exit(1)',
        'sys.exit(1)\n        # For expansion queries, just log error and continue with other queries\n        if q_idx > 0 and use_expansion:\n            click.echo(f"[warning] Search failed for expanded query {q_idx+1}: {e}", err=True)\n            continue'
    )
    
    modified_content = modified_content.replace(search_block, modified_search)
    
    # 5. Add result deduplication and processing after the search block
    dedup_code = '''
# After processing all queries, deduplicate results by ID if we used expansion
if use_expansion and len(expanded_queries) > 1 and all_results:
    # Deduplicate by ID and keep highest score
    from types import SimpleNamespace
    seen_ids = {}
    unique_results = []
    
    for result in all_results:
        result_id = getattr(result, 'id', None)
        if result_id is None:
            continue
            
        score = getattr(result, 'score', 0.0)
        if result_id not in seen_ids or score > seen_ids[result_id]:
            seen_ids[result_id] = score
            # Keep this result
            unique_results.append(result)
    
    # Sort by score descending and limit to k
    scored = sorted(unique_results, key=lambda x: getattr(x, 'score', 0.0), reverse=True)[:k]
    click.echo(f"[info] Combined {len(all_results)} results from {len(expanded_queries)} queries into {len(scored)} unique results")
else:
    # Just use results from the original query
    scored = all_results
'''
    
    # Find the position after the search block
    pos = modified_content.find(search_match.group(0)) + len(search_match.group(0))
    # Add the deduplication code after the search block
    modified_content = modified_content[:pos] + dedup_code + modified_content[pos:]
    
    # Write the modified content back
    with open(QUERY_RAG_PATH, 'w') as f:
        f.write(modified_content)
    
    print("Query expansion integration complete!")


def integrate_rag_evaluation():
    """Integrate RAG self-evaluation into query_rag.py."""
    print("\nIntegrating RAG Self-Evaluation...")
    
    with open(QUERY_RAG_PATH, 'r') as f:
        content = f.read()
    
    # Find the position where the final answer is generated
    gen_answer_match = re.search(r'# Raw mode: just plain response[\s\S]*?click\.echo\(llm_resp\)', content)
    if not gen_answer_match:
        print("Could not find answer generation code")
        return
    
    # Position after the answer generation
    pos = gen_answer_match.end()
    
    # Add evaluation code
    evaluation_code = '''
        
        # Evaluate RAG response if requested
        if evaluate:
            try:
                from advanced_rag import evaluate_rag_quality
                click.echo("\\n[info] Evaluating RAG response quality...")
                
                # Format the retrieved chunks for evaluation
                chunk_texts = []
                for point in scored[:5]:  # Use top 5 chunks for evaluation
                    payload = getattr(point, 'payload', {}) or {}
                    chunk_text = payload.get('chunk_text', '')
                    if chunk_text:
                        chunk_texts.append(chunk_text)
                
                # Run the evaluation
                eval_result = evaluate_rag_quality(
                    query=query_text,
                    retrieved_chunks=chunk_texts,
                    generated_answer=llm_resp,
                    openai_client=openai_client
                )
                
                # Display the evaluation results
                click.echo("\\n=== RAG Quality Evaluation ===")
                
                # Display scores
                scores = eval_result.get('scores', {})
                if scores:
                    click.echo("\\nScores (1-10):")
                    for metric, score in scores.items():
                        click.echo(f"  {metric.capitalize()}: {score}")
                
                # Display feedback
                feedback = eval_result.get('feedback', {})
                if feedback:
                    strengths = feedback.get('strengths', [])
                    if strengths:
                        click.echo("\\nStrengths:")
                        for s in strengths:
                            click.echo(f"  ✓ {s}")
                    
                    weaknesses = feedback.get('weaknesses', [])
                    if weaknesses:
                        click.echo("\\nWeaknesses:")
                        for w in weaknesses:
                            click.echo(f"  ✗ {w}")
                    
                    suggestions = feedback.get('improvement_suggestions', [])
                    if suggestions:
                        click.echo("\\nImprovement Suggestions:")
                        for s in suggestions:
                            click.echo(f"  → {s}")
                
            except ImportError as e:
                click.echo(f"[warning] RAG evaluation failed (advanced_rag module not found): {e}", err=True)
                click.echo("[info] To use RAG evaluation, install the advanced_rag module")
            except Exception as e:
                click.echo(f"[warning] RAG evaluation failed: {e}", err=True)'''
    
    # Add the evaluation code
    modified_content = content[:pos] + evaluation_code + content[pos:]
    
    # Write the modified content back
    with open(QUERY_RAG_PATH, 'w') as f:
        f.write(modified_content)
    
    print("RAG evaluation integration complete!")


def main():
    """Main execution function."""
    print("Advanced RAG Integration Script")
    print("===============================")
    
    # Check for advanced_rag.py
    if not check_advanced_rag_module():
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            print("Aborting integration.")
            return
    
    # Create backups
    backup_files()
    
    # Integrate features
    integrate_semantic_chunking()
    integrate_query_expansion()
    integrate_rag_evaluation()
    
    print("\nIntegration complete!")
    print(f"Backup files are saved in the {BACKUP_DIR} directory")
    print("\nTest the enhanced functionality with:")
    print("  1. /inject <URL> - Now uses semantic chunking by default")
    print("  2. /ask --use-expansion how does RAG work? - Uses query expansion")
    print("  3. /ask --evaluate what is retrieval augmented generation? - Includes quality evaluation")


if __name__ == "__main__":
    main()