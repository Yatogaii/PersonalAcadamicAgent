import os
import sys
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parser.pdf_parser import parse_pdf, parse_pdf_chunks, SectionCategory, flatten_pdf_tree

def test_parse_demo():
    pdf_path = os.path.join(os.path.dirname(__file__), "demo.pdf")
    
    # Try with HTML conversion
    result = parse_pdf(pdf_path)
    
    if isinstance(result, list):
        print("\n" + "="*30)
        print("HIERARCHICAL EXTRACTION RESULTS (VIA HTML)")
        print("="*30)
        
        def print_tree(nodes, level=0):
            for node in nodes:
                indent = "  " * level
                # Count roughly how much text
                content_len = sum(len(c) for c in node.get("content", []))
                print(f"{indent}- {node['title']} (Page {node['page']+1}): {content_len} chars")
                print_tree(node["children"], level + 1)
        
        print_tree(result)

def test_parse_chunks_demo():
    pdf_path = os.path.join(os.path.dirname(__file__), "demo.pdf")
    print(f"\nTesting flat chunk parsing for: {pdf_path}")
    
    # Simulate passing a paper title that matches "RAG"
    # User feedback: "RAG is appearing as a keyword in the title"
    paper_title = "Retrieval-Augmented Generation (RAG) for Knowledge-Intensive NLP Tasks"
    print(f"Simulated Paper Title: {paper_title}")
    
    # We need to manually call flatten because parse_pdf_chunks doesn't accept title yet
    # Wait, parse_pdf_chunks calls flatten_pdf_tree(tree), but flatten_pdf_tree has default paper_title=""
    # Let's modify parse_pdf_chunks to accept paper_title or just call flatten directly here
    
    tree = parse_pdf(pdf_path)
    chunks = flatten_pdf_tree(tree, paper_title=paper_title)
    
    print("\n" + "="*30)
    print(f"FLAT CHUNK EXTRACTION RESULTS (Total: {len(chunks)})")
    print("="*30)
    
    if not chunks:
        print("No chunks extracted.")
        return

    # 1. Check continuity of chunk_index
    indices = [c['chunk_index'] for c in chunks]
    if indices:
        is_continuous = indices == list(range(min(indices), max(indices)+1))
        print(f"Chunk indices continuous: {is_continuous} (Range: {min(indices)} - {max(indices)})")
    
    # 2. Category Statistics
    categories = [c['section_category'] for c in chunks]
    cat_counts = Counter(categories)
    print("\nCategory Distribution:")
    for cat_id, count in cat_counts.items():
        try:
            cat_name = SectionCategory(cat_id).name
        except ValueError:
            cat_name = f"UNKNOWN({cat_id})"
        print(f"  - {cat_name} ({cat_id}): {count} chunks")

    # 3. Print all chunks to debug classification
    print("\nAll Chunks Classification:")
    for chunk in chunks:
        cat_id = chunk['section_category']
        try:
            cat_name = SectionCategory(cat_id).name
        except ValueError:
            cat_name = f"UNKNOWN({cat_id})"
        print(f"  [{cat_name}] {chunk['section_title']} (Parent: {chunk['parent_section']})")

    # 3. Print sample chunks from different categories
    print("\nSample Chunks (First occurrence of each category):")
    
    seen_cats = set()
    for chunk in chunks:
        cat = chunk['section_category']
        if cat not in seen_cats:
            seen_cats.add(cat)
            try:
                cat_name = SectionCategory(cat).name
            except ValueError:
                cat_name = f"UNKNOWN({cat})"
                
            print(f"\n[Sample for {cat_name}]")
            print(f"  ID: {chunk['chunk_index']}")
            print(f"  Section: {chunk['section_title']}")
            print(f"  Parent: {chunk['parent_section']}")
            print(f"  Page: {chunk['page_number']}")
            print(f"  Content Preview: {chunk['text'][:150].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    # test_parse_demo()
    test_parse_chunks_demo()
