import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parser.pdf_parser import parse_pdf

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
        
        # Find Introduction
        def find_node(nodes, keyword):
            for node in nodes:
                if keyword.lower() in node["title"].lower():
                    return node
                found = find_node(node["children"], keyword)
                if found: return found
            return None
            
        intro_node = find_node(result, "introduction")
        if intro_node:
            print("\n" + "-"*20)
            print(f"CONTENT OF '{intro_node['title']}':")
            print("-"*20)
            text = "\n".join(intro_node["content"])
            print(text[:500] + "..." if len(text) > 500 else text)
        else:
            print("\nWarning: 'Introduction' section was not detected.")

if __name__ == "__main__":
    test_parse_demo()
