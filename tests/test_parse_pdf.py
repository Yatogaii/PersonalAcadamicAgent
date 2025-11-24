import fitz  # PyMuPDF
import os
import re

def get_pdf_outline(doc):
    """
    Extracts the outline (bookmarks) from a PDF using PyMuPDF.
    Returns a nested list of dictionaries representing the structure.
    Each dict has: 'title', 'page', 'children'.
    """
    toc = doc.get_toc()
    # toc item: [lvl, title, page] or [lvl, title, page, dest] depending on version/file
    # PyMuPDF documentation says get_toc(simple=True) returns [lvl, title, page]
    # Default is simple=True
    
    root = []
    stack = [] # Stack of (level, list_to_append_to)
    
    # Initialize stack with root level
    # Level 1 is usually top level
    stack.append((0, root))
    
    for item in toc:
        # Handle variable length tuple
        if len(item) == 3:
            lvl, title, page = item
        elif len(item) >= 4:
            lvl, title, page = item[0], item[1], item[2]
        else:
            continue
            
        # page is 1-based in get_toc output usually, let's verify
        # PyMuPDF docs say page number is 1-based. We convert to 0-based.
        # If page is -1, it means no destination
        if page > 0:
            page_idx = page - 1
        else:
            page_idx = 0 # Fallback
        
        node = {"title": title, "page": page_idx, "children": []}
        
        # Find the correct parent list
        while stack and stack[-1][0] >= lvl:
            stack.pop()
            
        if not stack:
            # Should not happen if hierarchy is correct starting at 1
            # Fallback to root
            root.append(node)
            stack.append((lvl, node["children"]))
        else:
            parent_list = stack[-1][1]
            parent_list.append(node)
            stack.append((lvl, node["children"]))
            
    return root

def parse_pdf_with_outline(pdf_path):
    """
    Parses PDF using its outline/bookmarks if available.
    Returns a hierarchical structure with content.
    """
    print(f"Checking for outline in {pdf_path}...")
    doc = fitz.open(pdf_path)
    
    outline_tree = get_pdf_outline(doc)
    
    if not outline_tree:
        print("No outline found. Falling back to regex-based parsing (not implemented in this version).")
        return {}
    
    print(f"Found outline with structure.")
    
    # Helper to flatten tree for sequential processing
    flat_nodes = []
    def _flatten(nodes):
        for node in nodes:
            flat_nodes.append(node)
            _flatten(node["children"])
    _flatten(outline_tree)
    
    total_pages = doc.page_count
    
    for i, node in enumerate(flat_nodes):
        start_page_idx = node["page"]
        current_title = node["title"]
        
        # Determine end page and next title for boundary checking
        if i < len(flat_nodes) - 1:
            next_node = flat_nodes[i+1]
            end_page_idx = next_node["page"]
            next_title = next_node["title"]
        else:
            end_page_idx = total_pages - 1 # Last page inclusive
            next_title = None
        
        node["content"] = []
        
        # Iterate through pages involved in this section
        for p_idx in range(start_page_idx, end_page_idx + 1):
            if p_idx >= total_pages: break
            
            page = doc[p_idx]
            # get_text("blocks") returns list of (x0, y0, x1, y1, "lines", block_no, block_type)
            # block_type=0 is text, 1 is image
            blocks = page.get_text("blocks")
            
            # Sort blocks? PyMuPDF usually returns them in reading order (column-wise)
            # But we can ensure it by sorting by vertical then horizontal if needed
            # Actually default is usually good.
            
            for b in blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                if block_type != 0: continue # Skip images
                
                text = text.strip()
                if not text: continue
                
                # Logic to filter content based on titles
                # This is tricky because a block might contain the title AND content
                # Or the title might be its own block.
                
                # Simple heuristic: 
                # If we are on the start page, we skip blocks until we find the title block
                # If we are on the end page, we stop when we find the next title block
                
                # Note: This is a simplification. Robust implementation needs fuzzy matching.
                
                # For now, we just append all blocks in the page range.
                # To be more precise, we would need to track "state" (collecting vs not collecting)
                # But since we are iterating by section, we assume the page range is mostly correct.
                # The boundary pages are the problem.
                
                # Let's try to exclude the title itself if it's a standalone block
                if current_title.lower() in text.lower() and len(text) < len(current_title) + 20:
                    # Likely the header block
                    continue
                    
                if next_title and next_title.lower() in text.lower() and len(text) < len(next_title) + 20:
                    # Likely the next header block, and since blocks are in order, 
                    # subsequent blocks on this page likely belong to next section.
                    # But wait, we are in the loop for the CURRENT section.
                    # So if we hit the next title, we should STOP for this page.
                    if p_idx == end_page_idx:
                        break
                
                node["content"].append(text)

    return outline_tree

def parse_pdf_sections(pdf_path):
    """
    Parses a PDF and attempts to extract sections based on common headers.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return

    print(f"Analyzing {pdf_path}...")

    # Common section headers in academic papers
    # We use regex to match things like "1. Introduction" or just "Introduction"
    # The regex allows for optional numbering and case insensitivity
    # Updated to be more permissive for PDF extraction artifacts
    header_patterns = {
        "Abstract": re.compile(r'^\s*(?:abstract)(?:.*)?$', re.IGNORECASE),
        "Introduction": re.compile(r'^\s*(?:1\.?|I\.?)?\s*introduction(?:.*)?$', re.IGNORECASE),
        "Related Work": re.compile(r'^\s*(?:2\.?|II\.?)?\s*related\s+work(?:.*)?$', re.IGNORECASE),
        "Methodology": re.compile(r'^\s*(?:3\.?|III\.?)?\s*(?:methodology|method|approach)(?:.*)?$', re.IGNORECASE),
        "Experiments": re.compile(r'^\s*(?:4\.?|IV\.?)?\s*(?:experiments|results|evaluation)(?:.*)?$', re.IGNORECASE),
        "Conclusion": re.compile(r'^\s*(?:5\.?|V\.?)?\s*(?:conclusion|conclusions|discussion)(?:.*)?$', re.IGNORECASE),
        "References": re.compile(r'^\s*(?:references|bibliography)(?:.*)?$', re.IGNORECASE),
    }

    extracted_content = {key: [] for key in header_patterns.keys()}
    extracted_content["Preamble"] = [] # For text before the first header
    
    current_section = "Preamble"

    # Fallback to regex-based parsing using PyMuPDF
    # This is a simplified version that just dumps text for now
    # Implementing full regex parsing with blocks is complex but doable
    
    doc = fitz.open(pdf_path)
    extracted_content = {}
    
    # ... implementation omitted for brevity as we are focusing on outline ...
    print("Regex fallback not fully implemented in this PyMuPDF version.")
    return {}

def test_parse_demo():
    pdf_path = os.path.join(os.path.dirname(__file__), "demo.pdf")
    
    # Try with outline first
    result = parse_pdf_with_outline(pdf_path)
    
    if isinstance(result, list):
        print("\n" + "="*30)
        print("HIERARCHICAL EXTRACTION RESULTS")
        print("="*30)
        
        def print_tree(nodes, level=0):
            for node in nodes:
                indent = "  " * level
                count = len(node.get("content", []))
                print(f"{indent}- {node['title']} (Page {node['page']+1}): {count} pages/blocks")
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
            
    elif isinstance(result, dict):
        sections = result
        print("\n" + "="*30)
        print("EXTRACTION RESULTS")
        print("="*30)
        
        # Print a summary of what was found
        for section, lines in sections.items():
            # lines here are actually full pages or large chunks
            count = len(lines)
            print(f"Section '{section}': {count} blocks extracted.")
            
        # Look for Introduction-like section dynamically
        intro_key = next((k for k in sections.keys() if "introduction" in k.lower()), None)
        
        if intro_key:
            print("\n" + "-"*20)
            print(f"CONTENT OF '{intro_key}':")
            print("-"*20)
            # Print first 500 characters
            intro_text = "\n".join(sections[intro_key])
            print(intro_text[:500] + "..." if len(intro_text) > 500 else intro_text)
        else:
            print("\nWarning: 'Introduction' section was not detected.")

if __name__ == "__main__":
    test_parse_demo()
