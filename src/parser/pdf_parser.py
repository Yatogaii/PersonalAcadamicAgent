import fitz  # PyMuPDF
from bs4 import BeautifulSoup
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

def clean_text(text):
    """
    Cleans the extracted text:
    1. Removes citations like [1], [ 1 ], [1, 2], [ \n 1 \n ].
    2. Fixes hyphenation (word-\nword -> wordword).
    3. Removes meaningless newlines and collapses spaces.
    """
    # Remove citations
    # Pattern matches [ followed by digits, commas, hyphens, whitespace followed by ]
    text = re.sub(r'\[\s*[\d\s,\-]+\s*\]', '', text)
    
    # Fix hyphenation: word-\nword -> wordword
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Replace newlines with space
    text = text.replace('\n', ' ')
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove space before punctuation
    text = re.sub(r'\s+([,.;:?!])', r'\1', text)
    
    return text.strip()

def parse_pdf(pdf_path):
    """
    Parses PDF by first converting pages to HTML.
    Returns the hierarchical structure with cleaned content.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Converting {pdf_path} to HTML for analysis...")
    doc = fitz.open(pdf_path)
    
    outline_tree = get_pdf_outline(doc)
    
    if not outline_tree:
        print("No outline found.")
        return []
    
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
            
            # Convert page to HTML
            # This preserves reading order (columns) and provides style info
            html_content = page.get_text("html")
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract text from HTML
            # We can use separator to keep paragraphs distinct
            page_text = soup.get_text(separator="\n")
            
            # Simple cleaning to prepare for regex matching
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            cleaned_text = "\n".join(lines)
            
            # Find start index
            start_idx = 0
            if p_idx == start_page_idx:
                # Try to find title
                # We use a loose match because HTML conversion might add spaces/newlines
                pattern = re.escape(current_title).replace(r'\ ', r'\s+')
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    start_idx = match.end()
            
            # Find end index
            end_idx = len(cleaned_text)
            if p_idx == end_page_idx and next_title:
                pattern = re.escape(next_title).replace(r'\ ', r'\s+')
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    end_idx = match.start()
            
            if start_idx < end_idx:
                chunk = cleaned_text[start_idx:end_idx].strip()
                if chunk:
                    # Apply deep cleaning to the content chunk
                    cleaned_chunk = clean_text(chunk)
                    if cleaned_chunk:
                        node["content"].append(cleaned_chunk)

    return outline_tree
