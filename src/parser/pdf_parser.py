import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import os
import re
from enum import IntEnum

class SectionCategory(IntEnum):
    ABSTRACT = 0
    INTRODUCTION = 1
    METHOD = 2
    EVALUATION = 3
    CONCLUSION = 4
    OTHER = 5
    RELATED_WORK = 6

def classify_section(title: str, paper_title: str = "") -> int:
    """
    Classifies a section title into one of the predefined categories.
    """
    title_lower = title.lower()
    
    if re.search(r'abstract', title_lower):
        return SectionCategory.ABSTRACT
    elif re.search(r'related work', title_lower):
        return SectionCategory.RELATED_WORK
    elif re.search(r'introduction|background|motivation', title_lower):
        return SectionCategory.INTRODUCTION
    elif re.search(r'method|approach|architecture|proposed|framework', title_lower):
        return SectionCategory.METHOD
    elif re.search(r'experiment|result|evaluation|ablation|performance|comparison', title_lower):
        return SectionCategory.EVALUATION
    elif re.search(r'conclusion|discussion|summary|future work', title_lower):
        return SectionCategory.CONCLUSION
    
    # Dynamic Method Matching: Check if section title contains significant words from paper title
    if paper_title:
        # Extract words, ignore common stop words
        stop_words = {'a', 'an', 'the', 'for', 'and', 'of', 'in', 'on', 'with', 'to', 'at', 'by', 'from'}
        paper_words = [w.lower() for w in re.findall(r'\w+', paper_title) if w.lower() not in stop_words and len(w) > 2]
        
        # Check if any significant word from paper title appears in section title
        # We require at least one significant word match, but maybe we should be stricter?
        # Let's try to match if the section title is mostly composed of title keywords
        # Or if the section title *is* a substring of the paper title (fuzzy)
        
        # Simple heuristic: If section title contains a significant keyword from paper title
        # AND it's not just a generic word (which we filtered via stop_words, but maybe not enough)
        for word in paper_words:
            if word in title_lower:
                # Double check it's not a false positive like "Introduction" (already handled)
                return SectionCategory.METHOD

    return SectionCategory.OTHER

def flatten_pdf_tree(outline_tree, paper_title=""):
    """
    Flattens the hierarchical outline tree into a list of chunks with metadata.
    Adds chunk_index, section_category, hierarchy_level, etc.
    """
    chunks = []
    global_chunk_idx = 0
    
    def _traverse(nodes, parent_title="", parent_category=SectionCategory.OTHER):
        nonlocal global_chunk_idx
        for node in nodes:
            current_title = node["title"]
            
            # Strict Inheritance: If parent is a specific category (not OTHER), 
            # children inherit it automatically without re-classification.
            # This ensures 2.1, 2.2 etc. belong to the same category as 2.
            if parent_category != SectionCategory.OTHER:
                category = parent_category
            else:
                category = classify_section(current_title, paper_title)
            
            # If content exists, create a chunk
            # Note: In the current implementation, 'content' is a list of strings (paragraphs/chunks)
            # We might want to join them or treat each as a separate chunk.
            # Let's treat each non-empty content string as a chunk for now.
            
            if "content" in node and node["content"]:
                for content_part in node["content"]:
                    if not content_part.strip():
                        continue
                        
                    chunk = {
                        "chunk_index": global_chunk_idx,
                        "text": content_part,
                        "section_title": current_title,
                        "parent_section": parent_title,
                        "section_category": int(category),
                        "page_number": node["page"] + 1, # 1-based
                        # hierarchy_level is not explicitly in node, but we could infer or pass it down
                        # For now, let's assume the parser structure implies level via recursion depth if we tracked it
                        # But since we are flattening a pre-built tree, we might need to adjust get_pdf_outline to store level
                    }
                    chunks.append(chunk)
                    global_chunk_idx += 1
            
            # Recurse
            _traverse(node["children"], current_title, category)

    _traverse(outline_tree)
    return chunks

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

def parse_pdf_chunks(pdf_path):
    """
    Parses PDF and returns a flat list of chunks with structure metadata.
    This is the main entry point for the RAG system.
    """
    tree = parse_pdf(pdf_path)
    # We might want to extract the paper title from the first node or filename
    # For now, let's leave it empty or infer later
    return flatten_pdf_tree(tree)
