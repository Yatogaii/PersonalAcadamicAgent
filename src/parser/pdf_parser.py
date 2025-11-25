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
    
    Also attempts to find Abstract if not in TOC.
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
    
    has_abstract = False
    first_section_page = 0
    
    for item in toc:
        # Handle variable length tuple
        if len(item) == 3:
            lvl, title, page = item
        elif len(item) >= 4:
            lvl, title, page = item[0], item[1], item[2]
        else:
            continue
        
        # Check if abstract exists in TOC
        if 'abstract' in title.lower():
            has_abstract = True
            
        # page is 1-based in get_toc output usually, let's verify
        # PyMuPDF docs say page number is 1-based. We convert to 0-based.
        # If page is -1, it means no destination
        if page > 0:
            page_idx = page - 1
        else:
            page_idx = 0 # Fallback
        
        # Track the first section page
        if first_section_page == 0 and lvl == 1:
            first_section_page = page_idx
        
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
    
    # If no Abstract in TOC, try to find it on the first page(s)
    if not has_abstract and root:
        abstract_node = _find_abstract(doc, first_section_page)
        if abstract_node:
            root.insert(0, abstract_node)
            
    return root


def _find_abstract(doc, first_section_page: int) -> dict | None:
    """
    Try to find Abstract section in pages before the first TOC section.
    Many papers have Abstract on page 1 but it's not in the TOC.
    """
    # Search in first few pages (before the first section)
    search_pages = min(first_section_page + 1, 3)  # At most first 3 pages
    
    for page_idx in range(search_pages):
        page = doc[page_idx]
        text = page.get_text()
        
        # Look for "Abstract" as a section header
        # Common patterns: "Abstract", "ABSTRACT", "Abstract."
        match = re.search(r'\n\s*(Abstract|ABSTRACT)\s*\n', text)
        if match:
            return {
                "title": "Abstract",
                "page": page_idx,
                "children": []
            }
        
        # Also try to find it at the beginning (some PDFs have it without explicit header)
        # Pattern: Abstract followed by the actual abstract text
        match = re.search(r'(^|\n)\s*(Abstract|ABSTRACT)[:\.]?\s*\n', text)
        if match:
            return {
                "title": "Abstract",
                "page": page_idx,
                "children": []
            }
    
    return None

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
        doc.close()
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
    
    # Pre-extract all page texts for efficiency and better boundary handling
    page_texts = []
    for p_idx in range(total_pages):
        page = doc[p_idx]
        html_content: str = page.get_text("html")  # type: ignore
        soup = BeautifulSoup(html_content, "html.parser")
        page_text = soup.get_text(separator="\n")
        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        page_texts.append("\n".join(lines))
    
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
        
        # Collect content across pages
        full_content = []
        
        for p_idx in range(start_page_idx, end_page_idx + 1):
            if p_idx >= total_pages:
                break
            
            cleaned_text = page_texts[p_idx]
            
            # Determine start position in this page's text
            start_idx = 0
            if p_idx == start_page_idx:
                # Find current section title - use the improved function
                title_pos = _find_section_title_position(cleaned_text, current_title)
                if title_pos >= 0:
                    # Find the end of the title line, starting from title_pos
                    # Use pattern that matches the title from the correct position
                    search_text = cleaned_text[title_pos:]
                    pattern = _make_title_pattern(current_title, require_section_number=False)
                    match = re.search(pattern, search_text, re.IGNORECASE)
                    if match:
                        start_idx = title_pos + match.end()
                        # Skip any leading whitespace/newlines after title
                        while start_idx < len(cleaned_text) and cleaned_text[start_idx] in '\n\r\t ':
                            start_idx += 1
            
            # Determine end position in this page's text
            end_idx = len(cleaned_text)
            
            # If this is the last page for this section AND there's a next section
            if p_idx == end_page_idx and next_title:
                # Use improved position finding that prefers section-numbered titles
                pos = _find_section_title_position(cleaned_text, next_title, start_idx)
                if pos >= 0:
                    end_idx = pos
            
            # Also check for ANY subsequent section title on this page
            # (handles case where multiple sections are on same page)
            if p_idx == start_page_idx:
                # Look for other section titles that come after current one
                for j in range(i + 1, len(flat_nodes)):
                    other_node = flat_nodes[j]
                    if other_node["page"] == p_idx:
                        # Use improved position finding
                        pos = _find_section_title_position(cleaned_text, other_node["title"], start_idx)
                        if pos >= 0 and pos < end_idx:
                            end_idx = pos
                    elif other_node["page"] > p_idx:
                        break
            
            if start_idx < end_idx:
                chunk = cleaned_text[start_idx:end_idx].strip()
                if chunk:
                    full_content.append(chunk)
        
        # Combine all content and clean
        if full_content:
            combined = "\n".join(full_content)
            cleaned_chunk = clean_text(combined)
            if cleaned_chunk:
                node["content"].append(cleaned_chunk)

    doc.close()
    return outline_tree


def _make_title_pattern(title: str, require_section_number: bool = False) -> str:
    """
    Create a regex pattern for matching section titles.
    Handles variations in spacing and numbering.
    
    Args:
        title: The section title to match
        require_section_number: If True, requires a section number before the title
                               (useful to avoid matching words in body text)
    """
    # Escape special regex characters
    escaped = re.escape(title)
    # Allow flexible whitespace
    pattern = escaped.replace(r'\ ', r'\s+')
    
    if require_section_number:
        # Require a section number like "3.1" or "3.2.1" before the title
        # This prevents matching words like "attention" in body text
        pattern = r'(?:^|\n)\s*(\d+\.(?:\d+\.?)*)\s*' + pattern
    else:
        # Allow optional section numbers
        pattern = r'(?:^|\n)\s*(?:\d+\.?\d*\.?\s*)?' + pattern
    
    return pattern


def _find_section_title_position(text: str, title: str, start_from: int = 0) -> int:
    """
    Find the position of a section title in text.
    First tries to match with section number requirement, falls back to looser match.
    
    Returns:
        Position of the title start, or -1 if not found
    """
    search_text = text[start_from:]
    
    # First try: require section number (e.g., "3.2 Attention")
    pattern = _make_title_pattern(title, require_section_number=True)
    match = re.search(pattern, search_text, re.IGNORECASE)
    if match:
        return start_from + match.start()
    
    # Second try: match title at line start without section number
    # But ensure it's at the beginning of a line, not mid-sentence
    # Pattern: newline + optional spaces + title (not preceded by lowercase letter)
    escaped = re.escape(title).replace(r'\ ', r'\s+')
    pattern = r'(?:^|\n)\s*' + escaped + r'\s*(?:\n|$)'
    match = re.search(pattern, search_text, re.IGNORECASE)
    if match:
        return start_from + match.start()
    
    return -1


def parse_pdf_chunks(pdf_path):
    """
    Parses PDF and returns a flat list of chunks with structure metadata.
    This is the main entry point for the RAG system.
    """
    tree = parse_pdf(pdf_path)
    # We might want to extract the paper title from the first node or filename
    # For now, let's leave it empty or infer later
    return flatten_pdf_tree(tree)
