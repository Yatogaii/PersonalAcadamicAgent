"""
Contextual Chunking 的 Prompt 模板
"""

CONTEXTUAL_CHUNK_PROMPT = """
You are an assistant that helps situate a chunk of text within the context of a larger document.

<document_title>
{title}
</document_title>

<document>
{full_document}
</document>

Here is the chunk we want to situate:

<chunk>
{chunk_text}
</chunk>

Please provide a short, succinct context (1-2 sentences) that helps situate this chunk within the overall document. 
The context should:
1. Explain where this content appears in the document structure
2. Briefly mention what topic or concept is being discussed
3. NOT repeat the content of the chunk itself

Respond with ONLY the context text, no explanations or formatting.
"""


def build_contextual_chunk_prompt(
    title: str,
    full_document: str,
    chunk_text: str,
    max_doc_length: int = 8000
) -> str:
    """
    构建 Contextual Chunking 的 prompt
    
    Args:
        title: 论文标题
        full_document: 全文内容
        chunk_text: 当前 chunk 文本
        max_doc_length: 全文最大长度（截断）
        
    Returns:
        完整的 prompt
    """
    # 截断全文（保留开头，因为结构信息在前面）
    if len(full_document) > max_doc_length:
        full_document = full_document[:max_doc_length] + "\n... [document truncated]"
    
    return CONTEXTUAL_CHUNK_PROMPT.format(
        title=title,
        full_document=full_document,
        chunk_text=chunk_text
    )
