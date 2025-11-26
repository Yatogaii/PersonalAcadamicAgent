from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from settings import settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


@dataclass
class ChunkResult:
    """
    分块结果，包含原始文本、上下文前缀和结构化元数据
    
    用于统一表示来自 pdf_parser 的结构化 chunks，并支持 contextual chunking
    """
    # 核心内容
    chunk_text: str
    contextual_prefix: str = ""
    
    # 索引
    chunk_index: int = 0
    
    # 结构化元数据（来自 pdf_parser）
    section_title: str = ""
    section_category: int = 0  # 0=Abstract, 1=Intro, 2=Method, etc.
    parent_section: str = ""
    page_number: int = 1
    
    # 调试信息
    debug_info: dict = field(default_factory=dict)
    
    @property
    def text_for_embedding(self) -> str:
        """用于 embedding 的文本：如果有 contextual_prefix 则拼接"""
        if self.contextual_prefix:
            return f"{self.contextual_prefix}\n\n{self.chunk_text}"
        return self.chunk_text
    
    def to_dict(self) -> dict:
        """转换为字典格式，兼容 milvus.insert_paper_chunks"""
        return {
            # ChunkResult 格式字段
            "chunk_text": self.chunk_text,
            "contextual_prefix": self.contextual_prefix,
            "text_for_embedding": self.text_for_embedding,
            # 兼容传统 pdf_parser 格式
            "text": self.chunk_text,
            "chunk_index": self.chunk_index,
            "section_title": self.section_title,
            "section_category": self.section_category,
            "parent_section": self.parent_section,
            "page_number": self.page_number,
        }
    
    @classmethod
    def from_pdf_parser_chunk(cls, chunk: dict, contextual_prefix: str = "") -> "ChunkResult":
        """从 pdf_parser.flatten_pdf_tree 的输出创建 ChunkResult"""
        return cls(
            chunk_text=chunk["text"],
            contextual_prefix=contextual_prefix,
            chunk_index=chunk["chunk_index"],
            section_title=chunk.get("section_title", ""),
            section_category=chunk.get("section_category", 0),
            parent_section=chunk.get("parent_section", ""),
            page_number=chunk.get("page_number", 1),
        )


class Chunker:
    """
    Chunker 作为 pdf_parser 的后处理器
    
    流程:
    1. pdf_parser.parse_pdf() → 结构化树
    2. pdf_parser.flatten_pdf_tree() → 基础 chunks (list[dict])
    3. Chunker.process_chunks() → list[ChunkResult] (带 contextual_prefix)
    """
    
    def __init__(self, llm_client: Optional["BaseChatModel"] = None):
        self.strategy = settings.chunk_strategy
        self.llm_client = llm_client

    def process_chunks(
        self, 
        chunks: list[dict], 
        full_text: str = "",
        title: str = ""
    ) -> list[ChunkResult]:
        """
        处理来自 pdf_parser.flatten_pdf_tree 的 chunks
        
        Args:
            chunks: pdf_parser.flatten_pdf_tree() 返回的 chunks 列表
            full_text: 文档全文（用于 contextual chunking）
            title: 文档标题
            
        Returns:
            list[ChunkResult]: 处理后的分块结果
        """
        if self.strategy == "paragraph":
            # 直接转换，保留结构信息
            return self._process_paragraph(chunks)
        elif self.strategy == "contextual":
            # 为每个 chunk 生成 contextual prefix
            return self._process_contextual(chunks, full_text, title)
        else:
            # 其他策略暂时按 paragraph 处理
            return self._process_paragraph(chunks)
    
    def _process_paragraph(self, chunks: list[dict]) -> list[ChunkResult]:
        """段落策略：直接转换，保留原有结构信息"""
        return [
            ChunkResult.from_pdf_parser_chunk(chunk)
            for chunk in chunks
        ]
    
    def _process_contextual(
        self, 
        chunks: list[dict], 
        full_text: str,
        title: str
    ) -> list[ChunkResult]:
        """
        Contextual Chunking: 为每个 chunk 生成上下文前缀
        """
        if not self.llm_client:
            # 没有 LLM，回退到 paragraph 策略
            return self._process_paragraph(chunks)
        
        # 如果没有提供全文，从 chunks 拼接
        if not full_text:
            full_text = self._reconstruct_full_text(chunks)
        
        result_chunks: list[ChunkResult] = []
        
        for chunk in chunks:
            chunk_text = chunk["text"]
            section_title = chunk.get("section_title", "")
            
            try:
                # 生成 contextual prefix
                prefix = self._generate_context_prefix(
                    title=title,
                    full_document=full_text,
                    chunk_text=chunk_text,
                    section_title=section_title
                )
                result_chunks.append(ChunkResult.from_pdf_parser_chunk(
                    chunk, 
                    contextual_prefix=prefix
                ))
            except Exception as e:
                # 生成失败时，使用空前缀
                result = ChunkResult.from_pdf_parser_chunk(chunk, contextual_prefix="")
                result.debug_info = {"error": str(e)}
                result_chunks.append(result)
        
        return result_chunks
    
    def _reconstruct_full_text(self, chunks: list[dict]) -> str:
        """从 chunks 重建全文"""
        sections: dict[str, list[str]] = {}
        for chunk in chunks:
            section = chunk.get("section_title", "Unknown")
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk["text"])
        
        parts = []
        for section_title, texts in sections.items():
            parts.append(f"## {section_title}\n")
            parts.extend(texts)
            parts.append("")
        
        return "\n\n".join(parts)
    
    def _generate_context_prefix(
        self,
        title: str,
        full_document: str,
        chunk_text: str,
        section_title: str = "",
        max_doc_length: int = 8000
    ) -> str:
        """
        使用 LLM 生成 chunk 的上下文前缀
        """
        if not self.llm_client:
            return ""
        
        # 截断全文
        if len(full_document) > max_doc_length:
            full_document = full_document[:max_doc_length] + "\n... [document truncated]"
        
        section_hint = f"\nThis chunk is from the section: {section_title}" if section_title else ""
        
        prompt = f"""You are an assistant that helps situate a chunk of text within the context of a larger document.

<document_title>
{title}
</document_title>

<document>
{full_document}
</document>

Here is the chunk we want to situate:{section_hint}

<chunk>
{chunk_text}
</chunk>

Please provide a short, succinct context (1-2 sentences) that helps situate this chunk within the overall document. 
The context should:
1. Explain where this content appears in the document structure
2. Briefly mention what topic or concept is being discussed
3. NOT repeat the content of the chunk itself

Respond with ONLY the context text, no explanations or formatting."""

        response = self.llm_client.invoke(prompt)
        # 处理 LangChain 的响应格式
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                if isinstance(first_item, str):
                    return first_item.strip()
                elif isinstance(first_item, dict) and 'text' in first_item:
                    return str(first_item['text']).strip()
        return str(response).strip()

    # ============== 兼容旧 API（直接从文本分块） ==============
    
    def chunk(self, text: str, title: str = "") -> list[ChunkResult]:
        '''
        直接从文本分块（不经过 pdf_parser）
        
        注意：这种方式不会有结构化元数据（section_title 等）
        
        Args:
            text: 要分块的文本
            title: 文档标题（用于 contextual chunking）
            
        Returns:
            list[ChunkResult]: 分块结果列表
        '''
        if self.strategy == "paragraph":
            return self._chunk_by_paragraph(text)
        elif self.strategy == "fixed_size":
            return self._chunk_by_fixed_size(text, size=settings.chunk_fixed_size)
        elif self.strategy == "sentence":
            return self._chunk_by_sentence(text)
        elif self.strategy == "contextual":
            return self._chunk_by_contextual(text, title)
        else:
            raise ValueError(f"Unsupported chunk strategy: {self.strategy}")

    def _chunk_by_paragraph(self, text: str) -> list[ChunkResult]:
        paragraphs = [para.strip() for para in text.split('\n\n') if para.strip()]
        return [
            ChunkResult(chunk_text=para, chunk_index=idx)
            for idx, para in enumerate(paragraphs)
        ]

    def _chunk_by_fixed_size(self, text: str, size: int) -> list[ChunkResult]:
        chunks = [text[i:i+size] for i in range(0, len(text), size)]
        return [
            ChunkResult(chunk_text=chunk, chunk_index=idx)
            for idx, chunk in enumerate(chunks)
        ]

    def _chunk_by_sentence(self, text: str) -> list[ChunkResult]:
        paras = [para.strip() for para in text.split('\n\n') if para.strip()]
        sentences = [
            sentence.strip() 
            for para in paras 
            for sentence in para.split('.') 
            if sentence.strip()
        ]
        return [
            ChunkResult(chunk_text=sentence, chunk_index=idx)
            for idx, sentence in enumerate(sentences)
        ]
        
    def _chunk_by_contextual(self, text: str, title: str = "") -> list[ChunkResult]:
        """
        Contextual Chunking: 先按段落分块，再用 LLM 为每个 chunk 生成上下文前缀
        """
        base_chunks = self._chunk_by_paragraph(text)
        
        if not self.llm_client:
            return base_chunks
        
        result_chunks: list[ChunkResult] = []
        
        for idx, chunk in enumerate(base_chunks):
            try:
                prefix = self._generate_context_prefix(
                    title=title,
                    full_document=text,
                    chunk_text=chunk.chunk_text
                )
                result_chunks.append(ChunkResult(
                    chunk_text=chunk.chunk_text,
                    contextual_prefix=prefix,
                    chunk_index=idx,
                ))
            except Exception as e:
                result_chunks.append(ChunkResult(
                    chunk_text=chunk.chunk_text,
                    contextual_prefix="",
                    chunk_index=idx,
                    debug_info={"error": str(e)}
                ))
        
        return result_chunks