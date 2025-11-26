"""
Chunk Processor

负责:
1. 解析 PDF
2. 分块 (paragraph / contextual)
3. 缓存分块结果
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, asdict, field

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.config import EvaluationConfig, ChunkStrategy
from evaluation.data_preparation.data_exporter import PaperSource


@dataclass
class ProcessedChunk:
    """处理后的 Chunk"""
    doc_id: str
    chunk_index: int
    
    # 原始内容
    original_text: str
    
    # 结构信息
    section_title: str = ""
    section_category: int = 0  # 0=Abstract, 1=Intro, 2=Method, etc.
    parent_section: str = ""
    page_number: int = 0
    
    # Contextual Chunking 生成的上下文 (可选)
    context_prefix: str = ""
    
    @property
    def text_for_embedding(self) -> str:
        """用于 embedding 的文本"""
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.original_text}"
        return self.original_text


@dataclass
class ChunkCache:
    """单篇论文的 Chunk 缓存"""
    doc_id: str
    title: str
    strategy: str  # "paragraph" or "contextual"
    chunks: list[ProcessedChunk] = field(default_factory=list)
    
    def to_jsonl_lines(self) -> list[str]:
        """转换为 JSONL 行"""
        lines = []
        for chunk in self.chunks:
            lines.append(json.dumps(asdict(chunk), ensure_ascii=False))
        return lines


class ChunkProcessor:
    """Chunk 处理器"""
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        llm_client: Optional["BaseChatModel"] = None
    ):
        """
        Args:
            config: 评估配置
            llm_client: LLM 客户端 (用于 Contextual Chunking)
        """
        self.config = config or EvaluationConfig()
        self.llm = llm_client
        self.config.ensure_dirs()
    
    # ============== 主入口 ==============
    
    def process_paper(
        self,
        paper: PaperSource,
        strategy: ChunkStrategy,
        force: bool = False
    ) -> ChunkCache:
        """
        处理单篇论文
        
        Args:
            paper: 论文元数据
            strategy: 分块策略
            force: 是否强制重新处理（忽略缓存）
            
        Returns:
            ChunkCache
        """
        # TODO: 实现
        # 1. 检查缓存是否存在
        # 2. 如果存在且 force=False，直接加载缓存
        # 3. 否则：
        #    a. 获取 PDF 路径
        #    b. 解析 PDF
        #    c. 按策略分块
        #    d. 保存缓存
        raise NotImplementedError
    
    def process_batch(
        self,
        papers: list[PaperSource],
        strategy: ChunkStrategy,
        show_progress: bool = True
    ) -> dict[str, ChunkCache]:
        """
        批量处理论文
        
        Args:
            papers: 论文列表
            strategy: 分块策略
            show_progress: 是否显示进度
            
        Returns:
            {doc_id: ChunkCache}
        """
        # TODO: 实现
        raise NotImplementedError
    
    # ============== 分块策略 ==============
    
    def _chunk_paragraph(self, doc_id: str, title: str, pdf_path: Path) -> list[ProcessedChunk]:
        """
        段落分块 (传统方式)
        
        使用现有的 pdf_parser 解析
        """
        # TODO: 实现
        # 1. 调用 pdf_parser.parse_pdf()
        # 2. 调用 pdf_parser.flatten_pdf_tree()
        # 3. 转换为 ProcessedChunk 列表
        raise NotImplementedError
    
    def _chunk_contextual(self, doc_id: str, title: str, pdf_path: Path) -> list[ProcessedChunk]:
        """
        Contextual Chunking
        
        先段落分块，再用 LLM 为每个 chunk 生成上下文
        """
        # TODO: 实现
        # 1. 先调用 _chunk_paragraph 获取基础分块
        # 2. 获取全文内容
        # 3. 对每个 chunk 调用 LLM 生成 context_prefix
        # 4. 返回带 context 的 chunks
        raise NotImplementedError
    
    def _generate_context(self, chunk_text: str, full_document: str, title: str) -> str:
        """
        用 LLM 为单个 chunk 生成上下文
        
        Args:
            chunk_text: 当前 chunk 的文本
            full_document: 全文内容
            title: 论文标题
            
        Returns:
            上下文前缀文本
        """
        # TODO: 实现
        # 调用 LLM，使用 CONTEXTUAL_CHUNK_PROMPT
        raise NotImplementedError
    
    # ============== 缓存管理 ==============
    
    def _get_cache_path(self, doc_id: str, strategy: ChunkStrategy) -> Path:
        """获取缓存文件路径"""
        return self.config.chunks_dir / strategy.value / f"{doc_id}.jsonl"
    
    def _cache_exists(self, doc_id: str, strategy: ChunkStrategy) -> bool:
        """检查缓存是否存在"""
        return self._get_cache_path(doc_id, strategy).exists()
    
    def _save_cache(self, cache: ChunkCache) -> None:
        """保存缓存到文件"""
        # TODO: 实现
        raise NotImplementedError
    
    def _load_cache(self, doc_id: str, strategy: ChunkStrategy) -> Optional[ChunkCache]:
        """从文件加载缓存"""
        # TODO: 实现
        raise NotImplementedError
    
    # ============== 统计 ==============
    
    def get_processing_stats(
        self,
        papers: list[PaperSource],
        strategy: ChunkStrategy
    ) -> dict:
        """
        统计处理状态
        
        Returns:
            {
                "total": 100,
                "cached": 80,
                "pending": 20,
                "total_chunks": 3200
            }
        """
        # TODO: 实现
        raise NotImplementedError
