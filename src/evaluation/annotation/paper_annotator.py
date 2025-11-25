"""
Paper-level Annotator

负责为每篇论文生成:
- summary: 核心贡献总结
- keywords: 关键词
- research_area: 研究领域
"""

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from rag.retriever import RAG
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.schemas import PaperAnnotation


class PaperAnnotator:
    """Paper-level 标注器"""
    
    def __init__(
        self, 
        rag_client: "RAG",
        llm_client: "BaseChatModel",
        output_path: str = "saved_summaries/papers_index.jsonl"
    ):
        """
        Args:
            rag_client: RAG 客户端，用于获取论文数据
            llm_client: LLM 客户端，用于生成标注
            output_path: 标注结果输出路径
        """
        self.rag_client = rag_client
        self.llm = llm_client
        self.output_path = Path(output_path)
        
    def annotate_single(self, doc_id: str, title: str, abstract: str) -> PaperAnnotation:
        """
        标注单篇论文
        
        Args:
            doc_id: 论文 ID
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            PaperAnnotation 对象
        """
        # TODO: 实现
        # 1. 构建 prompt
        # 2. 调用 LLM
        # 3. 解析输出
        # 4. 返回 PaperAnnotation
        raise NotImplementedError
    
    def annotate_all(
        self, 
        conference: str = None, 
        year: int = None,
        batch_size: int = 10,
        resume: bool = True
    ) -> int:
        """
        批量标注所有论文
        
        Args:
            conference: 可选，只标注特定会议
            year: 可选，只标注特定年份
            batch_size: 批量大小（控制进度保存频率）
            resume: 是否从上次中断处继续
            
        Returns:
            标注的论文数量
        """
        # TODO: 实现
        # 1. 获取待标注论文列表
        # 2. 如果 resume=True，跳过已标注的
        # 3. 批量调用 annotate_single
        # 4. 定期保存到 output_path
        raise NotImplementedError
    
    def load_existing(self) -> dict[str, PaperAnnotation]:
        """加载已有的标注结果"""
        # TODO: 实现
        raise NotImplementedError
    
    def save(self, annotations: list[PaperAnnotation]) -> None:
        """保存标注结果到 JSONL 文件"""
        # TODO: 实现
        raise NotImplementedError
