"""
Section-level Annotator

负责为已加载 PDF 的论文生成:
- method_summary: 方法总结
- method_keywords: 方法关键词
- eval_summary: 评估总结  
- eval_keywords: 评估关键词
"""

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from rag.retriever import RAG
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.schemas import PaperAnnotation


class SectionAnnotator:
    """Section-level 标注器"""
    
    # Section category 映射
    SECTION_CATEGORY_METHOD = 2      # Method/Design
    SECTION_CATEGORY_EVALUATION = 3  # Evaluation/Experiment
    
    def __init__(
        self,
        rag_client: "RAG",
        llm_client: "BaseChatModel",
        index_path: str = "saved_summaries/papers_index.jsonl"
    ):
        """
        Args:
            rag_client: RAG 客户端
            llm_client: LLM 客户端
            index_path: Paper 标注文件路径（读取并更新）
        """
        self.rag_client = rag_client
        self.llm = llm_client
        self.index_path = Path(index_path)
    
    def annotate_section(
        self, 
        doc_id: str, 
        section_category: int
    ) -> tuple[str, list[str]]:
        """
        标注单个 section
        
        Args:
            doc_id: 论文 ID
            section_category: section 类型 (2=Method, 3=Evaluation)
            
        Returns:
            (summary, keywords) 元组
        """
        # TODO: 实现
        # 1. 从 RAG 获取该 section 的所有 chunks
        # 2. 拼接 chunks 文本
        # 3. 调用 LLM 生成 summary 和 keywords
        raise NotImplementedError
    
    def annotate_paper_sections(self, doc_id: str) -> dict:
        """
        标注单篇论文的 Method 和 Evaluation sections
        
        Returns:
            {
                "method_summary": str,
                "method_keywords": list,
                "eval_summary": str,
                "eval_keywords": list
            }
        """
        # TODO: 实现
        raise NotImplementedError
    
    def annotate_loaded_papers(self, batch_size: int = 10) -> int:
        """
        标注所有已加载 PDF 的论文
        
        只处理 has_pdf_loaded=True 且尚未有 section 标注的论文
        
        Returns:
            标注的论文数量
        """
        # TODO: 实现
        # 1. 读取 papers_index.jsonl
        # 2. 筛选需要标注的论文
        # 3. 批量标注
        # 4. 更新并保存
        raise NotImplementedError
    
    def get_section_chunks(self, doc_id: str, section_category: int) -> list[str]:
        """从 RAG 获取指定 section 的所有 chunks 文本"""
        # TODO: 实现
        raise NotImplementedError
