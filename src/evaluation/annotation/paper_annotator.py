"""
Paper-level Annotator

负责为每篇论文生成:
- summary: 核心贡献总结
- keywords: 关键词
- research_area: 研究领域
"""

from typing import TYPE_CHECKING, Optional
from pathlib import Path
import json

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.schemas import PaperAnnotation
from evaluation.config import EvaluationConfig
from evaluation.data_preparation import PaperSource


class PaperAnnotator:
    """Paper-level 标注器"""
    
    def __init__(
        self, 
        llm_client: "BaseChatModel",
        config: Optional[EvaluationConfig] = None
    ):
        """
        Args:
            llm_client: LLM 客户端，用于生成标注
            config: 评估配置
        """
        self.llm = llm_client
        self.config = config or EvaluationConfig()
        self.config.ensure_dirs()
        
    def annotate_single(self, paper: PaperSource) -> PaperAnnotation:
        """
        标注单篇论文
        
        Args:
            paper: 论文元数据
            
        Returns:
            PaperAnnotation 对象
        """
        # TODO: 实现
        # 1. 构建 prompt (使用 title + abstract)
        # 2. 调用 LLM
        # 3. 解析 JSON 输出
        # 4. 返回 PaperAnnotation
        raise NotImplementedError
    
    def annotate_all(
        self, 
        papers: list[PaperSource],
        batch_size: int = 10,
        resume: bool = True
    ) -> list[PaperAnnotation]:
        """
        批量标注所有论文
        
        Args:
            papers: 论文列表
            batch_size: 批量大小（控制进度保存频率）
            resume: 是否从上次中断处继续
            
        Returns:
            PaperAnnotation 列表
        """
        # TODO: 实现
        # 1. 如果 resume=True，加载已有标注，跳过已处理的
        # 2. 批量调用 annotate_single
        # 3. 定期保存到 config.summaries_file
        raise NotImplementedError
    
    def load_existing(self) -> dict[str, PaperAnnotation]:
        """
        加载已有的标注结果
        
        Returns:
            {doc_id: PaperAnnotation}
        """
        # TODO: 实现
        raise NotImplementedError
    
    def save(self, annotations: list[PaperAnnotation]) -> None:
        """保存标注结果到 JSONL 文件"""
        # TODO: 实现
        raise NotImplementedError
    
    def get_stats(self, annotations: list[PaperAnnotation]) -> dict:
        """
        统计标注结果
        
        Returns:
            {
                "total": 100,
                "by_research_area": {"security": 30, "ML": 40, ...},
                "avg_keywords": 4.2
            }
        """
        # TODO: 实现
        raise NotImplementedError
