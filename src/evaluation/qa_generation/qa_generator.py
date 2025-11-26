"""
QA Generator

负责从标注结果生成测试用的 QA pairs
"""

from typing import TYPE_CHECKING, Optional
from pathlib import Path
import json

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.schemas import QAPair, GroundTruth, Difficulty, AnswerSource, PaperAnnotation
from evaluation.config import EvaluationConfig


class QAGenerator:
    """QA 生成器"""
    
    def __init__(
        self,
        llm_client: "BaseChatModel",
        config: Optional[EvaluationConfig] = None
    ):
        """
        Args:
            llm_client: LLM 客户端
            config: 评估配置
        """
        self.llm = llm_client
        self.config = config or EvaluationConfig()
        self.config.ensure_dirs()
    
    def generate(
        self,
        annotations: list[PaperAnnotation],
        num_questions: Optional[int] = None,
        difficulty_distribution: Optional[dict] = None
    ) -> GroundTruth:
        """
        生成 QA pairs
        
        Args:
            annotations: 论文标注列表
            num_questions: 生成的问题数量，默认使用 config
            difficulty_distribution: 难度分布，默认使用 config
            
        Returns:
            GroundTruth 对象
        """
        num_questions = num_questions or self.config.num_qa_pairs
        difficulty_distribution = difficulty_distribution or self.config.difficulty_distribution
        
        # TODO: 实现
        # 1. 按 research_area 分组
        # 2. 计算各难度的问题数量
        # 3. 生成 Easy / Medium / Hard 问题
        # 4. 汇总并保存
        raise NotImplementedError
    
    def generate_easy_questions(
        self,
        annotations: list[PaperAnnotation],
        count: int
    ) -> list[QAPair]:
        """
        生成 Easy 级别问题
        
        特点: 关键词直接匹配，单论文检索
        示例: "哪篇论文研究了 XXX？", "XXX 方法是哪篇论文提出的？"
        """
        # TODO: 实现
        raise NotImplementedError
    
    def generate_medium_questions(
        self,
        annotations: list[PaperAnnotation],
        count: int
    ) -> list[QAPair]:
        """
        生成 Medium 级别问题
        
        特点: 需要语义理解，可能需要 section-level 检索
        示例: "XXX 论文的核心方法是什么？", "XXX 的实验结果如何？"
        """
        # TODO: 实现
        raise NotImplementedError
    
    def generate_hard_questions(
        self,
        annotations: list[PaperAnnotation],
        count: int
    ) -> list[QAPair]:
        """
        生成 Hard 级别问题
        
        特点: 跨论文综合，需要多步检索
        示例: "比较 A 和 B 方法的优缺点", "在 XXX 领域有哪些主要方法？"
        """
        # TODO: 实现
        raise NotImplementedError
    
    def save(self, ground_truth: GroundTruth) -> Path:
        """
        保存 Ground Truth 到 JSON 文件
        
        Returns:
            保存的文件路径
        """
        # TODO: 实现
        raise NotImplementedError
    
    def load(self) -> Optional[GroundTruth]:
        """
        从文件加载 Ground Truth
        
        Returns:
            GroundTruth 对象，如果文件不存在返回 None
        """
        # TODO: 实现
        raise NotImplementedError
