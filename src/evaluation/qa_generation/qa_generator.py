"""
QA Generator

负责从标注结果生成测试用的 QA pairs
"""

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.schemas import QAPair, GroundTruth, Difficulty, AnswerSource


class QAGenerator:
    """QA 生成器"""
    
    def __init__(
        self,
        llm_client: "BaseChatModel",
        index_path: str = "saved_summaries/papers_index.jsonl",
        output_path: str = "src/evaluation/ground_truth.json"
    ):
        """
        Args:
            llm_client: LLM 客户端
            index_path: 标注文件路径
            output_path: Ground Truth 输出路径
        """
        self.llm = llm_client
        self.index_path = Path(index_path)
        self.output_path = Path(output_path)
    
    def generate(
        self,
        num_questions: int = 100,
        difficulty_distribution: dict = None,
        source_distribution: dict = None
    ) -> GroundTruth:
        """
        生成 QA pairs
        
        Args:
            num_questions: 生成的问题数量
            difficulty_distribution: 难度分布，如 {"easy": 0.3, "medium": 0.5, "hard": 0.2}
            source_distribution: 来源分布，如 {"abstract": 0.3, "method": 0.4, "evaluation": 0.3}
            
        Returns:
            GroundTruth 对象
        """
        # TODO: 实现
        # 1. 加载所有标注
        # 2. 按 research_area 分组
        # 3. 为每组生成 QA
        # 4. 汇总并保存
        raise NotImplementedError
    
    def generate_easy_questions(
        self,
        annotations: list,
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
        annotations: list,
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
        annotations: list,
        count: int
    ) -> list[QAPair]:
        """
        生成 Hard 级别问题
        
        特点: 跨论文综合，需要多步检索
        示例: "比较 A 和 B 方法的优缺点", "在 XXX 领域有哪些主要方法？"
        """
        # TODO: 实现
        raise NotImplementedError
    
    def load_annotations(self) -> list:
        """加载标注结果"""
        # TODO: 实现
        raise NotImplementedError
    
    def save(self, ground_truth: GroundTruth) -> None:
        """保存 Ground Truth 到 JSON 文件"""
        # TODO: 实现
        raise NotImplementedError
