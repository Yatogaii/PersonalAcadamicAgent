"""
Evaluation Runner

负责执行评估并生成报告
"""

from typing import TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import uuid

if TYPE_CHECKING:
    from rag.retriever import RAG
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.schemas import (
    GroundTruth, QAPair, EvaluationReport,
    L1Result, L2Result, L3Result, Difficulty
)


class EvaluationRunner:
    """评估执行器"""
    
    def __init__(
        self,
        rag_client: "RAG",
        llm_client: "BaseChatModel" = None,
        ground_truth_path: str = "src/evaluation/ground_truth.json",
        report_dir: str = "src/evaluation/reports"
    ):
        """
        Args:
            rag_client: RAG 客户端
            llm_client: LLM 客户端（用于 L3 评估的答案质量判断）
            ground_truth_path: Ground Truth 文件路径
            report_dir: 报告输出目录
        """
        self.rag_client = rag_client
        self.llm = llm_client
        self.ground_truth_path = Path(ground_truth_path)
        self.report_dir = Path(report_dir)
    
    def run_all(self) -> EvaluationReport:
        """
        运行完整评估（L1 + L2 + L3）
        
        Returns:
            EvaluationReport
        """
        # TODO: 实现
        # 1. 加载 Ground Truth
        # 2. 运行 L1 评估
        # 3. 运行 L2 评估
        # 4. 运行 L3 评估
        # 5. 汇总报告
        raise NotImplementedError
    
    def run_l1_paper_discovery(self, qa_pairs: list[QAPair]) -> L1Result:
        """
        L1: Paper Discovery 评估
        
        测试 search_abstracts() 的效果
        只使用 Easy 和部分 Medium 问题
        """
        # TODO: 实现
        # 1. 筛选适合的 QA pairs
        # 2. 对每个 query 调用 search_abstracts
        # 3. 计算 Precision@K, Recall@K, MRR, Hit Rate
        raise NotImplementedError
    
    def run_l2_section_retrieval(self, qa_pairs: list[QAPair]) -> L2Result:
        """
        L2: Section Retrieval 评估
        
        测试 search_by_section() 的效果
        使用有 answer_source 标注的 Medium 问题
        """
        # TODO: 实现
        # 1. 筛选有 section 来源的 QA pairs
        # 2. 对每个 query 调用 search_by_section (指定 category)
        # 3. 计算 Method 和 Evaluation section 的检索准确率
        raise NotImplementedError
    
    def run_l3_end_to_end(self, qa_pairs: list[QAPair]) -> L3Result:
        """
        L3: End-to-End QA 评估
        
        测试完整 RAG 流程（检索 + 生成）
        使用所有难度的问题
        """
        # TODO: 实现
        # 1. 对每个 query 执行完整 RAG 流程
        # 2. 用 LLM 评估生成的答案与 reference_answer 的匹配度
        # 3. 计算各难度的准确率
        raise NotImplementedError
    
    def load_ground_truth(self) -> GroundTruth:
        """加载 Ground Truth"""
        # TODO: 实现
        raise NotImplementedError
    
    def save_report(self, report: EvaluationReport) -> Path:
        """保存评估报告"""
        # TODO: 实现
        raise NotImplementedError
    
    def print_report(self, report: EvaluationReport) -> None:
        """打印评估报告到控制台"""
        print("\n" + "=" * 70)
        print("RAG Evaluation Report")
        print("=" * 70)
        print(f"Run ID: {report.run_id}")
        print(f"Time: {report.run_at}")
        print(f"Total QA Pairs: {report.total_qa_pairs}")
        print("-" * 70)
        
        # L1 Results
        print("\n[L1] Paper Discovery:")
        l1 = report.l1_paper_discovery
        print(f"  Precision@5:  {l1.precision_at_5:.4f}")
        print(f"  Precision@10: {l1.precision_at_10:.4f}")
        print(f"  Recall@10:    {l1.recall_at_10:.4f}")
        print(f"  MRR:          {l1.mrr:.4f}")
        print(f"  Hit Rate:     {l1.hit_rate:.4f}")
        print(f"  Latency:      {l1.mean_latency_ms:.1f} ms")
        
        # L2 Results
        print("\n[L2] Section Retrieval:")
        l2 = report.l2_section_retrieval
        print(f"  Method Precision:     {l2.method_precision:.4f}")
        print(f"  Evaluation Precision: {l2.eval_precision:.4f}")
        print(f"  Overall Precision:    {l2.overall_precision:.4f}")
        
        # L3 Results
        print("\n[L3] End-to-End QA:")
        l3 = report.l3_end_to_end
        print(f"  Easy Accuracy:   {l3.easy_accuracy:.4f}")
        print(f"  Medium Accuracy: {l3.medium_accuracy:.4f}")
        print(f"  Hard Accuracy:   {l3.hard_accuracy:.4f}")
        print(f"  Overall:         {l3.overall_accuracy:.4f}")
        print(f"  Faithfulness:    {l3.faithfulness:.4f}")
        
        print("=" * 70 + "\n")
