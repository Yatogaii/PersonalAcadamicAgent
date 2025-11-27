"""
Evaluation Runner

负责执行评估并生成报告
"""

from typing import TYPE_CHECKING, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import uuid
import json
import time

from logging_config import logger

if TYPE_CHECKING:
    from rag.retriever import RAG
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.schemas import (
    GroundTruth, QAPair, EvaluationReport,
    L1Result, L2Result, L3Result, Difficulty, AnswerSource
)
from evaluation.config import EvaluationConfig


class EvaluationRunner:
    """评估执行器"""
    
    def __init__(
        self,
        rag_client: "RAG",
        llm_client: "BaseChatModel" = None,
        config: Optional[EvaluationConfig] = None,
    ):
        """
        Args:
            rag_client: RAG 客户端（MilvusProvider）
            llm_client: LLM 客户端（用于 L3 评估的答案质量判断）
            config: 评估配置
        """
        self.rag = rag_client
        self.llm = llm_client
        self.config = config or EvaluationConfig()
        
        # 确保报告目录存在
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def run_all(self, ground_truth: Optional[GroundTruth] = None) -> EvaluationReport:
        """
        运行完整评估（L1 + L2）
        
        Args:
            ground_truth: Ground Truth，如果不提供则从文件加载
            
        Returns:
            EvaluationReport
        """
        # 1. 加载 Ground Truth
        if ground_truth is None:
            ground_truth = self.load_ground_truth()
        
        qa_pairs = ground_truth.qa_pairs
        logger.info(f"Running evaluation with {len(qa_pairs)} QA pairs")
        
        # 2. 运行 L1 评估
        l1_result = self.run_l1_paper_discovery(qa_pairs)
        
        # 3. 运行 L2 评估
        l2_result = self.run_l2_chunk_retrieval(qa_pairs)
        
        # 4. 构建报告
        report = EvaluationReport(
            run_id=str(uuid.uuid4())[:8],
            run_at=datetime.now().isoformat(),
            total_qa_pairs=len(qa_pairs),
            rag_provider=self.rag.__class__.__name__,
            embedding_model=getattr(self.rag, 'embedding_model', 'unknown'),
            l1_paper_discovery=l1_result,
            l2_section_retrieval=l2_result,
            l3_end_to_end=L3Result(),  # 暂时空着
        )
        
        return report
    
    def run_l1_paper_discovery(self, qa_pairs: list[QAPair]) -> L1Result:
        """
        L1: Paper Discovery 评估
        
        测试 search_abstracts() 能否找到正确的论文
        """
        # 只用有 expected_doc_ids 的问题
        valid_pairs = [q for q in qa_pairs if q.expected_doc_ids]
        
        if not valid_pairs:
            logger.warning("No valid QA pairs for L1 evaluation")
            return L1Result()
        
        logger.info(f"L1 evaluation with {len(valid_pairs)} QA pairs")
        
        # 指标收集
        precisions_at_5 = []
        precisions_at_10 = []
        recalls_at_5 = []
        recalls_at_10 = []
        reciprocal_ranks = []
        hits = []
        latencies = []
        
        for qa in valid_pairs:
            expected_docs = set(qa.expected_doc_ids)
            
            # 计时
            start = time.time()
            
            # 调用检索
            results = self.rag.search_abstracts(qa.question, k=10)
            
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            # 提取检索到的 doc_ids
            retrieved_docs = [r.get("doc_id", "") for r in results]
            retrieved_set_5 = set(retrieved_docs[:5])
            retrieved_set_10 = set(retrieved_docs[:10])
            
            # Precision@K
            p5 = len(expected_docs & retrieved_set_5) / 5 if retrieved_docs else 0
            p10 = len(expected_docs & retrieved_set_10) / 10 if retrieved_docs else 0
            precisions_at_5.append(p5)
            precisions_at_10.append(p10)
            
            # Recall@K
            r5 = len(expected_docs & retrieved_set_5) / len(expected_docs)
            r10 = len(expected_docs & retrieved_set_10) / len(expected_docs)
            recalls_at_5.append(r5)
            recalls_at_10.append(r10)
            
            # MRR (Mean Reciprocal Rank)
            rr = 0.0
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in expected_docs:
                    rr = 1.0 / (i + 1)
                    break
            reciprocal_ranks.append(rr)
            
            # Hit Rate (至少找到一个)
            hit = 1 if (expected_docs & retrieved_set_10) else 0
            hits.append(hit)
        
        # 计算平均值
        n = len(valid_pairs)
        result = L1Result(
            precision_at_5=sum(precisions_at_5) / n,
            precision_at_10=sum(precisions_at_10) / n,
            recall_at_5=sum(recalls_at_5) / n,
            recall_at_10=sum(recalls_at_10) / n,
            mrr=sum(reciprocal_ranks) / n,
            hit_rate=sum(hits) / n,
            mean_latency_ms=sum(latencies) / n,
        )
        
        logger.info(f"L1 results: P@5={result.precision_at_5:.3f}, "
                   f"R@10={result.recall_at_10:.3f}, MRR={result.mrr:.3f}")
        
        return result
    
    def run_l2_chunk_retrieval(self, qa_pairs: list[QAPair]) -> L2Result:
        """
        L2: Chunk/Section Retrieval 评估
        
        测试 search_by_section() 能否找到正确的 chunks
        """
        # 只用有 expected_chunk_ids 的问题
        valid_pairs = [q for q in qa_pairs if q.expected_chunk_ids]
        
        if not valid_pairs:
            logger.warning("No valid QA pairs with expected_chunk_ids for L2 evaluation")
            return L2Result()
        
        logger.info(f"L2 evaluation with {len(valid_pairs)} QA pairs")
        
        # 分类收集
        method_precisions = []
        method_recalls = []
        eval_precisions = []
        eval_recalls = []
        overall_precisions = []
        overall_recalls = []
        
        for qa in valid_pairs:
            expected_chunks = set(qa.expected_chunk_ids)
            expected_doc = qa.expected_doc_ids[0] if qa.expected_doc_ids else None
            
            # 根据 answer_source 确定 section_category
            section_cat = self._source_to_category(qa.answer_source)
            
            # 调用检索
            results = self.rag.search_by_section(
                query=qa.question,
                doc_id=expected_doc,
                section_category=section_cat,
                k=10
            )
            
            # 提取检索到的 chunk_ids
            retrieved_chunks = set(r.get("chunk_id", -1) for r in results)
            retrieved_chunks.discard(-1)  # 移除 paper-level 结果
            
            # 计算 Precision 和 Recall
            if retrieved_chunks:
                precision = len(expected_chunks & retrieved_chunks) / len(retrieved_chunks)
            else:
                precision = 0.0
            
            recall = len(expected_chunks & retrieved_chunks) / len(expected_chunks)
            
            overall_precisions.append(precision)
            overall_recalls.append(recall)
            
            # 按 section 类型分类统计
            if qa.answer_source == AnswerSource.METHOD:
                method_precisions.append(precision)
                method_recalls.append(recall)
            elif qa.answer_source == AnswerSource.EVALUATION:
                eval_precisions.append(precision)
                eval_recalls.append(recall)
        
        # 计算平均值
        result = L2Result(
            overall_precision=sum(overall_precisions) / len(overall_precisions) if overall_precisions else 0,
            overall_recall=sum(overall_recalls) / len(overall_recalls) if overall_recalls else 0,
            method_precision=sum(method_precisions) / len(method_precisions) if method_precisions else 0,
            method_recall=sum(method_recalls) / len(method_recalls) if method_recalls else 0,
            eval_precision=sum(eval_precisions) / len(eval_precisions) if eval_precisions else 0,
            eval_recall=sum(eval_recalls) / len(eval_recalls) if eval_recalls else 0,
        )
        
        logger.info(f"L2 results: Overall P={result.overall_precision:.3f}, "
                   f"R={result.overall_recall:.3f}")
        
        return result
    
    def _source_to_category(self, source: AnswerSource) -> Optional[int]:
        """将 AnswerSource 映射到 section_category"""
        mapping = {
            AnswerSource.ABSTRACT: 0,
            AnswerSource.INTRODUCTION: 1,
            AnswerSource.METHOD: 2,
            AnswerSource.EVALUATION: 4,
            AnswerSource.MULTIPLE: None,  # 不限制
        }
        return mapping.get(source)
    
    def load_ground_truth(self) -> GroundTruth:
        """加载 Ground Truth"""
        path = self.config.ground_truth_file
        
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        qa_pairs = []
        for q in data.get("qa_pairs", []):
            qa_pairs.append(QAPair(
                id=q["id"],
                question=q["question"],
                difficulty=Difficulty(q["difficulty"]),
                expected_doc_ids=q["expected_doc_ids"],
                expected_chunk_ids=q.get("expected_chunk_ids"),
                answer_source=AnswerSource(q["answer_source"]),
                reference_answer=q.get("reference_answer", ""),
                is_multi_paper=q.get("is_multi_paper", False),
            ))
        
        ground_truth = GroundTruth(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            total_papers=data.get("total_papers", 0),
            qa_pairs=qa_pairs,
            difficulty_distribution=data.get("difficulty_distribution", {}),
            source_distribution=data.get("source_distribution", {}),
        )
        
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {path}")
        return ground_truth
    
    def save_report(self, report: EvaluationReport) -> Path:
        """保存评估报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{report.run_id}_{timestamp}.json"
        output_path = self.config.reports_dir / filename
        
        # 转换为可序列化格式
        data = {
            "run_id": report.run_id,
            "run_at": report.run_at,
            "total_qa_pairs": report.total_qa_pairs,
            "rag_provider": report.rag_provider,
            "embedding_model": report.embedding_model,
            "l1_paper_discovery": asdict(report.l1_paper_discovery),
            "l2_section_retrieval": asdict(report.l2_section_retrieval),
            "l3_end_to_end": asdict(report.l3_end_to_end),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved report to {output_path}")
        return output_path
    
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
