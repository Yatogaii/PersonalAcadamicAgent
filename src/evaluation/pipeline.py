"""
Evaluation Pipeline - 完整的评估流水线

整合从数据准备到评估的完整流程
"""

from typing import TYPE_CHECKING, Optional
from pathlib import Path
from datetime import datetime
import json

if TYPE_CHECKING:
    from rag.retriever import RAG
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.config import EvaluationConfig, ChunkStrategy, IndexType, ExperimentConfig
from evaluation.schemas import (
    PaperAnnotation, GroundTruth, EvaluationReport,
    L1Result, L2Result, L3Result
)
from evaluation.data_preparation import (
    DataExporter, PDFDownloader, ChunkProcessor, CollectionBuilder, PaperSource
)
from evaluation.data_preparation.pipeline import DataPreparationPipeline
from evaluation.annotation import PaperAnnotator
from evaluation.qa_generation import QAGenerator


class EvaluationPipeline:
    """
    完整的评估流水线
    
    Usage:
        # 初始化
        pipeline = EvaluationPipeline(
            source_rag_client=rag,
            llm_client=llm,
            config=config
        )
        
        # 运行完整流程
        pipeline.run_full_pipeline()
        
        # 或者分步运行
        pipeline.step1_prepare_data()
        pipeline.step2_annotate_papers()
        pipeline.step3_generate_qa()
        pipeline.step4_run_evaluation()
    """
    
    def __init__(
        self,
        source_rag_client: "RAG",
        llm_client: "BaseChatModel",
        config: Optional[EvaluationConfig] = None
    ):
        """
        Args:
            source_rag_client: 业务库 RAG 客户端
            llm_client: LLM 客户端
            config: 评估配置
        """
        self.source_rag = source_rag_client
        self.llm = llm_client
        self.config = config or EvaluationConfig()
        self.config.ensure_dirs()
        
        # 初始化子组件
        self.data_pipeline = DataPreparationPipeline(
            source_rag_client, llm_client, self.config
        )
        self.annotator = PaperAnnotator(llm_client, self.config)
        self.qa_generator = QAGenerator(llm_client, self.config)
        self.collection_builder = CollectionBuilder(self.config)
    
    # ============== 完整流程 ==============
    
    def run_full_pipeline(
        self,
        sample_size: Optional[int] = None,
        skip_existing: bool = True
    ) -> EvaluationReport:
        """
        运行完整的评估流程
        
        Args:
            sample_size: 抽样数量
            skip_existing: 是否跳过已处理的数据
            
        Returns:
            EvaluationReport
        """
        print("\n" + "=" * 70)
        print("EVALUATION PIPELINE")
        print("=" * 70)
        
        # Step 1: 数据准备
        papers = self.step1_prepare_data(sample_size, skip_existing)
        
        # Step 2: 标注
        annotations = self.step2_annotate_papers(papers, skip_existing)
        
        # Step 3: 生成 QA
        ground_truth = self.step3_generate_qa(annotations)
        
        # Step 4: 运行评估
        report = self.step4_run_evaluation(ground_truth)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        
        return report
    
    # ============== 分步执行 ==============
    
    def step1_prepare_data(
        self,
        sample_size: Optional[int] = None,
        skip_existing: bool = True
    ) -> list[PaperSource]:
        """
        Step 1: 数据准备
        
        - 导出论文元数据
        - 下载 PDF
        - 分块处理
        - 入库
        """
        print("\n" + "-" * 50)
        print("STEP 1: Data Preparation")
        print("-" * 50)
        
        # TODO: 调用 data_pipeline
        raise NotImplementedError
    
    def step2_annotate_papers(
        self,
        papers: list[PaperSource],
        skip_existing: bool = True
    ) -> list[PaperAnnotation]:
        """
        Step 2: 论文标注
        
        为每篇论文生成 summary, keywords, research_area
        """
        print("\n" + "-" * 50)
        print("STEP 2: Paper Annotation")
        print("-" * 50)
        
        # TODO: 调用 annotator
        raise NotImplementedError
    
    def step3_generate_qa(
        self,
        annotations: list[PaperAnnotation]
    ) -> GroundTruth:
        """
        Step 3: 生成 QA Ground Truth
        """
        print("\n" + "-" * 50)
        print("STEP 3: QA Generation")
        print("-" * 50)
        
        # TODO: 调用 qa_generator
        raise NotImplementedError
    
    def step4_run_evaluation(
        self,
        ground_truth: GroundTruth
    ) -> EvaluationReport:
        """
        Step 4: 运行评估
        
        对所有实验配置运行评估
        """
        print("\n" + "-" * 50)
        print("STEP 4: Evaluation")
        print("-" * 50)
        
        experiments = self.config.get_all_experiments()
        all_results = {}
        
        for exp in experiments:
            print(f"\nRunning experiment: {exp.full_name}")
            
            # 切换 index (如果需要)
            self.collection_builder.rebuild_index(exp.chunk_strategy, exp.index_type)
            
            # 运行评估
            result = self._run_single_experiment(exp, ground_truth)
            all_results[exp.full_name] = result
        
        # 汇总报告
        report = self._aggregate_results(all_results, ground_truth)
        self._save_report(report)
        
        return report
    
    def _run_single_experiment(
        self,
        experiment: ExperimentConfig,
        ground_truth: GroundTruth
    ) -> dict:
        """运行单个实验"""
        # TODO: 实现
        # 1. 获取对应的 RAG 客户端
        # 2. 根据 enable_agentic_rag 选择检索方式
        # 3. 对每个 QA pair 运行检索
        # 4. 计算 L1/L2/L3 指标
        raise NotImplementedError
    
    def _aggregate_results(
        self,
        all_results: dict,
        ground_truth: GroundTruth
    ) -> EvaluationReport:
        """汇总所有实验结果"""
        # TODO: 实现
        raise NotImplementedError
    
    def _save_report(self, report: EvaluationReport) -> Path:
        """保存评估报告"""
        # TODO: 实现
        raise NotImplementedError
    
    # ============== 工具方法 ==============
    
    def get_status(self) -> dict:
        """
        获取当前流水线状态
        
        Returns:
            {
                "papers_exported": 100,
                "pdfs_downloaded": 95,
                "chunks_cached": {"paragraph": 100, "contextual": 80},
                "papers_annotated": 100,
                "qa_pairs_generated": 100,
                "collections": ["papers_eval_paragraph", "papers_eval_contextual"]
            }
        """
        # TODO: 实现
        raise NotImplementedError
    
    def clean_all(self) -> None:
        """清理所有评估数据（谨慎使用）"""
        # TODO: 实现
        raise NotImplementedError
