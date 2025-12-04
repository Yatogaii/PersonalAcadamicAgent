"""
对比实验运行器

自动化运行所有配置组合的评估实验:
- 2 种 Chunk 策略 (paragraph, contextual)
- 3 种 Index 类型 (FLAT, HNSW, IVF)
- 2 种 RAG 模式 (basic, agentic)

总计: 2 × 3 × 2 = 12 种配置
"""

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import json

from logging_config import logger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.config import (
    EvaluationConfig, ExperimentConfig, 
    ChunkStrategy, IndexType
)
from evaluation.schemas import EvaluationReport, L1Result, L2Result, L3Result
from evaluation.runner import EvaluationRunner
from evaluation.data_preparation.collection_builder import CollectionBuilder
from evaluation.data_preparation.pipeline import DataPreparationPipeline


@dataclass
class ExperimentResult:
    """单个实验结果"""
    config: ExperimentConfig
    report: EvaluationReport
    
    def to_dict(self) -> dict:
        return {
            "config": {
                "name": self.config.name,
                "chunk_strategy": self.config.chunk_strategy.value,
                "index_type": self.config.index_type.value,
                "enable_agentic_rag": self.config.enable_agentic_rag,
            },
            "metrics": {
                "l1_mrr": self.report.l1_paper_discovery.mrr,
                "l1_hit_rate": self.report.l1_paper_discovery.hit_rate,
                "l1_precision_at_5": self.report.l1_paper_discovery.precision_at_5,
                "l1_recall_at_10": self.report.l1_paper_discovery.recall_at_10,
                "l1_latency_ms": self.report.l1_paper_discovery.mean_latency_ms,
                "l2_precision": self.report.l2_section_retrieval.overall_precision,
                "l2_recall": self.report.l2_section_retrieval.overall_recall,
                "l3_accuracy": self.report.l3_end_to_end.overall_accuracy,
                "l3_faithfulness": self.report.l3_end_to_end.faithfulness,
            }
        }


@dataclass
class ComparisonReport:
    """对比实验汇总报告"""
    run_id: str
    run_at: str
    total_experiments: int
    results: list[ExperimentResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_at": self.run_at,
            "total_experiments": self.total_experiments,
            "results": [r.to_dict() for r in self.results]
        }
    
    def to_markdown_table(self) -> str:
        """生成 Markdown 对比表格"""
        headers = [
            "Config", "Chunk", "Index", "Agentic",
            "MRR", "Hit@10", "P@5", "L2_P", "L3_Acc", "Latency(ms)"
        ]
        
        rows = []
        for r in self.results:
            m = r.to_dict()["metrics"]
            rows.append([
                r.config.name,
                r.config.chunk_strategy.value,
                r.config.index_type.value,
                "✓" if r.config.enable_agentic_rag else "",
                f"{m['l1_mrr']:.3f}",
                f"{m['l1_hit_rate']:.3f}",
                f"{m['l1_precision_at_5']:.3f}",
                f"{m['l2_precision']:.3f}",
                f"{m['l3_accuracy']:.3f}",
                f"{m['l1_latency_ms']:.0f}",
            ])
        
        # 生成表格
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            table += "| " + " | ".join(row) + " |\n"
        
        return table


class AgenticRAGAdapter:
    """
    Agentic RAG 适配器
    
    包装 Searcher 使其符合 EvaluationRunner 的接口
    """
    
    def __init__(self, base_rag_client, llm_client):
        """
        Args:
            base_rag_client: 底层 MilvusProvider
            llm_client: LLM 客户端
        """
        self.base_rag = base_rag_client
        self.llm = llm_client
        self._searcher = None
    
    @property
    def searcher(self):
        """延迟初始化 Searcher"""
        if self._searcher is None:
            from agents.searcher import Searcher
            from settings import settings
            # 临时启用 agentic 模式
            original = settings.enable_agentic_rag
            settings.enable_agentic_rag = True
            self._searcher = Searcher()
            settings.enable_agentic_rag = original
        return self._searcher
    
    def search_abstracts(self, query: str, k: int = 10) -> list[dict]:
        """
        使用 Agentic 搜索
        
        注意：Agentic 模式会返回分析结果而非原始文档列表
        需要从 intermediate_steps 中提取实际检索到的文档
        """
        # 使用底层 RAG 执行搜索（Agentic 模式需要在实际使用时实现）
        # 这里简化处理：直接调用 base_rag，模拟 agentic 行为
        # TODO: 实现真正的 agentic 评估
        return self.base_rag.search_abstracts(query, k)
    
    def search_by_section(self, query: str, doc_id: str = None, 
                         section_category: int = None, k: int = 10) -> list[dict]:
        """使用底层 RAG 的 section 搜索"""
        return self.base_rag.search_by_section(query, doc_id, section_category, k)
    
    # 代理其他属性
    def __getattr__(self, name):
        return getattr(self.base_rag, name)


class ComparisonRunner:
    """对比实验运行器"""
    
    def __init__(
        self,
        llm_client: Optional["BaseChatModel"] = None,
        config: Optional[EvaluationConfig] = None
    ):
        self.config = config or EvaluationConfig()
        self.llm = llm_client
        self.builder = CollectionBuilder(self.config)
        
        # 缓存文件路径
        self._cache_file = self.config.reports_dir / ".comparison_cache.json"
    
    def _load_cache(self) -> dict[str, dict]:
        """加载已完成实验的缓存"""
        if not self._cache_file.exists():
            return {}
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}
    
    def _save_cache(self, cache: dict[str, dict]) -> None:
        """保存缓存"""
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    
    def _cache_result(self, exp_name: str, result: ExperimentResult) -> None:
        """缓存单个实验结果"""
        cache = self._load_cache()
        cache[exp_name] = result.to_dict()
        self._save_cache(cache)
        logger.info(f"  ✓ Cached: {exp_name}")
    
    def clear_cache(self) -> None:
        """清除缓存"""
        if self._cache_file.exists():
            self._cache_file.unlink()
            logger.info("Cache cleared")
        
    def prepare_data(
        self,
        strategies: list[ChunkStrategy] = None,
        sample_size: int = None,
    ) -> dict:
        """
        准备所有 chunk 策略的数据
        
        Args:
            strategies: 要准备的策略列表
            sample_size: 抽样大小
            
        Returns:
            每种策略的处理结果
        """
        from rag.milvus import MilvusProvider
        
        strategies = strategies or self.config.chunk_strategies
        source_rag = MilvusProvider()
        
        pipeline = DataPreparationPipeline(
            source_rag_client=source_rag,
            llm_client=self.llm,
            config=self.config
        )
        
        result = pipeline.run(
            strategies=strategies,
            sample_size=sample_size,
            drop_existing=True
        )
        
        return {
            "papers_exported": result.papers_exported,
            "papers_success": result.papers_success,
            "chunks_saved": result.chunks_saved,
        }
    
    def run_single_experiment(
        self,
        experiment: ExperimentConfig,
        run_l3: bool = True
    ) -> ExperimentResult:
        """
        运行单个实验配置
        
        Args:
            experiment: 实验配置
            run_l3: 是否运行 L3 评估
            
        Returns:
            ExperimentResult
        """
        from rag.milvus import MilvusProvider
        
        logger.info(f"Running experiment: {experiment.name}")
        
        # 1. 重建 collection（使用指定的索引类型）
        self.builder.create_collection(
            chunk_strategy=experiment.chunk_strategy,
            index_type=experiment.index_type,
            drop_if_exists=True
        )
        
        # 2. 从 chunks 文件重建数据
        pipeline = DataPreparationPipeline(
            source_rag_client=MilvusProvider(),  # 业务库
            llm_client=self.llm,
            config=self.config
        )
        pipeline.rebuild_from_chunks(experiment.chunk_strategy)
        
        # 3. 切换到评估 collection 并运行评估
        with self.builder.use_chunk_strategy(experiment.chunk_strategy):
            milvus = MilvusProvider()
            
            # 根据是否 agentic 选择 RAG 客户端
            if experiment.enable_agentic_rag:
                rag_client = AgenticRAGAdapter(milvus, self.llm)
            else:
                rag_client = milvus
            
            # 创建 Runner
            runner = EvaluationRunner(
                rag_client=rag_client,
                llm_client=self.llm if run_l3 else None,
                config=self.config
            )
            
            # 运行评估
            report = runner.run_all()
        
        return ExperimentResult(config=experiment, report=report)
    
    def run_all_experiments(
        self,
        experiments: list[ExperimentConfig] = None,
        run_l3: bool = True,
        skip_data_preparation: bool = False,
        resume: bool = True,
    ) -> ComparisonReport:
        """
        运行所有实验配置
        
        Args:
            experiments: 实验配置列表，None 则使用默认全部
            run_l3: 是否运行 L3 评估
            skip_data_preparation: 跳过数据准备（使用已有数据）
            resume: 是否从缓存恢复（跳过已完成的实验）
            
        Returns:
            ComparisonReport
        """
        import uuid
        
        experiments = experiments or self.config.get_all_experiments()
        
        # 加载缓存
        cache = self._load_cache() if resume else {}
        cached_count = len([e for e in experiments if e.name in cache])
        
        logger.info(f"Running {len(experiments)} experiments")
        logger.info(f"L3 evaluation: {'enabled' if run_l3 else 'disabled'}")
        if resume and cached_count > 0:
            logger.info(f"Resume mode: {cached_count} experiments already cached, will skip")
        
        # 按 chunk 策略分组
        strategies_needed = set(e.chunk_strategy for e in experiments)
        
        # 准备数据（如果需要）
        if not skip_data_preparation:
            logger.info("Preparing data for all chunk strategies...")
            self.prepare_data(strategies=list(strategies_needed))
        
        # 运行每个实验
        results = []
        for i, exp in enumerate(experiments, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i}/{len(experiments)}: {exp.name}")
            logger.info(f"{'='*60}")
            
            # 检查缓存
            if resume and exp.name in cache:
                logger.info(f"  ⏭️  Skipped (cached)")
                # 从缓存恢复结果
                cached_data = cache[exp.name]
                result = self._restore_from_cache(exp, cached_data)
                if result:
                    results.append(result)
                    m = cached_data["metrics"]
                    logger.info(f"  (Cached) MRR={m['l1_mrr']:.3f}, HitRate={m['l1_hit_rate']:.3f}, "
                               f"L3_Acc={m['l3_accuracy']:.3f}")
                continue
            
            try:
                result = self.run_single_experiment(exp, run_l3=run_l3)
                results.append(result)
                
                # 缓存结果
                self._cache_result(exp.name, result)
                
                # 打印中间结果
                m = result.to_dict()["metrics"]
                logger.info(f"  MRR={m['l1_mrr']:.3f}, HitRate={m['l1_hit_rate']:.3f}, "
                           f"L3_Acc={m['l3_accuracy']:.3f}")
            except Exception as e:
                logger.error(f"Experiment {exp.name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 生成汇总报告
        comparison = ComparisonReport(
            run_id=str(uuid.uuid4())[:8],
            run_at=datetime.now().isoformat(),
            total_experiments=len(experiments),
            results=results
        )
        
        return comparison
    
    def _restore_from_cache(
        self, 
        exp: ExperimentConfig, 
        cached_data: dict
    ) -> Optional[ExperimentResult]:
        """从缓存恢复 ExperimentResult"""
        try:
            m = cached_data["metrics"]
            # 创建一个简化的 report（只包含关键指标）
            report = EvaluationReport(
                run_id=cached_data.get("run_id", "cached"),
                run_at=cached_data.get("run_at", ""),
                total_qa_pairs=0,
                rag_provider="cached",
                embedding_model="cached",
                l1_paper_discovery=L1Result(
                    precision_at_5=m.get("l1_precision_at_5", 0),
                    precision_at_10=0,
                    recall_at_5=0,
                    recall_at_10=m.get("l1_recall_at_10", 0),
                    mrr=m["l1_mrr"],
                    hit_rate=m["l1_hit_rate"],
                    mean_latency_ms=m.get("l1_latency_ms", 0),
                    per_query_results=[]
                ),
                l2_section_retrieval=L2Result(
                    overall_precision=m.get("l2_precision", 0),
                    overall_recall=m.get("l2_recall", 0),
                    method_precision=0,
                    eval_precision=0,
                    per_query_results=[]
                ),
                l3_end_to_end=L3Result(
                    easy_accuracy=0,
                    medium_accuracy=0,
                    hard_accuracy=0,
                    overall_accuracy=m.get("l3_accuracy", 0),
                    faithfulness=m.get("l3_faithfulness", 0),
                    relevance=0,
                    per_query_results=[]
                )
            )
            return ExperimentResult(config=exp, report=report)
        except Exception as e:
            logger.warning(f"Failed to restore from cache: {e}")
            return None
    
    def save_comparison(self, comparison: ComparisonReport) -> Path:
        """保存对比报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{comparison.run_id}_{timestamp}.json"
        output_path = self.config.reports_dir / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 同时保存 Markdown 表格
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# RAG Comparison Report\n\n")
            f.write(f"Run ID: {comparison.run_id}\n")
            f.write(f"Time: {comparison.run_at}\n")
            f.write(f"Total Experiments: {comparison.total_experiments}\n\n")
            f.write("## Results\n\n")
            f.write(comparison.to_markdown_table())
        
        logger.info(f"Saved comparison to {output_path}")
        logger.info(f"Saved markdown to {md_path}")
        
        return output_path
    
    def print_comparison(self, comparison: ComparisonReport) -> None:
        """打印对比结果"""
        print("\n" + "=" * 80)
        print("RAG Comparison Results")
        print("=" * 80)
        print(f"Run ID: {comparison.run_id}")
        print(f"Time: {comparison.run_at}")
        print(f"Total: {comparison.total_experiments} experiments")
        print("\n" + comparison.to_markdown_table())
