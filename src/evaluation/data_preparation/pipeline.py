"""
Data Preparation Pipeline

整合数据准备的完整流程:
1. 从业务库导出论文元数据
2. 使用 PDFLoader 下载 + 解析 + 入库（复用现有逻辑）
3. 通过回调保存 chunks 到本地（供 QA 生成使用）
"""

import json
from typing import TYPE_CHECKING, Optional
from pathlib import Path
from dataclasses import dataclass, field

from logging_config import logger

if TYPE_CHECKING:
    from rag.milvus import MilvusProvider
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.config import EvaluationConfig, ChunkStrategy
from evaluation.data_preparation.data_exporter import DataExporter, PaperSource
from evaluation.data_preparation.collection_builder import CollectionBuilder


@dataclass
class PipelineResult:
    """Pipeline 执行结果"""
    papers_exported: int = 0
    papers_processed: dict[str, int] = field(default_factory=dict)  # strategy -> count
    papers_success: dict[str, int] = field(default_factory=dict)
    papers_failed: dict[str, int] = field(default_factory=dict)
    chunks_saved: dict[str, int] = field(default_factory=dict)


class DataPreparationPipeline:
    """
    数据准备流水线
    
    核心思路：复用 PDFLoader，通过以下方式适配评估场景：
    1. 使用 CollectionBuilder.use_chunk_strategy() 切换 settings
    2. 使用 PDFLoader 的 on_chunks_processed 回调保存 chunks 到本地
    """
    
    def __init__(
        self,
        source_rag_client: "MilvusProvider",
        llm_client: Optional["BaseChatModel"] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Args:
            source_rag_client: 业务库 MilvusProvider 客户端
            llm_client: LLM 客户端 (用于 Contextual Chunking)
            config: 评估配置
        """
        self.config = config or EvaluationConfig()
        self.source_rag_client = source_rag_client
        self.llm_client = llm_client
        
        # 确保目录存在
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.exporter = DataExporter(source_rag_client, self.config)
        self.collection_builder = CollectionBuilder(self.config)
        
        # 用于跟踪保存的 chunks
        self._chunks_saved_count = 0
    
    def run(
        self,
        strategies: Optional[list[ChunkStrategy]] = None,
        sample_size: Optional[int] = None,
        drop_existing: bool = False,
    ) -> PipelineResult:
        """
        运行数据准备流程
        
        Args:
            strategies: 要处理的分块策略列表，默认全部
            sample_size: 抽样数量（None = 全量）
            drop_existing: 是否删除已有的评估 collection
            
        Returns:
            PipelineResult
        """
        strategies = strategies or self.config.chunk_strategies
        result = PipelineResult()
        
        # Step 1: 导出论文元数据
        logger.info("=" * 60)
        logger.info("Step 1: Exporting papers from source database...")
        logger.info("=" * 60)
        
        papers = self._export_papers(sample_size)
        result.papers_exported = len(papers)
        logger.info(f"Exported {len(papers)} papers with pdf_url")
        
        if not papers:
            logger.warning("No papers to process!")
            return result
        
        # Step 2: 对每种策略，处理论文
        for strategy in strategies:
            logger.info("=" * 60)
            logger.info(f"Step 2: Processing with strategy [{strategy.value}]...")
            logger.info("=" * 60)
            
            stats = self._process_papers_for_strategy(
                papers=papers,
                strategy=strategy,
                drop_existing=drop_existing
            )
            
            result.papers_processed[strategy.value] = stats["processed"]
            result.papers_success[strategy.value] = stats["success"]
            result.papers_failed[strategy.value] = stats["failed"]
            result.chunks_saved[strategy.value] = stats["chunks_saved"]
        
        # 打印总结
        self._print_summary(result)
        
        return result
    
    def _export_papers(self, sample_size: Optional[int]) -> list[PaperSource]:
        """导出论文元数据"""
        import random
        
        # 尝试从文件加载
        if self.config.source_file.exists():
            logger.info(f"Loading papers from existing file: {self.config.source_file}")
            papers = self.exporter.load_from_file()
        else:
            # 从数据库导出（全量）
            papers = self.exporter.export()
            # 保存到文件
            self.exporter.export_to_file()
        
        # 如果需要抽样
        if sample_size and sample_size < len(papers):
            random.seed(42)
            papers = random.sample(papers, sample_size)
            logger.info(f"Sampled {len(papers)} papers from {len(self.exporter.load_from_file())}")
        
        return papers
    
    def _process_papers_for_strategy(
        self,
        papers: list[PaperSource],
        strategy: ChunkStrategy,
        drop_existing: bool
    ) -> dict:
        """
        使用指定策略处理所有论文
        
        核心：通过 Context Manager 切换 settings，然后复用 PDFLoader
        
        注意：需要先把 paper-level 记录复制到评估 collection，
        因为 PDFLoader.get_paper_metadata() 会从当前 collection 查询
        """
        from rag.milvus import MilvusProvider
        from rag.pdf_loader import PDFLoader, LoadStatus
        
        stats = {"processed": 0, "success": 0, "failed": 0, "chunks_saved": 0}
        
        # 1. 创建评估 collection
        self.collection_builder.create_collection(
            strategy, 
            drop_if_exists=drop_existing
        )
        
        # 2. 准备 chunks 保存目录
        chunks_dir = self.config.chunks_dir / strategy.value
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. 创建保存 chunks 的回调
        self._chunks_saved_count = 0
        
        def save_chunks_callback(doc_id: str, chunks: list[dict], title: str):
            """保存 chunks 到本地文件"""
            save_path = chunks_dir / f"{doc_id}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "doc_id": doc_id,
                    "title": title,
                    "strategy": strategy.value,
                    "chunks": chunks
                }, f, ensure_ascii=False, indent=2)
            self._chunks_saved_count += 1
        
        # 4. 使用 Context Manager 切换到评估 collection 和 chunk 策略
        with self.collection_builder.use_chunk_strategy(strategy):
            # 创建新的 MilvusProvider（会使用修改后的 settings）
            eval_rag_client = MilvusProvider()
            
            # 5. 复制 paper-level 记录到评估 collection
            # PDFLoader 需要从 collection 查询 metadata
            logger.info(f"Copying {len(papers)} paper-level records to eval collection...")
            self._copy_paper_records(papers, eval_rag_client)
            
            # 创建 PDFLoader，传入回调
            pdf_loader = PDFLoader(
                rag_client=eval_rag_client,
                llm_client=self.llm_client,
                on_chunks_processed=save_chunks_callback
            )
            
            # 设置 PDF 缓存目录（跨策略共享）
            pdf_loader.set_cache_dir(self.config.pdf_dir)
            
            # 获取所有 doc_ids
            doc_ids = [p.doc_id for p in papers]
            
            # 批量处理
            logger.info(f"Processing {len(doc_ids)} papers...")
            results = pdf_loader.load_papers(doc_ids)
            
            # 统计结果
            for doc_id, load_result in results.items():
                stats["processed"] += 1
                if load_result.status == LoadStatus.SUCCESS:
                    stats["success"] += 1
                elif load_result.status == LoadStatus.ALREADY_EXISTS:
                    stats["success"] += 1  # 已存在也算成功
                else:
                    stats["failed"] += 1
        
        stats["chunks_saved"] = self._chunks_saved_count
        
        logger.info(f"Strategy [{strategy.value}]: "
                   f"processed={stats['processed']}, "
                   f"success={stats['success']}, "
                   f"failed={stats['failed']}, "
                   f"chunks_saved={stats['chunks_saved']}")
        
        return stats
    
    def _print_summary(self, result: PipelineResult) -> None:
        """打印处理结果摘要"""
        logger.info("=" * 60)
        logger.info("Pipeline Complete!")
        logger.info("=" * 60)
        logger.info(f"Papers exported: {result.papers_exported}")
        
        for strategy in result.papers_processed.keys():
            logger.info(f"\n[{strategy}]:")
            logger.info(f"  Processed: {result.papers_processed.get(strategy, 0)}")
            logger.info(f"  Success: {result.papers_success.get(strategy, 0)}")
            logger.info(f"  Failed: {result.papers_failed.get(strategy, 0)}")
            logger.info(f"  Chunks saved: {result.chunks_saved.get(strategy, 0)}")
    
    def _copy_paper_records(self, papers: list[PaperSource], eval_rag_client: "MilvusProvider"):
        """
        复制 paper-level 记录到评估 collection
        
        PDFLoader 需要从 collection 查询 metadata（pdf_url 等）
        """
        for paper in papers:
            # 使用 insert_document 插入 paper-level 记录
            # 但需要保持原有的 doc_id
            eval_rag_client.client.insert(
                collection_name=eval_rag_client.collection,
                data={
                    eval_rag_client.doc_id_field: paper.doc_id,
                    eval_rag_client.vector_field: eval_rag_client.embedding_client.embed_query(
                        f"Title: {paper.title}\nAbstract: {paper.abstract}"
                    ),
                    eval_rag_client.title_field: paper.title,
                    eval_rag_client.text_field: paper.abstract,
                    eval_rag_client.url_field: paper.url,
                    eval_rag_client.pdf_url_field: paper.pdf_url,
                    eval_rag_client.conference_name_field: paper.conference_name,
                    eval_rag_client.conference_year_field: paper.conference_year,
                    eval_rag_client.conference_round_field: paper.conference_round,
                    eval_rag_client.chunk_id_field: -1,  # paper-level
                    eval_rag_client.section_category_field: 0,
                    eval_rag_client.parent_section_field: "",
                    eval_rag_client.page_number_field: 1,
                }
            )
        logger.info(f"Copied {len(papers)} paper-level records")
    
    # ============== 单独步骤方法（便于调试） ==============
    
    def export_only(self, sample_size: Optional[int] = None) -> list[PaperSource]:
        """只执行导出步骤"""
        return self.exporter.export(sample_size=sample_size)
    
    def get_collection_stats(self, strategy: ChunkStrategy) -> Optional[dict]:
        """获取指定策略的 collection 统计"""
        stats = self.collection_builder.get_collection_stats(strategy)
        if stats:
            return {
                "name": stats.name,
                "total_records": stats.total_records,
                "total_papers": stats.total_papers,
                "index_type": stats.index_type,
            }
        return None

    def rebuild_from_chunks(
        self,
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        drop_existing: bool = True,
    ) -> int:
        """
        从已保存的 chunks 文件重建评估 collection
        
        当 chunks 文件已存在但 collection 数据不一致时使用
        
        Args:
            strategy: 分块策略
            drop_existing: 是否删除已有 collection
            
        Returns:
            成功插入的 chunks 数量
        """
        from rag.milvus import MilvusProvider
        
        chunks_dir = self.config.chunks_dir / strategy.value
        
        if not chunks_dir.exists():
            logger.error(f"Chunks directory not found: {chunks_dir}")
            return 0
        
        chunk_files = list(chunks_dir.glob("*.json"))
        if not chunk_files:
            logger.error(f"No chunk files found in {chunks_dir}")
            return 0
        
        logger.info(f"Rebuilding collection from {len(chunk_files)} chunk files...")
        
        # 1. 创建/重建 collection
        self.collection_builder.create_collection(strategy, drop_if_exists=drop_existing)
        
        total_chunks = 0
        
        # 2. 切换到评估 collection
        with self.collection_builder.use_chunk_strategy(strategy):
            eval_rag_client = MilvusProvider()
            
            for chunk_file in chunk_files:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                doc_id = data["doc_id"]
                title = data["title"]
                chunks = data["chunks"]
                
                # 加载 source paper 信息（从文件或数据库）
                paper_info = self._get_paper_info(doc_id)
                
                # 插入 paper-level 记录
                eval_rag_client.client.insert(
                    collection_name=eval_rag_client.collection,
                    data={
                        eval_rag_client.doc_id_field: doc_id,
                        eval_rag_client.vector_field: eval_rag_client.embedding_client.embed_query(
                            f"Title: {title}\nAbstract: {paper_info.get('abstract', '')[:500]}"
                        ),
                        eval_rag_client.title_field: title,
                        eval_rag_client.text_field: paper_info.get("abstract", ""),
                        eval_rag_client.url_field: paper_info.get("url", ""),
                        eval_rag_client.pdf_url_field: paper_info.get("pdf_url", ""),
                        eval_rag_client.conference_name_field: paper_info.get("conference_name", ""),
                        eval_rag_client.conference_year_field: paper_info.get("conference_year", 0),
                        eval_rag_client.conference_round_field: paper_info.get("conference_round", ""),
                        eval_rag_client.chunk_id_field: -1,
                        eval_rag_client.section_category_field: 0,
                        eval_rag_client.parent_section_field: "",
                        eval_rag_client.page_number_field: 1,
                    }
                )
                
                # 插入 chunks
                eval_rag_client.insert_paper_chunks(doc_id, chunks, title)
                total_chunks += len(chunks)
                
                logger.info(f"Inserted {len(chunks)} chunks for {doc_id[:8]}...")
        
        logger.info(f"Rebuild complete: {total_chunks} chunks from {len(chunk_files)} papers")
        return total_chunks
    
    def _get_paper_info(self, doc_id: str) -> dict:
        """获取论文信息（从 source_papers.jsonl 或数据库）"""
        # 先尝试从文件加载
        if self.config.source_file.exists():
            with open(self.config.source_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if data["doc_id"] == doc_id:
                        return data
        
        # 回退到数据库查询
        metadata = self.source_rag_client.get_paper_metadata(doc_id)
        return metadata or {}
