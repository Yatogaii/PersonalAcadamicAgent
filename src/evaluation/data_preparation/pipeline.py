"""
Evaluation Pipeline

整合数据准备的完整流程
"""

from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from rag.retriever import RAG
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.config import EvaluationConfig, ChunkStrategy
from evaluation.data_preparation import (
    DataExporter,
    PDFDownloader,
    ChunkProcessor,
    CollectionBuilder,
    PaperSource,
)


class DataPreparationPipeline:
    """
    数据准备流水线
    
    整合从导出到入库的完整流程
    """
    
    def __init__(
        self,
        source_rag_client: "RAG",
        llm_client: Optional["BaseChatModel"] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Args:
            source_rag_client: 业务库 RAG 客户端
            llm_client: LLM 客户端 (用于 Contextual Chunking)
            config: 评估配置
        """
        self.config = config or EvaluationConfig()
        self.config.ensure_dirs()
        
        # 初始化各组件
        self.exporter = DataExporter(source_rag_client, self.config)
        self.downloader = PDFDownloader(self.config)
        self.chunk_processor = ChunkProcessor(self.config, llm_client)
        self.collection_builder = CollectionBuilder(self.config)
    
    def run_full_pipeline(
        self,
        strategies: Optional[list[ChunkStrategy]] = None,
        sample_size: Optional[int] = None,
        skip_download: bool = False,
        skip_existing_chunks: bool = True
    ) -> dict:
        """
        运行完整的数据准备流程
        
        Args:
            strategies: 要处理的分块策略列表，默认全部
            sample_size: 抽样数量
            skip_download: 是否跳过 PDF 下载（使用已有的）
            skip_existing_chunks: 是否跳过已缓存的 chunks
            
        Returns:
            处理结果统计
        """
        strategies = strategies or self.config.chunk_strategies
        
        result = {
            "papers_exported": 0,
            "pdfs_downloaded": 0,
            "chunks_processed": {},
            "records_inserted": {},
        }
        
        # Step 1: 导出论文元数据
        print("=" * 60)
        print("Step 1: Exporting papers from source database...")
        print("=" * 60)
        papers = self._step_export(sample_size)
        result["papers_exported"] = len(papers)
        print(f"Exported {len(papers)} papers\n")
        
        # Step 2: 下载 PDF
        if not skip_download:
            print("=" * 60)
            print("Step 2: Downloading PDFs...")
            print("=" * 60)
            download_stats = self._step_download(papers)
            result["pdfs_downloaded"] = download_stats.get("success", 0)
            print(f"Downloaded {result['pdfs_downloaded']} PDFs\n")
        else:
            print("Step 2: Skipping PDF download\n")
        
        # Step 3 & 4: 分块 + 入库 (按策略分别处理)
        for strategy in strategies:
            print("=" * 60)
            print(f"Step 3-4: Processing chunks [{strategy.value}]...")
            print("=" * 60)
            
            # 创建 collection
            collection_name = self.collection_builder.create_collection(
                strategy, drop_if_exists=True
            )
            print(f"Created collection: {collection_name}")
            
            # 处理 chunks
            chunk_caches = self._step_chunk(papers, strategy, skip_existing_chunks)
            result["chunks_processed"][strategy.value] = sum(
                len(c.chunks) for c in chunk_caches.values()
            )
            
            # 入库
            inserted = self.collection_builder.insert_batch(
                list(chunk_caches.values()), strategy
            )
            result["records_inserted"][strategy.value] = inserted
            
            print(f"Processed {result['chunks_processed'][strategy.value]} chunks")
            print(f"Inserted {inserted} records\n")
        
        print("=" * 60)
        print("Data preparation complete!")
        print("=" * 60)
        self._print_summary(result)
        
        return result
    
    def _step_export(self, sample_size: Optional[int]) -> list[PaperSource]:
        """Step 1: 导出论文"""
        # TODO: 实现
        raise NotImplementedError
    
    def _step_download(self, papers: list[PaperSource]) -> dict:
        """Step 2: 下载 PDF"""
        # TODO: 实现
        raise NotImplementedError
    
    def _step_chunk(
        self,
        papers: list[PaperSource],
        strategy: ChunkStrategy,
        skip_existing: bool
    ) -> dict:
        """Step 3: 分块处理"""
        # TODO: 实现
        raise NotImplementedError
    
    def _print_summary(self, result: dict) -> None:
        """打印处理结果摘要"""
        print(f"\nSummary:")
        print(f"  Papers exported: {result['papers_exported']}")
        print(f"  PDFs downloaded: {result['pdfs_downloaded']}")
        for strategy, count in result['chunks_processed'].items():
            print(f"  Chunks [{strategy}]: {count}")
        for strategy, count in result['records_inserted'].items():
            print(f"  Records [{strategy}]: {count}")
