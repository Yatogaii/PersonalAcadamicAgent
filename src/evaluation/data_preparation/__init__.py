"""
数据准备模块

负责:
1. 从业务库导出论文元数据
2. 下载 PDF
3. 分块 (paragraph / contextual)
4. 缓存分块结果
5. 创建 evaluation collection 并入库
"""

from .data_exporter import DataExporter, PaperSource
from .pdf_downloader import PDFDownloader, DownloadResult, DownloadStatus
from .chunk_processor import ChunkProcessor, ProcessedChunk, ChunkCache
from .collection_builder import CollectionBuilder, CollectionStats

__all__ = [
    "DataExporter",
    "PaperSource",
    "PDFDownloader",
    "DownloadResult",
    "DownloadStatus",
    "ChunkProcessor",
    "ProcessedChunk",
    "ChunkCache",
    "CollectionBuilder",
    "CollectionStats",
]
