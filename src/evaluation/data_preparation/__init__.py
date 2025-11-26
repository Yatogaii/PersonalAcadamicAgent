"""
数据准备模块

负责:
1. 从业务库导出论文元数据
2. 通过 PDFLoader 下载 + 解析 + 入库（复用现有逻辑）
3. 通过回调保存 chunks 到本地（供 QA 生成使用）
4. 创建 evaluation collection 并管理索引
"""

from .data_exporter import DataExporter, PaperSource
from .collection_builder import CollectionBuilder, CollectionStats
from .pipeline import DataPreparationPipeline, PipelineResult

__all__ = [
    # 数据导出
    "DataExporter",
    "PaperSource",
    # Collection 管理
    "CollectionBuilder",
    "CollectionStats",
    # Pipeline
    "DataPreparationPipeline",
    "PipelineResult",
]
