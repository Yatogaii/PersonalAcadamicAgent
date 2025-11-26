"""
src/evaluation/__init__.py

RAG 评估框架

模块结构:
- config: 评估配置
- schemas: 数据结构定义
- data_preparation/: 数据准备模块
- annotation/: LLM 标注模块 (TODO)
- qa_generation/: QA 生成模块 (TODO)
- runner: 评估执行器 (TODO)
- pipeline: 完整评估流水线 (TODO)
"""

from .config import (
    EvaluationConfig,
    ExperimentConfig,
    ChunkStrategy,
    IndexType,
)

from .schemas import (
    PaperAnnotation,
    QAPair,
    GroundTruth,
    Difficulty,
    AnswerSource,
    EvaluationReport,
    L1Result,
    L2Result,
    L3Result,
)

# 已实现的模块
from .data_preparation import (
    DataExporter,
    PaperSource,
    CollectionBuilder,
    CollectionStats,
    DataPreparationPipeline,
    PipelineResult,
)

# TODO: 待实现
# from .rag_evaluator import ...
# from .runner import EvaluationRunner
# from .pipeline import EvaluationPipeline

__all__ = [
    # 配置
    "EvaluationConfig",
    "ExperimentConfig",
    "ChunkStrategy",
    "IndexType",
    # Schema
    "PaperAnnotation",
    "QAPair",
    "GroundTruth",
    "Difficulty",
    "AnswerSource",
    "EvaluationReport",
    "L1Result",
    "L2Result",
    "L3Result",
    # Data Preparation
    "DataExporter",
    "PaperSource",
    "CollectionBuilder",
    "CollectionStats",
    "DataPreparationPipeline",
    "PipelineResult",
]
