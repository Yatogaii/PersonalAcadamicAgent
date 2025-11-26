"""
src/evaluation/__init__.py

RAG 评估框架

模块结构:
- config: 评估配置
- schemas: 数据结构定义
- data_preparation/: 数据准备模块
- annotation/: LLM 标注模块
- qa_generation/: QA 生成模块
- runner: 评估执行器
- pipeline: 完整评估流水线
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

from .rag_evaluator import (
    RAGEvaluator,
    EvaluationResult,
    AggregatedMetrics,
    SemanticSimilarityEvaluator,
    NDCGCalculator,
)

from .runner import EvaluationRunner
from .pipeline import EvaluationPipeline

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
    # 评估器
    "RAGEvaluator",
    "EvaluationResult", 
    "AggregatedMetrics",
    "SemanticSimilarityEvaluator",
    "NDCGCalculator",
    # Runner & Pipeline
    "EvaluationRunner",
    "EvaluationPipeline",
]
