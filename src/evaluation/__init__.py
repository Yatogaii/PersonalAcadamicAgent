"""
src/evaluation/__init__.py

RAG 评估框架

模块结构:
- rag_evaluator: 基础评估指标计算
- schemas: 数据结构定义
- annotation/: LLM 标注模块
- qa_generation/: QA 生成模块
- runner: 评估执行器
"""

from .rag_evaluator import (
    RAGEvaluator,
    EvaluationResult,
    AggregatedMetrics,
    SemanticSimilarityEvaluator,
    NDCGCalculator,
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

from .runner import EvaluationRunner

__all__ = [
    # 基础评估
    "RAGEvaluator",
    "EvaluationResult", 
    "AggregatedMetrics",
    "SemanticSimilarityEvaluator",
    "NDCGCalculator",
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
    # Runner
    "EvaluationRunner",
]
