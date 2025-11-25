"""
数据结构定义

所有标注、QA、评估结果的 schema 都在这里定义。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============== 标注相关 Schema ==============

@dataclass
class PaperAnnotation:
    """Paper-level 标注结果"""
    doc_id: str
    title: str
    conference: str = ""
    year: int = 0
    
    # LLM 生成的内容
    summary: str = ""                    # 1-2 句核心贡献总结
    keywords: list[str] = field(default_factory=list)  # 3-5 个关键词
    research_area: str = ""              # 研究领域分类
    
    # Section-level (可选，PDF 加载后填充)
    method_summary: Optional[str] = None
    method_keywords: list[str] = field(default_factory=list)
    eval_summary: Optional[str] = None
    eval_keywords: list[str] = field(default_factory=list)
    
    # 元信息
    annotated_at: str = ""               # 标注时间
    has_pdf_loaded: bool = False         # PDF 是否已加载


# ============== QA 相关 Schema ==============

class Difficulty(str, Enum):
    EASY = "easy"       # 关键词匹配即可
    MEDIUM = "medium"   # 需要语义理解
    HARD = "hard"       # 跨论文综合

class AnswerSource(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHOD = "method"
    EVALUATION = "evaluation"
    MULTIPLE = "multiple"  # 跨多个 section


@dataclass
class QAPair:
    """单个 QA 测试用例"""
    id: int
    question: str
    difficulty: Difficulty
    
    # Ground Truth
    expected_doc_ids: list[str]          # 应该检索到的论文
    answer_source: AnswerSource          # 答案来源 section
    reference_answer: str                # 参考答案
    
    # 可选
    expected_section_category: Optional[int] = None  # 期望的 section_category
    is_multi_paper: bool = False         # 是否需要跨论文检索


@dataclass
class GroundTruth:
    """完整的 Ground Truth 测试集"""
    version: str = "1.0"
    created_at: str = ""
    total_papers: int = 0
    qa_pairs: list[QAPair] = field(default_factory=list)
    
    # 统计信息
    difficulty_distribution: dict = field(default_factory=dict)
    source_distribution: dict = field(default_factory=dict)


# ============== 评估结果 Schema ==============

@dataclass 
class L1Result:
    """L1: Paper Discovery 评估结果"""
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    mean_latency_ms: float = 0.0


@dataclass
class L2Result:
    """L2: Section Retrieval 评估结果"""
    # 按 section 类型的准确率
    method_precision: float = 0.0
    method_recall: float = 0.0
    eval_precision: float = 0.0
    eval_recall: float = 0.0
    
    # 整体
    overall_precision: float = 0.0
    overall_recall: float = 0.0


@dataclass
class L3Result:
    """L3: End-to-End QA 评估结果"""
    # 按难度分
    easy_accuracy: float = 0.0
    medium_accuracy: float = 0.0
    hard_accuracy: float = 0.0
    
    # 整体
    overall_accuracy: float = 0.0
    answer_relevance: float = 0.0    # LLM 评分
    faithfulness: float = 0.0        # 答案是否忠实于检索内容


@dataclass
class EvaluationReport:
    """完整评估报告"""
    run_id: str
    run_at: str
    
    # 配置信息
    total_qa_pairs: int = 0
    rag_provider: str = ""
    embedding_model: str = ""
    
    # 分层结果
    l1_paper_discovery: L1Result = field(default_factory=L1Result)
    l2_section_retrieval: L2Result = field(default_factory=L2Result)
    l3_end_to_end: L3Result = field(default_factory=L3Result)
    
    # 详细结果（可选）
    detailed_results: list[dict] = field(default_factory=list)
