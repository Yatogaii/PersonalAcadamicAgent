"""
Evaluation 配置

定义评估实验的所有配置项
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ChunkStrategy(str, Enum):
    """分块策略"""
    PARAGRAPH = "paragraph"      # 传统段落分块
    CONTEXTUAL = "contextual"    # Contextual Chunking (需要 LLM)


class IndexType(str, Enum):
    """Milvus Index 类型"""
    FLAT = "FLAT"    # 精确搜索，适合小数据集
    HNSW = "HNSW"    # 近似搜索，速度快
    IVF = "IVF_FLAT" # 倒排索引


@dataclass
class ExperimentConfig:
    """单个实验配置"""
    name: str
    chunk_strategy: ChunkStrategy
    index_type: IndexType
    enable_agentic_rag: bool = False
    
    @property
    def collection_name(self) -> str:
        """生成 collection 名称 (只按 chunk 策略分)"""
        return f"papers_eval_{self.chunk_strategy.value}"
    
    @property
    def full_name(self) -> str:
        """完整实验名称"""
        agentic = "_agentic" if self.enable_agentic_rag else ""
        return f"{self.chunk_strategy.value}_{self.index_type.value.lower()}{agentic}"


@dataclass 
class EvaluationConfig:
    """评估总配置"""
    
    # === 数据路径 ===
    data_dir: Path = field(default_factory=lambda: Path("evaluation/data"))
    
    @property
    def source_file(self) -> Path:
        """业务库导出的论文元数据"""
        return self.data_dir / "papers_source.jsonl"
    
    @property
    def pdf_dir(self) -> Path:
        """下载的 PDF 存储目录"""
        return self.data_dir / "pdfs"
    
    @property
    def chunks_dir(self) -> Path:
        """分块缓存目录"""
        return self.data_dir / "chunks"
    
    @property
    def summaries_file(self) -> Path:
        """论文标注结果"""
        return self.data_dir / "papers_summaries.jsonl"
    
    @property
    def ground_truth_file(self) -> Path:
        """Ground Truth QA pairs"""
        return self.data_dir / "ground_truth.json"
    
    @property
    def reports_dir(self) -> Path:
        """评估报告目录"""
        return self.data_dir / "reports"
    
    # === 数据源配置 ===
    source_collection: str = "papers"  # 业务库 collection 名称
    sample_size: Optional[int] = None  # None = 全量, 50/100/500 = 抽样
    
    # === Chunk 配置 ===
    chunk_strategies: list[ChunkStrategy] = field(
        default_factory=lambda: [ChunkStrategy.PARAGRAPH, ChunkStrategy.CONTEXTUAL]
    )
    fixed_chunk_size: int = 512  # fixed_size 分块时的大小
    
    # === Index 配置 ===
    index_types: list[IndexType] = field(
        default_factory=lambda: [IndexType.FLAT, IndexType.HNSW, IndexType.IVF]
    )
    
    # === Embedding 配置 ===
    embedding_model: str = "qwen3-embedding:4b"
    embedding_dim: int = 2560
    
    # === LLM 配置 (用于 Contextual Chunking 和标注) ===
    llm_model: str = "qwen3:8b"  # 本地 Ollama 模型
    llm_base_url: str = "http://localhost:11434"
    
    # === QA 生成配置 ===
    num_qa_pairs: int = 100
    difficulty_distribution: dict = field(
        default_factory=lambda: {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    )
    
    # === 评估配置 ===
    top_k_values: list[int] = field(default_factory=lambda: [5, 10])
    
    def get_all_experiments(self) -> list[ExperimentConfig]:
        """生成所有实验配置组合"""
        experiments = []
        for chunk in self.chunk_strategies:
            for index in self.index_types:
                for agentic in [False, True]:
                    exp = ExperimentConfig(
                        name=f"{chunk.value}_{index.value.lower()}_{'agentic' if agentic else 'basic'}",
                        chunk_strategy=chunk,
                        index_type=index,
                        enable_agentic_rag=agentic
                    )
                    experiments.append(exp)
        return experiments
    
    def get_collections(self) -> list[str]:
        """获取需要创建的 collection 列表 (按 chunk 策略)"""
        return [f"papers_eval_{chunk.value}" for chunk in self.chunk_strategies]
    
    def ensure_dirs(self) -> None:
        """确保所有目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        (self.chunks_dir / "paragraph").mkdir(exist_ok=True)
        (self.chunks_dir / "contextual").mkdir(exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


# 默认配置单例
default_config = EvaluationConfig()
