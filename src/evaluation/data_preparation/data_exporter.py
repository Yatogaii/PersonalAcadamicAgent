"""
Data Exporter

从业务库导出论文元数据，作为评估的数据源
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, asdict

if TYPE_CHECKING:
    from rag.retriever import RAG

from evaluation.config import EvaluationConfig


@dataclass
class PaperSource:
    """论文元数据"""
    doc_id: str
    title: str
    abstract: str
    pdf_url: str
    url: str = ""
    conference_name: str = ""
    conference_year: int = 0
    conference_round: str = ""


class DataExporter:
    """从业务库导出数据"""
    
    def __init__(self, rag_client: "RAG", config: EvaluationConfig = None):
        """
        Args:
            rag_client: 业务库的 RAG 客户端
            config: 评估配置
        """
        self.rag_client = rag_client
        self.config = config or EvaluationConfig()
    
    def export(
        self,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        sample_size: Optional[int] = None
    ) -> list[PaperSource]:
        """
        从业务库导出论文元数据
        
        Args:
            conference: 筛选特定会议
            year: 筛选特定年份
            sample_size: 抽样数量，None 表示全量
            
        Returns:
            PaperSource 列表
        """
        # TODO: 实现
        # 1. 从 Milvus 查询 paper-level 记录 (chunk_id == -1)
        # 2. 按条件筛选
        # 3. 抽样（如果需要）
        # 4. 转换为 PaperSource 对象
        raise NotImplementedError
    
    def export_to_file(
        self,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> int:
        """
        导出并保存到 JSONL 文件
        
        Args:
            output_path: 输出路径，默认使用 config 中的路径
            **kwargs: 传递给 export() 的参数
            
        Returns:
            导出的论文数量
        """
        # TODO: 实现
        # 1. 调用 export()
        # 2. 保存到 JSONL 文件
        raise NotImplementedError
    
    def load_from_file(self, input_path: Optional[Path] = None) -> list[PaperSource]:
        """
        从 JSONL 文件加载论文元数据
        
        Args:
            input_path: 输入路径，默认使用 config 中的路径
            
        Returns:
            PaperSource 列表
        """
        # TODO: 实现
        raise NotImplementedError
    
    def get_stats(self, papers: list[PaperSource]) -> dict:
        """统计导出数据的信息"""
        # TODO: 实现
        # 返回: 总数, 按会议分布, 按年份分布, 有 pdf_url 的比例等
        raise NotImplementedError
