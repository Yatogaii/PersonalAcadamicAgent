"""
Data Exporter

从业务库导出论文元数据，作为评估的数据源
"""

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, asdict

from logging_config import logger

if TYPE_CHECKING:
    from rag.milvus import MilvusProvider

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
    
    def __init__(self, rag_client: "MilvusProvider", config: EvaluationConfig = None):
        """
        Args:
            rag_client: 业务库的 MilvusProvider 客户端
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
        
        只导出有 pdf_url 的论文（paper-level 记录，chunk_id == -1）
        
        Args:
            conference: 筛选特定会议（可选）
            year: 筛选特定年份（可选）
            sample_size: 抽样数量，None 表示全量
            
        Returns:
            PaperSource 列表
        """
        # 构建查询条件：paper-level 记录 (chunk_id == -1) 且有 pdf_url
        filters = [f"{self.rag_client.chunk_id_field} == -1"]
        
        if conference:
            filters.append(f'{self.rag_client.conference_name_field} == "{conference}"')
        if year:
            filters.append(f'{self.rag_client.conference_year_field} == {year}')
        
        filter_expr = " && ".join(filters)
        
        logger.info(f"Querying papers with filter: {filter_expr}")
        
        # 查询所有符合条件的 paper-level 记录
        results = self.rag_client.client.query(
            collection_name=self.rag_client.collection,
            filter=filter_expr,
            output_fields=[
                self.rag_client.doc_id_field,
                self.rag_client.title_field,
                self.rag_client.text_field,  # abstract
                self.rag_client.pdf_url_field,
                self.rag_client.url_field,
                self.rag_client.conference_name_field,
                self.rag_client.conference_year_field,
                self.rag_client.conference_round_field,
            ],
            limit=10000  # 足够大，获取全部
        )
        
        logger.info(f"Found {len(results)} paper-level records")
        
        # 过滤有 pdf_url 的记录，并转换为 PaperSource
        papers: list[PaperSource] = []
        for r in results:
            pdf_url = r.get(self.rag_client.pdf_url_field, "")
            if not pdf_url or not pdf_url.strip():
                continue  # 跳过没有 pdf_url 的
            
            papers.append(PaperSource(
                doc_id=r.get(self.rag_client.doc_id_field, ""),
                title=r.get(self.rag_client.title_field, ""),
                abstract=r.get(self.rag_client.text_field, ""),
                pdf_url=pdf_url,
                url=r.get(self.rag_client.url_field, ""),
                conference_name=r.get(self.rag_client.conference_name_field, ""),
                conference_year=r.get(self.rag_client.conference_year_field, 0),
                conference_round=r.get(self.rag_client.conference_round_field, ""),
            ))
        
        logger.info(f"Found {len(papers)} papers with pdf_url")
        
        # 抽样（如果需要）
        if sample_size and sample_size < len(papers):
            random.seed(42)  # 固定种子，保证可复现
            papers = random.sample(papers, sample_size)
            logger.info(f"Sampled {len(papers)} papers")
        
        return papers
    
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
        papers = self.export(**kwargs)
        
        output_path = output_path or self.config.source_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
        
        logger.info(f"Exported {len(papers)} papers to {output_path}")
        return len(papers)
    
    def load_from_file(self, input_path: Optional[Path] = None) -> list[PaperSource]:
        """
        从 JSONL 文件加载论文元数据
        
        Args:
            input_path: 输入路径，默认使用 config 中的路径
            
        Returns:
            PaperSource 列表
        """
        input_path = input_path or self.config.source_file
        
        if not input_path.exists():
            raise FileNotFoundError(f"Source file not found: {input_path}")
        
        papers: list[PaperSource] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    papers.append(PaperSource(**data))
        
        logger.info(f"Loaded {len(papers)} papers from {input_path}")
        return papers
    
    def get_stats(self, papers: list[PaperSource]) -> dict:
        """
        统计导出数据的信息
        
        Returns:
            包含统计信息的字典
        """
        if not papers:
            return {"total": 0}
        
        # 按会议统计
        by_conference: dict[str, int] = {}
        for p in papers:
            conf = p.conference_name or "unknown"
            by_conference[conf] = by_conference.get(conf, 0) + 1
        
        # 按年份统计
        by_year: dict[int, int] = {}
        for p in papers:
            year = p.conference_year or 0
            by_year[year] = by_year.get(year, 0) + 1
        
        return {
            "total": len(papers),
            "by_conference": by_conference,
            "by_year": dict(sorted(by_year.items())),
        }
