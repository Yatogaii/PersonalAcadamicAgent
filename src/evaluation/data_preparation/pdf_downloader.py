"""
PDF Downloader

下载论文 PDF 文件，支持断点续传和并发下载
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    pass

from evaluation.config import EvaluationConfig
from evaluation.data_preparation.data_exporter import PaperSource


class DownloadStatus(str, Enum):
    """下载状态"""
    SUCCESS = "success"
    ALREADY_EXISTS = "already_exists"
    FAILED = "failed"
    NO_URL = "no_url"


@dataclass
class DownloadResult:
    """下载结果"""
    doc_id: str
    status: DownloadStatus
    file_path: Optional[Path] = None
    error_message: str = ""


class PDFDownloader:
    """PDF 下载器"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Args:
            config: 评估配置
        """
        self.config = config or EvaluationConfig()
        self.config.ensure_dirs()
    
    def download_single(self, paper: PaperSource) -> DownloadResult:
        """
        下载单个 PDF
        
        Args:
            paper: 论文元数据
            
        Returns:
            DownloadResult
        """
        # TODO: 实现
        # 1. 检查是否已存在
        # 2. 检查是否有 pdf_url
        # 3. 下载 PDF
        # 4. 保存到 config.pdf_dir / {doc_id}.pdf
        raise NotImplementedError
    
    def download_batch(
        self,
        papers: list[PaperSource],
        max_concurrent: int = 5,
        show_progress: bool = True
    ) -> dict[str, DownloadResult]:
        """
        批量下载 PDF
        
        Args:
            papers: 论文列表
            max_concurrent: 最大并发数
            show_progress: 是否显示进度条
            
        Returns:
            {doc_id: DownloadResult}
        """
        # TODO: 实现
        # 1. 过滤已下载的
        # 2. 并发下载
        # 3. 返回结果汇总
        raise NotImplementedError
    
    def get_local_path(self, doc_id: str) -> Optional[Path]:
        """获取本地 PDF 路径（如果存在）"""
        path = self.config.pdf_dir / f"{doc_id}.pdf"
        return path if path.exists() else None
    
    def get_download_stats(self, papers: list[PaperSource]) -> dict:
        """
        统计下载状态
        
        Returns:
            {
                "total": 100,
                "downloaded": 80,
                "pending": 15,
                "no_url": 5
            }
        """
        # TODO: 实现
        raise NotImplementedError
