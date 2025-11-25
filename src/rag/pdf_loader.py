"""
PDF Lazy Loader - 按需下载、解析 PDF 并插入 chunks 到 RAG

Usage:
    loader = PDFLoader(rag_client)
    results = loader.load_papers(["doc_id_1", "doc_id_2"])
"""

from typing import TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile
import time
import httpx

from logging_config import logger

if TYPE_CHECKING:
    from rag.retriever import RAG


class LoadStatus(Enum):
    """PDF 加载状态"""
    SUCCESS = "success"           # 成功加载并入库
    ALREADY_EXISTS = "already_exists"  # chunks 已存在，跳过
    DOWNLOAD_FAILED = "download_failed"  # PDF 下载失败
    PARSE_FAILED = "parse_failed"      # PDF 解析失败
    NO_PDF_URL = "no_pdf_url"          # 没有 pdf_url
    NOT_FOUND = "not_found"            # doc_id 不存在


@dataclass
class LoadResult:
    """单个论文的加载结果"""
    doc_id: str
    status: LoadStatus
    message: str = ""
    chunks_count: int = 0


class PDFLoader:
    """
    负责按需下载、解析 PDF 并插入 chunks 到 RAG。
    
    流程:
    1. 检查 doc_id 的 chunks 是否已存在
    2. 获取 paper metadata（包含 pdf_url）
    3. 下载 PDF
    4. 解析 PDF 为 chunks
    5. 插入 chunks 到 RAG
    """
    
    # 常见的学术 PDF 网站需要的 headers
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }
    
    def __init__(self, rag_client: "RAG"):
        self.rag_client = rag_client
        self.download_timeout = 120  # PDF 可能较大，给足够时间
        self.max_retries = 3
        self.retry_delay = 2  # 重试间隔秒数
        
        # 可选：本地缓存目录
        self.cache_dir: Path | None = None
    
    def set_cache_dir(self, cache_dir: str | Path):
        """设置 PDF 缓存目录"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_papers(self, doc_ids: list[str]) -> dict[str, LoadResult]:
        """
        批量加载论文 PDF。
        
        Args:
            doc_ids: 要加载的论文 doc_id 列表
            
        Returns:
            dict[doc_id, LoadResult] - 每个论文的加载结果
        """
        results: dict[str, LoadResult] = {}
        
        for doc_id in doc_ids:
            result = self._load_single_paper(doc_id)
            results[doc_id] = result
            
            status_emoji = {
                LoadStatus.SUCCESS: "✅",
                LoadStatus.ALREADY_EXISTS: "⏭️",
                LoadStatus.DOWNLOAD_FAILED: "❌",
                LoadStatus.PARSE_FAILED: "❌",
                LoadStatus.NO_PDF_URL: "⚠️",
                LoadStatus.NOT_FOUND: "❌",
            }.get(result.status, "❓")
            
            logger.info(f"{status_emoji} PDF load [{doc_id[:8]}...]: {result.status.value} - {result.message}")
        
        return results
    
    def _load_single_paper(self, doc_id: str) -> LoadResult:
        """加载单个论文的 PDF"""
        
        # Step 1: 检查是否已存在 chunks
        if self.rag_client.check_pdf_chunks_exist(doc_id):
            return LoadResult(
                doc_id=doc_id,
                status=LoadStatus.ALREADY_EXISTS,
                message="PDF chunks already indexed"
            )
        
        # Step 2: 获取 paper metadata
        metadata = self.rag_client.get_paper_metadata(doc_id)
        if not metadata:
            return LoadResult(
                doc_id=doc_id,
                status=LoadStatus.NOT_FOUND,
                message="Paper not found in database"
            )
        
        pdf_url = metadata.get("pdf_url", "")
        if not pdf_url:
            return LoadResult(
                doc_id=doc_id,
                status=LoadStatus.NO_PDF_URL,
                message="No PDF URL available"
            )
        
        title = metadata.get("title", "")
        logger.info(f"Loading PDF for: {title[:50]}...")
        
        # Step 3: 下载 PDF（带重试）
        pdf_bytes = self._download_pdf_with_retry(pdf_url)
        if pdf_bytes is None:
            return LoadResult(
                doc_id=doc_id,
                status=LoadStatus.DOWNLOAD_FAILED,
                message=f"Failed to download PDF from {pdf_url}"
            )
        
        logger.info(f"Downloaded PDF: {len(pdf_bytes) / 1024:.1f} KB")
        
        # Step 4: 解析 PDF
        try:
            chunks = self._parse_pdf(pdf_bytes, title)
        except Exception as e:
            logger.error(f"PDF parse error: {e}")
            return LoadResult(
                doc_id=doc_id,
                status=LoadStatus.PARSE_FAILED,
                message=f"Failed to parse PDF: {str(e)[:100]}"
            )
        
        if not chunks:
            return LoadResult(
                doc_id=doc_id,
                status=LoadStatus.PARSE_FAILED,
                message="No content extracted from PDF (possibly scanned/image PDF)"
            )
        
        logger.info(f"Parsed {len(chunks)} chunks from PDF")
        
        # Step 5: 插入 chunks
        try:
            self._insert_chunks(doc_id, chunks, title)
        except Exception as e:
            logger.error(f"Insert chunks error: {e}")
            return LoadResult(
                doc_id=doc_id,
                status=LoadStatus.PARSE_FAILED,
                message=f"Failed to insert chunks: {str(e)[:100]}"
            )
        
        return LoadResult(
            doc_id=doc_id,
            status=LoadStatus.SUCCESS,
            message=f"Successfully indexed {len(chunks)} chunks",
            chunks_count=len(chunks)
        )
    
    def _download_pdf_with_retry(self, pdf_url: str) -> bytes | None:
        """带重试的 PDF 下载"""
        for attempt in range(self.max_retries):
            try:
                pdf_bytes = self._download_pdf(pdf_url)
                if pdf_bytes:
                    return pdf_bytes
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1}/{self.max_retries} failed: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))  # 递增延迟
        
        return None
    
    def _download_pdf(self, pdf_url: str) -> bytes | None:
        """
        下载 PDF 文件。
        
        处理:
        - 重定向
        - 自定义 User-Agent（避免被拒绝）
        - 超时控制
        - SSL 问题容错
        """
        # 检查本地缓存
        if self.cache_dir:
            cache_file = self._get_cache_path(pdf_url)
            if cache_file.exists():
                logger.info(f"Using cached PDF: {cache_file}")
                return cache_file.read_bytes()
        
        # 尝试多种 SSL 配置
        ssl_configs = [
            True,   # 默认验证
            False,  # 跳过验证（某些学术网站证书有问题）
        ]
        
        for verify_ssl in ssl_configs:
            try:
                with httpx.Client(
                    timeout=httpx.Timeout(self.download_timeout),
                    follow_redirects=True,
                    headers=self.DEFAULT_HEADERS,
                    verify=verify_ssl,
                ) as client:
                    response = client.get(pdf_url)
                    response.raise_for_status()
                    
                    # 验证是否为 PDF
                    content_type = response.headers.get("content-type", "")
                    if "pdf" not in content_type.lower() and not response.content[:4] == b"%PDF":
                        logger.warning(f"Response may not be PDF. Content-Type: {content_type}")
                        # 仍然尝试解析，可能 content-type 不准确
                    
                    pdf_bytes = response.content
                    
                    # 保存到缓存
                    if self.cache_dir and pdf_bytes:
                        cache_file = self._get_cache_path(pdf_url)
                        cache_file.write_bytes(pdf_bytes)
                        logger.info(f"Cached PDF to: {cache_file}")
                    
                    if not verify_ssl:
                        logger.warning("Downloaded with SSL verification disabled")
                    
                    return pdf_bytes
                    
            except httpx.TimeoutException:
                logger.error(f"Download timeout after {self.download_timeout}s: {pdf_url}")
                return None
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code}: {pdf_url}")
                return None
            except Exception as e:
                if verify_ssl:
                    # 如果是 SSL 错误，尝试禁用验证
                    logger.warning(f"Download failed with SSL verification, retrying without: {e}")
                    continue
                else:
                    logger.error(f"Download error: {e}")
                    return None
        
        return None
    
    def _get_cache_path(self, pdf_url: str) -> Path:
        """根据 URL 生成缓存文件路径"""
        import hashlib
        url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:16]
        assert self.cache_dir is not None
        return self.cache_dir / f"{url_hash}.pdf"
    
    def _parse_pdf(self, pdf_bytes: bytes, paper_title: str) -> list[dict]:
        """
        解析 PDF 为 chunks。
        
        使用 parser/pdf_parser.py 的解析函数
        """
        from parser.pdf_parser import parse_pdf, flatten_pdf_tree
        
        # 写入临时文件（PyMuPDF 需要文件路径）
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            temp_path = f.name
        
        try:
            # 解析 PDF 获取结构化树
            outline_tree = parse_pdf(temp_path)
            
            if not outline_tree:
                logger.warning("No outline/structure found in PDF, trying fallback...")
                # Fallback: 尝试简单的全文提取
                chunks = self._fallback_parse(temp_path, paper_title)
                return chunks
            
            # 扁平化为 chunks
            chunks = flatten_pdf_tree(outline_tree, paper_title)
            return chunks
            
        finally:
            # 清理临时文件
            Path(temp_path).unlink(missing_ok=True)
    
    def _fallback_parse(self, pdf_path: str, paper_title: str) -> list[dict]:
        """
        Fallback 解析：当 PDF 没有 outline 时，按页面分割
        """
        import fitz
        from parser.pdf_parser import clean_text, SectionCategory
        
        doc = fitz.open(pdf_path)
        chunks = []
        chunk_idx = 0
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text: str = page.get_text("text")  # type: ignore
            
            if not text or not text.strip():
                continue
            
            # 清理文本
            cleaned = clean_text(text)
            if not cleaned or len(cleaned) < 50:  # 跳过太短的内容
                continue
            
            # 简单的章节推断（基于页码）
            if page_num == 0:
                category = SectionCategory.ABSTRACT
            elif page_num <= 2:
                category = SectionCategory.INTRODUCTION
            else:
                category = SectionCategory.OTHER
            
            chunk = {
                "chunk_index": chunk_idx,
                "text": cleaned,
                "section_title": f"Page {page_num + 1}",
                "parent_section": "",
                "section_category": int(category),
                "page_number": page_num + 1,
            }
            chunks.append(chunk)
            chunk_idx += 1
        
        doc.close()
        return chunks
    
    def _insert_chunks(self, doc_id: str, chunks: list[dict], paper_title: str):
        """
        插入 chunks 到 RAG。
        """
        self.rag_client.insert_paper_chunks(doc_id, chunks, paper_title)
        logger.success(f"Inserted {len(chunks)} chunks for doc_id: {doc_id[:8]}...")


def get_pdf_loader(rag_client: "RAG") -> PDFLoader:
    """工厂函数，获取 PDF Loader 实例"""
    return PDFLoader(rag_client)
