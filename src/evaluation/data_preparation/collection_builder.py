"""
Collection Builder

负责:
1. 创建 evaluation collection
2. 将处理好的 chunks 入库
3. 管理 index 重建
"""

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from rag.retriever import RAG

from evaluation.config import EvaluationConfig, ChunkStrategy, IndexType
from evaluation.data_preparation.chunk_processor import ChunkCache, ProcessedChunk


@dataclass
class CollectionStats:
    """Collection 统计信息"""
    name: str
    total_records: int
    total_papers: int
    index_type: str
    embedding_dim: int


class CollectionBuilder:
    """Collection 构建器"""
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Args:
            config: 评估配置
        """
        self.config = config or EvaluationConfig()
        self._clients: dict[str, "RAG"] = {}  # collection_name -> client
    
    # ============== Collection 管理 ==============
    
    def create_collection(
        self,
        chunk_strategy: ChunkStrategy,
        drop_if_exists: bool = False
    ) -> str:
        """
        创建 evaluation collection
        
        Args:
            chunk_strategy: 分块策略（决定 collection 名称）
            drop_if_exists: 是否删除已存在的 collection
            
        Returns:
            collection 名称
        """
        # TODO: 实现
        # 1. 生成 collection 名称: papers_eval_{strategy}
        # 2. 如果存在且 drop_if_exists，删除
        # 3. 创建新 collection（使用与业务库相同的 schema）
        # 4. 创建 client 并缓存
        raise NotImplementedError
    
    def drop_collection(self, chunk_strategy: ChunkStrategy) -> bool:
        """删除 collection"""
        # TODO: 实现
        raise NotImplementedError
    
    def collection_exists(self, chunk_strategy: ChunkStrategy) -> bool:
        """检查 collection 是否存在"""
        # TODO: 实现
        raise NotImplementedError
    
    def get_client(self, chunk_strategy: ChunkStrategy) -> "RAG":
        """获取指定 collection 的 RAG 客户端"""
        # TODO: 实现
        raise NotImplementedError
    
    # ============== 数据入库 ==============
    
    def insert_chunks(
        self,
        chunk_cache: ChunkCache,
        paper_title: str = "",
        show_progress: bool = True
    ) -> int:
        """
        将 chunks 插入到对应 collection
        
        Args:
            chunk_cache: Chunk 缓存
            paper_title: 论文标题
            show_progress: 是否显示进度
            
        Returns:
            插入的记录数
        """
        # TODO: 实现
        # 1. 根据 chunk_cache.strategy 获取 collection
        # 2. 计算 embedding
        # 3. 批量插入
        raise NotImplementedError
    
    def insert_batch(
        self,
        chunk_caches: list[ChunkCache],
        strategy: ChunkStrategy,
        show_progress: bool = True
    ) -> int:
        """
        批量入库多篇论文的 chunks
        
        Args:
            chunk_caches: Chunk 缓存列表
            strategy: 分块策略
            show_progress: 是否显示进度
            
        Returns:
            总插入记录数
        """
        # TODO: 实现
        raise NotImplementedError
    
    # ============== Index 管理 ==============
    
    def rebuild_index(
        self,
        chunk_strategy: ChunkStrategy,
        index_type: IndexType
    ) -> bool:
        """
        重建 collection 的 index
        
        Args:
            chunk_strategy: 分块策略（决定哪个 collection）
            index_type: 目标 index 类型
            
        Returns:
            是否成功
        """
        # TODO: 实现
        # 1. 获取 collection
        # 2. drop 现有 index
        # 3. 创建新 index
        raise NotImplementedError
    
    def get_current_index_type(self, chunk_strategy: ChunkStrategy) -> Optional[IndexType]:
        """获取当前 collection 的 index 类型"""
        # TODO: 实现
        raise NotImplementedError
    
    # ============== 统计 ==============
    
    def get_collection_stats(self, chunk_strategy: ChunkStrategy) -> CollectionStats:
        """获取 collection 统计信息"""
        # TODO: 实现
        raise NotImplementedError
    
    def list_all_collections(self) -> list[str]:
        """列出所有 evaluation collection"""
        # TODO: 实现
        raise NotImplementedError
