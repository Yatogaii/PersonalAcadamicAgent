"""
Collection Builder

负责:
1. 创建 evaluation collection
2. 管理 index 重建
3. 提供 collection 切换的 Context Manager
"""

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import sqlite3

from logging_config import logger
from settings import settings

if TYPE_CHECKING:
    from pymilvus import MilvusClient
    from rag.milvus import MilvusProvider

from evaluation.config import EvaluationConfig, ChunkStrategy, IndexType


# Index 参数配置
INDEX_PARAMS = {
    IndexType.FLAT: {
        "index_type": "FLAT",
        "params": {}
    },
    IndexType.HNSW: {
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256}
    },
    IndexType.IVF: {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    },
}

SEARCH_PARAMS = {
    IndexType.FLAT: {},
    IndexType.HNSW: {"ef": 64},
    IndexType.IVF: {"nprobe": 16},
}


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
        self._client: Optional["MilvusClient"] = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None
    
    @property
    def client(self) -> "MilvusClient":
        """获取 Milvus 客户端（延迟创建）"""
        if settings.rag_provider == "sqlite":
            raise RuntimeError("Milvus client requested while using sqlite provider.")
        if self._client is None:
            from pymilvus import MilvusClient
            self._client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token
            )
        return self._client

    def _sqlite_connection(self) -> sqlite3.Connection:
        if self._sqlite_conn is None:
            try:
                import sqlite_vec
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "sqlite-vec is not installed. Install with `uv add --optional sqlite sqlite-vec`."
                ) from exc

            db_path = Path(settings.sqlite_path).expanduser()
            if str(db_path) != ":memory:":
                db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._sqlite_conn = conn

        return self._sqlite_conn
    
    def _get_collection_name(self, chunk_strategy: ChunkStrategy) -> str:
        """生成 collection 名称"""
        return f"papers_eval_{chunk_strategy.value}"

    # ============== SQLite Helpers ==============

    def _sqlite_table_exists(self, table_name: str) -> bool:
        conn = self._sqlite_connection()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        )
        return cur.fetchone() is not None

    def _sqlite_drop_table(self, table_name: str) -> None:
        conn = self._sqlite_connection()
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()

    def _sqlite_create_table(self, table_name: str) -> None:
        columns = [
            f"{settings.milvus_vector_field} float[{self.config.embedding_dim}]",
            f"{settings.milvus_doc_id_field} TEXT",
            f"{settings.milvus_chunk_id_field} INTEGER",
            f"{settings.milvus_section_category_field} INTEGER",
            f"{settings.milvus_conference_name_field} TEXT",
            f"{settings.milvus_conference_year_field} INTEGER",
            f"{settings.milvus_conference_round_field} TEXT",
            f"+{settings.milvus_title_field} TEXT",
            f"+{settings.milvus_text_field} TEXT",
            f"+{settings.milvus_url_field} TEXT",
            f"+{settings.milvus_pdf_url_field} TEXT",
            f"+{settings.milvus_parent_section_field} TEXT",
            f"+{settings.milvus_page_number_field} INTEGER",
        ]

        metric = (
            settings.sqlite_distance_metric
            or settings.milvus_vector_index_metric_type
            or ""
        ).strip().upper()

        if metric == "COSINE":
            columns.append("distance_metric=cosine")
        elif metric and metric != "L2":
            logger.warning("Unsupported sqlite-vec metric '%s', falling back to L2.", metric)

        sql = f"CREATE VIRTUAL TABLE IF NOT EXISTS {table_name} USING vec0({', '.join(columns)})"
        conn = self._sqlite_connection()
        conn.execute(sql)
        conn.commit()
    
    # ============== Collection 管理 ==============
    
    def create_collection(
        self,
        chunk_strategy: ChunkStrategy,
        index_type: IndexType = IndexType.FLAT,
        drop_if_exists: bool = False
    ) -> str:
        """
        创建 evaluation collection
        
        使用与业务库相同的 schema
        
        Args:
            chunk_strategy: 分块策略（决定 collection 名称）
            index_type: 索引类型
            drop_if_exists: 是否删除已存在的 collection
            
        Returns:
            collection 名称
        """
        collection_name = self._get_collection_name(chunk_strategy)

        if settings.rag_provider == "sqlite":
            if self._sqlite_table_exists(collection_name):
                if drop_if_exists:
                    logger.info(f"Dropping existing table: {collection_name}")
                    self._sqlite_drop_table(collection_name)
                else:
                    logger.info(f"Table already exists: {collection_name}")
                    return collection_name

            self._sqlite_create_table(collection_name)
            logger.info(f"Created table: {collection_name} (sqlite-vec)")
            return collection_name
        
        from rag.milvus import MilvusProvider
        
        # 检查是否存在
        if self.client.has_collection(collection_name):
            if drop_if_exists:
                logger.info(f"Dropping existing collection: {collection_name}")
                self.client.drop_collection(collection_name)
            else:
                logger.info(f"Collection already exists: {collection_name}")
                return collection_name
        
        # 使用 MilvusProvider 的 schema
        provider = MilvusProvider()
        schema = provider._create_schema()
        
        # 创建 index params
        index_config = INDEX_PARAMS.get(index_type, INDEX_PARAMS[IndexType.FLAT])
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=settings.milvus_vector_field,
            index_type=index_config["index_type"],
            metric_type=settings.milvus_vector_index_metric_type,
            index_name="vector_index",
            params=index_config["params"],
        )
        
        # 创建 collection
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        
        logger.info(f"Created collection: {collection_name} with index: {index_type.value}")
        return collection_name
    
    def drop_collection(self, chunk_strategy: ChunkStrategy) -> bool:
        """删除 collection"""
        collection_name = self._get_collection_name(chunk_strategy)
        if settings.rag_provider == "sqlite":
            if not self._sqlite_table_exists(collection_name):
                logger.warning(f"Table does not exist: {collection_name}")
                return False
            self._sqlite_drop_table(collection_name)
            logger.info(f"Dropped table: {collection_name}")
            return True

        if not self.client.has_collection(collection_name):
            logger.warning(f"Collection does not exist: {collection_name}")
            return False
        
        self.client.drop_collection(collection_name)
        logger.info(f"Dropped collection: {collection_name}")
        return True
    
    def collection_exists(self, chunk_strategy: ChunkStrategy) -> bool:
        """检查 collection 是否存在"""
        collection_name = self._get_collection_name(chunk_strategy)
        if settings.rag_provider == "sqlite":
            return self._sqlite_table_exists(collection_name)
        return self.client.has_collection(collection_name)
    
    # ============== Settings 切换 ==============
    
    @contextmanager
    def use_eval_collection(self, chunk_strategy: ChunkStrategy):
        """
        Context Manager: 临时切换到 evaluation collection
        
        Usage:
            with builder.use_eval_collection(ChunkStrategy.PARAGRAPH):
                # settings.milvus_collection 已切换
                loader = PDFLoader(rag_client)
                loader.load_papers(doc_ids)
            # 退出后自动恢复原 collection
        """
        eval_collection = self._get_collection_name(chunk_strategy)

        if settings.rag_provider == "sqlite":
            original_table = settings.sqlite_table
            try:
                settings.sqlite_table = eval_collection
                logger.info(f"Switched to eval table: {eval_collection}")
                yield eval_collection
            finally:
                settings.sqlite_table = original_table
                logger.info(f"Restored to original table: {original_table}")
            return

        original_collection = settings.milvus_collection
        try:
            settings.milvus_collection = eval_collection
            logger.info(f"Switched to eval collection: {eval_collection}")
            yield eval_collection
        finally:
            settings.milvus_collection = original_collection
            logger.info(f"Restored to original collection: {original_collection}")
    
    @contextmanager
    def use_chunk_strategy(self, chunk_strategy: ChunkStrategy):
        """
        Context Manager: 临时切换 chunk_strategy 和 collection
        
        同时切换:
        - settings.milvus_collection
        - settings.chunk_strategy
        """
        eval_collection = self._get_collection_name(chunk_strategy)
        original_strategy = settings.chunk_strategy

        if settings.rag_provider == "sqlite":
            original_table = settings.sqlite_table
            try:
                settings.sqlite_table = eval_collection
                settings.chunk_strategy = chunk_strategy.value
                logger.info(f"Switched to: table={eval_collection}, strategy={chunk_strategy.value}")
                yield eval_collection
            finally:
                settings.sqlite_table = original_table
                settings.chunk_strategy = original_strategy
                logger.info(f"Restored to: table={original_table}, strategy={original_strategy}")
            return

        original_collection = settings.milvus_collection
        try:
            settings.milvus_collection = eval_collection
            settings.chunk_strategy = chunk_strategy.value
            logger.info(f"Switched to: collection={eval_collection}, strategy={chunk_strategy.value}")
            yield eval_collection
        finally:
            settings.milvus_collection = original_collection
            settings.chunk_strategy = original_strategy
            logger.info(f"Restored to: collection={original_collection}, strategy={original_strategy}")
    
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
        collection_name = self._get_collection_name(chunk_strategy)

        if settings.rag_provider == "sqlite":
            logger.info("SQLite vec0 uses brute-force search; rebuild_index skipped.")
            return True
        
        if not self.client.has_collection(collection_name):
            logger.error(f"Collection does not exist: {collection_name}")
            return False
        
        vector_field = settings.milvus_vector_field
        
        try:
            # 1. Release collection（如果已 load）
            self.client.release_collection(collection_name)
            
            # 2. Drop 现有 index
            self.client.drop_index(
                collection_name=collection_name,
                index_name="vector_index"
            )
            logger.info(f"Dropped old index on {collection_name}")
            
            # 3. 创建新 index
            index_config = INDEX_PARAMS.get(index_type, INDEX_PARAMS[IndexType.FLAT])
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=vector_field,
                index_type=index_config["index_type"],
                metric_type=settings.milvus_vector_index_metric_type,
                index_name="vector_index",
                params=index_config["params"],
            )
            
            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params,
            )
            
            # 4. Load collection
            self.client.load_collection(collection_name)
            
            logger.info(f"Rebuilt index on {collection_name}: {index_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            return False
    
    def get_current_index_type(self, chunk_strategy: ChunkStrategy) -> Optional[IndexType]:
        """获取当前 collection 的 index 类型"""
        collection_name = self._get_collection_name(chunk_strategy)

        if settings.rag_provider == "sqlite":
            return None
        
        if not self.client.has_collection(collection_name):
            return None
        
        try:
            indexes = self.client.list_indexes(collection_name)
            if not indexes:
                return None
            
            # 获取 index 信息
            index_info = self.client.describe_index(
                collection_name=collection_name,
                index_name="vector_index"
            )
            
            index_type_str = index_info.get("index_type", "FLAT")
            
            # 映射到 IndexType enum
            for it in IndexType:
                if it.value == index_type_str:
                    return it
            
            return IndexType.FLAT
            
        except Exception as e:
            logger.warning(f"Failed to get index type: {e}")
            return None
    
    # ============== 统计 ==============
    
    def get_collection_stats(self, chunk_strategy: ChunkStrategy) -> Optional[CollectionStats]:
        """获取 collection 统计信息"""
        collection_name = self._get_collection_name(chunk_strategy)
        if settings.rag_provider == "sqlite":
            if not self._sqlite_table_exists(collection_name):
                return None
            try:
                conn = self._sqlite_connection()
                total_records = conn.execute(
                    f"SELECT COUNT(*) FROM {collection_name}"
                ).fetchone()[0]
                total_papers = conn.execute(
                    f"SELECT COUNT(*) FROM {collection_name} WHERE {settings.milvus_chunk_id_field} = -1"
                ).fetchone()[0]
                return CollectionStats(
                    name=collection_name,
                    total_records=total_records,
                    total_papers=total_papers,
                    index_type="vec0",
                    embedding_dim=self.config.embedding_dim,
                )
            except Exception as e:
                logger.error(f"Failed to get table stats: {e}")
                return None

        if not self.client.has_collection(collection_name):
            return None
        
        try:
            # 获取总记录数
            stats = self.client.get_collection_stats(collection_name)
            total_records = stats.get("row_count", 0)
            
            # 获取 unique doc_id 数量（论文数）
            # 查询所有 paper-level 记录
            results = self.client.query(
                collection_name=collection_name,
                filter=f"{settings.milvus_chunk_id_field} == -1",
                output_fields=[settings.milvus_doc_id_field],
                limit=10000
            )
            total_papers = len(results)
            
            # 获取 index 类型
            index_type = self.get_current_index_type(chunk_strategy)
            
            return CollectionStats(
                name=collection_name,
                total_records=total_records,
                total_papers=total_papers,
                index_type=index_type.value if index_type else "unknown",
                embedding_dim=self.config.embedding_dim,
            )
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return None
    
    def list_all_collections(self) -> list[str]:
        """列出所有 evaluation collection"""
        if settings.rag_provider == "sqlite":
            conn = self._sqlite_connection()
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'papers_eval_%'"
            )
            return [row[0] for row in cur.fetchall()]

        all_collections = self.client.list_collections()
        # 只返回评估相关的 collection
        return [c for c in all_collections if c.startswith("papers_eval_")]
