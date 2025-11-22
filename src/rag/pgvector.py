import uuid
import psycopg2
from psycopg2.extras import execute_values
from rag.retriever import RAG, Chunk
from rag.feature_extractor import FeatureExtractor
from settings import settings

# https://pixion.co/blog/vector-database-benchmark-chroma-vs-milvus-vs-pgvector-vs-redis
# https://www.firecrawl.dev/blog/best-vector-databases-2025

class PGVectorProvider(RAG):
    def __init__(self) -> None:
        super().__init__()
        self.user = settings.postgres_user
        self.password = settings.postgres_password
        self.host = settings.postgres_host
        self.port = settings.postgres_port
        self.dbname = settings.postgres_db
        self.table_name = settings.postgres_table_name
        self.dim = settings.embedding_dim
        
        # Embedding model configuration
        self.embedding_client = FeatureExtractor(
            provider=settings.embedding_provider,
            api_key=settings.HF_TOKEN,
            model=settings.embedding_model
        )

        self.conn = self._get_connection()
        self._ensure_table_exists()

    def _get_connection(self):
        try:
            conn = psycopg2.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                dbname=self.dbname
            )
            conn.autocommit = True
            return conn
        except Exception as e:
            raise RuntimeError(f"Failed to connect to PostgreSQL: {e}")

    def _ensure_table_exists(self):
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id BIGSERIAL PRIMARY KEY,
                doc_id UUID NOT NULL,
                title TEXT,
                abstract TEXT,
                url TEXT,
                conference_name TEXT,
                conference_year INTEGER,
                conference_round TEXT,
                embedding VECTOR({self.dim})
            );
            """
            cur.execute(create_table_query)
            
            # Create index (IVFFlat is common, or HNSW for better performance)
            # Using HNSW for better performance/recall trade-off
            # Note: HNSW index creation might take time for large datasets
            index_name = f"{self.table_name}_embedding_idx"
            # Check if index exists to avoid error or re-creation overhead
            # Simple check: just try creating with IF NOT EXISTS if supported or catch error
            # Postgres doesn't support CREATE INDEX IF NOT EXISTS until v9.5, but standard now.
            # However, for vector index, we usually specify opclass.
            
            # We'll use L2 distance (<->) as default to match Milvus L2 setting
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name} 
                ON {self.table_name} 
                USING hnsw (embedding vector_l2_ops);
            """)

    def query_relevant_documents(self, query: str) -> list[dict]:
        query_vector = self.embedding_client.embed_query(query)
        
        # Convert list of floats to string representation for pgvector '[1.0, 2.0, ...]'
        vector_str = str(query_vector)
        
        search_query = f"""
        SELECT title, abstract, doc_id, url, conference_name, conference_year, conference_round,
               (embedding <-> %s::vector) as distance
        FROM {self.table_name}
        ORDER BY distance ASC
        LIMIT {settings.milvus_top_k};
        """
        
        with self.conn.cursor() as cur:
            cur.execute(search_query, (vector_str,))
            rows = cur.fetchall()
            
        res = []
        for row in rows:
            # row: title, abstract, doc_id, url, conf_name, conf_year, conf_round, distance
            res.append({
                "title": row[0],
                "abstract": row[1],
                "doc_id": str(row[2]),
                # "score": 1 - row[7] # Convert distance to similarity if needed, but interface just returns dicts usually
            })
        return res

    def insert_document(self, title: str, abstract: str, url: str='', conference_name: str='', conference_year: int=0, conference_round: str='all'):
        doc_vector = self.embedding_client.embed_query(f"Title: {title}\nAbstract: {abstract}")
        doc_id = str(uuid.uuid4())
        
        insert_query = f"""
        INSERT INTO {self.table_name} 
        (doc_id, title, abstract, url, conference_name, conference_year, conference_round, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        with self.conn.cursor() as cur:
            cur.execute(insert_query, (
                doc_id, 
                title, 
                abstract, 
                url, 
                conference_name, 
                conference_year, 
                conference_round, 
                str(doc_vector)
            ))

    def list_resources(self) -> list[str]:
        return [f"PostgreSQL Table: {self.table_name}"]

    def check_conference_exists(self, conference_name: str, year: int, round: str) -> bool:
        query = f"""
        SELECT 1 FROM {self.table_name}
        WHERE conference_name = %s AND conference_year = %s AND conference_round = %s
        LIMIT 1
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (conference_name, year, round))
            return cur.fetchone() is not None

    def get_existing_rounds(self, conference_name: str, year: int) -> list[str]:
        query = f"""
        SELECT DISTINCT conference_round FROM {self.table_name}
        WHERE conference_name = %s AND conference_year = %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (conference_name, year))
            rows = cur.fetchall()
        return [row[0] for row in rows if row[0]]

    def get_conference_papers(self, conference_name: str, year: int, round: str, limit: int = 10) -> list[Chunk]:
        query = f"""
        SELECT title, abstract, url 
        FROM {self.table_name}
        WHERE conference_name = %s AND conference_year = %s AND conference_round = %s
        LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (conference_name, year, round, limit))
            rows = cur.fetchall()
            
        chunks: list[Chunk] = []
        for row in rows:
            title = row[0]
            abstract = row[1]
            url = row[2]
            content = f"Title: {title}\nAbstract: {abstract}"
            metadata = {
                "title": title,
                "abstract": abstract,
                "url": url,
                "conference_name": conference_name,
                "conference_year": year,
                "conference_round": round,
            }
            chunks.append(Chunk(content=content, metadata=metadata, score=0.0))
            
        return chunks
