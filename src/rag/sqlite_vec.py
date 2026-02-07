import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from logging_config import logger
from models import get_llm_by_usage
from rag.retriever import RAG, Chunk
from settings import settings


class SQLiteVecProvider(RAG):
    def __init__(self) -> None:
        super().__init__()
        self.db_path = settings.sqlite_path
        self.table = settings.sqlite_table
        self.dim = settings.embedding_dim
        self.top_k = getattr(settings, "rag_top_k", None) or settings.milvus_top_k

        # Field names (keep aligned with settings for compatibility)
        self.doc_id_field = settings.milvus_doc_id_field
        self.chunk_id_field = settings.milvus_chunk_id_field
        self.section_category_field = settings.milvus_section_category_field
        self.conference_name_field = settings.milvus_conference_name_field
        self.conference_year_field = settings.milvus_conference_year_field
        self.conference_round_field = settings.milvus_conference_round_field
        self.title_field = settings.milvus_title_field
        self.text_field = settings.milvus_text_field
        self.url_field = settings.milvus_url_field
        self.pdf_url_field = settings.milvus_pdf_url_field
        self.parent_section_field = settings.milvus_parent_section_field
        self.page_number_field = settings.milvus_page_number_field
        self.vector_field = settings.milvus_vector_field

        # Distance metric
        self.distance_metric = (
            settings.sqlite_distance_metric
            or settings.milvus_vector_index_metric_type
            or ""
        ).strip()

        # Embedding model configuration
        self.embedding_client = get_llm_by_usage("contextual", model_type="embedding")

        self._connect()

    # ============== Connection / Setup ==============
    def _connect(self) -> None:
        db_path = Path(self.db_path).expanduser()
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._load_vec_extension()
        self._ensure_table_exists()

    def _load_vec_extension(self) -> None:
        try:
            import sqlite_vec
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sqlite-vec is not installed. Install with `uv add --optional sqlite sqlite-vec`."
            ) from exc

        try:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
        except (AttributeError, sqlite3.OperationalError) as exc:
            raise RuntimeError(
                "Failed to load sqlite-vec extension. "
                "If you're on macOS, you may need a Python build that allows "
                "extension loading (e.g., python.org installer or pysqlite3-binary)."
            ) from exc
        finally:
            try:
                self.conn.enable_load_extension(False)
            except sqlite3.OperationalError:
                # Some builds may not allow toggling after failure; ignore.
                pass

    def _ensure_table_exists(self) -> None:
        columns = [
            f"{self.vector_field} float[{self.dim}]",
            f"{self.doc_id_field} TEXT",
            f"{self.chunk_id_field} INTEGER",
            f"{self.section_category_field} INTEGER",
            f"{self.conference_name_field} TEXT",
            f"{self.conference_year_field} INTEGER",
            f"{self.conference_round_field} TEXT",
            f"+{self.title_field} TEXT",
            f"+{self.text_field} TEXT",
            f"+{self.url_field} TEXT",
            f"+{self.pdf_url_field} TEXT",
            f"+{self.parent_section_field} TEXT",
            f"+{self.page_number_field} INTEGER",
        ]

        metric = self.distance_metric.upper()
        if metric in {"COSINE"}:
            columns.append("distance_metric=cosine")
        elif metric and metric not in {"L2"}:
            logger.warning("Unsupported sqlite-vec metric '%s', falling back to L2.", metric)

        sql = f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} USING vec0({', '.join(columns)})"
        self.conn.execute(sql)
        self.conn.commit()

    # ============== Helpers ==============
    def _vector_to_sql(self, vector: list[float]) -> str:
        return json.dumps(vector)

    def _knn_search(self, query_vector: list[float], k: int, filters: list[tuple[str, list]] | None = None):
        where_clauses = [f"{self.vector_field} MATCH ?", "k = ?"]
        params: list = [self._vector_to_sql(query_vector), k]

        if filters:
            for clause, values in filters:
                where_clauses.append(clause)
                params.extend(values)

        sql = f"""
        SELECT
            {self.doc_id_field},
            {self.chunk_id_field},
            {self.section_category_field},
            {self.conference_name_field},
            {self.conference_year_field},
            {self.conference_round_field},
            {self.title_field},
            {self.text_field},
            {self.url_field},
            {self.pdf_url_field},
            {self.parent_section_field},
            {self.page_number_field},
            distance
        FROM {self.table}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY distance
        """

        cur = self.conn.execute(sql, params)
        return cur.fetchall()

    def _row_to_hit(self, row: sqlite3.Row) -> dict:
        return {
            "title": row[self.title_field] or "",
            "abstract": row[self.text_field] or "",
            "doc_id": row[self.doc_id_field] or "",
            "url": row[self.url_field] or "",
            "pdf_url": row[self.pdf_url_field] or "",
            "conference_name": row[self.conference_name_field] or "",
            "conference_year": row[self.conference_year_field] or 0,
            "conference_round": row[self.conference_round_field] or "",
            "section_category": row[self.section_category_field] or 0,
            "parent_section": row[self.parent_section_field] or "",
            "page_number": row[self.page_number_field] or 0,
            "score": row["distance"] if "distance" in row.keys() else 0.0,
            "chunk_id": row[self.chunk_id_field] if self.chunk_id_field in row.keys() else -1,
        }

    # ============== Core RAG Interface ==============
    def query_relevant_documents(self, query: str):
        query_vector = self.embedding_client.embed_query(query)
        rows = self._knn_search(query_vector, self.top_k)
        return [self._row_to_hit(r) for r in rows]

    def insert_document(
        self,
        title: str,
        abstract: str,
        url: str = "",
        pdf_url: str = "",
        conference_name: str = "",
        conference_year: int = 0,
        conference_round: str = "all",
    ) -> str:
        doc_vector = self.embedding_client.embed_query(f"Title: {title}\nAbstract: {abstract}")
        doc_id = str(uuid4())

        sql = f"""
        INSERT INTO {self.table} (
            {self.vector_field},
            {self.doc_id_field},
            {self.chunk_id_field},
            {self.section_category_field},
            {self.conference_name_field},
            {self.conference_year_field},
            {self.conference_round_field},
            {self.title_field},
            {self.text_field},
            {self.url_field},
            {self.pdf_url_field},
            {self.parent_section_field},
            {self.page_number_field}
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self.conn.execute(
            sql,
            (
                self._vector_to_sql(doc_vector),
                doc_id,
                -1,
                0,
                conference_name,
                conference_year,
                conference_round,
                title,
                abstract,
                url,
                pdf_url,
                "",
                1,
            ),
        )
        self.conn.commit()
        return doc_id

    def list_resources(self) -> list[str]:
        return [f"SQLite Vec Table: {self.table} ({self.db_path})"]

    def check_conference_exists(self, conference_name: str, year: int, round: str) -> bool:
        sql = f"""
        SELECT 1 FROM {self.table}
        WHERE {self.conference_name_field} = ?
          AND {self.conference_year_field} = ?
          AND {self.conference_round_field} = ?
          AND {self.chunk_id_field} = -1
        LIMIT 1
        """
        cur = self.conn.execute(sql, (conference_name, year, round))
        return cur.fetchone() is not None

    def get_existing_rounds(self, conference_name: str, year: int) -> list[str]:
        sql = f"""
        SELECT DISTINCT {self.conference_round_field}
        FROM {self.table}
        WHERE {self.conference_name_field} = ?
          AND {self.conference_year_field} = ?
          AND {self.chunk_id_field} = -1
        """
        cur = self.conn.execute(sql, (conference_name, year))
        return [row[0] for row in cur.fetchall() if row[0]]

    def get_conference_papers(self, conference_name: str, year: int, round: str, limit: int = 10) -> list[Chunk]:
        sql = f"""
        SELECT {self.title_field}, {self.text_field}, {self.url_field}, {self.pdf_url_field}
        FROM {self.table}
        WHERE {self.conference_name_field} = ?
          AND {self.conference_year_field} = ?
          AND {self.conference_round_field} = ?
          AND {self.chunk_id_field} = -1
        LIMIT ?
        """
        cur = self.conn.execute(sql, (conference_name, year, round, limit))
        rows = cur.fetchall()

        chunks: list[Chunk] = []
        for r in rows:
            title = r[self.title_field] or ""
            abstract = r[self.text_field] or ""
            url = r[self.url_field] or ""
            pdf_url = r[self.pdf_url_field] or ""
            content = f"Title: {title}\nAbstract: {abstract}"
            metadata = {
                "title": title,
                "abstract": abstract,
                "url": url,
                "pdf_url": pdf_url,
                "conference_name": conference_name,
                "conference_year": year,
                "conference_round": round,
            }
            chunks.append(Chunk(content=content, metadata=metadata, score=0.0))

        return chunks

    def insert_paper_chunks(self, doc_id: str, chunks: list[dict], paper_title: str = ""):
        if not chunks:
            return

        data_rows = []
        for chunk in chunks:
            if "chunk_text" in chunk:
                text = chunk["chunk_text"]
                contextual_prefix = chunk.get("contextual_prefix", "")
                if "text_for_embedding" in chunk:
                    embed_text = chunk["text_for_embedding"]
                elif contextual_prefix:
                    embed_text = f"{contextual_prefix}\n\n{text}"
                else:
                    embed_text = f"Title: {paper_title}\nContent: {text}"

                chunk_index = chunk.get("id", chunk.get("chunk_index", 0))
                section_category = chunk.get("section_category", 0)
                parent_section = chunk.get("parent_section", "")
                page_number = chunk.get("page_number", 1)
            else:
                text = chunk["text"]
                section_title = chunk.get("section_title", "")
                embed_text = f"Title: {paper_title}\nSection: {section_title}\nContent: {text}"

                chunk_index = chunk["chunk_index"]
                section_category = chunk["section_category"]
                parent_section = chunk["parent_section"]
                page_number = chunk["page_number"]

            vector = self.embedding_client.embed_query(embed_text)

            data_rows.append(
                (
                    self._vector_to_sql(vector),
                    doc_id,
                    chunk_index,
                    section_category,
                    "",
                    0,
                    "",
                    paper_title,
                    text,
                    "",
                    "",
                    parent_section,
                    page_number,
                )
            )

        sql = f"""
        INSERT INTO {self.table} (
            {self.vector_field},
            {self.doc_id_field},
            {self.chunk_id_field},
            {self.section_category_field},
            {self.conference_name_field},
            {self.conference_year_field},
            {self.conference_round_field},
            {self.title_field},
            {self.text_field},
            {self.url_field},
            {self.pdf_url_field},
            {self.parent_section_field},
            {self.page_number_field}
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        batch_size = 100
        for i in range(0, len(data_rows), batch_size):
            batch = data_rows[i : i + batch_size]
            self.conn.executemany(sql, batch)
            self.conn.commit()

    def get_context_window(self, doc_id: str, center_chunk_index: int, window_size: int = 1) -> str:
        start_idx = max(0, center_chunk_index - window_size)
        end_idx = center_chunk_index + window_size

        sql = f"""
        SELECT {self.text_field}, {self.chunk_id_field}
        FROM {self.table}
        WHERE {self.doc_id_field} = ?
          AND {self.chunk_id_field} >= ?
          AND {self.chunk_id_field} <= ?
        ORDER BY {self.chunk_id_field}
        """
        cur = self.conn.execute(sql, (doc_id, start_idx, end_idx))
        rows = cur.fetchall()

        return "\n\n".join([r[self.text_field] or "" for r in rows])

    def search_by_section(
        self,
        query: str,
        doc_id: str | None = None,
        section_category: int | None = None,
        k: int = 5,
    ) -> list[dict]:
        filters: list[tuple[str, list]] = []
        if doc_id:
            filters.append((f"{self.doc_id_field} = ?", [doc_id]))
        if section_category is not None:
            filters.append((f"{self.section_category_field} = ?", [section_category]))

        query_vector = self.embedding_client.embed_query(query)
        rows = self._knn_search(query_vector, k, filters=filters)
        return [self._row_to_hit(r) for r in rows]

    def search_abstracts(self, query: str, k: int = 5) -> list[dict]:
        query_vector = self.embedding_client.embed_query(query)

        rows = self._knn_search(
            query_vector,
            k,
            filters=[(f"{self.section_category_field} = ?", [0])],
        )
        paper_rows = self._knn_search(
            query_vector,
            k,
            filters=[(f"{self.chunk_id_field} = ?", [-1])],
        )

        merged: dict[str, sqlite3.Row] = {}
        for r in rows + paper_rows:
            doc_id = r[self.doc_id_field] or ""
            if not doc_id:
                continue
            if doc_id not in merged or r["distance"] < merged[doc_id]["distance"]:
                merged[doc_id] = r

        sorted_rows = sorted(merged.values(), key=lambda r: r["distance"])
        hits = [self._row_to_hit(r) for r in sorted_rows[:k]]

        # Deduplicate to match Milvus behavior
        seen = set()
        deduped = []
        for h in hits:
            if h["doc_id"] in seen:
                continue
            seen.add(h["doc_id"])
            deduped.append(h)
        return deduped

    def get_paper_introduction(self, doc_id: str) -> str:
        sql = f"""
        SELECT {self.text_field}, {self.chunk_id_field}
        FROM {self.table}
        WHERE {self.doc_id_field} = ?
          AND {self.section_category_field} = 1
        ORDER BY {self.chunk_id_field}
        LIMIT 10
        """
        cur = self.conn.execute(sql, (doc_id,))
        rows = cur.fetchall()
        if not rows:
            return ""

        intro_text = "\n\n".join([r[self.text_field] or "" for r in rows])
        if len(intro_text) > 1500:
            intro_text = intro_text[:1500] + "..."
        return intro_text

    # ============== Lazy Load PDF ==============
    def check_pdf_chunks_exist(self, doc_id: str) -> bool:
        sql = f"""
        SELECT 1 FROM {self.table}
        WHERE {self.doc_id_field} = ?
          AND {self.chunk_id_field} >= 0
        LIMIT 1
        """
        cur = self.conn.execute(sql, (doc_id,))
        return cur.fetchone() is not None

    def get_paper_metadata(self, doc_id: str) -> dict | None:
        sql = f"""
        SELECT
            {self.title_field},
            {self.text_field},
            {self.url_field},
            {self.pdf_url_field},
            {self.conference_name_field},
            {self.conference_year_field},
            {self.conference_round_field}
        FROM {self.table}
        WHERE {self.doc_id_field} = ?
          AND {self.chunk_id_field} = -1
        LIMIT 1
        """
        cur = self.conn.execute(sql, (doc_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "doc_id": doc_id,
            "title": row[self.title_field] or "",
            "abstract": row[self.text_field] or "",
            "url": row[self.url_field] or "",
            "pdf_url": row[self.pdf_url_field] or "",
            "conference_name": row[self.conference_name_field] or "",
            "conference_year": row[self.conference_year_field] or 0,
            "conference_round": row[self.conference_round_field] or "",
        }

    def get_papers_metadata_batch(self, doc_ids: list[str]) -> list[dict]:
        if not doc_ids:
            return []

        placeholders = ", ".join(["?"] * len(doc_ids))
        sql = f"""
        SELECT
            {self.doc_id_field},
            {self.title_field},
            {self.text_field},
            {self.url_field},
            {self.pdf_url_field},
            {self.conference_name_field},
            {self.conference_year_field},
            {self.conference_round_field}
        FROM {self.table}
        WHERE {self.chunk_id_field} = -1
          AND {self.doc_id_field} IN ({placeholders})
        """
        cur = self.conn.execute(sql, doc_ids)
        rows = cur.fetchall()

        return [
            {
                "doc_id": r[self.doc_id_field] or "",
                "title": r[self.title_field] or "",
                "abstract": r[self.text_field] or "",
                "url": r[self.url_field] or "",
                "pdf_url": r[self.pdf_url_field] or "",
                "conference_name": r[self.conference_name_field] or "",
                "conference_year": r[self.conference_year_field] or 0,
                "conference_round": r[self.conference_round_field] or "",
            }
            for r in rows
        ]
