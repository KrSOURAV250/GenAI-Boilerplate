"""
Async PGVectorStore pipeline for GenAI RAG ingest + retrieval.

Self-contained wrapper over langchain-postgres: shared SQLAlchemy pool,
schema/table setup, document upsert, similarity search, and HNSW indexing.
Keep the pool open for the process lifetime; call close() on exit.

Workflow:
  setup(embedding) → add_documents / add_texts → similarity_search*
  → apply_hnsw() after first bulk load if search latency needs it
  → reindex() only for optional maintenance (new inserts already update HNSW)
  → close() when the app shuts down

Notes:
  - Classmethods cache engine/store until close(); safe to call setup() again.
  - add_documents upserts when Document.id already exists.
  - Concurrent DB ops ≈ pool_size + max_overflow; further callers wait pool_timeout.
  - VECTOR_SIZE must match the embedding model; schema must exist before table init
    (use create_schema=True on first setup, or ensure_schema()).

Env (via PostgresSettings): POSTGRES_*, SCHEMA_NAME, TABLE_NAME, VECTOR_SIZE,
POSTGRES_POOL_*, HNSW_INDEX_NAME

Docs: https://docs.langchain.com/oss/python/integrations/vectorstores/pgvectorstore
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.schema import CreateSchema

from common_module.vector_mill.postgres_office import PostgresSettings


class VectorStorePipeline:
    """Factory/registry for PGEngine + PGVectorStore (schema → table → store)."""

    _async_engine: AsyncEngine | None = None
    _pg_engine: PGEngine | None = None
    _store: PGVectorStore | None = None

    # --- pool / engine ---

    @classmethod
    def async_engine(cls) -> AsyncEngine:
        """Return shared AsyncEngine pool (created once, then reused)."""
        if cls._async_engine is None:
            cls._async_engine = create_async_engine(
                PostgresSettings.connection_string(),
                pool_size=PostgresSettings.pool_size(),
                max_overflow=PostgresSettings.max_overflow(),
                pool_timeout=PostgresSettings.pool_timeout(),
            )
        return cls._async_engine

    @classmethod
    def pg_engine(cls) -> PGEngine:
        """Return shared PGEngine wrapping async_engine()."""
        if cls._pg_engine is None:
            cls._pg_engine = PGEngine.from_engine(engine=cls.async_engine())
        return cls._pg_engine

    # --- setup ---

    @classmethod
    async def setup(
        cls,
        embedding: Embeddings,
        *,
        create_schema: bool = False,
        create_table: bool = False,
    ) -> PGVectorStore:
        """Init schema/table/store once; later calls reuse the cached store."""
        if create_schema:
            await cls.ensure_schema()
        if create_table:
            await cls.init_table()
        if cls._store is None:
            cls._store = await PGVectorStore.create(
                engine=cls.pg_engine(),
                table_name=PostgresSettings.table_name(),
                schema_name=PostgresSettings.schema_name(),
                embedding_service=embedding,
            )
        return cls._store

    @classmethod
    async def ensure_schema(cls) -> None:
        """CREATE SCHEMA IF NOT EXISTS for SCHEMA_NAME (SQLAlchemy CreateSchema)."""
        async with cls.async_engine().begin() as conn:
            await conn.execute(
                CreateSchema(PostgresSettings.schema_name(), if_not_exists=True)
            )

    @classmethod
    async def init_table(cls) -> None:
        """Create the vectorstore table via PGEngine.ainit_vectorstore_table."""
        await cls.pg_engine().ainit_vectorstore_table(
            table_name=PostgresSettings.table_name(),
            vector_size=PostgresSettings.vector_size(),
            schema_name=PostgresSettings.schema_name(),
        )

    @classmethod
    def store(cls) -> PGVectorStore:
        """Return the cached store; raises if setup() was not called."""
        if cls._store is None:
            raise RuntimeError("Call await VectorStorePipeline.setup(embedding) first")
        return cls._store

    # --- manage ---

    @classmethod
    async def add_documents(cls, docs: list[Document]) -> list[str]:
        """Embed + upsert documents (same Document.id → update). → id list."""
        return await cls.store().aadd_documents(docs)

    @classmethod
    async def add_texts(
        cls,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Embed + upsert raw texts. Same id → update. → id list."""
        return await cls.store().aadd_texts(texts, metadatas=metadatas, ids=ids)

    @classmethod
    async def delete(cls, ids: list[str]) -> bool | None:
        """Delete rows by id list."""
        return await cls.store().adelete(ids)

    @classmethod
    async def get_by_ids(cls, ids: list[str]) -> list[Document]:
        """Fetch documents by id list."""
        return await cls.store().aget_by_ids(ids)

    # --- query ---

    @classmethod
    async def similarity_search(
        cls, query: str, k: int = 4, filter: dict | None = None, **kwargs
    ) -> list[Document]:
        """Natural-language similarity search → top-k Documents."""
        return await cls.store().asimilarity_search(query, k=k, filter=filter, **kwargs)

    @classmethod
    async def similarity_search_with_score(
        cls, query: str, k: int = 4, filter: dict | None = None, **kwargs
    ) -> list[tuple[Document, float]]:
        """Similarity search → list of (Document, distance/score)."""
        return await cls.store().asimilarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )

    @classmethod
    async def similarity_search_by_vector(
        cls, embedding: list[float], k: int = 4, filter: dict | None = None, **kwargs
    ) -> list[Document]:
        """Similarity search from a precomputed embedding vector → top-k Documents."""
        return await cls.store().asimilarity_search_by_vector(
            embedding, k=k, filter=filter, **kwargs
        )

    # --- index ---

    @classmethod
    async def apply_hnsw(cls, name: str | None = None, **kwargs) -> None:
        """Create HNSW index (prefer after first bulk load). Name from .env if omitted."""
        name = name or PostgresSettings.hnsw_index_name()
        await cls.store().aapply_vector_index(HNSWIndex(name=name, **kwargs))

    @classmethod
    async def reindex(cls, name: str | None = None) -> None:
        """Rebuild an existing HNSW index (optional maintenance; not required per insert)."""
        await cls.store().areindex(name or PostgresSettings.hnsw_index_name())

    @classmethod
    async def drop_index(cls, name: str | None = None) -> None:
        """Drop the HNSW index by name (default: HNSW_INDEX_NAME)."""
        await cls.store().adrop_vector_index(name or PostgresSettings.hnsw_index_name())

    # --- cleanup ---

    @classmethod
    async def close(cls) -> None:
        """Dispose the engine pool and clear cached engine/store handles."""
        if cls._async_engine is not None:
            await cls._async_engine.dispose()
        cls._async_engine = None
        cls._pg_engine = None
        cls._store = None
