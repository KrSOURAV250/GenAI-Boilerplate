"""
PGVectorStore pipeline (langchain-postgres).

Docs: https://docs.langchain.com/oss/python/integrations/vectorstores/pgvectorstore
Pool defaults: pool_size=5 + max_overflow=10 → 15 concurrent; 16th waits.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.schema import CreateSchema

from common_module.vector_mill.postgres_office import PostgresSettings


class VectorStorePipeline:
    """Lazy registry: engine pool → schema/table → PGVectorStore."""

    _async_engine: AsyncEngine | None = None
    _pg_engine: PGEngine | None = None
    _store: PGVectorStore | None = None

    # --- pool / engine (docs: from_engine or from_connection_string) ---

    @classmethod
    def async_engine(cls) -> AsyncEngine:
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
        if cls._pg_engine is None:
            cls._pg_engine = PGEngine.from_engine(engine=cls.async_engine())
        return cls._pg_engine

    # --- setup (docs: CreateSchema tip + ainit_vectorstore_table + create) ---

    @classmethod
    async def setup(
        cls,
        embedding: Embeddings,
        *,
        create_schema: bool = False,
        create_table: bool = False,
    ) -> PGVectorStore:
        """Init once; later calls reuse the cached store."""
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
        async with cls.async_engine().begin() as conn:
            await conn.execute(
                CreateSchema(PostgresSettings.schema_name(), if_not_exists=True)
            )

    @classmethod
    async def init_table(cls) -> None:
        await cls.pg_engine().ainit_vectorstore_table(
            table_name=PostgresSettings.table_name(),
            vector_size=PostgresSettings.vector_size(),
            schema_name=PostgresSettings.schema_name(),
        )

    @classmethod
    def store(cls) -> PGVectorStore:
        if cls._store is None:
            raise RuntimeError("Call await VectorStorePipeline.setup(embedding) first")
        return cls._store

    # --- manage (docs: aadd_documents / aadd_texts / adelete) ---

    @classmethod
    async def add_documents(cls, docs: list[Document]) -> list[str]:
        return await cls.store().aadd_documents(docs)

    @classmethod
    async def add_texts(
        cls,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        return await cls.store().aadd_texts(texts, metadatas=metadatas, ids=ids)

    @classmethod
    async def delete(cls, ids: list[str]) -> bool | None:
        return await cls.store().adelete(ids)

    @classmethod
    async def get_by_ids(cls, ids: list[str]) -> list[Document]:
        return await cls.store().aget_by_ids(ids)

    # --- query (docs: asimilarity_search*) ---

    @classmethod
    async def similarity_search(cls, query: str, k: int = 4, **kwargs) -> list[Document]:
        return await cls.store().asimilarity_search(query, k=k, **kwargs)

    @classmethod
    async def similarity_search_with_score(
        cls, query: str, k: int = 4, **kwargs
    ) -> list[tuple[Document, float]]:
        return await cls.store().asimilarity_search_with_score(query, k=k, **kwargs)

    @classmethod
    async def similarity_search_by_vector(
        cls, embedding: list[float], k: int = 4, **kwargs
    ) -> list[Document]:
        return await cls.store().asimilarity_search_by_vector(
            embedding, k=k, **kwargs
        )

    # --- index (docs: HNSWIndex + aapply_vector_index / areindex) ---

    @classmethod
    async def apply_hnsw(cls, name: str | None = None, **kwargs) -> None:
        name = name or PostgresSettings.hnsw_index_name()
        await cls.store().aapply_vector_index(HNSWIndex(name=name, **kwargs))

    @classmethod
    async def reindex(cls, name: str | None = None) -> None:
        await cls.store().areindex(name or PostgresSettings.hnsw_index_name())

    @classmethod
    async def drop_index(cls, name: str | None = None) -> None:
        await cls.store().adrop_vector_index(
            name or PostgresSettings.hnsw_index_name()
        )

    # --- cleanup ---

    @classmethod
    async def close(cls) -> None:
        if cls._async_engine is not None:
            await cls._async_engine.dispose()
        cls._async_engine = None
        cls._pg_engine = None
        cls._store = None
