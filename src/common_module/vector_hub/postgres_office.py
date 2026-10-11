"""
Lazy Postgres / pgvector settings for GenAI pipelines.

Self-contained: classmethods load values from .env once, then reuse the cache.
Call reset() after changing env vars

Env: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
     SCHEMA_NAME, TABLE_NAME, VECTOR_SIZE,
     POSTGRES_POOL_SIZE, POSTGRES_MAX_OVERFLOW, POSTGRES_POOL_TIMEOUT,
     HNSW_INDEX_NAME
"""

from decouple import config


class PostgresSettings:
    """Factory/registry for Postgres connection, pool, and HNSW index settings."""

    _user: str | None = None
    _password: str | None = None
    _host: str | None = None
    _port: str | None = None
    _db: str | None = None
    _schema_name: str | None = None
    _table_name: str | None = None
    _vector_size: int | None = None
    _pool_size: int | None = None
    _max_overflow: int | None = None
    _pool_timeout: int | None = None
    _hnsw_index_name: str | None = None
    _connection_string: str | None = None

    @classmethod
    def connection_string(cls, driver: str = "asyncpg") -> str:
        """Return shared SQLAlchemy URL (built once, then reused)."""
        if cls._connection_string is None:
            cls._connection_string = (
                f"postgresql+{driver}://{cls.user()}:{cls.password()}"
                f"@{cls.host()}:{cls.port()}/{cls.db()}"
            )
        return cls._connection_string

    @classmethod
    def user(cls) -> str:
        """Return POSTGRES_USER from .env."""
        if cls._user is None:
            cls._user = config("POSTGRES_USER")
        return cls._user

    @classmethod
    def password(cls) -> str:
        """Return POSTGRES_PASSWORD from .env."""
        if cls._password is None:
            cls._password = config("POSTGRES_PASSWORD")
        return cls._password

    @classmethod
    def host(cls) -> str:
        """Return POSTGRES_HOST (default: localhost)."""
        if cls._host is None:
            cls._host = config("POSTGRES_HOST", default="localhost")
        return cls._host

    @classmethod
    def port(cls) -> str:
        """Return POSTGRES_PORT (default: 6024)."""
        if cls._port is None:
            cls._port = config("POSTGRES_PORT", default="6024")
        return cls._port

    @classmethod
    def db(cls) -> str:
        """Return POSTGRES_DB from .env."""
        if cls._db is None:
            cls._db = config("POSTGRES_DB")
        return cls._db

    @classmethod
    def schema_name(cls) -> str:
        """Return SCHEMA_NAME (default: public). Must exist before table init."""
        if cls._schema_name is None:
            cls._schema_name = config("SCHEMA_NAME", default="public")
        return cls._schema_name

    @classmethod
    def table_name(cls) -> str:
        """Return TABLE_NAME for the vector store (default: vectorstore)."""
        if cls._table_name is None:
            cls._table_name = config("TABLE_NAME", default="vectorstore")
        return cls._table_name

    @classmethod
    def vector_size(cls) -> int:
        """Return VECTOR_SIZE; must match the embedding model dimension."""
        if cls._vector_size is None:
            cls._vector_size = config("VECTOR_SIZE", default=1024, cast=int)
        return cls._vector_size

    @classmethod
    def pool_size(cls) -> int:
        """Return persistent pool size (default: 5)."""
        if cls._pool_size is None:
            cls._pool_size = config("POSTGRES_POOL_SIZE", default=5, cast=int)
        return cls._pool_size

    @classmethod
    def max_overflow(cls) -> int:
        """Return overflow connections beyond pool_size (default: 10)."""
        if cls._max_overflow is None:
            cls._max_overflow = config("POSTGRES_MAX_OVERFLOW", default=10, cast=int)
        return cls._max_overflow

    @classmethod
    def pool_timeout(cls) -> int:
        """Return seconds a waiter blocks for a free connection (default: 30)."""
        if cls._pool_timeout is None:
            cls._pool_timeout = config("POSTGRES_POOL_TIMEOUT", default=30, cast=int)
        return cls._pool_timeout

    @classmethod
    def hnsw_index_name(cls) -> str:
        """Return HNSW_INDEX_NAME used by apply_hnsw / reindex / drop_index."""
        if cls._hnsw_index_name is None:
            cls._hnsw_index_name = config("HNSW_INDEX_NAME", default="my-hnsw-index")
        return cls._hnsw_index_name

    @classmethod
    def reset(cls) -> None:
        """Clear the cache so the next accessors reload from .env."""
        cls._user = None
        cls._password = None
        cls._host = None
        cls._port = None
        cls._db = None
        cls._schema_name = None
        cls._table_name = None
        cls._vector_size = None
        cls._pool_size = None
        cls._max_overflow = None
        cls._pool_timeout = None
        cls._hnsw_index_name = None
        cls._connection_string = None
