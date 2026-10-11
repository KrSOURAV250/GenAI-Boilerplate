"""Shared StrOutputParser for plain-text LLM responses."""

from langchain_core.output_parsers import StrOutputParser


class StrParser:
    """Factory/registry for LangChain StrOutputParser."""

    _instance: StrOutputParser | None = None

    @classmethod
    def get(cls) -> StrOutputParser:
        """Return the shared parser (created once, then reused)."""
        if cls._instance is None:
            cls._instance = cls.create()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached parser so the next get() rebuilds it."""
        cls._instance = None

    @staticmethod
    def create() -> StrOutputParser:
        """Build a new StrOutputParser instance."""
        return StrOutputParser()
