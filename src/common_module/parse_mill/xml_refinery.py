"""Shared XMLOutputParser for XML LLM responses."""

from langchain_core.output_parsers import XMLOutputParser


class XmlParser:
    """Factory/registry for LangChain XMLOutputParser."""

    _instance: XMLOutputParser | None = None

    @classmethod
    def get(cls) -> XMLOutputParser:
        """Return the shared parser (created once, then reused)."""
        if cls._instance is None:
            cls._instance = cls.create()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached parser so the next get() rebuilds it."""
        cls._instance = None

    @staticmethod
    def create() -> XMLOutputParser:
        """Build a new XMLOutputParser instance."""
        return XMLOutputParser()
