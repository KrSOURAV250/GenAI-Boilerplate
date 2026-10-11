"""Shared JsonOutputParser for JSON LLM responses."""

from langchain_core.output_parsers import JsonOutputParser


class JsonParser:
    """Factory/registry for LangChain JsonOutputParser."""

    _instance: JsonOutputParser | None = None

    @classmethod
    def get(cls) -> JsonOutputParser:
        """Return the shared parser (created once, then reused)."""
        if cls._instance is None:
            cls._instance = cls.create()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached parser so the next get() rebuilds it."""
        cls._instance = None

    @staticmethod
    def create() -> JsonOutputParser:
        """Build a new JsonOutputParser instance."""
        return JsonOutputParser()
