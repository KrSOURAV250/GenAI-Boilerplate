"""
Create and cache ChatOpenAI clients for GenAI pipelines.

Env: OPENAI_API_KEY, DEFAULT_LLM_NAME, DEFAULT_REASONING_EFFORT
"""

from decouple import config
from langchain_openai import ChatOpenAI
from langfuse import observe


class LLMRegistry:
    """Factory/registry for LangChain ChatOpenAI clients."""

    _default_llm: ChatOpenAI | None = None

    @classmethod
    def get_default(cls) -> ChatOpenAI:
        """Return the shared default LLM (created once, then reused)."""
        if cls._default_llm is None:
            cls._default_llm = cls.get_llm()
        return cls._default_llm

    @classmethod
    def reset(cls) -> None:
        """Clear the cached default so the next get_default() rebuilds it."""
        cls._default_llm = None

    @staticmethod
    @observe
    def get_llm(
        model_name: str | None = None,
        temperature: float = 0.2,
        reasoning_effort: str = config("DEFAULT_REASONING_EFFORT"),
        api_key: str | None = None,
    ) -> ChatOpenAI:
        """Build a ChatOpenAI client; omitted args fall back to .env defaults."""
        if not model_name:
            model_name = config("DEFAULT_LLM_NAME")
        if not api_key:
            api_key = config("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            reasoning={"effort": reasoning_effort},
        )
