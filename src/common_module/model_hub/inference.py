"""
LLM invoke helpers for GenAI pipelines (text + structured output).

Uses LLMRegistry + StrParser by default. Structured calls fall back to
OutputFixingParser when with_structured_output fails.
"""

from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from common_module.model_hub.llm_factory import ChatOpenAI, LLMRegistry
from common_module.parse_mill.str_refinery import StrOutputParser, StrParser


class LLMRunner:
    """Run LLM calls: sync/async text, batch, and structured Pydantic output."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        parser: StrOutputParser | None = None,
    ) -> None:
        """Wire llm/parser; defaults to LLMRegistry.get_default() + StrParser.get()."""
        self.llm = llm or LLMRegistry.get_default()
        self.parser = parser or StrParser.get()

    def invoke(self, prompt: str | list[BaseMessage]) -> str:
        """Sync call → parsed string."""
        response = self.llm.invoke(prompt)
        return self.parser.invoke(input=response)

    async def ainvoke(self, prompt: str | list[BaseMessage]) -> str:
        """Async call → parsed string."""
        response = await self.llm.ainvoke(prompt)
        return await self.parser.ainvoke(input=response)

    async def abatch(
        self, prompts: list[str | list[BaseMessage]]
    ) -> list[str | Exception]:
        """Async batch → list of parsed strings (or Exception per failed item)."""
        responses: list[AIMessage | Exception] = await self.llm.abatch(
            prompts, return_exceptions=True
        )

        results: list[str | Exception] = [
            r if isinstance(r, Exception) else "" for r in responses  # temp
        ]

        ok_indexes: list[int] = [
            i for i, r in enumerate(responses) if not isinstance(r, Exception)
        ]
        ok_inputs: list[AIMessage] = [responses[i] for i in ok_indexes]

        if ok_inputs:
            parsed: list[str | Exception] = await self.parser.abatch(
                ok_inputs, return_exceptions=True
            )
            for i, item in zip(ok_indexes, parsed):
                results[i] = item

        return results

    async def structured_ainvoke(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        """Async call → Pydantic model; repairs output if structured path fails."""
        structured_llm = self.llm.with_structured_output(schema=schema)

        try:
            response = await structured_llm.ainvoke(input=prompt)
        except Exception:
            parser = PydanticOutputParser(pydantic_object=schema)
            fixing_parser = OutputFixingParser.from_llm(llm=self.llm, parser=parser)
            fallback_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"
            raw_response = await self.llm.ainvoke(input=fallback_prompt)
            response = await fixing_parser.ainvoke(input=raw_response)

        return response

    async def structured_abatch(
        self,
        prompts: list[str],
        schema: type[BaseModel],
    ) -> list[BaseModel | Exception]:
        """Async batch → Pydantic models; retries only failed items via fallback."""
        structured_llm = self.llm.with_structured_output(schema=schema)

        try:
            results: list[BaseModel | Exception] = await structured_llm.abatch(
                prompts, return_exceptions=True
            )
        except Exception:
            # whole structured path failed → fallback for every prompt
            return await self._structured_fallback_abatch(prompts, schema)

        fail_indexes: list[int] = [
            i for i, r in enumerate(results) if isinstance(r, Exception)
        ]
        if not fail_indexes:
            return results

        # only retry the failed items
        fallback: list[BaseModel | Exception] = await self._structured_fallback_abatch(
            [prompts[i] for i in fail_indexes], schema
        )
        for i, item in zip(fail_indexes, fallback):
            results[i] = item

        return results

    async def _structured_fallback_abatch(
        self,
        prompts: list[str],
        schema: type[BaseModel],
    ) -> list[BaseModel | Exception]:
        """Fallback batch parse using format instructions + OutputFixingParser."""
        parser: PydanticOutputParser[BaseModel] = PydanticOutputParser(
            pydantic_object=schema
        )
        fixing_parser: object = OutputFixingParser.from_llm(llm=self.llm, parser=parser)
        format_instructions: str = parser.get_format_instructions()

        fallback_prompts: list[str] = [f"{p}\n\n{format_instructions}" for p in prompts]
        raw_responses: list[AIMessage | Exception] = await self.llm.abatch(
            fallback_prompts, return_exceptions=True
        )

        results: list[BaseModel | Exception] = [
            r if isinstance(r, Exception) else None  # type: ignore[list-item]
            for r in raw_responses
        ]

        ok_indexes: list[int] = [
            i for i, r in enumerate(raw_responses) if not isinstance(r, Exception)
        ]
        ok_inputs: list[AIMessage | Exception] = [raw_responses[i] for i in ok_indexes]

        if ok_inputs:
            parsed: list[BaseModel | Exception] = await fixing_parser.abatch(
                ok_inputs, return_exceptions=True
            )
            for i, item in zip(ok_indexes, parsed):
                results[i] = item

        return results

