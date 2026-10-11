"""BFS extractor for PageIndex tree structures.

Traverses a hierarchical document tree (from PageIndex) level by level,
using an LLM to filter relevant nodes at each level. Collects relevant
leaf nodes as the final output.
"""

import asyncio
from dataclasses import dataclass, field

from langchain.chat_models import BaseChatModel
from langfuse import observe
from pydantic import BaseModel

from src.model_hub import get_llm
from ...prompts.general_prompts import GeneralPrompt
from ...schemas.response_formats import NodeRelevance

MAX_NODE_EVALUATION_ATTEMPTS = 3
INITIAL_BACKOFF_SECONDS = 1.0


@dataclass
class BFSExtractionResult:
    relevant_leaves: list[dict[str, object]] = field(default_factory=list)
    failed_nodes: list[dict[str, str]] = field(default_factory=list)


class RelevantNodesExtractor:
    """Extracts relevant leaf nodes from a PageIndex document tree using BFS.

    Walks the tree level by level. At each level, all nodes are evaluated
    for relevance via a single batched LLM call. Relevant nodes with
    children are expanded into the next level; relevant leaf nodes are
    collected as results.
    """

    def __init__(self, prompt_template: str, schema: type[BaseModel]):
        llm: BaseChatModel = get_llm()
        self._structured_llm = llm.with_structured_output(schema=schema)
        self._prompt_template: str = prompt_template

    def _build_node_path(self, parent_path: str | None, title: str) -> str:
        """Builds the full document path for a node."""
        return f"{parent_path} -> {title}" if parent_path else title

    def _build_prompt(self, node: dict[str, object], parent_path: str | None) -> str:
        """Builds the LLM prompt for a single node."""
        title: str = node.get("title", "")
        path_str = self._build_node_path(parent_path=parent_path, title=title)
        return self._prompt_template.format(
            path=path_str,
            title=title,
            summary=node.get("summary", ""),
        )

    def _node_failure_info(
        self, node: dict[str, object], parent_path: str | None
    ) -> dict[str, str]:
        title: str = str(node.get("title", ""))
        return {
            "title": title,
            "path": self._build_node_path(parent_path=parent_path, title=title),
        }

    @observe
    async def _evaluate_nodes(
        self, nodes_with_paths: list[tuple[dict[str, object], str | None]]
    ) -> tuple[
        list[tuple[dict[str, object], str | None]],
        list[dict[str, str]],
    ]:
        """Evaluates a batch of nodes for relevance.

        Retries failed nodes up to MAX_NODE_EVALUATION_ATTEMPTS with
        exponential backoff, then records permanently failed nodes.
        """
        relevant: list[tuple[dict[str, object], str | None]] = list()
        pending: list[tuple[dict[str, object], str | None]] = list(nodes_with_paths)
        attempt_counts: dict[int, int] = {}
        permanently_failed: list[dict[str, str]] = list()
        batch_attempt = 0

        while pending:
            if batch_attempt > 0:
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** (batch_attempt - 1))
                await asyncio.sleep(backoff)
            prompts: list[str] = [
                self._build_prompt(node=node, parent_path=parent_path)
                for node, parent_path in pending
            ]

            responses: list[BaseModel | Exception | None] = (
                await self._structured_llm.abatch(
                    inputs=prompts, return_exceptions=True
                )
            )

            retry: list[tuple[dict[str, object], str | None]] = list()
            for (node, parent_path), response in zip(pending, responses):
                if isinstance(response, NodeRelevance):
                    if response.is_relevant:
                        relevant.append((node, parent_path))
                else:
                    node_id = id(node)
                    attempt_counts[node_id] = attempt_counts.get(node_id, 0) + 1
                    if attempt_counts[node_id] >= MAX_NODE_EVALUATION_ATTEMPTS:
                        permanently_failed.append(
                            self._node_failure_info(node=node, parent_path=parent_path)
                        )
                    else:
                        retry.append((node, parent_path))
            pending = retry
            batch_attempt += 1
        return relevant, permanently_failed

    @observe
    async def extract(self, doc_tree: list[dict[str, object]]) -> BFSExtractionResult:
        """Performs BFS extraction on the document tree.

        Args:
            doc_tree: Top-level list of nodes from PageIndex document structure.

        Returns:
            Relevant leaf nodes and any nodes that could not be evaluated.
        """
        relevant_leaves: list[dict[str, object]] = []
        failed_nodes: list[dict[str, str]] = []

        current_level: list[tuple[dict[str, object], str | None]] = [
            (node, None) for node in doc_tree
        ]

        while current_level:
            relevant, level_failed = await self._evaluate_nodes(
                nodes_with_paths=current_level
            )
            failed_nodes.extend(level_failed)

            next_level: list[tuple[dict[str, object], str | None]] = []
            for node, parent_path in relevant:
                title: str = node.get("title", "")
                node_path: str = self._build_node_path(
                    parent_path=parent_path, title=title
                )
                children: object | None = node.get("nodes")

                if children:
                    parent_start: int | None = node.get("start_index")
                    parent_end: int | None = node.get("end_index")
                    first_child_start: int | None = children[0].get("start_index")
                    last_child_end: int | None = children[-1].get("end_index")

                    # Gap at the beginning: parent has content before its first child

                    if (
                        parent_start is not None
                        and first_child_start is not None
                        and parent_start < first_child_start
                    ):
                        gap_node: dict[str, object] = {
                            "title": f"{title} (preamble)",
                            "start_index": parent_start,
                            "end_index": first_child_start - 1,
                            "is_gap_content": True,
                        }
                        relevant_leaves.append(gap_node)
                    # Gap at the end: parent has content after its last child

                    if (
                        parent_end is not None
                        and last_child_end is not None
                        and parent_end > last_child_end
                    ):
                        gap_node = {
                            "title": f"{title} (closing)",
                            "start_index": last_child_end + 1,
                            "end_index": parent_end,
                            "is_gap_content": True,
                        }
                        relevant_leaves.append(gap_node)
                    for child in children:
                        next_level.append((child, node_path))
                else:
                    relevant_leaves.append(node)
            current_level = next_level
        return BFSExtractionResult(
            relevant_leaves=relevant_leaves,
            failed_nodes=failed_nodes,
        )
