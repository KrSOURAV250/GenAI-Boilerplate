import json

from decouple import config
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langgraph.graph import END
from langgraph.types import Command

from ..model_hub import get_llm
from ..pgidx.bfs_extractor import (
    BFSExtractionResult,
    MAX_NODE_EVALUATION_ATTEMPTS,
    RelevantNodesExtractor,
)
from ..pgidx.custom_client import CustomPageIndexClient
from ..pgidx.pgindex import extract_page_numbers, get_page_ranges
from ..prompts.general_prompts import GeneralPrompt
from ..schemas.response_formats import NodeRelevance
from .states import DocumentToContextState


def document_to_tree_node(state: DocumentToContextState):
    file_path: str = state.get("document_path")
    client = CustomPageIndexClient(
        model=config("PGIDX_MODEL_NAME"),
        api_key=config("PGIDX_API_KEY"),
        workspace=config("PGIDX_WORKSPACE", default="pageindex-results"),
        toc_check_page_num=config("TOC_CHECK_PAGE_NUM", default=20, cast=int),
    )
    document_id: str = client.index(file_path=file_path)
    to_update: dict[str, str] = {"document_id": document_id}
    goto = "extract_relevant_sections_node"
    return Command(update=to_update, goto=goto)


async def extract_relevant_sections_node(state: DocumentToContextState):
    document_id: str = state.get("document_id", "")
    client = CustomPageIndexClient(
        workspace=config("PGIDX_WORKSPACE", default="pageindex-results"),
    )
    doc_structure_str = client.get_document_structure(doc_id=document_id)

    doc_structure = json.loads(s=doc_structure_str)

    extractor = RelevantNodesExtractor(
        prompt_template=GeneralPrompt.get_node_relevance_prompt(), schema=NodeRelevance
    )
    extraction_result: BFSExtractionResult = await extractor.extract(
        doc_tree=doc_structure
    )

    if extraction_result.failed_nodes:
        to_update = None
        goto = "handle_failed_nodes_node"
    elif extraction_result.relevant_leaves:
        to_update: dict[str, list[dict[str, object]]] = {
            "relevant_nodes": extraction_result.relevant_leaves
        }
        goto = "get_page_numbers_node"
    else:
        to_update = None
        goto = "handle_no_rules_found_node"
    return Command(update=to_update, goto=goto)


def handle_no_rules_found_node(state: DocumentToContextState):
    """Handles the case when no relevant nodes are found."""
    to_update = {
        "extracted_rules": {
            "success": False,
            "message": "No Rules were found in the document.",
        }
    }
    goto = END
    return Command(update=to_update, goto=goto)


def handle_failed_nodes_node(state: DocumentToContextState):
    """Handles nodes that failed relevance evaluation even after retries."""
    to_update = {
        "extracted_rules": {
            "success": False,
            "message": (
                "Failed to evaluate document sections after "
                f"{MAX_NODE_EVALUATION_ATTEMPTS} attempts. "
                "Please retry with a better LLM model or something else."
            ),
        }
    }
    goto = END
    return Command(update=to_update, goto=goto)


def get_page_numbers_node(state: DocumentToContextState):
    """Extracts the page numbers from the relevant nodes."""
    relevant_nodes: list[dict[str, object]] = state.get("relevant_nodes", [])

    document_id: str = state.get("document_id", "")
    client = CustomPageIndexClient(
        workspace=config("PGIDX_WORKSPACE", default="pageindex-results")
    )
    document_str: str = client.get_document(doc_id=document_id)
    document: dict[str, object] = json.loads(s=document_str)
    page_count: object | None = document.get("page_count")

    page_numbers: list[int] = extract_page_numbers(
        relevant_nodes=relevant_nodes, page_count=page_count, confidence=False
    )
    page_ranges: list[tuple[int, int] | int] = get_page_ranges(page_nums=page_numbers)
    to_update: dict[str, list[tuple[int, int] | int]] = {
        "page_ranges_to_extract_content": page_ranges
    }
    goto = "extract_content_node"
    return Command(update=to_update, goto=goto)


def extract_content_node(state: DocumentToContextState):
    """Extracts the content from the document based on the page ranges."""
    document_id: str = state.get("document_id", "")
    page_ranges: list[tuple[int, int] | int] = state.get(
        "page_ranges_to_extract_content", []
    )
    client = CustomPageIndexClient(
        workspace=config("PGIDX_WORKSPACE", default="pageindex-results"),
    )
    pages_str: str = ""
    for i in page_ranges:
        if isinstance(i, tuple):
            pages_str += f"{i[0]}-{i[1]},"
        elif isinstance(i, int):
            pages_str += f"{i},"
    if pages_str.endswith(","):
        pages_str = pages_str[:-1]
    page_content_str: str = client.get_page_content(doc_id=document_id, pages=pages_str)
    page_content: list[dict[str, object]] = json.loads(s=page_content_str)
    to_update: dict[str, list[dict[str, object]]] = {"page_content": page_content}
    goto = "make_context_node"
    return Command(update=to_update, goto=goto)


def make_context_node(state: DocumentToContextState):
    page_content: list[dict[str, object]] = state.get("page_content", [])
    context = ""
    for page in page_content:
        context += f"\n{page.get("content")}\n"
    to_update: dict[str, str] = {"context": context}
    goto = "extract_rules_node"
    return Command(update=to_update, goto=goto)
