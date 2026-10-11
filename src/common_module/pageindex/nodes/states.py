from typing import TypedDict


class DocumentToContextState(TypedDict):
    document_path: str
    document_id: str
    relevant_nodes: list[dict[str, object]]
    page_ranges_to_extract_content: list[tuple[int, int] | int]
    page_content: list[dict[str, object]]
    context: str
    extracted_rules: dict[str, object] | None
