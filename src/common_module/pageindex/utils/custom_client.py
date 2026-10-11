import asyncio
import concurrent.futures
import os
import uuid

import PyPDF2
from pageindex import PageIndexClient
from pageindex.client import md_to_tree
from pageindex.page_index import page_index

_DEFAULT_PAGE_INDEX_KWARGS = {
    "if_add_node_summary": "yes",
    "if_add_node_text": "yes",
    "if_add_node_id": "yes",
    "if_add_doc_description": "yes",
}


class CustomPageIndexClient(PageIndexClient):
    """
    Extends PageIndexClient to support passing all `page_index()` arguments
    without duplicating or changing the original index() logic.

    Extra kwargs are forwarded directly to `page_index()` on the PDF indexing
    path, avoiding a global monkey-patch on `pageindex.client.page_index`.
    """

    def __init__(
        self,
        toc_check_page_num: int | None = None,
        max_page_num_each_node: int | None = None,
        max_token_num_each_node: int | None = None,
        if_add_node_id: str | None = None,
        if_add_node_summary: str | None = None,
        if_add_doc_description: str | None = None,
        if_add_node_text: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._extra_page_index_kwargs = {
            k: v
            for k, v in {
                "toc_check_page_num": toc_check_page_num,
                "max_page_num_each_node": max_page_num_each_node,
                "max_token_num_each_node": max_token_num_each_node,
                "if_add_node_id": if_add_node_id,
                "if_add_node_summary": if_add_node_summary,
                "if_add_doc_description": if_add_doc_description,
                "if_add_node_text": if_add_node_text,
            }.items()
            if v is not None
        }

    def index(self, file_path: str, mode: str = "auto") -> str:
        """Index a document. Returns a document_id."""
        # Persist a canonical absolute path so workspace reloads do not
        # reinterpret caller-relative paths against the workspace directory.
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file_path)[1].lower()

        is_pdf = ext == ".pdf"
        is_md = ext in [".md", ".markdown"]

        if mode == "pdf" or (mode == "auto" and is_pdf):
            page_index_kwargs = {
                **_DEFAULT_PAGE_INDEX_KWARGS,
                **self._extra_page_index_kwargs,
            }
            result = page_index(
                doc=file_path,
                model=self.model,
                **page_index_kwargs,
            )
            # Extract per-page text so queries don't need the original PDF
            pages = []
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages, 1):
                    pages.append({"page": i, "content": page.extract_text() or ""})

            self.documents[doc_id] = {
                "id": doc_id,
                "type": "pdf",
                "path": file_path,
                "doc_name": result.get("doc_name", ""),
                "doc_description": result.get("doc_description", ""),
                "page_count": len(pages),
                "structure": result["structure"],
                "pages": pages,
            }

        elif mode == "md" or (mode == "auto" and is_md):
            coro = md_to_tree(
                md_path=file_path,
                if_thinning=False,
                if_add_node_summary="yes",
                summary_token_threshold=200,
                model=self.model,
                if_add_doc_description="yes",
                if_add_node_text="yes",
                if_add_node_id="yes",
            )
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            self.documents[doc_id] = {
                "id": doc_id,
                "type": "md",
                "path": file_path,
                "doc_name": result.get("doc_name", ""),
                "doc_description": result.get("doc_description", ""),
                "line_count": result.get("line_count", 0),
                "structure": result["structure"],
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        if self.workspace:
            self._save_doc(doc_id)
        return doc_id
