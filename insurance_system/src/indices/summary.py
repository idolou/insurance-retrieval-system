import os
from typing import Any, List, Optional

from llama_index.core import (
    Document,
    StorageContext,
    SummaryIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize

from insurance_system.src.config import SUMMARY_STORAGE_DIR


def create_summary_index(
    documents: List[Document], persist_dir: str = SUMMARY_STORAGE_DIR
) -> SummaryIndex:
    """Creates a Summary Index for high-level queries."""
    # Summary index stores all nodes.
    # For efficiency in a real timeline, we might chunk first, but SummaryIndex defaults are often okay for small doc sets.
    # However, to be "MapReduce" style explicitly, we often control the query engine side more.
    # But constructing the index is standard.

    index = SummaryIndex.from_documents(documents)

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    index.storage_context.persist(persist_dir=persist_dir)

    return index


def get_summary_query_engine(
    persist_dir: str = SUMMARY_STORAGE_DIR, llm: Optional[Any] = None
) -> BaseQueryEngine:
    """Returns a query engine that uses tree summarization (MapReduce style)."""
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

    # "tree_summarize" is LlamaIndex's version of MapReduce summarization
    # (hierarchically summarizing chunks)
    return index.as_query_engine(
        response_mode="tree_summarize", use_async=True, llm=llm
    )
