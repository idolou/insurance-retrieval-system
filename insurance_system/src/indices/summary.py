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


class SummaryIndexError(Exception):
    """Base exception for summary index errors."""

    pass


def create_summary_index(
    documents: List[Document], persist_dir: str = SUMMARY_STORAGE_DIR
) -> SummaryIndex:
    """
    Creates a Summary Index for high-level queries.

    Args:
        documents: List of documents to index.
        persist_dir: Directory path for persisting the index.

    Returns:
        SummaryIndex instance.

    Raises:
        SummaryIndexError: If index creation fails.
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    try:
        index = SummaryIndex.from_documents(documents)

        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir, exist_ok=True)

        try:
            index.storage_context.persist(persist_dir=persist_dir)
        except Exception as e:
            raise SummaryIndexError(f"Index persistence failed: {e}") from e

        return index
    except ValueError:
        raise
    except Exception as e:
        raise SummaryIndexError(f"Index creation failed: {e}") from e


def get_summary_query_engine(
    persist_dir: str = SUMMARY_STORAGE_DIR, llm: Optional[Any] = None
) -> BaseQueryEngine:
    """Returns a query engine that uses tree summarization (MapReduce style)."""
    if not os.path.exists(persist_dir):
        error_msg = f"Summary index storage directory not found: {persist_dir}"
        raise FileNotFoundError(error_msg)

    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

        # "tree_summarize" is LlamaIndex's version of MapReduce summarization
        # (hierarchically summarizing chunks)
        query_engine = index.as_query_engine(
            response_mode="tree_summarize", use_async=True, llm=llm
        )
        return query_engine
    except FileNotFoundError:
        raise
    except Exception as e:
        raise SummaryIndexError(f"Query engine creation failed: {e}") from e
