import os
from typing import Any, Dict, List, Optional

from llama_index.core import (
    Document,
    StorageContext,
    SummaryIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import MetadataMode, NodeWithScore

from insurance_system.src.utils.config import LLM_MODEL, SUMMARY_STORAGE_DIR


class SummaryIndexError(Exception):
    """Base exception for summary index errors."""

    pass


def _precompute_mapreduce_summaries(
    documents: List[Document], llm: Optional[Any] = None
) -> Dict[str, str]:
    """
    Pre-compute summaries using MapReduce strategy.

    Map Phase: Summarize each document chunk individually.
    Reduce Phase: Combine chunk summaries hierarchically.

    Args:
        documents: List of documents to summarize.
        llm: Optional LLM instance for summarization.

    Returns:
        Dictionary mapping document IDs to their summaries.
    """
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI

    if llm is None:
        llm = OpenAI(model=LLM_MODEL) if not Settings.llm else Settings.llm

    # Map Phase: Summarize each document chunk
    chunk_summaries: Dict[str, str] = {}
    map_prompt_template = (
        "Summarize the following document chunk, focusing on key events, "
        "dates, entities, and important details:\n\n{text}\n\nSummary:"
    )

    print("  📝 Map Phase: Summarizing document chunks...")
    for i, doc in enumerate(documents):
        doc_id = doc.doc_id or f"doc_{i}"
        doc_text = doc.get_content(metadata_mode=MetadataMode.NONE)

        # Create summary prompt
        summary_prompt = map_prompt_template.format(text=doc_text[:4000])

        try:
            # Generate chunk summary using LLM
            response = llm.complete(summary_prompt)
            chunk_summaries[doc_id] = str(response).strip()
            if (i + 1) % 5 == 0:
                print(f"    Processed {i + 1}/{len(documents)} chunks...")
        except Exception as e:
            # Fallback: use first 500 chars if summarization fails
            chunk_summaries[doc_id] = doc_text[:500] + "..."
            print(f"    Warning: Failed to summarize chunk {doc_id}: {e}")

    # Reduce Phase: Combine chunk summaries hierarchically
    print("  🔄 Reduce Phase: Combining summaries...")
    if len(chunk_summaries) == 0:
        return {}

    # If only one document, return its summary
    if len(chunk_summaries) == 1:
        return chunk_summaries

    # Combine all chunk summaries into a single document-level summary
    all_summaries = "\n\n".join(
        [f"Chunk {i+1}:\n{summary}" for i, summary in enumerate(chunk_summaries.values())]
    )

    reduce_prompt_template = (
        "Combine the following document chunk summaries into a coherent, "
        "high-level summary of the entire document. Focus on the overall narrative, "
        "key timeline, and main themes:\n\n{summaries}\n\nCombined Summary:"
    )

    try:
        reduce_prompt = reduce_prompt_template.format(summaries=all_summaries[:8000])
        response = llm.complete(reduce_prompt)
        combined_summary = str(response).strip()

        # Store combined summary with a special key
        result: Dict[str, str] = {"_combined": combined_summary}
        # Also keep individual chunk summaries for reference
        result.update(chunk_summaries)
        return result
    except Exception as e:
        # Fallback: concatenate summaries
        print(f"    Warning: Failed to combine summaries: {e}")
        combined = "\n\n".join(chunk_summaries.values())[:2000]
        return {"_combined": combined, **chunk_summaries}


def create_summary_index(
    documents: List[Document],
    persist_dir: str = SUMMARY_STORAGE_DIR,
    llm: Optional[Any] = None,
    use_mapreduce: bool = True,
) -> SummaryIndex:
    """
    Creates a Summary Index for high-level queries with optional MapReduce pre-computation.

    Args:
        documents: List of documents to index.
        persist_dir: Directory path for persisting the index.
        llm: Optional LLM instance for summarization.
        use_mapreduce: If True, pre-compute summaries using MapReduce strategy.

    Returns:
        SummaryIndex instance.

    Raises:
        SummaryIndexError: If index creation fails.
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    try:
        # Create index from documents
        index = SummaryIndex.from_documents(documents)

        # Pre-compute summaries using MapReduce if requested
        if use_mapreduce:
            print("  🗺️  Pre-computing summaries using MapReduce strategy...")
            summaries = _precompute_mapreduce_summaries(documents, llm=llm)

            # Store summaries in index metadata
            if summaries:
                # Store in docstore metadata
                docstore = index.storage_context.docstore
                for doc_id, summary in summaries.items():
                    if doc_id in docstore.docs:
                        node = docstore.docs[doc_id]
                        if hasattr(node, "metadata"):
                            node.metadata["precomputed_summary"] = summary
                        elif isinstance(node, dict):
                            node["metadata"] = node.get("metadata", {})
                            node["metadata"]["precomputed_summary"] = summary

                # Also store in a separate metadata file for easy access
                import json

                metadata_file = os.path.join(persist_dir, "mapreduce_summaries.json")
                os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
                with open(metadata_file, "w") as f:
                    json.dump(summaries, f, indent=2)
                print(f"  ✅ Stored {len(summaries)} pre-computed summaries")

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
    persist_dir: str = SUMMARY_STORAGE_DIR,
    llm: Optional[Any] = None,
    use_precomputed: bool = True,
) -> BaseQueryEngine:
    """
    Returns a query engine that uses pre-computed MapReduce summaries or tree summarization.

    Args:
        persist_dir: Directory path for summary index storage.
        llm: Optional LLM instance for query engine.
        use_precomputed: If True, use pre-computed summaries when available.

    Returns:
        BaseQueryEngine instance.
    """
    if not os.path.exists(persist_dir):
        error_msg = f"Summary index storage directory not found: {persist_dir}"
        raise FileNotFoundError(error_msg)

    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

        # Check if pre-computed summaries exist
        summaries_file = os.path.join(persist_dir, "mapreduce_summaries.json")
        has_precomputed = use_precomputed and os.path.exists(summaries_file)

        if has_precomputed:
            # Use pre-computed summaries with a simple wrapper
            import json

            with open(summaries_file, "r") as f:
                summaries = json.load(f)

            # Get combined summary
            combined_summary = summaries.get("_combined", "")

            if combined_summary:
                # Get LLM
                from llama_index.core import Response, Settings
                from llama_index.core.schema import TextNode
                from llama_index.llms.openai import OpenAI

                if llm is None:
                    llm = Settings.llm if Settings.llm else OpenAI(model=LLM_MODEL)

                # Create a simple wrapper class that uses pre-computed summary
                class MapReduceQueryEngineWrapper(BaseQueryEngine):
                    """Simple wrapper that uses pre-computed MapReduce summaries."""

                    def __init__(self, summary_text: str, llm_instance: Any, fallback_engine: Any):
                        # Initialize with callback_manager from fallback engine or create default
                        from llama_index.core.callbacks import CallbackManager
                        callback_manager = getattr(fallback_engine, 'callback_manager', None)
                        if callback_manager is None:
                            callback_manager = CallbackManager([])
                        super().__init__(callback_manager=callback_manager)
                        self.summary_text = summary_text
                        self.llm = llm_instance
                        self.fallback_engine = fallback_engine

                    def _query(self, query_bundle: Any) -> Any:
                        """Internal query method (abstract method implementation)."""
                        # Extract query string
                        query_str = str(query_bundle.query_str) if hasattr(query_bundle, 'query_str') else str(query_bundle)

                        # Use LLM to answer query based on pre-computed summary
                        prompt = (
                            f"Based on the following pre-computed document summary, "
                            f"answer the query.\n\n"
                            f"Summary:\n{self.summary_text}\n\n"
                            f"Query: {query_str}\n\n"
                            f"Answer:"
                        )

                        try:
                            response_text = str(self.llm.complete(prompt))
                            source_node = TextNode(
                                text=self.summary_text, metadata={"type": "mapreduce_summary"}
                            )
                            return Response(response=response_text, source_nodes=[source_node])
                        except Exception:
                            # Fallback to tree_summarize if LLM call fails
                            return self.fallback_engine.query(query_str)

                    async def _aquery(self, query_bundle: Any) -> Any:
                        """Internal async query method (abstract method implementation)."""
                        # Extract query string
                        query_str = str(query_bundle.query_str) if hasattr(query_bundle, 'query_str') else str(query_bundle)

                        # Use LLM to answer query based on pre-computed summary
                        prompt = (
                            f"Based on the following pre-computed document summary, "
                            f"answer the query.\n\n"
                            f"Summary:\n{self.summary_text}\n\n"
                            f"Query: {query_str}\n\n"
                            f"Answer:"
                        )

                        try:
                            response_text = str(self.llm.complete(prompt))
                            source_node = TextNode(
                                text=self.summary_text, metadata={"type": "mapreduce_summary"}
                            )
                            return Response(response=response_text, source_nodes=[source_node])
                        except Exception:
                            # Fallback to tree_summarize if LLM call fails
                            return await self.fallback_engine.aquery(query_str)

                    def _get_prompt_modules(self) -> Dict[str, Any]:
                        """Get prompt modules (abstract method implementation)."""
                        return {}

                # Create fallback engine
                fallback_engine = index.as_query_engine(
                    response_mode="tree_summarize", use_async=True, llm=llm
                )

                # Return wrapper
                query_engine = MapReduceQueryEngineWrapper(
                    combined_summary, llm, fallback_engine
                )
                return query_engine
        else:
            # Fallback to tree_summarize (on-demand MapReduce)
            query_engine = index.as_query_engine(
                response_mode="tree_summarize", use_async=True, llm=llm
            )
            return query_engine
    except FileNotFoundError:
        raise
    except Exception as e:
        raise SummaryIndexError(f"Query engine creation failed: {e}") from e
