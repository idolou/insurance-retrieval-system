from typing import Any, Optional

from llama_index.core.tools import QueryEngineTool, ToolMetadata

from insurance_system.src.config import SUMMARY_STORAGE_DIR
from insurance_system.src.indices.summary import get_summary_query_engine


class SummaryAgentError(Exception):
    """Base exception for SummaryAgent errors."""

    pass


class SummaryAgent:
    """
    Agent specialized for generating high-level summaries and timelines.

    Uses summary indexing with tree summarization for document-wide synthesis.
    """

    def __init__(
        self, persist_dir: str = SUMMARY_STORAGE_DIR, llm: Optional[Any] = None
    ) -> None:
        """
        Initialize the Summary Agent.

        Args:
            persist_dir: Directory path for summary index storage.
            llm: Optional LLM instance for query engine.

        Raises:
            SummaryAgentError: If agent initialization fails.
        """
        try:
            self.query_engine = get_summary_query_engine(persist_dir, llm=llm)
        except Exception as e:
            raise SummaryAgentError(f"Agent initialization failed: {e}") from e

    def get_tool(self) -> QueryEngineTool:
        """
        Get the QueryEngineTool for this agent.

        Returns:
            QueryEngineTool configured for summary generation.

        Raises:
            SummaryAgentError: If tool creation fails.
        """
        try:
            return QueryEngineTool(
                query_engine=self.query_engine,
                metadata=ToolMetadata(
                    name="summary_expert",
                    description=(
                        "Use this ONLY for broad, high-level summaries of the entire claim case. "
                        "Do not use this for specific questions like costs or dates. "
                        "Use this for 'tell me the story', 'summarize', or 'overview'."
                    ),
                ),
            )
        except Exception as e:
            raise SummaryAgentError(f"Tool creation failed: {e}") from e
