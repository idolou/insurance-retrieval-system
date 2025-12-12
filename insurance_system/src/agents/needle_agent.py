from typing import Any, Optional

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from insurance_system.src.indices.hierarchical import get_hierarchical_query_engine


class NeedleAgentError(Exception):
    """Base exception for NeedleAgent errors."""

    pass


class NeedleAgent:
    """
    Agent specialized for retrieving specific facts from documents.

    Uses hierarchical indexing with auto-merging for precise fact retrieval.
    """

    def __init__(
        self, retriever: AutoMergingRetriever, llm: Optional[Any] = None
    ) -> None:
        """
        Initialize the Needle Agent.

        Args:
            retriever: AutoMergingRetriever instance for hierarchical retrieval.
            llm: Optional LLM instance for query engine.

        Raises:
            NeedleAgentError: If agent initialization fails.
        """
        try:
            self.query_engine = get_hierarchical_query_engine(retriever, llm=llm)
        except Exception as e:
            raise NeedleAgentError(f"Agent initialization failed: {e}") from e

    def get_tool(self) -> QueryEngineTool:
        """
        Get the QueryEngineTool for this agent.

        Returns:
            QueryEngineTool configured for fact retrieval.

        Raises:
            NeedleAgentError: If tool creation fails.
        """
        try:
            return QueryEngineTool(
                query_engine=self.query_engine,
                metadata=ToolMetadata(
                    name="needle_expert",
                    description=(
                        "The DEFAULT tool. Use this for retrieving specific facts, numbers, dates, costs, names, "
                        "or any precise details from the claim documents. "
                        "If the user asks 'what', 'when', 'who', 'how much', use this."
                    ),
                ),
            )
        except Exception as e:
            raise NeedleAgentError(f"Tool creation failed: {e}") from e
