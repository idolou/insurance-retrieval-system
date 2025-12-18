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

            # Inject strict prompt for precision
            from llama_index.core import PromptTemplate

            qa_prompt_tmpl = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the query specifically and precisely.\n"
                "If the answer is finding a specific value, date, or name, provide it directly.\n"
                "If the information is not in the context, say 'Information not found'.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            self.query_engine.update_prompts(
                {
                    "response_synthesizer:text_qa_template": PromptTemplate(
                        qa_prompt_tmpl
                    )
                }
            )

        except Exception as e:
            raise NeedleAgentError(f"Agent initialization failed: {e}") from e

    def robust_query(self, query_str: str) -> str:
        """
        Query with validation.
        """
        response = self.query_engine.query(query_str)

        # Simple check: if no source nodes, we might want to inform the user
        if not response.source_nodes:
            if "not found" not in str(response).lower():
                return f"No specific information found in documents for: {query_str}"

        return str(response)

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
