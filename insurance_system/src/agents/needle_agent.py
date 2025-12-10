from typing import Any, Optional

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from insurance_system.src.indices.hierarchical import \
    get_hierarchical_query_engine


class NeedleAgent:
    def __init__(
        self, retriever: AutoMergingRetriever, llm: Optional[Any] = None
    ) -> None:
        self.query_engine = get_hierarchical_query_engine(retriever, llm=llm)

    def get_tool(self) -> QueryEngineTool:
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
