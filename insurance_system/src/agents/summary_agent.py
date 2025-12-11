from typing import Any, Optional

from llama_index.core.tools import QueryEngineTool, ToolMetadata

from insurance_system.src.config import SUMMARY_STORAGE_DIR
from insurance_system.src.indices.summary import get_summary_query_engine


class SummaryAgent:
    def __init__(
        self, persist_dir: str = SUMMARY_STORAGE_DIR, llm: Optional[Any] = None
    ) -> None:
        self.query_engine = get_summary_query_engine(persist_dir, llm=llm)

    def get_tool(self) -> QueryEngineTool:
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
