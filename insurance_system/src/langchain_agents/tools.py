from typing import List

from langchain_core.tools import Tool

from insurance_system.src.agents.needle_agent import NeedleAgent
from insurance_system.src.agents.summary_agent import SummaryAgent
from insurance_system.src.config import (HIERARCHICAL_STORAGE_DIR,
                                         SUMMARY_STORAGE_DIR)
from insurance_system.src.indices.hierarchical import \
    load_hierarchical_retriever
from insurance_system.src.langchain_agents.mcp_tools import (
    get_langchain_time_tools, get_langchain_weather_tools)


def get_langchain_tools() -> List[Tool]:
    """
    Initialize LlamaIndex agents and wrap them as LangChain tools.
    """
    # 1. Initialize LlamaIndex Components
    hierarchical_retriever = load_hierarchical_retriever(
        persist_dir=HIERARCHICAL_STORAGE_DIR
    )

    # 2. Initialize Agents
    needle_agent = NeedleAgent(hierarchical_retriever)
    summary_agent = SummaryAgent(persist_dir=SUMMARY_STORAGE_DIR)

    # 3. Wrap as LangChain Tools

    def run_needle(query: str) -> str:
        return needle_agent.robust_query(query)

    def run_summary(query: str) -> str:
        return str(summary_agent.query_engine.query(query))

    tools = [
        Tool(
            name="needle_expert",
            func=run_needle,
            description=(
                "The DEFAULT tool. Use this for retrieving specific facts, numbers, dates, costs, names, "
                "or any precise details from the claim documents. "
                "If the user asks 'what', 'when', 'who', 'how much', use this."
            ),
        ),
        Tool(
            name="summary_expert",
            func=run_summary,
            description=(
                "Use this ONLY for broad, high-level summaries of the entire claim case. "
                "Do not use this for specific questions like costs or dates. "
                "Use this for 'tell me the story', 'summarize', or 'overview'."
            ),
        ),
    ]

    tools.extend(get_langchain_time_tools())
    # tools.extend(get_langchain_weather_tools())

    # Add our custom robust weather tool
    from insurance_system.src.langchain_agents.weather_tool import \
        get_historical_weather

    tools.append(get_historical_weather)

    return tools
