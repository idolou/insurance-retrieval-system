from typing import Any, List, Optional

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.llms.openai import OpenAI

from insurance_system.src.agents.needle_agent import NeedleAgent
from insurance_system.src.agents.simple_agent import SimpleAgent
from insurance_system.src.agents.summary_agent import SummaryAgent
from insurance_system.src.mcp.tools import get_mcp_tools
from insurance_system.src.prompts import MANAGER_SYSTEM_PROMPT


class ManagerAgent:
    def __init__(
        self,
        hierarchical_retriever: AutoMergingRetriever,
        summary_persist_dir: str = "storage/summary",
        llm: Optional[Any] = None,
    ) -> None:
        self.llm = llm or OpenAI(model="gpt-4o")

        # Initialize Sub-Agents
        self.needle_agent = NeedleAgent(hierarchical_retriever, llm=self.llm)
        self.summary_agent = SummaryAgent(summary_persist_dir, llm=self.llm)

        # Get Tools
        self.tools: List[Any] = [self.needle_agent.get_tool(), self.summary_agent.get_tool()]

        # Add MCP Tools (if any)
        self.tools.extend(get_mcp_tools())

        # Initialize the Manager (Router)
        self.agent = SimpleAgent(
            tools=self.tools,
            llm=self.llm,
            system_prompt=MANAGER_SYSTEM_PROMPT.template,
        )

    async def aquery(self, question: str) -> Any:
        response = await self.agent.achat(question)

        # Check if the response has 'sources' (AgentChatResponse) and if those sources have 'raw_output'
        # This is a best-effort attempt to expose the retrieval nodes to the top level
        if hasattr(response, "sources") and not hasattr(response, "source_nodes"):
            all_nodes = []
            for tool_output in response.sources:
                # For QueryEngineTool, raw_output is usually the Response object
                if hasattr(tool_output, "raw_output") and hasattr(
                    tool_output.raw_output, "source_nodes"
                ):
                    all_nodes.extend(tool_output.raw_output.source_nodes)

            if all_nodes:
                response.source_nodes = all_nodes

        return response

    def query(self, question: str) -> Any:
        response = self.agent.chat(question)

        if hasattr(response, "sources") and not hasattr(response, "source_nodes"):
            all_nodes = []
            for tool_output in response.sources:
                if hasattr(tool_output, "raw_output") and hasattr(
                    tool_output.raw_output, "source_nodes"
                ):
                    all_nodes.extend(tool_output.raw_output.source_nodes)

            if all_nodes:
                response.source_nodes = all_nodes

        return response
