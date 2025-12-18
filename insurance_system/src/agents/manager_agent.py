import logging
from typing import Any, List, Optional

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.llms.openai import OpenAI

from insurance_system.src.agents.needle_agent import NeedleAgent
from insurance_system.src.agents.simple_agent import SimpleAgent
from insurance_system.src.agents.summary_agent import SummaryAgent
from insurance_system.src.config import LLM_MODEL
from insurance_system.src.mcp.tools import get_math_tools, get_time_tools
from insurance_system.src.prompts import MANAGER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ManagerAgentError(Exception):
    """Base exception for ManagerAgent errors."""

    pass


class ManagerAgent:
    def _attach_source_nodes(self, response: Any) -> None:
        """Attach source_nodes from tool outputs to response."""
        if hasattr(response, "sources") and not hasattr(response, "source_nodes"):
            all_nodes = [
                node
                for tool_output in response.sources
                if hasattr(tool_output, "raw_output")
                and hasattr(tool_output.raw_output, "source_nodes")
                for node in tool_output.raw_output.source_nodes
            ]
            if all_nodes:
                response.source_nodes = all_nodes

    def _log_tools_called(self, response: Any) -> None:
        """Log which tools were called during the query."""
        if hasattr(response, "sources") and response.sources:
            tool_names = [source.tool_name for source in response.sources]
            logger.info("Tools called: %s", tool_names)
            print(f"Tools called: {tool_names}")

    def __init__(
        self,
        hierarchical_retriever: AutoMergingRetriever,
        summary_persist_dir: str = "storage/summary",
        llm: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Manager Agent.

        Args:
            hierarchical_retriever: The hierarchical retriever for fact queries.
            summary_persist_dir: Directory path for summary index storage.
            llm: Optional[Any] = None. Defaults to OpenAI with configured model.

        Raises:
            ManagerAgentError: If agent initialization fails.
        """
        try:
            self.llm = llm or OpenAI(model=LLM_MODEL)

            # Initialize Sub-Agents
            self.needle_agent = NeedleAgent(hierarchical_retriever, llm=self.llm)
            self.summary_agent = SummaryAgent(summary_persist_dir, llm=self.llm)

            # Get Tools
            self.tools: List[Any] = [
                self.needle_agent.get_tool(),
                self.summary_agent.get_tool(),
            ]

            # Add MCP Tools (Time, Math)
            try:
                time_tools = get_time_tools()
                math_tools = get_math_tools()
                self.tools.extend(time_tools)
                self.tools.extend(math_tools)
                logger.info("Loaded %d MCP tools", len(time_tools) + len(math_tools))
            except Exception as e:
                logger.warning("MCP tools unavailable: %s", e)

            logger.info(
                "Total tools available: %s", [t.metadata.name for t in self.tools]
            )

            # Initialize the Manager (Router)
            self.agent = SimpleAgent(
                tools=self.tools,
                llm=self.llm,
                system_prompt=MANAGER_SYSTEM_PROMPT.template,
            )
        except Exception as e:
            raise ManagerAgentError(f"Agent initialization failed: {e}") from e

    async def aquery(self, question: str) -> Any:
        """
        Asynchronously query the agent.

        Args:
            question: The user's query string.

        Returns:
            Agent response object.

        Raises:
            ManagerAgentError: If query execution fails.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        try:
            response = await self.agent.achat(question)
            self._attach_source_nodes(response)
            self._log_tools_called(response)
            return response
        except ValueError:
            raise
        except Exception as e:
            raise ManagerAgentError(f"Query execution failed: {e}") from e

    def query(self, question: str) -> Any:
        """
        Synchronously query the agent.

        Args:
            question: The user's query string.

        Returns:
            Agent response object.

        Raises:
            ManagerAgentError: If query execution fails.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        try:
            response = self.agent.chat(question)
            self._attach_source_nodes(response)
            self._log_tools_called(response)
            return response
        except ValueError:
            raise
        except Exception as e:
            raise ManagerAgentError(f"Query execution failed: {e}") from e
