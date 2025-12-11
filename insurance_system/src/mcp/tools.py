import asyncio
import os
from typing import Any, List

from llama_index.core.tools import FunctionTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from insurance_system.src.config import MCP_SERVER_PATH


async def run_sequential_thinking(
    thought: str, thoughtHistory: List[str], step: int, totalSteps: int
) -> str:
    """
    Executes a step of sequential thinking by connecting to the official MCP server.
    This helps the agent break down complex problems into steps.
    """
    # Define connection to the Local Python MCP Server
    # We use the same python interpreter
    import sys

    server_script = MCP_SERVER_PATH

    server_params = StdioServerParameters(
        command=sys.executable, args=[server_script], env=os.environ.copy()
    )

    # Establish connection just for this tool call
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Call the tool on the server
            result = await session.call_tool(
                "sequentialThinking",
                arguments={
                    "thought": thought,
                    "thoughtHistory": thoughtHistory,
                    "step": step,
                    "totalSteps": totalSteps,
                },
            )

            # Extract text content
            final_text = []
            if result.content:
                for content in result.content:
                    if content.type == "text":
                        final_text.append(content.text)

            return "\n".join(final_text)


def get_mcp_tools() -> List[FunctionTool]:
    """
    Returns a list of MCP-powered tools.
    """
    # Wrap the async function in a LlamaIndex FunctionTool
    tool = FunctionTool.from_defaults(
        async_fn=run_sequential_thinking,
        name="sequentialThinking",
        description=(
            "A tool for dynamic step-by-step thinking. Use this to break down complex problems. "
            "Arguments: thought (str), thoughtHistory (List[str]), step (int), totalSteps (int). "
            "Use this tool MULTIPLE times to iterate through a problem."
        ),
    )
    return [tool]
