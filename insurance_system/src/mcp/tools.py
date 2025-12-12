import asyncio
import os
import sys
from typing import Any, List

from llama_index.core.tools import FunctionTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from insurance_system.src.config import MCP_SERVER_PATH


class MCPToolError(Exception):
    """Base exception for MCP tool errors."""

    pass


async def run_sequential_thinking(
    thought: str, thoughtHistory: List[str], step: int, totalSteps: int
) -> str:
    """
    Executes a step of sequential thinking by connecting to the MCP server.

    Args:
        thought: Current thought/step description.
        thoughtHistory: List of previous thoughts.
        step: Current step number (1-indexed).
        totalSteps: Total number of steps planned.

    Returns:
        Processed thought response from MCP server.

    Raises:
        MCPToolError: If MCP server connection or execution fails.
        ValueError: If input validation fails.
    """
    # Input validation
    if not thought or not thought.strip():
        raise ValueError("Thought cannot be empty")
    if step < 1:
        raise ValueError("Step must be >= 1")
    if totalSteps < 1:
        raise ValueError("TotalSteps must be >= 1")
    if step > totalSteps:
        raise ValueError("Step cannot exceed totalSteps")
    try:
        server_script = MCP_SERVER_PATH

        if not os.path.exists(server_script):
            error_msg = f"MCP server script not found at {server_script}"
            raise MCPToolError(error_msg)

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

                response = "\n".join(final_text)
                return response
    except ValueError:
        raise
    except FileNotFoundError as e:
        error_msg = f"MCP server script not found: {e}"
        raise MCPToolError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to execute sequential thinking via MCP: {e}"
        raise MCPToolError(error_msg) from e
        raise MCPToolError(error_msg) from e


async def run_module_mcp_tool(
    module_name: str, tool_name: str, arguments: dict
) -> str:
    """
    Generic runner for Python module-based MCP tools (e.g., mcp-server-time).
    """
    import sys

    # Use the current python executable (venv) to run the module
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", module_name],
        env=os.environ.copy(),
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(tool_name, arguments=arguments)

                final_text = []
                if result.content:
                    for content in result.content:
                        if content.type == "text":
                            final_text.append(content.text)

                return "\n".join(final_text)

    except Exception as e:
        raise MCPToolError(f"MCP tool {module_name}.{tool_name} failed: {e}") from e


def get_time_tools() -> List[FunctionTool]:
    """
    Returns tools from mcp-server-time.
    """

    async def get_current_time(timezone: str = "UTC") -> str:
        return await run_module_mcp_tool(
            "mcp_server_time", "get_current_time", {"timezone": timezone}
        )

    async def convert_time(
        time: str, source_timezone: str, target_timezone: str
    ) -> str:
        return await run_module_mcp_tool(
            "mcp_server_time",
            "convert_time",
            {
                "time": time,
                "source_timezone": source_timezone,
                "target_timezone": target_timezone,
            },
        )

    t1 = FunctionTool.from_defaults(
        async_fn=get_current_time,
        name="get_current_time",
        description="Get current time in a specific timezone (default UTC).",
    )
    t2 = FunctionTool.from_defaults(
        async_fn=convert_time,
        name="convert_time",
        description="Convert time between timezones.",
    )
    return [t1, t2]


def get_math_tools() -> List[FunctionTool]:
    """
    Returns tools from mcp-server-math.
    """

    async def math_add(a: float, b: float) -> str:
        # Maps to 'sum' which takes a list
        return await run_module_mcp_tool(
            "mcp_server_math", "sum", {"numbers": [a, b]}
        )

    async def math_multiply(a: float, b: float) -> str:
        # Maps to 'product' which takes a list
        return await run_module_mcp_tool(
            "mcp_server_math", "product", {"numbers": [a, b]}
        )

    t1 = FunctionTool.from_defaults(
        async_fn=math_add,
        name="math_add",
        description="Add two numbers. Useful for calculating costs.",
    )
    t2 = FunctionTool.from_defaults(
        async_fn=math_multiply,
        name="math_multiply",
        description="Multiply two numbers.",
    )
    return [t1, t2]

def get_mcp_tools() -> List[FunctionTool]:
    """
    Returns a list of MCP-powered tools.

    Returns:
        List of FunctionTool instances for MCP tools.

    Raises:
        MCPToolError: If tool creation fails.
    """
    try:
        # Wrap the async function in a LlamaIndex FunctionTool
        tool = FunctionTool.from_defaults(
            async_fn=run_sequential_thinking,
            name="sequentialThinking",
            description=(
                "An MCP (Model Context Protocol) tool for dynamic step-by-step thinking and problem decomposition. "
                "Use this to break down complex, multi-step problems into smaller, manageable steps. "
                "This is particularly useful for complex reasoning tasks that require planning. "
                "Arguments: thought (str), thoughtHistory (List[str]), step (int), totalSteps (int). "
                "Use this tool MULTIPLE times to iterate through a problem step-by-step."
            ),
        )
        return [tool]
    except Exception as e:
        raise MCPToolError(f"Tool creation failed: {e}") from e
