import asyncio
import logging
import os
import sys
from typing import Any, Callable, Dict, List

from llama_index.core.tools import FunctionTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPToolError(Exception):
    """Base exception for MCP tool errors."""

    pass


async def run_module_mcp_tool(module_name: str, tool_name: str, arguments: dict) -> str:
    """
    Generic runner for Python module-based MCP tools (e.g., mcp-server-time).

    Args:
        module_name: Python module name to run (e.g., "mcp_server_time").
        tool_name: Name of the tool to call.
        arguments: Dictionary of arguments to pass to the tool.

    Returns:
        Text response from the MCP tool.

    Raises:
        MCPToolError: If MCP tool execution fails.
    """
    # Disable tokenizers parallelism to avoid fork warnings
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", module_name],
        env=env,
    )

    try:
        logger.debug(
            "Calling MCP tool %s.%s with arguments: %s",
            module_name,
            tool_name,
            arguments,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)

                final_text = [
                    content.text for content in result.content if content.type == "text"
                ]
                response = "\n".join(final_text)
                logger.debug(
                    "MCP tool %s.%s returned: %s",
                    module_name,
                    tool_name,
                    response[:200],
                )
                return response

    except Exception as e:
        error_msg = (
            f"MCP tool {module_name}.{tool_name} failed with arguments {arguments}: {e}"
        )
        logger.error(error_msg, exc_info=True)
        raise MCPToolError(error_msg) from e


def _create_tool_wrapper(
    module_name: str, tool_name: str, tool_schema: Dict[str, Any]
) -> FunctionTool:
    """
    Dynamically create a FunctionTool wrapper for an MCP tool.

    Args:
        module_name: MCP server module name.
        tool_name: Name of the tool.
        tool_schema: Tool schema from MCP server.

    Returns:
        FunctionTool instance wrapping the MCP tool.
    """
    properties = tool_schema.get("properties", {})

    # Check if tool expects "numbers" array (math tools)
    expects_numbers = "numbers" in properties

    if expects_numbers:
        # Math tools: accept numbers as a list
        async def tool_wrapper(numbers: List[float]) -> str:
            return await run_module_mcp_tool(
                module_name, tool_name, {"numbers": numbers}
            )

    else:
        # Other tools: pass kwargs directly
        async def tool_wrapper(**kwargs: Any) -> str:
            try:
                return await run_module_mcp_tool(module_name, tool_name, kwargs)
            except MCPToolError as e:
                # Re-raise with more context
                raise MCPToolError(
                    f"Error calling {tool_name}: {str(e)}. "
                    f"Check that arguments match the tool schema."
                ) from e

    # Extract description from schema
    description = tool_schema.get("description", f"{tool_name} tool from {module_name}")

    # Enhance description with format hints for better LLM understanding
    if module_name == "mcp_server_time":
        if tool_name == "convert_time":
            description += (
                " IMPORTANT: First use 'needle_expert' to find the time from documents. "
                "Then convert using: time (24-hour format HH:MM only, e.g., '15:45' not '3:45 PM'), "
                "source_timezone (IANA name like 'America/Chicago' not 'CST'), "
                "target_timezone (IANA name like 'Europe/Berlin' not 'Berlin'). "
                "Common IANA names: 'America/Chicago' (CST), 'Europe/Berlin' (CET), 'Asia/Tokyo' (JST)."
            )
        elif tool_name == "get_current_time":
            description += " Use IANA timezone names (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo')."

    return FunctionTool.from_defaults(
        async_fn=tool_wrapper,
        name=tool_name,
        description=description,
    )


async def _discover_mcp_tools(module_name: str) -> List[FunctionTool]:
    """
    Automatically discover and wrap all tools from an MCP server.

    Args:
        module_name: Python module name of the MCP server.

    Returns:
        List of FunctionTool instances for all discovered tools.

    Raises:
        MCPToolError: If tool discovery fails.
    """
    # Disable tokenizers parallelism to avoid fork warnings
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", module_name],
        env=env,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()

                wrapped_tools = []
                for tool in tools_response.tools:
                    schema = tool.inputSchema
                    if isinstance(schema, dict):
                        # Use tool description if available, otherwise use schema description
                        description = tool.description or schema.get("description", "")
                        if description:
                            schema = {**schema, "description": description}
                        wrapped_tool = _create_tool_wrapper(
                            module_name, tool.name, schema
                        )
                        wrapped_tools.append(wrapped_tool)

                return wrapped_tools

    except Exception as e:
        raise MCPToolError(f"Failed to discover tools from {module_name}: {e}") from e


def _run_async_discovery(coro):
    """Safely run async discovery, handling event loop context."""
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - can't use asyncio.run()
        # Create a task instead (but this won't work for sync callers)
        raise RuntimeError("Cannot discover tools from async context")
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)


def get_time_tools() -> List[FunctionTool]:
    """
    Returns all tools from mcp-server-time (auto-discovered).

    Returns:
        List of FunctionTool instances for time operations.
    """
    return _run_async_discovery(_discover_mcp_tools("mcp_server_time"))


def get_math_tools() -> List[FunctionTool]:
    """
    Returns all tools from mcp-server-math (auto-discovered).

    Returns:
        List of FunctionTool instances for math operations.
    """
    return _run_async_discovery(_discover_mcp_tools("mcp_server_math"))
