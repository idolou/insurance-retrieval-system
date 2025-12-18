import asyncio
import logging
import os
import sys
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPToolError(Exception):
    """Base exception for MCP tool errors."""

    pass


async def run_module_mcp_tool(module_name: str, tool_name: str, arguments: dict) -> str:
    """
    Generic runner for Python module-based MCP tools.
    """
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
                return response

    except Exception as e:
        error_msg = (
            f"MCP tool {module_name}.{tool_name} failed with arguments {arguments}: {e}"
        )
        logger.error(error_msg, exc_info=True)
        raise MCPToolError(error_msg) from e
