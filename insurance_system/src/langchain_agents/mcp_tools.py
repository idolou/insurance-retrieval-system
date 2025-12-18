import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Type

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, create_model

from insurance_system.src.utils.mcp_utils import (MCPToolError,
                                                  run_module_mcp_tool)

logger = logging.getLogger(__name__)


def _create_pydantic_model_from_schema(
    name: str, schema: Dict[str, Any]
) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model from a JSON schema.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields = {}
    for field_name, field_schema in properties.items():
        field_type_str = field_schema.get("type", "string")

        # Simple type mapping
        if field_type_str == "string":
            py_type = str
        elif field_type_str in ["number", "integer"]:
            py_type = float
        elif field_type_str == "array":
            py_type = List[float]
        elif field_type_str == "boolean":
            py_type = bool
        else:
            py_type = Any

        description = field_schema.get("description", "")

        if field_name in required:
            default = ...
        else:
            default = None

        fields[field_name] = (py_type, Field(default=default, description=description))

    return create_model(f"{name}Input", **fields)


def _create_langchain_tool_wrapper(
    module_name: str, tool_name: str, tool_schema: Dict[str, Any]
) -> StructuredTool:
    """
    Dynamically create a LangChain StructuredTool wrapper for an MCP tool.
    """
    # Create Pydantic model for input validation
    input_model = _create_pydantic_model_from_schema(tool_name, tool_schema)

    # Wrapper function
    async def tool_wrapper(**kwargs: Any) -> str:
        # Sanitize time input for 'convert_time' to handle AM/PM
        if tool_name == "convert_time" and "time" in kwargs:
            raw_time = str(kwargs["time"]).strip().upper()
            # Simple conversion for standard formats like "10:22 AM"
            try:
                if "AM" in raw_time or "PM" in raw_time:
                    from datetime import datetime

                    # Parse likely formats
                    for fmt in ("%I:%M %p", "%I:%M%p", "%I %p"):
                        try:
                            dt = datetime.strptime(raw_time, fmt)
                            kwargs["time"] = dt.strftime("%H:%M")
                            break
                        except ValueError:
                            continue
            except Exception:
                pass  # Fallback to original input if parsing fails

        return await run_module_mcp_tool(module_name, tool_name, kwargs)

    # Extract and enhance description
    description = tool_schema.get("description", f"{tool_name} from {module_name}")

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
            description += " Use IANA timezone names (e.g., 'America/New_York')."

    return StructuredTool.from_function(
        func=None,
        coroutine=tool_wrapper,
        name=tool_name,
        description=description,
        args_schema=input_model,
    )


async def _discover_langchain_mcp_tools(module_name: str) -> List[StructuredTool]:
    """
    Discover tools from an MCP server and wrap them for LangChain.
    """
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
                        # Enhance schema description if missing from schema but present in tool
                        desc = tool.description or schema.get("description", "")
                        if desc:
                            schema = {**schema, "description": desc}

                        wrapped_tool = _create_langchain_tool_wrapper(
                            module_name, tool.name, schema
                        )
                        wrapped_tools.append(wrapped_tool)

                return wrapped_tools

    except Exception as e:
        logger.error(f"Failed to discover MCP tools from {module_name}: {e}")
        return []


def _run_sync_discovery(coro):
    """Run async discovery synchronously."""
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError("Cannot discover tools from async context sync-ly")
    except RuntimeError:
        return asyncio.run(coro)


def get_langchain_time_tools() -> List[StructuredTool]:
    return _run_sync_discovery(_discover_langchain_mcp_tools("mcp_server_time"))


def get_langchain_weather_tools() -> List[StructuredTool]:
    return _run_sync_discovery(_discover_langchain_mcp_tools("mcp_weather_server"))
