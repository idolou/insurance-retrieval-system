
import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def list_tools(module_name):
    print(f"\n--- Checking tools for {module_name} ---")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", module_name],
        env=os.environ.copy()
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"Tool: {tool.name}")
                    print(f"  Description: {tool.description}")
                    print(f"  Schema: {tool.inputSchema}")
    except Exception as e:
        print(f"Error checking {module_name}: {e}")

async def main():
    await list_tools("mcp_server_time")
    await list_tools("mcp_server_math")

if __name__ == "__main__":
    asyncio.run(main())
