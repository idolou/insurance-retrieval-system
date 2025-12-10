from mcp.server.fastmcp import FastMCP
from typing import List

# Create an MCP server
mcp = FastMCP("SequentialThinking")

@mcp.tool()
def sequentialThinking(thought: str, thoughtHistory: List[str], step: int, totalSteps: int) -> str:
    """
    A tool for dynamic step-by-step thinking. Use this to break down complex problems.
    Returns the thought processed by the server.
    """
    # Logic effectively just echoes back but in a real server could persist state
    response = [
        f"Server Processed Step {step}/{totalSteps}",
        f"Thought: {thought}",
        f"History Depth: {len(thoughtHistory)}"
    ]
    return "\n".join(response)

if __name__ == "__main__":
    mcp.run()
