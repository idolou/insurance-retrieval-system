"""
Central configuration for the Insurance Retrieval System.
"""

import os
from typing import List

# Indexing Configuration
CHUNK_SIZES: List[int] = [2048, 512, 128]  # [Root, Intermediate, Leaf]
CHUNK_OVERLAP: int = 20  # Default overlap between chunks
SIMILARITY_TOP_K: int = 60  # Number of leaf nodes to retrieve

# Model Configuration - Configurable via environment variables
EMBEDDING_MODEL: str = "text-embedding-3-small"
LLM_MODEL: str = "gpt-4o"
MANAGER_MODEL: str = "gpt-4o-mini"
EVALUATOR_MODEL: str = "claude-3-7-sonnet-20250219"

# Reranker Configuration
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANKER_TOP_N: int = 20
USE_RERANKER: bool = True


# Paths Configuration


from dotenv import load_dotenv

load_dotenv()

# Map CLAUDE_API_KEY to ANTHROPIC_API_KEY if present
if os.getenv("CLAUDE_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("CLAUDE_API_KEY")


# Points to 'insurance_system' directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
HIERARCHICAL_STORAGE_DIR = os.path.join(STORAGE_DIR, "hierarchical")
SUMMARY_STORAGE_DIR = os.path.join(STORAGE_DIR, "summary")

MCP_SERVER_PATH = os.path.join(PROJECT_ROOT, "mcp_server.py")

# Environment / Debug Flags
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
VERBOSE: bool = os.getenv("VERBOSE", "false").lower() == "true"
