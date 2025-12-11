"""
Central configuration for the Insurance Retrieval System.
"""

# Indexing Configuration
CHUNK_SIZES = [2048, 512, 128]  # [Root, Intermediate, Leaf]
CHUNK_OVERLAP = 20  # Default overlap between chunks
SIMILARITY_TOP_K = 6  # Number of leaf nodes to retrieve

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"


# Reranker Configuration
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANKER_TOP_N = 20

# Paths Configuration
import os

from dotenv import load_dotenv

load_dotenv()

# Points to 'insurance_system' directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
HIERARCHICAL_STORAGE_DIR = os.path.join(STORAGE_DIR, "hierarchical")
SUMMARY_STORAGE_DIR = os.path.join(STORAGE_DIR, "summary")

MCP_SERVER_PATH = os.path.join(PROJECT_ROOT, "mcp_server.py")

# Environment / Debug Flags
DEBUG = False
VERBOSE = False
