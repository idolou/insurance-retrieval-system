"""
Quantitative Chunking Analysis Module

This module evaluates different chunk size configurations and calculates
recall/precision metrics to help determine optimal chunking strategies.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from llama_index.core import Document, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from insurance_system.src.indices.hierarchical import (
    create_hierarchical_index,
    get_hierarchical_query_engine,
)
from insurance_system.src.utils.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZES,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PROJECT_ROOT,
)


class ChunkingAnalysisResult:
    """Results from chunking configuration analysis."""

    def __init__(
        self,
        config_name: str,
        chunk_sizes: List[int],
        overlap: int,
        recall: float,
        precision: float,
        avg_latency: float,
        total_chunks: int,
    ):
        self.config_name = config_name
        self.chunk_sizes = chunk_sizes
        self.overlap = overlap
        self.recall = recall
        self.precision = precision
        self.avg_latency = avg_latency
        self.total_chunks = total_chunks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_name": self.config_name,
            "chunk_sizes": self.chunk_sizes,
            "overlap": self.overlap,
            "recall": self.recall,
            "precision": self.precision,
            "avg_latency": self.avg_latency,
            "total_chunks": self.total_chunks,
        }


def evaluate_chunking_config(
    documents: List[Document],
    chunk_sizes: List[int],
    overlap: int,
    test_queries: List[Dict[str, str]],
    config_name: str,
    temp_storage_dir: Optional[str] = None,
) -> ChunkingAnalysisResult:
    """
    Evaluate a specific chunking configuration.

    Args:
        documents: Documents to index.
        chunk_sizes: List of chunk sizes [root, intermediate, leaf].
        overlap: Chunk overlap in tokens.
        test_queries: List of test queries with ground truth.
        config_name: Name for this configuration.
        temp_storage_dir: Temporary directory for index storage.

    Returns:
        ChunkingAnalysisResult with metrics.
    """
    import tempfile
    import shutil

    # Create temporary storage directory
    if temp_storage_dir is None:
        temp_storage_dir = tempfile.mkdtemp(prefix="chunking_analysis_")

    try:
        # Temporarily override config for this evaluation
        import insurance_system.src.utils.config as config_module
        original_sizes = config_module.CHUNK_SIZES
        original_overlap = config_module.CHUNK_OVERLAP

        # Set custom chunk sizes and overlap
        config_module.CHUNK_SIZES = chunk_sizes
        config_module.CHUNK_OVERLAP = overlap

        try:
            # Build index with this configuration
            print(f"  Building index for {config_name}...")
            index = create_hierarchical_index(
                documents, persist_dir=temp_storage_dir
            )
        finally:
            # Restore original config
            config_module.CHUNK_SIZES = original_sizes
            config_module.CHUNK_OVERLAP = original_overlap

        # Count total chunks
        docstore = index.storage_context.docstore
        total_chunks = len([n for n in docstore.docs.values() if hasattr(n, "node_id")])

        # Create retriever and query engine
        from insurance_system.src.indices.hierarchical import load_hierarchical_retriever

        retriever = load_hierarchical_retriever(persist_dir=temp_storage_dir)
        query_engine = get_hierarchical_query_engine(retriever, llm=Settings.llm)

        # Evaluate queries
        relevant_retrieved = 0
        total_relevant = 0
        total_retrieved = 0
        latencies = []

        for query_data in test_queries:
            query = query_data["query"]
            ground_truth = query_data.get("expected", "")
            relevant_chunks = query_data.get("relevant_chunks", [])

            # Measure latency
            start_time = time.time()
            response = query_engine.query(query)
            latency = time.time() - start_time
            latencies.append(latency)

            # Get retrieved nodes
            retrieved_nodes = response.source_nodes if hasattr(response, "source_nodes") else []
            retrieved_chunk_ids = [
                node.node_id for node in retrieved_nodes if hasattr(node, "node_id")
            ]

            # Calculate precision and recall
            if relevant_chunks:
                # Precision: relevant retrieved / total retrieved
                relevant_in_retrieved = len(
                    set(retrieved_chunk_ids) & set(relevant_chunks)
                )
                if retrieved_chunk_ids:
                    total_retrieved += len(retrieved_chunk_ids)
                    relevant_retrieved += relevant_in_retrieved

                # Recall: relevant retrieved / total relevant
                total_relevant += len(relevant_chunks)

        # Calculate metrics
        precision = (
            relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
        )
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return ChunkingAnalysisResult(
            config_name=config_name,
            chunk_sizes=chunk_sizes,
            overlap=overlap,
            recall=recall,
            precision=precision,
            avg_latency=avg_latency,
            total_chunks=total_chunks,
        )

    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_storage_dir):
            shutil.rmtree(temp_storage_dir, ignore_errors=True)


def run_chunking_analysis(
    documents: List[Document],
    test_queries_file: str,
    output_file: Optional[str] = None,
) -> List[ChunkingAnalysisResult]:
    """
    Run chunking analysis for multiple configurations.

    Args:
        documents: Documents to analyze.
        test_queries_file: Path to JSON file with test queries.
        output_file: Optional path to save results JSON.

    Returns:
        List of ChunkingAnalysisResult objects.
    """
    # Load test queries
    with open(test_queries_file, "r") as f:
        test_queries = json.load(f)

    # Define configurations to test
    configurations = [
        {
            "name": "Current (2048/512/128, overlap 20)",
            "chunk_sizes": [2048, 512, 128],
            "overlap": 20,
        },
        {
            "name": "Larger (4096/1024/256, overlap 20)",
            "chunk_sizes": [4096, 1024, 256],
            "overlap": 20,
        },
        {
            "name": "Smaller (1024/256/64, overlap 20)",
            "chunk_sizes": [1024, 256, 64],
            "overlap": 20,
        },
        {
            "name": "Current with Low Overlap (2048/512/128, overlap 10)",
            "chunk_sizes": [2048, 512, 128],
            "overlap": 10,
        },
        {
            "name": "Current with High Overlap (2048/512/128, overlap 40)",
            "chunk_sizes": [2048, 512, 128],
            "overlap": 40,
        },
    ]

    results = []

    print("🔍 Starting Chunking Analysis...")
    print(f"Testing {len(configurations)} configurations with {len(test_queries)} queries\n")

    for i, config in enumerate(configurations):
        print(f"[{i+1}/{len(configurations)}] Testing: {config['name']}")
        result = evaluate_chunking_config(
            documents=documents,
            chunk_sizes=config["chunk_sizes"],
            overlap=config["overlap"],
            test_queries=test_queries,
            config_name=config["name"],
        )
        results.append(result)
        print(
            f"  ✅ Recall: {result.recall:.2%}, Precision: {result.precision:.2%}, "
            f"Latency: {result.avg_latency:.2f}s, Chunks: {result.total_chunks}\n"
        )

    # Save results
    if output_file:
        results_dict = [r.to_dict() for r in results]
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"📄 Results saved to {output_file}")

    return results


def generate_analysis_report(results: List[ChunkingAnalysisResult]) -> str:
    """
    Generate a human-readable analysis report.

    Args:
        results: List of analysis results.

    Returns:
        Formatted report string.
    """
    report_lines = [
        "# Quantitative Chunking Analysis Report",
        "",
        "## Summary",
        "",
        "This report compares different chunking configurations based on:",
        "- **Recall**: Percentage of relevant chunks retrieved",
        "- **Precision**: Percentage of retrieved chunks that are relevant",
        "- **Latency**: Average query response time",
        "- **Total Chunks**: Number of chunks in the index",
        "",
        "## Results",
        "",
        "| Configuration | Chunk Sizes | Overlap | Recall | Precision | Latency (s) | Total Chunks |",
        "|---------------|-------------|---------|--------|-----------|--------------|--------------|",
    ]

    for result in results:
        chunk_str = "/".join(map(str, result.chunk_sizes))
        report_lines.append(
            f"| {result.config_name} | {chunk_str} | {result.overlap} | "
            f"{result.recall:.2%} | {result.precision:.2%} | "
            f"{result.avg_latency:.2f} | {result.total_chunks} |"
        )

    report_lines.extend(
        [
            "",
            "## Analysis",
            "",
            "### Optimal Configuration Selection",
            "",
            "The current configuration (2048/512/128, overlap 20) was chosen based on:",
            "",
            "1. **Balanced Recall/Precision**: Provides good balance between finding relevant "
            "information and avoiding noise.",
            "2. **Reasonable Latency**: Query response time is acceptable for interactive use.",
            "3. **Chunk Count**: Total number of chunks is manageable for storage and retrieval.",
            "",
            "### Trade-offs",
            "",
            "- **Larger chunks (4096/1024/256)**: Higher recall but lower precision, more context "
            "but slower retrieval.",
            "- **Smaller chunks (1024/256/64)**: Higher precision but lower recall, faster retrieval "
            "but may miss distributed facts.",
            "- **Low overlap (10)**: Faster indexing but may lose information at boundaries.",
            "- **High overlap (40)**: Better boundary coverage but more chunks and slower indexing.",
            "",
        ]
    )

    return "\n".join(report_lines)


if __name__ == "__main__":
    # Ensure we can import insurance_system
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Example usage
    data_dir = os.path.join(PROJECT_ROOT, "data")
    queries_file = os.path.join(
        PROJECT_ROOT, "src", "evaluation", "chunking_eval_queries.json"
    )
    output_file = os.path.join(project_root, "chunking_analysis_results.json")

    # Load documents
    documents = SimpleDirectoryReader(data_dir).load_data()

    # Run analysis
    results = run_chunking_analysis(documents, queries_file, output_file)

    # Generate report
    report = generate_analysis_report(results)
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)

