# Insurance Claim Retrieval System - Final Submission

## Project Overview

This project implements an **Agentic RAG (Retrieval-Augmented Generation) System** designed to automate the investigation of complex insurance claims. By leveraging a **Hub-and-Spoke architecture** with **LangGraph** orchestration, the system can retrieve specific facts ("needle in a haystack"), generate high-level summaries, and perform external tasks via **Model Context Protocol (MCP)** tools.

## Key Capabilities

1.  **Orchestration**: A central "Manager Agent" (LangGraph Supervisor) intelligently routes user queries to specialized sub-agents or tools.
2.  **Hierarchical Retrieval**: Uses an Auto-Merging strategy to retrieve small, precise chunks (128 tokens) while providing full context (parent chunks) to the LLM.
3.  **Advanced Table Extraction**: Integrates **LlamaParse** to accurately retrieve dense numerical data (e.g., sensor logs) from PDF tables, preserving row/column structure.
4.  **Real-Time Reasoning**: The system streams its "thought process" live to the user, displaying tool selection and execution steps transparently with a rich CLI experience.
5.  **Extensibility**: Fully integrated with standard **MCP Servers** (Time, Weather) allowing the agent to answer context-dependent questions like "Was the weather consistent with the claim date?".

## Architecture

- **Manager Agent (`manager.py`)**: The central brain using OpenAI's tool-calling to route queries. It enforces strict guidelines to prevent "lazy" tool usage.
- **Needle Agent (`needle_agent.py`)**: Specialized for precision. Uses `AutoMergingRetriever` and `ChromaDB` to find specific dates, costs, and entities.
- **Summary Agent (`summary_agent.py`)**: Specialized for narratives. Uses `LlamaIndex SummaryIndex` to synthesize full-document stories.
- **MCP & External Tools (`mcp_tools.py`)**:
  - **Time Server**: IANA timezone conversion.
  - **Weather Tool**: Custom tool fetching historical weather data from Open-Meteo.

## Evaluation Results

The system was rigorously evaluated using an **LLM-as-a-Judge** framework (Claude 3.7 Sonnet) across a diverse test suite of 10 queries, covering fact retrieval, timeline summarization, table extraction, and external tool usage.

| Metric                 | Score     | Description                                                                        |
| :--------------------- | :-------- | :--------------------------------------------------------------------------------- |
| **Answer Correctness** | **80.0%** | (8/10) Agent successfully retrieved specific facts (e.g., "$12,400.00", "Nov 16"). |
| **Context Relevancy**  | **90.0%** | (9/10) Retrieved chunks were highly relevant to the query.                         |
| **Context Recall**     | **90.0%** | (9/10) The system successfully retrieved the necessary ground-truth facts.         |

_(Detailed evaluation logs provided in `evaluation_results_langchain.json`)_
