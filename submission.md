# Insurance Claim Retrieval System - Final Submission

## Project Overview

This project implements an **Agentic RAG (Retrieval-Augmented Generation) System** designed to automate the investigation of complex insurance claims. By leveraging a **Hub-and-Spoke architecture** with **LangGraph** orchestration, the system can retrieve specific facts ("needle in a haystack"), generate high-level summaries, and perform external tasks via **Model Context Protocol (MCP)** tools.

## Key Capabilities

1.  **Orchestration**: A central "Manager Agent" intelligently routes user queries to specialized sub-agents or tools.
2.  **Hierarchical Retrieval**: Uses an Auto-Merging strategy to retrieve small, precise chunks (128 tokens) while providing full context (parent chunks) to the LLM, achieving **100% recall** on test datasets.
3.  **Real-Time Reasoning**: The system streams its "thought process" live to the user, displaying tool selection and execution steps transparently.
4.  **Extensibility**: Fully integrated with standard **MCP Servers** (Time, Weather) allowing the agent to answer context-dependent questions like "What was the weather?" or "Time difference between cities?".

## Architecture

- **LangGraph Supervisor**: State machine managing the conversation flow.
- **Needle Expert**: LlamaIndex `AutoMergingRetriever` for fact extraction.
- **Summary Expert**: LlamaIndex `SummaryIndex` for holistic queries.
- **MCP Tools**: Dynamic integration of `mcp-time` and `mcp-weather` servers.

## Evaluation Results

The system was rigorously evaluated using an **LLM-as-a-Judge** framework (Claude 3.7 Sonnet):

| Metric          | Score    | Description                                                         |
| :-------------- | :------- | :------------------------------------------------------------------ |
| **Correctness** | **100%** | Answers are factually accurate against the claims document.         |
| **Relevancy**   | **100%** | Retrieved context is strictly relevant to the query.                |
| **Recall**      | **100%** | The system successfully retrieved all necessary ground-truth facts. |

_(Detailed evaluation logs provided in `evaluation_results_langchain.json`)_
