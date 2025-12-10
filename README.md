# Insurance Claim Timeline Retrieval System

A **Multi-Agent RAG System** built with **LlamaIndex** to investigate complex insurance claims. It combines hierarchical retrieval for specific facts with summary indexing for high-level context, all orchestrated by a routing agent.

---

## 🏗️ Architecture Explanation

The system follows a **Hub-and-Spoke Agentic Architecture**:
-   **Manager Agent (Router)**: The central brain. It analyzes the user's intent and routes queries to the most appropriate sub-agent or tool.
-   **Specialized Sub-Agents**:
    -   **Needle Agent**: Uses the Hierarchical Index to find precise "needle-in-a-haystack" facts.
    -   **Summary Agent**: Uses the Summary Index to generate broad, narrative answers.
-   **MCP Tools**: Integration of deterministic tools (e.g., Python functions) for calculations.

![Architecture Diagram](architecture_diagram.md) *(See architecture_diagram.md)*

---

## 📊 Data Segmentation Decisions

To handle the complexity of an insurance claim file (which contains dense logs, receipts, and narratives), we employ a **Multi-Granularity Strategy**:

1.  **Granularity**: We do not treat the document as a flat list of pages. We segment it into:
    -   **Roots (2048 tokens)**: Large context windows (e.g., full incident reports).
    -   **Intermediates (512 tokens)**: Logical sections (e.g., a specific table of receipts).
    -   **Leaves (128 tokens)**: Small, precise snippets (e.g., a single log entry timestamp).

2.  **Why?**:
    -   Small chunks allow us to match specific queries ("What happened at 10:22 AM?") with high similarity scores.
    -   Large chunks provide the necessary context for the LLM to generate a coherent answer once the relevant section is found.

## 🧠 Chunking Rationale & Index Schemas

### Chunking Levels
We use the `HierarchicalNodeParser` with the following configuration (`src/config.py`):
-   **Levels**: `[2048, 512, 128]`
-   **Overlap**: `20` tokens (ensures no semantic breaks at boundaries).

### Index Schemas
We utilize two distinct index types:

1.  **Hierarchical Vector Index (ChromaDB)**
    -   **Schema**: Stores embeddings ONLY for the **Leaf Nodes (128 tokens)**.
    -   **Reasoning**: Keeping the vector index small and precise reduces noise. We don't vector-index the large nodes because they result in "muddy" embeddings.

2.  **Summary Index (DocStore)**
    -   **Schema**: A linked list of all nodes.
    -   **Reasoning**: Used for "MapReduce" style queries where the agent needs to read *everything* to generate a summary.

### Recall Improvement Strategy
We implement **Auto-Merging Retrieval**:
-   If the retriever finds multiple leaf nodes that share the same parent, it **discards the leaves and returns the parent**.
-   **Benefit**: This drastically improves Context Recall. Instead of seeing 3 fragmented sentences, the LLM sees the full paragraph they belong to.

---

## 🤖 Agent Design + Prompt Structure

### Manager Agent (Router)
-   **Type**: ReAct / Tool-Use Agent with **Chain-of-Thought (CoT)** reasoning.
-   **Prompt**: "You are a Router. Think step-by-step. Use 'needle_expert' for facts. Use 'summary_expert' for broad overviews."
-   **Robustness**:
    -   **Few-Shot Examples**: The system prompt includes concrete examples of user queries and the expected thought process/tool selection.
    -   **Prompts as Functions**: All prompts are encapsulated in `src/prompts.py` using `PromptTemplate` objects, treating them as code artifacts rather than magic strings.
-   **Goal**: Prevent "hallucinated summaries" when the user asks for specific data.

### Sub-Agents
-   **Needle Agent**:
    -   **Description**: "The DEFAULT tool. Use for facts, dates, costs, names."
    -   **Tool**: `QueryEngineTool` -> `AutoMergingRetriever`.
-   **Summary Agent**:
    -   **Description**: "Use ONLY for broad summaries."
    -   **Tool**: `QueryEngineTool` -> `SummaryIndex` (TreeSummarize).

---

## 🛠️ MCP Usage Explanation

## 🛠️ MCP Usage Explanation
We integrate the **Model Context Protocol (MCP)** to give the agent structured reasoning capabilities.
-   **Server**: A local Python MCP server (`insurance_system/mcp_server.py`) implementing the `SequentialThinking` tool.
-   **Client**: The Manager Agent connects via the `mcp` Python SDK (`stdio_client`).
-   **Workflow**:
    1.  User asks a complex multi-step question.
    2.  Manager uses `sequentialThinking` tool to plan its approach step-by-step.
    3.  Server echoes back the thought process (simulated "thinking" state).
    4.  Manager then executes the plan using other tools.

---

## ⚖️ Evaluation Methodology + Examples

We use an **LLM-as-a-judge** approach (`src/evaluation/run_eval.py`).
-   **Judge**: GPT-4o (impartial prompt).
-   **Metrics** (Strict Compliance):
    1.  **Answer Correctness**: Does the answer match the ground truth factually?
    2.  **Context Relevancy**: Did the agent use relevant details/indexes to answer the query?
    3.  **Context Recall**: Did the agent retrieve *all* key facts from the ground truth?
-   **Test Suite**: 8 automated queries covering facts, summaries, and negative constraints.

### 🏆 System Evaluation Results

| Metric | Score | Pass Rate |
| :--- | :--- | :--- |
| **Answer Correctness** | **100.0%** | (8/8) |
| **Context Relevancy** | **100.0%** | (8/8) |
| **Context Recall** | **87.5%** | (7/8) |

> Full detailed results (including reasoning) are available in `src/evaluation/evaluation_results.json`.

---

## ⚠️ Limitations & Trade-offs
1.  **Cost vs. Latency**: The `SummaryAgent` reads the entire document. This is slow and costly (tokens) but necessary for accurate summaries. We mitigate this by defaulting to the `NeedleAgent`.
2.  **PDF Parsing**: `SimpleDirectoryReader` uses PyMuPDF, which may lose layout information for complex tables.
3.  **Indexing Time**: Hierarchical indexing takes 3x longer than flat indexing due to the number of nodes generated.

---

## 🚀 Setup & Usage

### 1. Installation
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY
```

### 2. Generate & Index Data
```bash
python3 insurance_system/generate_claim.py
python3 insurance_system/build_index.py
```

### 3. Run
```bash
python3 insurance_system/main.py
```

### 4. Evaluate
```bash
python3 insurance_system/src/evaluation/run_eval.py
```
