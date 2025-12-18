# Insurance Claim Timeline Retrieval System (LangChain + MCP)

A **Multi-Agent RAG System** built with **LangChain & LangGraph** to investigate complex insurance claims. It combines hierarchical retrieval for specific facts with summary indexing for high-level context, all orchestrated by a routing agent capable of using **Model Context Protocol (MCP)** tools.

---

---

## 📋 Table of Contents

1. [Architecture Explanation](#-architecture-explanation)
2. [Key Features](#-key-features)
3. [Design Decisions & Rationale](#-design-decisions--rationale)
   - [Hierarchical Indexing](#1-hierarchical-indexing-configuration)
   - [Smart Routing Strategy](#3-smart-routing-strategy)
   - [Needle Agent Precision](#4-needle-agent-precision)
4. [Index Schemas](#-index-schemas)
5. [Limitations & Trade-offs](#-limitations--trade-offs)
6. [Evaluation](#-evaluation)
7. [Setup & Usage](#-setup--usage)

---

## 🏗️ Architecture Explanation

The system follows a **Hub-and-Spoke Agentic Architecture**:

- **Manager Agent (LangGraph Supervisor)**: The central brain. It routes user queries to the most appropriate tool or sub-agent.
- **Specialized Sub-Agents (Tools)**:
  - **Needle Expert**: Uses the Hierarchical Index (Auto-Merging Retriever) for precise facts.
  - **Summary Expert**: Uses the Summary Index for broad, narrative answers.
- **MCP Tools**: Integration of external capabilities via the Model Context Protocol.
  - **Time Server**: Provides current time and timezone conversion.
  - **Weather Server**: Provides real-time weather data and historical checks.

---

## 🚀 Key Features

1. **LangGraph Orchestration**:

   - Utilizes a state machine for robust orchestration, providing explicit control over state transitions and superior observability.
   - Uses OpenAI's tool-calling capabilities for reliable routing.

2. **MCP Integration (Model Context Protocol)**:
   - **Extensibility**: Tools are not hardcoded. We use MCP servers to dynamically discover and register tools.
   - **Supported MCP Servers**: `mcp-time`, `mcp-weather` (includes custom historical weather tool).

---

## 🧠 Design Decisions & Rationale

We made specific architectural choices to balance **precision** (finding specific facts) with **context** (understanding the story).

### 1. Hierarchical Indexing Configuration

**The Problem**: In standard RAG, large chunks (e.g., 1000 tokens) dilute the meaning of small facts (like a specific date or dollar amount), making them hard to find via vector similarity. Small chunks (e.g., 100 tokens) find the fact but lose the surrounding context needed to answer "why".

**The Solution**: We implementation a **3-Level Hierarchy**:

- **Root (2048 tokens)**: Full sections (e.g., "Scope of Work").
- **Intermediate (512 tokens)**: Paragraphs.
- **Leaf (128 tokens)**: Specific facts.

**Example**:

> _Query_: "What is the deductible?"
>
> 1. **Retrieval**: The system searches **Leaf Attributes**. It finds a tiny 128-token chunk containing "Deductible: $1,000". This has very high vector similarity.
> 2. **Context Expansion**: The **Auto-Merging Retriever** sees this leaf belongs to a larger "Policy Declarations" block. It retrieves the **parent 512-token chunk** instead, providing the LLM with the fact ($1,000) AND the context (Policy Type, Coverage Limits).

### 3. Smart Routing Strategy

**The Problem**: General-purpose agents often "hallucinate" tool usage—using a Summary tool for specific questions (resulting in vague answers) or a Needle tool for broad questions (resulting in missing the big picture).

**The Solution**: We injected a **Strict System Prompt** into the Supervisor Agent (Smart Router).

- **Rule**: "Always use Needle for 'how much', 'when', 'who'."
- **Rule**: "Use Summary ONLY for 'tell me the story'."

**Example**:

> _User_: "How much was the repair estimate?"
> _Router Decision_: "The user is asking for a specific amount ('How much'). I MUST route this to `needle_expert`."
> _Result_: Precision tool is called. No time wasted on summary generation.

### 4. Needle Agent Precision

**The Problem**: Even with good retrieval, LLMs can be "lazy" and gloss over specific details if the prompt is too generic.

**The Solution**: The `NeedleAgent` uses a **Precise System Prompt**: "Answer specifically and precisely. If the answer is a specific value, date, or name, provide it directly."

**Example**:

> _Context_: "The sofa cleaning was approved for $250 on Nov 22."
> _Standard Agent Answer_: "The document mentions the sofa cleaning was approved." (Vague)
> _Our Needle Agent Answer_: "The sofa replacement was partially approved for **$250.00**." (Precise)

---

---

## 🗂️ Index Schemas

### 1. Hierarchical Index (ChromaDB)

Optimized for precise fact retrieval.

- **Collection**: `hierarchical_claims`
- **Metadata Fields**:
  - `document_id`: "HO-2024-8892"
  - `chunk_type`: "root" | "intermediate" | "leaf"
  - `parent_id`: ID of the parent node (for auto-merging)
  - `page_label`: Source page number

### 2. Summary Index (LlamaIndex)

Optimized for high-level narrative queries.

- **Structure**: List Index
- **Content**: Full document text synthesized into high-level summaries.
- **Usage**: Accessed by the `Summary Expert` agent when the user asks broad questions like "Tell me the story of what happened."

---

## ⚠️ Limitations & Trade-offs

(Deep Dive into Architectural Bottlenecks)

### 1. The "Serial Router" Bottleneck

**Architecture Fact**: The `ManagerAgent` (Supervisor) must process every query before any retrieval happens.
**Implication**: This creates a strict **Serial Dependency**.

- _Latency_: Minimum latency is always `Latency(Supervisor) + Latency(SubAgent)`. We cannot speculatively run retrieval in parallel with intent classification.
- _Single Point of Failure_: If the Supervisor misinterprets a query (e.g., routing "What is the deductible?" to `summary_expert`), the downstream agent has no chance to recover. The graph is currently a DAG (Directed Acyclic Graph) without a "correction loop" to circle back if a sub-agent fails.

### 2. The "Leaf Isolation" Problem in Hierarchical Retrieval

**Architecture Fact**: We index 128-token leaves for precision.
**Trade-off**: While this finds exact numbers, it can miss **Distributed Facts**.

- _Scenario_: If a sentence says "The total cost was..." and the actual number "$10,000" appears in the _next_ 128-token chunk, the vector similarity for the number chunk might be low for the query "What was the cost?".
- _Mitigation_: We rely on `CHUNK_OVERLAP=20`, but if the semantic gap is larger than 20 tokens, we might miss the connection entirely.

### 3. Context Pollution via Auto-Merging

**Architecture Fact**: When a leaf matches, we retrieve the **Parent (512 or 2048 tokens)**.
**Trade-off**: This assumes the Parent is _mostly_ relevant.

- _Risk_: A 2048-token parent "Scope of Work" might contain the target fact (1 line) and 100 lines of irrelevant repair codes.
- _Effect_: Inspecting the parent node injects noise into the LLM's context window. This increases the chance of "Lost in the Middle" phenomenon where the LLM ignores the specific fact because it's buried in a large, mostly irrelevant chunk.

### 4. Error Propagation in Tool Chains

**Architecture Fact**: We use a `Supervisor -> Tool` pattern.
**Limitation**: The current Supervisor does not see the _content_ of the tool output before deciding it's done. It hands off to the tool, which returns a string.

- _Real limitation_: If `NeedleAgent` returns "No information found", the Supervisor currently accepts that as the final answer. It lacks a **Reflective Loop** to say "Wait, if Needle failed, maybe I should try Summary?" (though we could implement this in LangGraph, it is a current architectural gap).

### 5. Static Indexing vs. Dynamic Claims

**Architecture Fact**: We build a static vector index at startup.
**Real World Friction**: Insurance claims are dynamic. New documents (e.g., a "Supplement 1" estimate) arrive daily.

- _Bottleneck_: Our architecture requires a **Full Re-index** to incorporate new files. There is no "Incremental Indexing" pipeline implemented. In a production scenario, this would be a major blocking factor for real-time claim handling.

---

## ⚖️ Evaluation

We use an **LLM-as-a-judge** approach (`src/evaluation/run_eval_langchain.py`) to rigorously test the system. The evaluation output is enhanced with the `rich` library for readability.

- **Judge**: Claude 3.7 Sonnet (via Anthropic API) or GPT-4o.
- **Methodology**:
  - We evaluate **Answer Correctness**, **Context Relevancy**, and **Context Recall**.
  - The test suite includes complex queries that require:
    - **Fact Retrieval**: "What is the deductible?"
    - **Summarization**: "Summarize the timeline."
    - **Tool Usage**: "What is the weather in Berlin?" (tests MCP integration).
    - **Claim Verification**: "Check if the claim story matches the weather at that location/date." (Historical weather check).

### 🏆 Evaluation Results

| Metric                 | Score     | Pass Rate |
| :--------------------- | :-------- | :-------- |
| **Answer Correctness** | **88.9%** | (8/9)     |
| **Context Relevancy**  | **77.8%** | (7/9)     |
| **Context Recall**     | **66.7%** | (6/9)     |

> Detailed results: `evaluation_results_langchain.json`

---

## ⚙️ Setup & Usage

### 1. Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY and ANTHROPIC_API_KEY (for evaluation)
```

### 2. Generate Data & Index

```bash
python3 insurance_system/generate_claim.py
python3 insurance_system/build_index.py
```

### 3. Run the Agent (Interactive CLI)

```bash
python3 insurance_system/main_langchain.py
```

_Try queries like:_

- "Summarize the claim."
- "What is the weather in Berlin?"
- "What time is it in London?"

### 4. Run Evaluation

```bash
python3 insurance_system/src/evaluation/run_eval_langchain.py
```
