# Architecture Diagram

```mermaid
graph TD
    User[User] -->|Query| Manager[Manager Agent (LangGraph Supervisor)]

    subgraph "Data Layer"
        PDF[Original PDF] -->|Chunking| HNP[Hierarchical Node Parser]
        HNP -->|Leaf Nodes (128t)| Chroma[ChromaDB Vector Store]
        HNP -->|All Nodes| DocStore[Document Store]

        Chroma -->|Retrieves Leafs| AMR[Auto-Merging Retriever]
        AMR -->|Fetches Parents| DocStore

        DocStore -->|Source for| SumIndex[Summary Index]
    end

    subgraph "Agent Layer"
        Manager -->|Route: Facts/Dates| Needle[Needle Expert]
        Manager -->|Route: High-level/Timeline| Summary[Summary Expert]

        Needle -->|Query| NeedleEngine[Hierarchical Query Engine]
        NeedleEngine -->|Retrieve| AMR

        Summary -->|Query| SumEngine[Summary Query Engine]
        SumEngine -->|MapReduce| SumIndex
    end

    subgraph "MCP Tools Layer"
         Manager -.->|Tool Call| Time[MCP Tool: Time Server]
         Manager -.->|Tool Call| Weather[MCP Tool: Live Weather]
         Manager -.->|Tool Call| HistWeather[Custom Tool: Historical Weather]
    end

    Needle -->|Response| Manager
    Summary -->|Response| Manager
    Time -->|Current Time| Manager
    Weather -->|Forecast| Manager
    HistWeather -->|Archive Data| Manager

    Manager -->|Final Answer| User
```
