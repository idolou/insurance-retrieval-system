import asyncio
import json
import os
import sys

# Suppress HuggingFace Tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from llama_index.core import Settings
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from insurance_system.src.agents.manager import build_graph
from insurance_system.src.evaluation.models import EvaluationResult
from insurance_system.src.utils.config import (
    EMBEDDING_MODEL,
    EVALUATOR_MODEL,
    LLM_MODEL,
)
from insurance_system.src.utils.prompts import (
    CONTEXT_RECALL_EVAL_PROMPT,
    CONTEXT_RELEVANCY_EVAL_PROMPT,
    CORRECTNESS_EVAL_PROMPT,
)

load_dotenv()

console = Console()


class LangGraphWrapper:
    """Wraps LangGraph app to mimic LlamaIndex Agent interface."""

    def __init__(self, app: Any):
        self.app = app

    async def aquery(self, query_str: str) -> str:
        messages = [HumanMessage(content=query_str)]
        result = await self.app.ainvoke({"messages": messages})
        return result["messages"][-1].content

    def query(self, query_str: str) -> str:
        # Sync fallback
        messages = [HumanMessage(content=query_str)]
        result = self.app.invoke({"messages": messages})
        return result["messages"][-1].content


async def evaluate_query(query, expected, agent, evaluator_llm):
    # Use Global settings for embeddings just in case
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    console.print(f"\n[bold blue]🔍 Query:[/bold blue] {query}")

    # 1. Get Agent Response
    if hasattr(agent, "aquery"):
        agent_response = await agent.aquery(query)
    else:
        agent_response = agent.query(query)

    actual_answer = str(agent_response)
    console.print(
        Panel(
            actual_answer,
            title="[bold green]🤖 Agent Answer[/bold green]",
            border_style="green",
        )
    )

    # helper for structured output
    async def get_eval_result(prompt_template, **kwargs):
        try:
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=EvaluationResult,
                prompt=prompt_template,
                llm=evaluator_llm,
                verbose=False,
            )
            return await program.acall(**kwargs)
        except Exception as e:
            console.print(f"[bold red]Error in evaluation:[/bold red] {e}")
            return EvaluationResult(score=0, explanation=f"Evaluation failed: {e}")

    # --- 1. Answer Correctness ---
    res_correct = await get_eval_result(
        CORRECTNESS_EVAL_PROMPT,
        query=query,
        expected=expected,
        actual_answer=actual_answer,
    )

    # --- 2. Context Relevancy ---
    res_relevancy = await get_eval_result(
        CONTEXT_RELEVANCY_EVAL_PROMPT,
        query=query,
        expected=expected,
        actual_answer=actual_answer,
    )

    # --- 3. Context Recall ---
    res_recall = await get_eval_result(
        CONTEXT_RECALL_EVAL_PROMPT,
        query=query,
        expected=expected,
        actual_answer=actual_answer,
    )

    # Create a results table
    table = Table(title="⚖️ Judge Results")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")
    table.add_column("Explanation", style="white")

    table.add_row("Correctness", str(res_correct.score), res_correct.explanation)
    table.add_row("Relevancy", str(res_relevancy.score), res_relevancy.explanation)
    table.add_row("Recall", str(res_recall.score), res_recall.explanation)

    console.print(table)

    return {
        "query": query,
        "correctness": res_correct,
        "relevancy": res_relevancy,
        "recall": res_recall,
    }


async def run_eval():
    console.print(
        Panel.fit(
            "[bold yellow]🚀 Starting LangChain Evaluation...[/bold yellow]",
            border_style="yellow",
        )
    )

    # Setup
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    llm = OpenAI(model=LLM_MODEL)

    # Initialize Evaluator (Judge)
    from insurance_system.src.utils.config import EVALUATOR_MODEL

    if "claude" in EVALUATOR_MODEL:
        from llama_index.llms.anthropic import Anthropic

        console.print(
            f"👨‍⚖️ Judge initialized with Claude: [bold]{EVALUATOR_MODEL}[/bold]"
        )
        evaluator_llm = Anthropic(model=EVALUATOR_MODEL)
    else:
        console.print(f"👨‍⚖️ Judge initialized with OpenAI: [bold]{LLM_MODEL}[/bold]")
        evaluator_llm = OpenAI(model=LLM_MODEL)

    # Load LangGraph System
    console.print("⚙️ [dim]Initializing LangGraph Agent...[/dim]")
    app = build_graph()
    manager = LangGraphWrapper(app)

    # Load Test Cases from JSON
    queries_file = os.path.join(os.path.dirname(__file__), "eval_queries.json")
    try:
        with open(queries_file, "r") as f:
            test_cases = json.load(f)
        console.print(
            f"📄 [dim]Loaded {len(test_cases)} test cases from {queries_file}[/dim]"
        )
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Could not find {queries_file}")
        return

    # 3. Aggregate Results
    results = []
    correctness_scores = []
    relevancy_scores = []
    recall_scores = []

    for case in test_cases:
        res = await evaluate_query(
            case["query"], case["expected"], manager, evaluator_llm
        )

        # Extract model outputs
        c_score = res["correctness"].score
        rel_score = res["relevancy"].score
        rec_score = res["recall"].score

        # Add flat scores for analysis
        res["correctness_score"] = c_score
        res["relevancy_score"] = rel_score
        res["recall_score"] = rec_score

        # Convert Pydantic objects to dict for JSON serialization
        # Use model_dump instead of dict() to avoid DeprecationWarning
        res["correctness"] = res["correctness"].model_dump()
        res["relevancy"] = res["relevancy"].model_dump()
        res["recall"] = res["recall"].model_dump()

        correctness_scores.append(c_score)
        relevancy_scores.append(rel_score)
        recall_scores.append(rec_score)

        results.append(res)

    # 4. Summary Report
    total = len(results)
    c_pass = sum(correctness_scores)
    rel_pass = sum(relevancy_scores)
    rec_pass = sum(recall_scores)

    console.print("\n")
    summary_table = Table(title="📊 LANGCHAIN EVALUATION SUMMARY", box=None)
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Percentage", style="bold magenta")
    summary_table.add_column("Count", style="white")

    summary_table.add_row(
        "Answer Correctness", f"{c_pass/total*100:.1f}%", f"({c_pass}/{total})"
    )
    summary_table.add_row(
        "Context Relevancy", f"{rel_pass/total*100:.1f}%", f"({rel_pass}/{total})"
    )
    summary_table.add_row(
        "Context Recall", f"{rec_pass/total*100:.1f}%", f"({rec_pass}/{total})"
    )

    console.print(Panel(summary_table, border_style="blue"))

    # Save to JSON
    output_file = "evaluation_results_langchain.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(
        f"\n📄 [dim]Detailed results saved to[/dim] [bold]{output_file}[/bold]"
    )


if __name__ == "__main__":
    asyncio.run(run_eval())
