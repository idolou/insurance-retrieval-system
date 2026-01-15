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
from rich.panel import Panel
from rich.table import Table

from insurance_system.src.agents.manager import build_graph
from insurance_system.src.evaluation.models import EvaluationResult
from insurance_system.src.utils.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
)
from insurance_system.src.utils.prompts import (
    CONTEXT_RECALL_EVAL_PROMPT,
    CONTEXT_RELEVANCY_EVAL_PROMPT,
    CORRECTNESS_EVAL_PROMPT,
    FAITHFULNESS_EVAL_PROMPT,
)

load_dotenv()

# Initialize Console with recording enabled to capture output for file saving
console = Console(record=True)


class LangGraphWrapper:
    """Wraps LangGraph app to mimic LlamaIndex Agent interface."""

    def __init__(self, app: Any):
        self.app = app
        self.last_tool_used = "unknown"
        self.last_context = ""

    def _extract_tool_usage(self, messages: list):
        """Extracts the first major tool used from the message history."""
        self.last_tool_used = "unknown"
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # Capture the first tool call
                tool_name = msg.tool_calls[0]["name"]
                # We care primarily about expert routing
                if "expert" in tool_name:
                    self.last_tool_used = tool_name.replace("insurance_system_src_agents_mcp_tools_", "") # Clean up if namespaced
                    # Use simple names
                    if "needle" in self.last_tool_used: self.last_tool_used = "needle"
                    if "summary" in self.last_tool_used: self.last_tool_used = "summary"
                    break
                elif "weather" in tool_name:
                    self.last_tool_used = "weather"
                elif "time" in tool_name:
                    self.last_tool_used = "time"
                else:
                    self.last_tool_used = tool_name

        # Extract Context (Tool Output)
        for msg in reversed(messages):
            if msg.type == "tool":
                self.last_context = msg.content
                break

    async def aquery(self, query_str: str) -> str:
        messages = [HumanMessage(content=query_str)]
        result = await self.app.ainvoke({"messages": messages})
        self._extract_tool_usage(result["messages"])
        return result["messages"][-1].content

    def query(self, query_str: str) -> str:
        # Sync fallback
        messages = [HumanMessage(content=query_str)]
        result = self.app.invoke({"messages": messages})
        self._extract_tool_usage(result["messages"])
        return result["messages"][-1].content


async def evaluate_query(query, expected, agent, evaluator_llm, console=None):
    # Use Global settings for embeddings just in case
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    if console is None:
        console = Console()

    console.print(f"\n[bold blue]🔍 Query:[/bold blue] {query}")

    # 1. Get Agent Response
    if hasattr(agent, "aquery"):
        agent_response = await agent.aquery(query)
    else:
        agent_response = agent.query(query)

    actual_answer = str(agent_response)
    
    # Print Router info if available
    if hasattr(agent, "last_tool_used"):
        console.print(f"Using Tool: [bold blue]{agent.last_tool_used}[/bold blue]")
        # console.print(f"Context Length: {len(getattr(agent, 'last_context', ''))} chars")

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

    # --- 4. Faithfulness ---
    # Only run if context is available
    context = getattr(agent, "last_context", "") or "No context available"
    res_faithfulness = await get_eval_result(
        FAITHFULNESS_EVAL_PROMPT,
        query=query,
        context=context,
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
    table.add_row("Faithfulness", str(res_faithfulness.score), res_faithfulness.explanation)

    console.print(table)

    return {
        "query": query,
        "correctness": res_correct,
        "relevancy": res_relevancy,
        "recall": res_recall,
        "faithfulness": res_faithfulness,
        "agent_used": getattr(agent, "last_tool_used", "unknown"),
        "agent_response": actual_answer
    }


async def run_eval():
    console.print(
        Panel.fit(
            "[bold yellow]🚀 Starting Evaluation...[/bold yellow]",
            border_style="yellow",
        )
    )

    # Setup
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

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

    # Load System
    console.print("⚙️ [dim]Initializing Agent...[/dim]")
    app = build_graph()
    manager = LangGraphWrapper(app)

    # Load Test Cases from JSON
    queries_file = os.path.join(os.path.dirname(__file__), "eval_queries.json")
    try:
        with open(queries_file, "r") as f:
            test_data = json.load(f)
        console.print(f"📄 [dim]Loaded test cases from {queries_file}[/dim]")
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Could not find {queries_file}")
        return

    # 3. Aggregate Results
    results = []
    correctness_scores = []
    relevancy_scores = []
    recall_scores = []
    faithfulness_scores = []

    # Iterate through categories
    for category, queries in test_data.items():
        console.print(f"\n[bold purple reversed] {category} [/bold purple reversed]")

        for case in queries:
            res = await evaluate_query(
                case["query"], case["expected"], manager, evaluator_llm
            )

            # Extract model outputs
            c_score = res["correctness"].score
            rel_score = res["relevancy"].score
            rec_score = res["recall"].score
            faith_score = res["faithfulness"].score

            # Add flat scores for analysis
            res["correctness_score"] = c_score
            res["relevancy_score"] = rel_score
            res["recall_score"] = rec_score
            res["faithfulness_score"] = faith_score
            res["category"] = category  # Add category to result

            # Convert Pydantic objects to dict for JSON serialization
            res["correctness"] = res["correctness"].model_dump()
            res["relevancy"] = res["relevancy"].model_dump()
            res["recall"] = res["recall"].model_dump()
            res["faithfulness"] = res["faithfulness"].model_dump()

            correctness_scores.append(c_score)
            relevancy_scores.append(rel_score)
            recall_scores.append(rec_score)
            faithfulness_scores.append(faith_score)

            results.append(res)

    # 4. Summary Report
    total = len(results)
    if total > 0:
        c_pass = sum(correctness_scores)
        rel_pass = sum(relevancy_scores)
        rec_pass = sum(recall_scores)
        faith_pass = sum(faithfulness_scores)

        console.print("\n")
        summary_table = Table(title="📊 EVALUATION SUMMARY", box=None)
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
        summary_table.add_row(
            "Faithfulness", f"{faith_pass/total*100:.1f}%", f"({faith_pass}/{total})"
        )

        console.print(Panel(summary_table, border_style="blue"))

    # Save to JSON
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(
        f"\n📄 [dim]Detailed results saved to[/dim] [bold]{output_file}[/bold]"
    )

    # Save Console Output to File
    console_output_file = "evaluation_summary.txt"
    console.save_text(console_output_file, clear=False)
    console.print(
        f"📄 [dim]Console output log saved to[/dim] [bold]{console_output_file}[/bold]"
    )

    # Save Console Output to File (HTML with colors)
    console_html_file = "evaluation_summary.html"
    console.save_html(console_html_file)
    console.print(
        f"🎨 [dim]Colored report saved to[/dim] [bold]{console_html_file}[/bold]"
    )


if __name__ == "__main__":
    asyncio.run(run_eval())
