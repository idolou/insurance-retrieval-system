import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from insurance_system.src.agents.manager import build_graph
from insurance_system.src.evaluation.hard_eval import HardEvaluator
from insurance_system.src.evaluation.hitl import run_hitl
from insurance_system.src.evaluation.llm_as_judge import LangGraphWrapper, run_eval

# Re-use existing run_eval logic but point to new dataset if needed

console = Console(record=True)
load_dotenv()

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "insurance_system", "src", "evaluation", "data", "comprehensive_eval_dataset.json")
    if not os.path.exists(path):
        # Fallback for running from project root
        path = "insurance_system/src/evaluation/data/comprehensive_eval_dataset.json"
    
    with open(path, "r") as f:
        return json.load(f)

async def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Suite")
    parser.add_argument("--mode", choices=["hard", "llm", "hitl", "all"], default="all", help="Evaluation mode")
    args = parser.parse_args()

    console.print(Panel(f"[bold yellow]🚀 Running Evaluation Mode: {args.mode.upper()}[/bold yellow]"))

    dataset = load_dataset()
    
    # Initialize Agent once if possible (or let modules handle it)
    console.print("⚙️ Initializing Agent...")
    app = build_graph()
    agent = LangGraphWrapper(app)

    if args.mode in ["hard", "all"]:
        console.print("\n[bold purple]=== Running Hard Evals (Guardrails) ===[/bold purple]")
        hard_eval = HardEvaluator(console=console)
        hard_cases = dataset.get("hard_evals", [])
        hard_eval.run_suite(hard_cases, agent)

    if args.mode in ["llm", "all"]:
        console.print("\n[bold purple]=== Running LLM-as-a-Judge ===[/bold purple]")
        # We need to temporarily decouple run_eval to use our new dataset or just run it as is
        # For this implementation, we will perform a direct integration of the Logic from run_eval
        # but specifically targeting the 'llm_evals' list from our new dataset.
        
        from insurance_system.src.evaluation.llm_as_judge import evaluate_query
        from llama_index.llms.openai import OpenAI
        from insurance_system.src.utils.config import LLM_MODEL, EVALUATOR_MODEL
        
        # Init Judge
        if "claude" in EVALUATOR_MODEL:
            from llama_index.llms.anthropic import Anthropic
            evaluator_llm = Anthropic(model=EVALUATOR_MODEL)
        else:
            evaluator_llm = OpenAI(model=LLM_MODEL)
            
        llm_cases = dataset.get("llm_evals", [])
        
        results = []
        for case in llm_cases:
            res = await evaluate_query(
                case["query"], 
                case["expected"], 
                agent, 
                evaluator_llm,
                console=console
            )
            results.append(res)
            
        # Save results to JSON
        output_file = "evaluation_results.json"
        
        # Convert Pydantic objects to dict for JSON serialization
        serializable_results = []
        for r in results:
            r_dict = r.copy()
            r_dict["correctness"] = r["correctness"].model_dump()
            r_dict["relevancy"] = r["relevancy"].model_dump()
            r_dict["recall"] = r["recall"].model_dump()
            
            # Add flat scores for convenience
            r_dict["correctness_score"] = r["correctness"].score
            r_dict["relevancy_score"] = r["relevancy"].score
            r_dict["recall_score"] = r["recall"].score
            serializable_results.append(r_dict)
            
        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
            
        console.print(f"\n📄 [dim]Detailed results saved to[/dim] [bold]{output_file}[/bold]")
        # But for now, we rely on the per-query output of run_eval

    if args.mode in ["hitl", "all"]:
        console.print("\n[bold purple]=== Running Human-in-the-Loop ===[/bold purple]")
        from insurance_system.src.evaluation.hitl import HITLGrader
        grader = HITLGrader(console=console)
        grader.load_existing_results()
        grader.run_grading_session(dataset.get("hitl_evals", []), agent)

    # Save Console Output to File
    console.save_text("evaluation_summary.txt", clear=False)
    console.save_html("evaluation_summary.html")
    console.print(f"\n📄 [dim]Console output saved to evaluation_summary.txt and .html[/dim]")

if __name__ == "__main__":
    asyncio.run(main())
