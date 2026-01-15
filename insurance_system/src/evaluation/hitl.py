import json
import os
import sys
from typing import Any, Dict, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from insurance_system.src.agents.manager import build_graph
from insurance_system.src.evaluation.llm_as_judge import LangGraphWrapper

console = Console()

class HITLGrader:
    def __init__(self, output_file: str = "hitl_results.json", console=None):
        self.output_file = output_file
        self.results = []
        self.console = console or Console()

    def load_existing_results(self):
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r") as f:
                    self.results = json.load(f)
            except:
                self.results = []
    
    def save_results(self):
        with open(self.output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        self.console.print(f"[green]Saved results to {self.output_file}[/green]")

    def run_grading_session(self, test_cases: List[Dict[str, Any]], agent_wrapper):
        self.console.print(Panel("[bold yellow]👨‍🏫 Starting Human-in-the-Loop Grading Session[/bold yellow]"))
        
        # Filter out already graded
        graded_ids = {r.get("id") for r in self.results}
        remaining_cases = [c for c in test_cases if c.get("id") not in graded_ids]
        
        if not remaining_cases:
            self.console.print("[green]All cases already graded![/green]")
            return

        for i, case in enumerate(remaining_cases):
            self.console.print(f"\n[bold cyan]--- Case {i+1}/{len(remaining_cases)} ---[/bold cyan]")
            self.console.print(f"[bold]Query:[/bold] {case['query']}")
            self.console.print(f"[dim]Goal: {case['description']}[/dim]")
            
            # Run Agent
            with self.console.status("[bold green]Thinking...[/bold green]"):
                try:
                    response = agent_wrapper.query(case['query'])
                except Exception as e:
                    response = f"Error: {str(e)}"
            
            # Print Router info if available
            if hasattr(agent_wrapper, "last_tool_used"):
                 self.console.print(f"Using Tool: [bold blue]{agent_wrapper.last_tool_used}[/bold blue]")

            self.console.print(Panel(Markdown(str(response)), title="🤖 Agent Response", border_style="blue"))
            
            # Ask for Grade
            grade = Prompt.ask(
                "Grade", 
                choices=["1", "2", "3", "4", "5", "skip", "exit"], 
                console=self.console
            )
            
            if grade == "exit":
                break
            if grade == "skip":
                continue
                
            feedback = Prompt.ask("Optional Feedback (Enter to skip)", console=self.console)
            
            self.results.append({
                "id": case["id"],
                "query": case["query"],
                "response": str(response),
                "human_score": int(grade),
                "human_feedback": feedback
            })
            
            self.save_results()  # Save after each to prevent data loss

def run_hitl():
    # Load Data
    data_path = os.path.join(os.path.dirname(__file__), "data", "comprehensive_eval_dataset.json")
    with open(data_path, "r") as f:
        dataset = json.load(f)
    
    hitl_cases = dataset.get("hitl_evals", [])
    
    # Init System
    console.print("⚙️ Initializing Agent...")
    app = build_graph()
    agent = LangGraphWrapper(app)
    
    grader = HITLGrader()
    grader.load_existing_results()
    grader.run_grading_session(hitl_cases, agent)

if __name__ == "__main__":
    run_hitl()
