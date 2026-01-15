import json
import re
import sys
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class HardEvaluationResult(BaseModel):
    query: str
    passed: bool
    error: Optional[str] = None
    type: str


class HardEvaluator:
    """
    Evaluator for strict, rule-based tests (Guardrails).
    Supports:
    - Regex matching
    - JSON schema validation
    - Blacklist (Forbidden words)
    """

    def __init__(self, console=None):
        self.console = console or Console()

    def verify_regex(self, pattern: str, text: str) -> bool:
        """Checks if text matches the regex pattern."""
        return bool(re.search(pattern, text))

    def verify_json(self, schema: Dict[str, Any], text: str) -> bool:
        """Checks if text is valid JSON and matches a simple schema structure."""
        try:
            # First try to find JSON blob if embedded in text
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "{" in text and "}" in text:
                 # Naive extraction for simple cases
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]
            
            data = json.loads(json_str)

            # Simple Schema Validation
            if schema.get("type") == "object":
                if not isinstance(data, dict):
                    return False
                for prop in schema.get("required", []):
                    if prop not in data:
                        return False
            elif schema.get("type") == "array":
                if not isinstance(data, list):
                    return False
            
            return True
        except json.JSONDecodeError:
            return False
        except Exception:
            return False

    def verify_blacklist(self, pattern: str, text: str) -> bool:
        """
        Checks if text DOES NOT contain the forbidden pattern.
        Returns True if SAFE (pattern NOT found).
        Returns False if UNSAFE (pattern FOUND).
        """
        return not bool(re.search(pattern, text))

    def run_eval(self, test_case: Dict[str, Any], actual_output: str) -> HardEvaluationResult:
        eval_type = test_case.get("type")
        pattern = test_case.get("expected_pattern")
        
        passed = False
        error = None

        if eval_type == "regex":
            passed = self.verify_regex(pattern, actual_output)
            if not passed:
                error = f"Failed to match regex: {pattern}"
        elif eval_type == "json":
            schema = test_case.get("schema", {})
            passed = self.verify_json(schema, actual_output)
            if not passed:
                error = "Failed JSON validation or schema check"
        elif eval_type == "blacklist":
            passed = self.verify_blacklist(pattern, actual_output)
            if not passed:
                error = f"Contains forbidden pattern: {pattern}"
        else:
            error = f"Unknown eval type: {eval_type}"

        return HardEvaluationResult(
            query=test_case["query"],
            passed=passed,
            error=error,
            type=eval_type
        )

    def run_suite(self, test_cases: List[Dict[str, Any]], agent_runner) -> List[HardEvaluationResult]:
        results = []
        
        self.console.print("[bold cyan]Running Hard Evals...[/bold cyan]")

        # Create table for final summary
        table = Table(title="🛡️ Hard Evals (Guardrails) - Summary")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Router", style="blue")
        table.add_column("Status", style="bold")
        table.add_column("Actual (Truncated)", style="dim white")
        table.add_column("Error", style="red")

        for i, case in enumerate(test_cases):
            self.console.print(f"\n[{i+1}/{len(test_cases)}] Checking [bold]{case.get('id')}[/bold]")
            self.console.print(f"[dim]Query: {case['query']}[/dim]")
            
            # Run Agent
            router_status = "N/A"
            used_tool = "unknown"
            try:
                if hasattr(agent_runner, "query"):
                   response = agent_runner.query(case["query"])
                else:
                   response = agent_runner(case["query"]) 
                
                # Check Router
                if hasattr(agent_runner, "last_tool_used"):
                    used_tool = agent_runner.last_tool_used
                    expected_agent = case.get("agent_type", "needle") # Default to needle
                    
                    # Normalize for comparison
                    if expected_agent == used_tool:
                         router_status = f"[green]{used_tool}[/green]"
                    else:
                         router_status = f"[red]{used_tool} (Exp: {expected_agent})[/red]"

                response_text = str(response)
            except Exception as e:
                response_text = ""
                self.console.print(f"[red]Error:[/red] {e}")

            # Run Eval
            result = self.run_eval(case, response_text)
            results.append(result)

            status_style = "green" if result.passed else "red"
            status_text = "PASS" if result.passed else "FAIL"
            
            # Print immediate status
            self.console.print(f"Status: [{status_style}]{status_text}[/{status_style}] | Router: {router_status}")
            
            # Print full answer for debugging
            self.console.print(Panel(response_text, title="Agent Answer", border_style="dim white", expand=False))

            if not result.passed:
                self.console.print(f"   [yellow]Reason:[/yellow] {result.error}")
            
            # Truncate output for table
            truncated_output = (response_text[:50] + '...') if len(response_text) > 50 else response_text
            
            table.add_row(
                case.get("id", "N/A"),
                case["type"],
                router_status,
                f"[{status_style}]{status_text}[/{status_style}]",
                truncated_output.replace("\n", " "),
                result.error or ""
            )

        self.console.print("\n")
        self.console.print(table)
        
        # summary
        passed_count = sum(1 for r in results if r.passed)
        total = len(results)
        self.console.print(f"\n[bold]Summary:[/bold] {passed_count}/{total} passed ({passed_count/total*100:.1f}%)")
        
        return results
