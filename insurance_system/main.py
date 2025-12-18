import asyncio
import os
import sys

# Suppress tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure module visibility
sys.path.append(os.getcwd())

import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables
load_dotenv()

from insurance_system.src.langchain_agents.graph import build_graph

# Initialize Rich Console
CONSOLE = Console()


async def main():
    CONSOLE.print(Panel.fit("[bold blue]Insurance Retrieval Agent[/bold blue]"))

    # Initialize Graph
    CONSOLE.print("⚙️ Initializing Insurance Claim Retrieval System...")
    try:
        app = build_graph()
    except Exception as e:
        CONSOLE.print(f"[bold red]❌ Error initializing graph:[/bold red] {e}")
        return

    CONSOLE.print(
        "[green]✅ System Ready![/green] Type [bold red]'exit'[/bold red] to quit."
    )
    CONSOLE.print("Type [bold yellow]'1'[/bold yellow] to see more sample queries.\n")

    while True:
        try:
            # Note: console.input is blocking, but acceptable for this CLI loop
            user_input = CONSOLE.input("\n[bold cyan]👤 User > [/bold cyan]")
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            CONSOLE.print("\n[bold red]Goodbye![/bold red]")
            break

        if user_input.lower() in ["exit", "quit", "q"]:
            CONSOLE.print("\n[green]Shutting down. Goodbye![/green]")
            break

        if user_input.lower() == "clear":
            CONSOLE.clear()
            continue

        if not user_input.strip():
            continue

        # Expand Samples logic matched from main.py
        if user_input.strip() == "1":
            CONSOLE.print("\n[bold yellow]Expanded Sample Queries:[/bold yellow]")
            CONSOLE.print(" [bold]Needle (Fact) Queries:[/bold]")
            CONSOLE.print(" - 'What is the date of the police report?'")
            CONSOLE.print(" - 'Who is the witness listed?'")
            CONSOLE.print(" - 'What is the deductible amount?'")
            CONSOLE.print(" - 'List all line items related to drywall repairs.'")

            CONSOLE.print("\n [bold]Summary (High-Level) Queries:[/bold]")
            CONSOLE.print(" - 'Give me a summary of the medical treatment history.'")
            CONSOLE.print(
                " - 'Explain the sequence of events leading to the settlement.'"
            )
            CONSOLE.print(" - 'What are the main arguments for liability?'")

            CONSOLE.print("\n [bold]Complex (Planning) Queries:[/bold]")
            CONSOLE.print(
                " - 'Identify the claimant and calculate the total payout vs the policy limit.'"
            )
            CONSOLE.print(
                " - 'Compare the initial estimate with the final settlement figure.'"
            )
            CONSOLE.print(
                " - 'What was the time in Berlin when the incident occurred?'"
            )
            continue

        # Run LangGraph app
        CONSOLE.print("\n[bold yellow]🤖 Agent >[/bold yellow]")

        # Start the spinner
        with CONSOLE.status(
            "[bold green]Thinking...[/bold green]", spinner="dots"
        ) as status:
            try:
                messages = [HumanMessage(content=user_input)]
                final_content = ""
                first_token_received = False

                # Use astream_events v2 to capture both tool calls and token streaming
                async for event in app.astream_events(
                    {"messages": messages}, version="v2"
                ):
                    kind = event["event"]

                    # 1. Tool Call Start
                    if kind == "on_tool_start":
                        tool_name = event["name"]
                        # Skip internal LangChain tools or trivial ones if any
                        if tool_name not in ["__start__", "_interruption"]:
                            inputs = event["data"].get("input")
                            CONSOLE.print(
                                f"[dim]🛠️ Called Tool: [bold cyan]{tool_name}[/bold cyan][/dim]"
                            )
                            CONSOLE.print(inputs, style="dim")

                    # 2. Tool Output
                    elif kind == "on_tool_end":
                        tool_name = event["name"]
                        output = event["data"].get("output")

                        # Extract content if it's a ToolMessage object
                        content = output
                        if hasattr(output, "content"):
                            content = output.content

                        CONSOLE.print(f"[dim]   → Result ({tool_name}):[/dim]")

                        # Try to parse as JSON for pretty (but faded) printing
                        try:
                            if isinstance(content, str):
                                json_obj = json.loads(content)
                                # Use json.dumps to get a string, then print it dim
                                formatted_json = json.dumps(json_obj, indent=2)
                                CONSOLE.print(f"[dim]{formatted_json}[/dim]")
                            elif isinstance(content, (dict, list)):
                                formatted_json = json.dumps(content, indent=2)
                                CONSOLE.print(f"[dim]{formatted_json}[/dim]")
                            else:
                                # Truncate if too long and not JSON
                                output_str = str(content)
                                if len(output_str) > 500:
                                    output_str = output_str[:500] + "... (truncated)"
                                CONSOLE.print(f"[dim]{output_str}[/dim]")
                        except Exception:
                            # Fallback for non-JSON text
                            output_str = str(content)
                            if len(output_str) > 500:
                                output_str = output_str[:500] + "... (truncated)"
                            CONSOLE.print(f"[dim]{output_str}[/dim]")

                    # 3. LLM Streaming (Main Answer)
                    elif kind == "on_chat_model_stream":
                        # Stop spinner on first token
                        if not first_token_received:
                            status.stop()
                            first_token_received = True

                        content = event["data"]["chunk"].content
                        if content:
                            # Print content directly to console as it arrives
                            CONSOLE.print(content, end="")
                            final_content += content

                # Print a newline at the end since streaming used end=""
                CONSOLE.print()
                CONSOLE.print("-" * 50, style="dim")

            except Exception as e:
                CONSOLE.print(f"\n[bold red]Agent Error:[/bold red] {e}")
                continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
