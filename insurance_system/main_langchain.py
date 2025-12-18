import os
import sys
import asyncio

# Suppress tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure module visibility
sys.path.append(os.getcwd())

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage

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
        # Run LangGraph app with streaming
        CONSOLE.print("\n[bold yellow]🤖 Agent >[/bold yellow]")

        try:
            messages = [HumanMessage(content=user_input)]
            final_content = ""

            # Use astream_events v2 to capture both tool calls and token streaming
            async for event in app.astream_events({"messages": messages}, version="v2"):
                kind = event["event"]

                # 1. Tool Call Start
                if kind == "on_tool_start":
                    tool_name = event["name"]
                    # Skip internal LangChain tools or trivial ones if any
                    if tool_name not in ["__start__", "_interruption"]:
                        inputs = event["data"].get("input")
                        CONSOLE.print(
                            f"[dim]🛠️ Called Tool: [bold cyan]{tool_name}[/bold cyan] with args: {inputs}[/dim]"
                        )

                # 2. Tool Output (Optional: print result)
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    output = event["data"].get("output")
                    # Truncate long outputs for readability
                    output_str = str(output)
                    if len(output_str) > 200:
                        output_str = output_str[:200] + "..."
                    CONSOLE.print(f"[dim]   → Result: {output_str}[/dim]")

                # 3. LLM Streaming (Main Answer)
                elif kind == "on_chat_model_stream":
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
