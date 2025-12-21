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

from insurance_system.src.agents.manager import build_graph

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
                " - 'Look up the claim details using your tools. Find the location and date of the loss. Then check the weather for that location and date. Finally, explicitly state if the weather explains the loss based on the facts.'"
            )
            CONSOLE.print(
                " - 'First find the incident location (City, State) from the documents. Then, given the incident started at 10:22 AM in that location's time zone, what time was it in New York?'"
            )
            continue

        # Run LangGraph app
        CONSOLE.print("\n[bold yellow]🤖 Agent >[/bold yellow]")

        from rich.live import Live
        from rich.spinner import Spinner

        # Buffer for the final answer
        response_buffer = ""
        # Initial renderable (Spinner)
        current_renderable = Spinner(
            "dots", text="[bold green]Thinking...[/bold green]"
        )

        # Persist conversation history
        if "chat_history" not in locals():
            chat_history = []

        # Add user message to history
        chat_history.append(HumanMessage(content=user_input))

        with Live(current_renderable, console=CONSOLE, refresh_per_second=10) as live:
            try:
                # State tracking
                is_streaming_answer = False

                # Stream events with FULL history
                async for event in app.astream_events(
                    {"messages": chat_history}, version="v2"
                ):
                    kind = event["event"]
                    kind = event["event"]

                    # 1. Tool Call Start
                    if kind == "on_tool_start":
                        tool_name = event["name"]
                        if tool_name not in ["__start__", "_interruption"]:
                            inputs = event["data"].get("input")
                            # Print ABOVE the live display
                            live.console.print(
                                f"[dim]🛠️ Called Tool: [bold cyan]{tool_name}[/bold cyan][/dim]"
                            )
                            live.console.print(inputs, style="dim")

                    # 2. Tool Output
                    elif kind == "on_tool_end":
                        tool_name = event["name"]
                        output = event["data"].get("output")
                        content = output
                        if hasattr(output, "content"):
                            content = output.content

                        live.console.print(f"[dim]   → Result ({tool_name}):[/dim]")

                        # Use simple truncation for tool output to keep it clean
                        output_str = str(content)
                        if len(output_str) > 500:
                            output_str = output_str[:500] + "... (truncated)"
                        live.console.print(f"[dim]{output_str}[/dim]")

                    # 3. LLM Streaming (Main Answer)
                    elif kind == "on_chat_model_stream":
                        chunk_content = event["data"]["chunk"].content
                        if chunk_content:
                            if not is_streaming_answer:
                                is_streaming_answer = True
                                # Switch from Spinner to Markdown
                                # We treat the accumulated text as Markdown
                                pass

                            response_buffer += chunk_content
                            live.update(Markdown(response_buffer))

                            live.update(Markdown(response_buffer))

                # After interaction completes, update history with the Agent's response
                # Ideally, we should get the final state from the graph to ensure we have the tool calls too.
                # But a simple way for CLI is to append the final AIMessage if we only care about text.
                # BETTER: Invoke locally or verify state.
                # For this simple loop, let's append the final text response.
                from langchain_core.messages import AIMessage

                chat_history.append(AIMessage(content=response_buffer))

            except Exception as e:
                live.console.print(f"\n[bold red]Agent Error:[/bold red] {e}")
                continue

        # End of stream, print separator
        CONSOLE.print("-" * 50, style="dim")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
