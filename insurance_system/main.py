import logging
import os
import sys

from dotenv import load_dotenv

# Configure Logging to reduce noise
logging.getLogger("llama_index").setLevel(logging.WARNING)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from insurance_system.src.agents.manager_agent import ManagerAgent
from insurance_system.src.config import (
    DEBUG,
    EMBEDDING_MODEL,
    HIERARCHICAL_STORAGE_DIR,
    LLM_MODEL,
    SUMMARY_STORAGE_DIR,
)
from insurance_system.src.indices.hierarchical import (
    create_hierarchical_index,
    load_hierarchical_retriever,
)
from insurance_system.src.indices.summary import create_summary_index

# Load environment variables
load_dotenv()


def main() -> None:
    print("⚙️ Initializing Insurance Claim Retrieval System...")

    # Check for API Keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please add 'OPENAI_API_KEY=sk-...' to your .env file.")
        return

    print("⬆️ Starting Insurance Retrieval System...")

    # Set Global Settings (OpenAI Only)
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    # Enable LlamaIndex Callback Handler for Debugging (Optional)
    if DEBUG:
        from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        Settings.callback_manager = callback_manager

    # Using centralized paths from config
    hierarchical_storage = HIERARCHICAL_STORAGE_DIR
    summary_storage = SUMMARY_STORAGE_DIR

    # Load Retrievers/Indices
    print("📂 Loading Indices (ChromaDB + OpenAI Embeddings)...")
    try:
        if not os.path.exists(hierarchical_storage) or not os.path.exists(
            summary_storage
        ):
            raise FileNotFoundError("Index storage not found.")

        hierarchical_retriever = load_hierarchical_retriever(
            persist_dir=hierarchical_storage
        )
    except Exception as e:
        print(f"❌ Error loading indices: {e}")
        print("💡 You strictly need to build indices first.")
        print("👉 Run build_index.py first.")
        return

    # Initialize Manager Agent
    print("🤖 Initializing Agents...")
    llm = OpenAI(model=LLM_MODEL)
    manager = ManagerAgent(
        hierarchical_retriever, summary_persist_dir=summary_storage, llm=llm
    )

    # Initialize Rich Console
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.text import Text

        CONSOLE = Console()
    except ImportError:
        print("❌ Error: 'rich' library not found.")
        print("👉 Please run: pip install rich")
        return

    CONSOLE.print(Panel.fit("[bold blue]Insurance Retrieval Agent[/bold blue]"))
    CONSOLE.print(
        "[green]✅ System Ready![/green] Type [bold red]'exit'[/bold red] to quit."
    )
    CONSOLE.print("Type [bold yellow]'1'[/bold yellow] to see more sample queries.\n")

    while True:
        try:
            user_input = CONSOLE.input("\n[bold cyan]👤 User > [/bold cyan]")
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            CONSOLE.print("\n[bold red]Goodbye![/bold red]")
            break

        if user_input.lower() in ["exit", "quit"]:
            CONSOLE.print("\n[bold yellow]Shutting down. Goodbye![/bold yellow]")
            break

        if not user_input.strip():
            continue

        # Expand Samples
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
            continue

        with CONSOLE.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            response = manager.query(user_input)

        # Render response as Markdown
        CONSOLE.print("\n[bold yellow]🤖 Agent >[/bold yellow]")
        CONSOLE.print(Markdown(str(response)))
        CONSOLE.print("-" * 50, style="dim")


if __name__ == "__main__":
    main()
