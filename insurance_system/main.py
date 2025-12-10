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
    EMBEDDING_MODEL, LLM_MODEL, HIERARCHICAL_STORAGE_DIR, SUMMARY_STORAGE_DIR
)
from insurance_system.src.indices.hierarchical import (
    create_hierarchical_index, load_hierarchical_retriever)
from insurance_system.src.indices.summary import create_summary_index

# Load environment variables
load_dotenv()


def main() -> None:
    print("🚀 Initializing Insurance Claim Retrieval System...")

    # Check for API Keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please add 'OPENAI_API_KEY=sk-...' to your .env file.")
        return

    print("🚀 Starting Insurance Retrieval System...")

    # Set Global Settings (OpenAI Only)
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    # Enable LlamaIndex Callback Handler for Debugging (Optional)
    if os.getenv("DEBUG", "False").lower() == "true":
        from llama_index.core.callbacks import (CallbackManager,
                                                LlamaDebugHandler)

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

    # Interactive Loop
    print("\n✅ System Ready! Type 'exit' to quit.")
    print("Sample queries:")
    print(" - 'Summarize the claim timeline.'")
    print(" - 'What is the total repair estimate?'")
    print(" - 'Does the driver have a pre-existing condition?'")

    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response = manager.query(user_input)
        print(f"🤖 Agent: {response}")


if __name__ == "__main__":
    main()
