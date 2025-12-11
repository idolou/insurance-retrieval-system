import logging
import os
import sys

from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from insurance_system.src.config import EMBEDDING_MODEL, LLM_MODEL, PROJECT_ROOT, HIERARCHICAL_STORAGE_DIR, SUMMARY_STORAGE_DIR
from insurance_system.src.indices.hierarchical import create_hierarchical_index
from insurance_system.src.indices.summary import create_summary_index

load_dotenv()


def build_indices() -> None:
    """
    Builds the hierarchical and summary indices from the claim documents in the 'data' folder.
    """
    print("🚀 Starting Data Indexing Process...")

    # 1. Verification
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please check your .env file.")
        return

    # Use PROJECT_ROOT (insurance_system folder) to locate data
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory '{data_dir}' not found.")
        print("Please ensure your PDFs are in this folder.")
        return

    # 2. Configure Settings (OpenAI)
    print("⚙️  Configuring OpenAI Embeddings...")
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    # 3. Load Documents
    print(f"📂 Loading Documents from {data_dir}...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"✅ Loaded {len(documents)} document(s).")

    # 4. Build Hierarchical Index
    print("\n🏗️  Building Hierarchical Index (Fact Retrieval)...")
    create_hierarchical_index(documents, persist_dir=HIERARCHICAL_STORAGE_DIR)
    print(f"✅ Hierarchical Index saved to {HIERARCHICAL_STORAGE_DIR}")

    # 5. Build Summary Index
    print("\n🏗️  Building Summary Index (High-level Retrieval)...")
    create_summary_index(documents, persist_dir=SUMMARY_STORAGE_DIR)
    print(f"✅ Summary Index saved to {SUMMARY_STORAGE_DIR}")

    print(
        "\n✨ Indexing Complete! You can now run the retrieval system using 'main.py'."
    )


if __name__ == "__main__":
    build_indices()
