import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from insurance_system.src.config import EMBEDDING_MODEL, LLM_MODEL, HIERARCHICAL_STORAGE_DIR, SUMMARY_STORAGE_DIR
from insurance_system.src.indices.hierarchical import load_hierarchical_retriever
from insurance_system.src.agents.manager_agent import ManagerAgent

load_dotenv()

def test_complex():
    print("Setting up Manager Agent...")
    try:
        Settings.llm = OpenAI(model=LLM_MODEL)
        Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
        
        # Load retriever
        retriever = load_hierarchical_retriever(persist_dir=HIERARCHICAL_STORAGE_DIR)
        
        # Initialize Agent
        manager = ManagerAgent(
            hierarchical_retriever=retriever,
            summary_persist_dir=SUMMARY_STORAGE_DIR,
            llm=OpenAI(model=LLM_MODEL)
        )
        
        question = "What was the time in Berlin when the incident occurred?"
        print(f"\nQuerying: {question}")
        
        # Using synchronous query for simplicity in script
        response = manager.query(question)
        print(f"\nResponse: {response}")
        
        # Check source nodes to see if it retrieved anything
        if hasattr(response, "source_nodes"):
             print(f"Retrieval Count: {len(response.source_nodes)}")

    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_complex()
