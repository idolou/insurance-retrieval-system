import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from insurance_system.src.agents.manager_agent import ManagerAgent
from insurance_system.src.config import (
    EMBEDDING_MODEL,
    HIERARCHICAL_STORAGE_DIR,
    LLM_MODEL,
    SUMMARY_STORAGE_DIR,
)
from insurance_system.src.indices.hierarchical import load_hierarchical_retriever
from insurance_system.src.prompts import (
    CONTEXT_RECALL_EVAL_PROMPT,
    CONTEXT_RELEVANCY_EVAL_PROMPT,
    CORRECTNESS_EVAL_PROMPT,
)

load_dotenv()


async def evaluate_query(query, expected, agent, evaluator_llm):
    # Use Global settings for embeddings just in case
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    print(f"\n🔍 Query: {query}")

    # 1. Get Agent Response
    if hasattr(agent, "aquery"):
        agent_response = await agent.aquery(query)
    else:
        agent_response = agent.query(query)

    actual_answer = str(agent_response)
    print(f"🤖 Agent Answer: {actual_answer}")

    # --- 1. Answer Correctness ---
    # Does answer match ground truth?
    correctness_prompt = CORRECTNESS_EVAL_PROMPT.format(
        query=query, expected=expected, actual_answer=actual_answer
    )
    res_correct = await evaluator_llm.acomplete(correctness_prompt)

    # --- 2. Context Relevancy ---
    # Did agent use the correct index and relevant segments? (Inferred)
    relevancy_prompt = CONTEXT_RELEVANCY_EVAL_PROMPT.format(
        query=query, actual_answer=actual_answer
    )
    res_relevancy = await evaluator_llm.acomplete(relevancy_prompt)

    # --- 3. Context Recall ---
    # Did the system retrieve the correct chunk(s)? (Inferred via completeness vs Ground Truth)
    # We check if the answer contains all the detailed facts from the expected answer.
    recall_prompt = CONTEXT_RECALL_EVAL_PROMPT.format(
        query=query, expected=expected, actual_answer=actual_answer
    )
    res_recall = await evaluator_llm.acomplete(recall_prompt)

    print(
        f"⚖️  Judge Results:\n"
        f"        Correctness: {res_correct.text.strip()}\n"
        f"        Relevancy:   {res_relevancy.text.strip()}\n"
        f"        Recall:      {res_recall.text.strip()}"
    )

    return {
        "query": query,
        "correctness": res_correct.text,
        "relevancy": res_relevancy.text,
        "recall": res_recall.text,
    }


async def run_eval():

    # Setup
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    llm = OpenAI(model=LLM_MODEL)
    evaluator_llm = OpenAI(model=LLM_MODEL)  # Judge

    # Load components
    from llama_index.core import SimpleDirectoryReader

    from insurance_system.src.indices.hierarchical import create_hierarchical_index
    from insurance_system.src.indices.summary import create_summary_index

    if not os.path.exists(HIERARCHICAL_STORAGE_DIR) or not os.path.exists(
        SUMMARY_STORAGE_DIR
    ):
        print("Building indexes...")
        documents = SimpleDirectoryReader("insurance_system/data").load_data()
        create_hierarchical_index(documents, persist_dir=HIERARCHICAL_STORAGE_DIR)
        create_summary_index(documents, persist_dir=SUMMARY_STORAGE_DIR)

    # Load the system
    print("🚀 Initializing Agent for Evaluation...")
    hierarchical_retriever = load_hierarchical_retriever(
        persist_dir=HIERARCHICAL_STORAGE_DIR
    )
    manager = ManagerAgent(
        hierarchical_retriever,
        summary_persist_dir=SUMMARY_STORAGE_DIR,
        llm=llm,
    )

    # Test Cases
    test_cases = [
        {
            "query": "What was the date of the incident?",
            "expected": "November 16, 2024",
        },
        {"query": "Who is the policyholder?", "expected": "Alex Johnson"},
        {"query": "What is the total repair estimate cost?", "expected": "$12,400.00"},
        {
            "query": "Does Sarah Smith have a pre-existing condition?",
            "expected": "No information about Sarah Smith found in the documents.",
        },
        {
            "query": "Summarize the claim timeline.",
            "expected": "Incident on Nov 16, 2024. Valve shutoff same day. Inspection on Nov 18. Settlement reached Nov 22. Final payment Nov 24.",
        },
        {
            "query": "How many days passed between the incident and the adjuster inspection?",
            "expected": "2 days (Nov 16 to Nov 18).",
        },
        {
            "query": "What is the specific model of the TV claimed?",
            "expected": 'Samsung 65" QLED TV',
        },
        {
            "query": "Was the sofa replacement approved fully?",
            "expected": "No, it was partially denied. Only $250 for cleaning was approved initially.",
        },
    ]

    import json
    import re

    # Helper to parse JSON from LLM response
    def parse_json_score(text):
        try:
            # Try direct JSON parse
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            return {"score": 0, "explanation": "Failed to parse JSON"}

    # 3. Aggregate Results
    results = []
    correctness_scores = []
    relevancy_scores = []
    recall_scores = []

    for case in test_cases:
        res = await evaluate_query(
            case["query"], case["expected"], manager, evaluator_llm
        )

        # Parse scores
        c_score = parse_json_score(res["correctness"])
        rel_score = parse_json_score(res["relevancy"])
        rec_score = parse_json_score(res["recall"])

        # Update result with parsed data
        res["correctness"] = c_score
        res["relevancy"] = rel_score
        res["recall"] = rec_score

        res["correctness_score"] = c_score.get("score", 0)
        res["relevancy_score"] = rel_score.get("score", 0)
        res["recall_score"] = rec_score.get("score", 0)

        correctness_scores.append(res["correctness_score"])
        relevancy_scores.append(res["relevancy_score"])
        recall_scores.append(res["recall_score"])

        results.append(res)

    # 4. Summary Report
    total = len(results)
    c_pass = sum(correctness_scores)
    rel_pass = sum(relevancy_scores)
    rec_pass = sum(recall_scores)

    print("\n" + "=" * 50)
    print("📊 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Queries: {total}")
    print(f"Answer Correctness: {c_pass/total*100:.1f}% ({c_pass}/{total})")
    print(f"Context Relevancy:  {rel_pass/total*100:.1f}% ({rel_pass}/{total})")
    print(f"Context Recall:     {rec_pass/total*100:.1f}% ({rec_pass}/{total})")
    print("=" * 50)

    # Save to JSON
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Detailed results saved to '{output_file}'")


if __name__ == "__main__":
    if asyncio.get_event_loop().is_closed():
        asyncio.run(run_eval())
    else:
        # If running in a notebook or already running loop environment
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_eval())
