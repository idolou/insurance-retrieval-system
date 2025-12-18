import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from insurance_system.src.agents.manager_agent import ManagerAgent
from insurance_system.src.config import (EMBEDDING_MODEL,
                                         HIERARCHICAL_STORAGE_DIR, LLM_MODEL,
                                         SUMMARY_STORAGE_DIR)
from insurance_system.src.indices.hierarchical import \
    load_hierarchical_retriever
from insurance_system.src.prompts import (CONTEXT_RECALL_EVAL_PROMPT,
                                          CONTEXT_RELEVANCY_EVAL_PROMPT,
                                          CORRECTNESS_EVAL_PROMPT)

load_dotenv()


from insurance_system.src.evaluation.models import EvaluationResult


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

    # helper for structured output
    async def get_eval_result(prompt_template, **kwargs):
        try:
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=EvaluationResult,
                prompt=prompt_template,
                llm=evaluator_llm,
                verbose=True,
            )
            return await program.acall(**kwargs)
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return EvaluationResult(score=0, explanation=f"Evaluation failed: {e}")

    # --- 1. Answer Correctness ---
    # Does answer match ground truth?
    res_correct = await get_eval_result(
        CORRECTNESS_EVAL_PROMPT,
        query=query,
        expected=expected,
        actual_answer=actual_answer,
    )

    # --- 2. Context Relevancy ---
    # Did agent use the correct index and relevant segments? (Inferred)
    res_relevancy = await get_eval_result(
        CONTEXT_RELEVANCY_EVAL_PROMPT,
        query=query,
        expected=expected,  # The prompt expects 'expected' though relevancy check only technically needs expected for context
        actual_answer=actual_answer,
    )

    # --- 3. Context Recall ---
    # Did the system retrieve the correct chunk(s)? (Inferred via completeness vs Ground Truth)
    # We check if the answer contains all the detailed facts from the expected answer.
    res_recall = await get_eval_result(
        CONTEXT_RECALL_EVAL_PROMPT,
        query=query,
        expected=expected,
        actual_answer=actual_answer,
    )

    print(
        f"⚖️  Judge Results:\n"
        f"        Correctness: Score={res_correct.score}, Exp={res_correct.explanation}\n"
        f"        Relevancy:   Score={res_relevancy.score}, Exp={res_relevancy.explanation}\n"
        f"        Recall:      Score={res_recall.score}, Exp={res_recall.explanation}"
    )

    return {
        "query": query,
        "correctness": res_correct,
        "relevancy": res_relevancy,
        "recall": res_recall,
    }


async def run_eval():

    # Setup
    Settings.llm = OpenAI(model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    llm = OpenAI(model=LLM_MODEL)

    # Initialize Evaluator (Judge)
    from insurance_system.src.config import EVALUATOR_MODEL

    if "claude" in EVALUATOR_MODEL:
        from llama_index.llms.anthropic import Anthropic

        print(f"👨‍⚖️ Judge initialized with Claude: {EVALUATOR_MODEL}")
        evaluator_llm = Anthropic(model=EVALUATOR_MODEL)
    else:
        print(f"👨‍⚖️ Judge initialized with OpenAI: {LLM_MODEL}")
        evaluator_llm = OpenAI(model=LLM_MODEL)

    # Load components
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.program import LLMTextCompletionProgram

    from insurance_system.src.indices.hierarchical import \
        create_hierarchical_index
    from insurance_system.src.indices.summary import create_summary_index

    if not os.path.exists(HIERARCHICAL_STORAGE_DIR) or not os.path.exists(
        SUMMARY_STORAGE_DIR
    ):
        print("Building indexes...")
        documents = SimpleDirectoryReader("insurance_system/data").load_data()
        create_hierarchical_index(documents, persist_dir=HIERARCHICAL_STORAGE_DIR)
        create_summary_index(documents, persist_dir=SUMMARY_STORAGE_DIR)

    # Load the system
    print("⚙️ Initializing Agent for Evaluation...")
    hierarchical_retriever = load_hierarchical_retriever(
        persist_dir=HIERARCHICAL_STORAGE_DIR
    )
    manager = ManagerAgent(
        hierarchical_retriever,
        summary_persist_dir=SUMMARY_STORAGE_DIR,
        llm=llm,
    )

    # Test Cases
    # NOTE: Update expected values based on the actual content of RAG_Enhanced_Claim_HO-2024-8892.pdf
    test_cases = [
        # Basic Fact Retrieval
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
        # Summary/Timeline Queries
        {
            "query": "Summarize the claim timeline.",
            "expected": "Incident on Nov 16, 2024. Valve shutoff same day. Inspection on Nov 18. Settlement reached Nov 22. Final payment Nov 24.",
        },
        # Calculation Queries (using math tools)
        {
            "query": "How many days passed between the incident and the adjuster inspection?",
            "expected": "2 days (Nov 16 to Nov 18).",
        },
        # Specific Detail Retrieval
        {
            "query": "What is the specific model of the TV claimed?",
            "expected": "Samsung QN90C Series",
        },
        {
            "query": "Was the sofa replacement approved fully?",
            "expected": "No, it was partially denied. Only $250 for cleaning was approved initially.",
        },
        # {
        #     "query": "What is the total repair cost plus 15% tax?",
        #     "expected": "$14,260.00 (or $12,400.00 + 15% = $14,260.00)",
        # },
        # # Time Conversion Queries (using convert_time tool)
        # {
        #     "query": "What was the time in Berlin when the incident occurred?",
        #     "expected": "The incident occurred at 10:22 AM CST (America/Chicago) on November 16, 2024, which is 5:22 PM CET (Europe/Berlin) on the same day.",
        # },
        # {
        #     "query": "What was the time in Tokyo when the incident occurred?",
        #     "expected": "The incident occurred at 10:22 AM CST (America/Chicago) on November 16, 2024, which is 1:22 AM JST (Asia/Tokyo) on November 17, 2024.",
        # },
    ]

    import json

    # 3. Aggregate Results
    results = []
    correctness_scores = []
    relevancy_scores = []
    recall_scores = []

    for case in test_cases:
        res = await evaluate_query(
            case["query"], case["expected"], manager, evaluator_llm
        )

        # Extract model outputs
        c_score = res["correctness"].score
        rel_score = res["relevancy"].score
        rec_score = res["recall"].score

        # Add flat scores for analysis
        res["correctness_score"] = c_score
        res["relevancy_score"] = rel_score
        res["recall_score"] = rec_score

        # Convert Pydantic objects to dict for JSON serialization
        res["correctness"] = res["correctness"].dict()
        res["relevancy"] = res["relevancy"].dict()
        res["recall"] = res["recall"].dict()

        correctness_scores.append(c_score)
        relevancy_scores.append(rel_score)
        recall_scores.append(rec_score)

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
