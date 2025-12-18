import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from langchain_core.messages import HumanMessage

from insurance_system.src.agents.manager import build_graph


async def test_query():
    app = build_graph()
    user_query = "What was the Total Vol recorded by Flow_Meter_01 at 11:15:00 AM?"
    print(f"Query: {user_query}")

    messages = [HumanMessage(content=user_query)]
    final_answer = ""
    async for event in app.astream_events({"messages": messages}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="")
                final_answer += content
    print("\n")


if __name__ == "__main__":
    asyncio.run(test_query())
