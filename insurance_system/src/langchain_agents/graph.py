import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from insurance_system.src.config import LLM_MODEL
from insurance_system.src.langchain_agents.tools import get_langchain_tools


# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# 2. Initialize Model & Tools
tools = get_langchain_tools()
tool_node = ToolNode(tools)

# Use OpenAI for the router/supervisor
model = ChatOpenAI(model=LLM_MODEL)
model = model.bind_tools(tools)


# 3. Define Nodes
from langchain_core.messages import SystemMessage


# 3. Define Nodes
def supervisor_node(state: AgentState):
    messages = state["messages"]

    # Define the system prompt for smart routing
    system_prompt = SystemMessage(
        content="""You are a helpful assistant that answers questions about insurance claims.

                You have access to tools to retrieve information from the claim documents. Choose the BEST tool based on the question type:

                RETRIEVAL TOOLS (choose carefully):
                - summary_expert: ONLY for broad narrative overviews and "what happened" questions. NOT for specific topics.
                - needle_expert: The DEFAULT tool. Use this for specific facts like dates, amounts, names, exact numbers, precise details.

                TOOL SELECTION GUIDE:
                - "Summarize the medical treatment" → needle_expert (specific topic) or summary_expert (if very high level) - PREFER needle_expert for details.
                - "What is this claim about?" → summary_expert
                - "What was the deductible?" → needle_expert
                - "Who were the witnesses?" → needle_expert
                - "Timeline of events" → summary_expert
                - questions about specific costs or dates → needle_expert

                MCP TOOLS (if available):
                - Use 'get_historical_weather' if the user asks about weather at the time of the incident.

                IMPORTANT INSTRUCTIONS:
                1. Always use a tool to get information before answering.
                2. If the user asks for specific details, DO NOT use the summary tool. Use the needle_expert.
                3. After getting the tool result, answer the user's question directly.
"""
    )

    # Prepend system prompt if it's not already the first message (simple check)
    # In LangGraph, we usually pass the full history. We can prepend the system message to the invocation.

    response = model.invoke([system_prompt] + list(messages))
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM call resulted in tool calls, we continue to the tool node
    if last_message.tool_calls:
        return "tools"
    # Otherwise we end
    return END


# 4. Build Graph
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
    )

    workflow.add_edge("tools", "supervisor")

    return workflow.compile()
