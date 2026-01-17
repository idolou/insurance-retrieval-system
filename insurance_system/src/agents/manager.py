"""
Manager Agent (Supervisor)

This module defines the LangGraph state machine that acts as the central Supervisor/Manager.
It orchestrates the conversation, routing user queries to the appropriate specialized agents
(Needle Agent for facts, Summary Agent for overviews) or external tools.
"""

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from insurance_system.src.agents.tools import get_langchain_tools
from insurance_system.src.utils.config import LLM_MODEL
from insurance_system.src.utils.prompts import MANAGER_SYSTEM_PROMPT


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
    system_prompt = SystemMessage(content=str(MANAGER_SYSTEM_PROMPT))

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
