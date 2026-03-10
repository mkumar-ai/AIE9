"""An agent graph with a post-response clarity check loop.

After the agent responds, a secondary node evaluates whether the response
is clear and easy to understand — free of unnecessary jargon.
If clear, end; otherwise, loop back and try again with a hard limit.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from app.state import MessagesState
from app.models import get_chat_model
from app.tools import get_tool_belt


class ClarityResult(BaseModel):
    is_clear: bool = Field(description="Whether the response is clear, simple, and free of unnecessary jargon")


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_clarity(state: MessagesState):
    """Decide whether to execute tools or run the clarity evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "clarity"


_clarity_prompt = ChatPromptTemplate.from_template(
    "Given an initial query and a final response, determine if the response "
    "is clear, simple, and easy to understand for a non-technical person. "
    "It should avoid unnecessary jargon and be straightforward.\n\n"
    "Initial Query:\n{initial_query}\n\n"
    "Final Response:\n{final_response}"
)


def clarity_node(state: MessagesState) -> dict:
    """Evaluate clarity of the latest response relative to the initial query."""
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="CLARITY:END")]}

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    structured_model = get_chat_model(model_name="gpt-4.1-mini").with_structured_output(ClarityResult)
    result = (_clarity_prompt | structured_model).invoke(
        {
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        }
    )

    decision = "Y" if result.is_clear else "N"
    return {"messages": [AIMessage(content=f"CLARITY:{decision}")]}


def clarity_decision(state: MessagesState):
    """Terminate on 'CLARITY:Y' or loop otherwise; guard against infinite loops."""
    if any(getattr(m, "content", "") == "CLARITY:END" for m in state["messages"][-1:]):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if "CLARITY:Y" in text:
        return "end"
    return "continue"


def build_graph():
    """Build an agent graph with an auxiliary clarity evaluation node."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("clarity", clarity_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_clarity,
        {"action": "action", "clarity": "clarity"},
    )
    graph.add_conditional_edges(
        "clarity",
        clarity_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()