"""A simple math assistant built with LangGraph.

The graph has two nodes:
  agent -> asks the LLM what to do next
  tools -> runs any calculator tools the LLM asked for
and loops between them until the LLM replies without a tool call.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> float | str:
    """Divide a by b."""
    if b == 0:
        return "Error: cannot divide by zero."
    return a / b


TOOLS = [add, subtract, multiply, divide]

SYSTEM_PROMPT = (
    "You are a friendly math assistant. Always use the provided tools for "
    "arithmetic instead of calculating in your head. Reply with a short answer."
)


def build_graph(model=None):
    """Build and compile the agent graph."""
    if model is None:
        model = init_chat_model(os.getenv("MODEL", "openai:gpt-4o-mini"))
    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: MessagesState):
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver())


def main():
    load_dotenv()
    graph = build_graph()
    config = {"configurable": {"thread_id": "local-session"}}

    print("Math agent ready! Ask me anything (type 'quit' to exit).")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue

        inputs = {"messages": [("user", user_input)]}
        for step in graph.stream(inputs, config, stream_mode="updates"):
            for node, update in step.items():
                for message in update["messages"]:
                    if node == "tools":
                        print(f"  [tool] {message.name} -> {message.content}")
                    elif message.tool_calls:
                        for call in message.tool_calls:
                            print(f"  [agent] calling {call['name']}({call['args']})")
                    else:
                        print(f"Agent: {message.content}")


if __name__ == "__main__":
    main()
