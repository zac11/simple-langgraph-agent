from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agent import build_graph, divide


class FakeToolModel(GenericFakeChatModel):
    """A scripted chat model so the graph can be tested without an API key."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_calls_tool_then_answers():
    model = FakeToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "multiply", "args": {"a": 6, "b": 7}, "id": "call_1"}
                    ],
                ),
                AIMessage(content="6 times 7 is 42."),
            ]
        )
    )
    graph = build_graph(model)
    config = {"configurable": {"thread_id": "test"}}

    result = graph.invoke({"messages": [("user", "What is 6 times 7?")]}, config)

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_messages] == ["42.0"]
    assert result["messages"][-1].content == "6 times 7 is 42."


def test_divide_by_zero_returns_error_message():
    assert divide.invoke({"a": 1, "b": 0}) == "Error: cannot divide by zero."
