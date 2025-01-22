from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


tools = [multiply]

llm = ChatOllama(model="granite3.1-dense")
llm_with_tools = llm.bind_tools(tools=tools)


class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built
    pass


# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools=tools))

# Add edge
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool_call -> tool_condition routes to tools
    # If the latest message (result) from assistant is not a tool_call -> tool_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# Invoke Graph
responses = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in responses["messages"]:
    m.pretty_print()


responses = graph.invoke({"messages": HumanMessage(content="Multiply 4 and 6")})
for m in responses["messages"]:
    m.pretty_print()
