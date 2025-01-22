from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """Add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract a from b.

    Args:
        a: first int
        b: second int
    """
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, subtract, multiply, divide]

llm = ChatOllama(model="granite3.1-dense")
llm_with_tools = llm.bind_tools(tools=tools)

memory = MemorySaver()

# Specify the thread
config = {"configurable": {"thread_id": "1"}}


class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built
    pass


# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing airthmetic on a set of inputs."
)


# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools=tools))

# Add edge
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool_call -> tool_condition routes to tools
    # If the latest message (result) from assistant is not a tool_call -> tool_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=memory)

# Invoke Graph
responses = graph.invoke({"messages": HumanMessage(content="Hello!")}, config=config)
for m in responses["messages"]:
    m.pretty_print()


responses = graph.invoke(
    {"messages": HumanMessage(content="Add 3 and 4.")}, config=config
)
for m in responses["messages"]:
    m.pretty_print()


responses = graph.invoke(
    {"messages": HumanMessage(content="Multiply that by 2.")}, config=config
)
for m in responses["messages"]:
    m.pretty_print()
