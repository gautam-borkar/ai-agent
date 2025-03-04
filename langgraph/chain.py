
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import MessagesState, StateGraph, START, END

llm = ChatOllama(model="granite3.1-dense")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# Invoke Graph
messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
print(messages)

messages = graph.invoke({"messages": HumanMessage(content="Multiply 4 and 6")})
print(messages)