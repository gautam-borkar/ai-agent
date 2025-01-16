
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import MessagesState, StateGraph, START, END

def multiply(a:int, b:int) -> int:
  """Multiple a and b
  
  Args:
    a: First int
    b: Second int
  """
  return a * b

llm_with_tools = ChatOllama(model="llama3.2", temperature=0).bind_tools(tools=[multiply])

class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build Graph
# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)

# Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# Invoke Graph
messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    m.pretty_print()

messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
for m in messages['messages']:
    m.pretty_print()
