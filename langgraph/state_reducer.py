from operator import add
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


# Graph State
class State(TypedDict):
    foo: Annotated[list[int], add]


def node_1(state: State) -> State:
    print("--- Node_1 ---")
    return {"foo": [state["foo"][-1] + 1]}


def node_2(state: State) -> State:
    print("--- Node_2 ---")
    return {"foo": [state["foo"][-1] + 1]}


def node_3(state=State) -> State:
    print("--- Node_3 ---")
    return {"foo": [state["foo"][-1] + 1]}


# Build graph
builder = StateGraph(state_schema=State)

# Node
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Add edge
builder.add_edge(start_key=START, end_key="node_1")
builder.add_edge(start_key="node_1", end_key="node_2")
builder.add_edge(start_key="node_1", end_key="node_3")
builder.add_edge(start_key="node_2", end_key=END)
builder.add_edge(start_key="node_3", end_key=END)

# Invoke Graph
graph = builder.compile()
response = graph.invoke({"foo": [1]})
print(response)
