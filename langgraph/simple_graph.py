import random

from IPython.display import Image, display
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    graph_state: str


def node_1(state: State) -> dict:
    print("--- Node 1 ---")
    return {"graph_state": state["graph_state"] + " I am"}


def node_2(state: State) -> dict:
    print("--- Node 2 ---")
    return {"graph_state": state["graph_state"] + " happy!"}


def node_3(state: State) -> dict:
    print("--- Node 3 ---")
    return {"graph_state": state["graph_state"] + " sad!"}

def decide_mood(state: State) -> Literal["node_2", "node_3"]:
    # Often, we will use state to decide the next node to visit
    user_input = state["graph_state"]

    # Here, let's just do a 50/50 split between nodes 2, 3
    if random.random() < 0.5:
        # 50% of time, we return Node 2
        return "node_2"
    
    # 50% of time, we return Node 3
    return "node_3"

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

# Invoke
final_state = graph.invoke({"graph_state": "Hi, this is Lance."})
print(final_state["graph_state"])
