from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage

llm = ChatOllama(model="granite3.1-dense")


class State(MessagesState):
    summary: str


# Define the logic to call the model (Langgraph node)
def call_model(state: State):

    # Get summary if it exist
    summary = state.get("summary", "")

    # If there us summary, then we add it
    if summary:

        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state[messages]
    else:
        messages = state["messages"]

    response = llm.invoke(messages)
    return {"messages": response}


# Summarize the messages
def summarize_messages(state: State):

    # First we get the existing summary
    summary = state.get("summary")

    # Create our summarization prompt
    if summary:

        # A summary already exists
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to pur history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State):
    """Return the next node to execute"""
    messages = state["messages"]

    # If there are more than six messages, then we summarize the coversation
    if len(messages) > 6:
        return "summarize_messages"

    # Otherwise we can just end
    return END


# Define a new graph
builder = StateGraph(state_schema=State)
builder.add_node("conversation", call_model)
builder.add_node(summarize_messages)

# Set the entrypoint as conversation
builder.add_edge(START, "conversation")
builder.add_conditional_edges("conversation", should_continue)
builder.add_edge("summarize_messages", END)

# Compile
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Create thread
config = {"configurable": {"thread_id": "1"}}

# Start conversation
input_message = HumanMessage(content="Hi! I'm Lance")
output_message = graph.invoke({"messages": [input_message]}, config=config)
[m.pretty_print() for m in output_message["messages"]]

input_message = HumanMessage(content="What is my name?")
output_message = graph.invoke({"messages": [input_message]}, config=config)
[m.pretty_print() for m in output_message["messages"]]

input_message = HumanMessage(content="I like 49ers!")
output_message = graph.invoke({"messages": [input_message]}, config=config)
[m.pretty_print() for m in output_message["messages"]]

input_message = HumanMessage(
    content="I like Nick Bosa, isn't he the highest paid defensive player?"
)
output_message = graph.invoke({"messages": [input_message]}, config=config)
[m.pretty_print() for m in output_message["messages"]]
