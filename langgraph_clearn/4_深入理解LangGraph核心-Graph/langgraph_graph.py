from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages, AnyMessage
from langchain.messages import AIMessage
from operator import add
from langchain_core.runnables import RunnableConfig
from langgraph.types import CachePolicy, Send, Command
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.cache.memory import InMemoryCache


"""
Graph主要包含三个元素：
    State：在整个应用当中共享的一种数据结构
    Node：一个处理数据的节点。
    Edge：表示Node之前的依赖关系。
"""

class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    return {"foo": state["user_input"] + " success"}

def node_2(state: OverallState) -> PrivateState:
    return {"bar": state["foo"] + " good "}

def node_3(state: PrivateState) -> OutputState:
    return {"graph_output": state["bar"] + "done"}

builder = StateGraph(state_schema=OverallState, input_schema=InputState, output_schema=OutputState)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()

result = graph.invoke({"user_input": "hello world"})
print(f"result: {result}")

# png_bytes = graph.get_graph().draw_mermaid_png()
# with open("graph_flow.png", "wb") as f:
#     f.write(png_bytes)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    list_field: Annotated[list[int], add]
    extra_field: int

def node1(state: State):
    new_message = AIMessage("Hello")
    return {"messages": [new_message], "list_field": [1, 2, 3], "extra_field": 1}

def node2(state: State):
    new_message = AIMessage("LangGraph")
    return {"messages": [new_message], "list_field": [4, 5, 6], "extra_field": 2}

graph = (StateGraph(State)
         .add_node("node1", node1)
         .add_node("node2", node2)
         .set_entry_point("node1")
         .add_edge("node1", "node2")
         .compile()
         )

input_message = {"role": "user", "content": "Hi"}
result = graph.invoke({"messages": [input_message], "list_field": [10, 20, 30]})

print(f"result: {result}")


class MemoryState(TypedDict):
    number: int
    user_id: str

class ConfigSchema(TypedDict):
    user_id: str

def memory_node1(state: MemoryState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    return {"number": state["number"] + 1, "user_id": user_id}

builder = StateGraph(MemoryState, context_schema=ConfigSchema)
builder.add_node("memory_node1", memory_node1, cache_policy=CachePolicy(ttl=5))

builder.add_edge(START, "memory_node1")
builder.add_edge("memory_node1", END)

graph = builder.compile(cache=InMemoryCache())
result = graph.invoke({"number": 1}, {"configurable": {"user_id": "123"}}, stream_mode="updates")
print(f"user_123: {result}")

result = graph.invoke({"number": 1}, {"configurable": {"user_id": "456"}}, stream_mode="updates")
print(f"user_456: {result}")


class ConditionState(TypedDict):
    number: int

def condition_node_1(state: ConditionState, config: RunnableConfig):
    return {"number": state["number"] + 1}

builder = StateGraph(ConditionState)

builder.add_node("condition_node_1", condition_node_1)

def routing_func(state: ConditionState) -> str:
    if state["number"] > 5:
        return "condition_node_1"
    else:
        return END

"""
通过map映射
def routing_func(state: ConditionState) -> bool:
    if state["number"] > 5:
        return True
    else:
        return False
    
builder.add_conditional_edges(START, routing_func, {True: "node_a", False: "node_b"})
"""

builder.add_edge("condition_node_1", END)

builder.add_conditional_edges(START, routing_func)

graph = builder.compile()
result = graph.invoke({"number": 6})
print(f"condition result: {result}")


"""
Send动态路由
    如果希望一个Node同时路由到多个Node，可以返回Send动态路由的方式实现
    参数：Node名称，Node输入
"""

class SendState(TypedDict):
    messages: Annotated[list[str], add]

class SendPrivateState(TypedDict):
    msg: str

def send_node_1(state: SendPrivateState) -> SendState:
    res = state["msg"] + " SendState"
    return {"messages": [res]}

builder = StateGraph(SendState)

builder.add_node("send_node_1", send_node_1)

def send_routing_func(state: SendState):
    result = []
    for message in state["messages"]:
        result.append(Send("send_node_1", {"msg": message}))
    return result

builder.add_conditional_edges(START, send_routing_func, ["send_node_1"])
builder.add_edge("send_node_1", END)

graph = builder.compile()
result = graph.invoke({"messages": ["hello", "graph"]})
print(f"send result: {result}")


# 通过Command返回
class CommandState(TypedDict):
    messages: Annotated[list[str], add]

def command_node_1(state: CommandState):
    new_message = []
    for message in state["messages"]:
        new_message.append(message + " command")
    return Command(
        goto=END,
        update={"messages": new_message}
    )

builder = StateGraph(CommandState)
builder.add_node("command_node_1", command_node_1)
builder.add_edge(START, "command_node_1")

graph = builder.compile()
result = graph.invoke({"messages": ["hello", "graph"]})
print(f"command result: {result}")


"""
Node可以是一个函数，也可以是一个子图
当触发了SubGraph代表的Node后，实际上是相当于重新调用了一次subgraph.invoke(state)方法。
"""

class SubGraphState(TypedDict):
    messages: Annotated[list[str], add]

def sub_node_1(state: SubGraphState) -> MessagesState:
    return MessagesState(messages=["response from subgraph"])

subgraph_builder = StateGraph(SubGraphState)
subgraph_builder.add_node("sub_node_1", sub_node_1)
subgraph_builder.add_edge(START, "sub_node_1")
subgraph_builder.add_edge("sub_node_1", END)

subgraph = subgraph_builder.compile()

builder = StateGraph(SubGraphState)
builder.add_node("subgraph_node", subgraph)
builder.add_edge(START, "subgraph_node")
builder.add_edge("subgraph_node", END)

graph = builder.compile()

result = graph.invoke({"messages": ["hello subgraph"]})
print(f"subgraph result: {result}")
