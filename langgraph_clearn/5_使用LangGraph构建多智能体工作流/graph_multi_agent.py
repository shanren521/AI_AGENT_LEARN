import os
from typing import TypedDict, Annotated, NotRequired
from operator import add
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Literal, interrupt
import uuid

load_dotenv()

model="qwen3.7-max"

llm = ChatOpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    model=model,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

"""
消息持久化
    checkpoint 短期记忆
    store 长期记忆
"""

builder = StateGraph(MessagesState)

builder.add_node(call_model)
builder.add_edge(START, "call_model")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 设置请求的线程id，用来区分对话，也是用来利用存储的记忆
config = {
    "configurable": {"thread_id": "1"}
}

def demo_first():
    response = graph.invoke(
            {"messages": [{"role": "user", "content": "讲一下最新的agent技术"}]},
        config=config)
    print(f"response: {response}")

"""
Human-In-Loop 人类介入
    在任务执行过程中可以中断，人工验证后，继续执行
    必须指定thread_id，恢复任务的时候根据这个id恢复
    通过interrupt()方法中断，用Command(resume=True)来恢复执行，不一定是True，也可以是其他的
"""

class HILState(TypedDict):
    messages: Annotated[list[AnyMessage], add]

def call_llm(state: HILState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
    
def human_approve(state: HILState) -> Command[Literal["call_llm", "__end__"]]:
    is_approved = interrupt(
        {
            "questions": "是否同意调用大语言模型?"
        }
    )
    if is_approved:
        return Command(goto="call_llm")
    else:
        return Command(goto="__end__")

def interrupt_demo():
    builder = StateGraph(HILState)
    builder.add_node("human_approve", human_approve)
    builder.add_node("call_llm", call_llm)
    
    builder.add_edge(START, "human_approve")
    
    checkpointer = InMemorySaver()
    
    graph = builder.compile(checkpointer=checkpointer)
    
    thread_config = {"configurable": {"thread_id": "1"}}
    response = graph.invoke({"messages": [HumanMessage("简单介绍最新流行的agent技术有哪些?")]}, config=thread_config)
    print(response)
    final_response = graph.invoke(Command(resume=True), config=thread_config)
    print(final_response)

"""
Time Travel 时间回溯
    运行graph时，需要提供初始的输入消息
    运行时，指定thread_id线程ID，并且要基于这个线程ID，再指定一个checkpoint检查点，执行后将在每一个Node执行后，生成一个check_point_id
    指定thread_id和check_point_id，进行任务重演。重演前，可以选择更新state
"""

class TimeTravelState(TypedDict):
    author: NotRequired[str]
    joke: NotRequired[str]

def author_node(state: TimeTravelState):
    prompt = "帮我推荐一位受人们欢迎的作家, 只给出人名"
    author = llm.invoke(prompt)
    return {"author": author}

def joke_node(state: TimeTravelState):
    prompt = f"用作家：{state['author']}的风格，写一个简短的笑话"
    joke = llm.invoke(prompt)
    return {"joke": joke}

def time_travel_demo():
    builder = StateGraph(TimeTravelState)
    builder.add_node(author_node)
    builder.add_node(joke_node)
    builder.add_edge(START, "author_node")
    builder.add_edge("author_node", "joke_node")
    builder.add_edge("joke_node", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer)

    config = {"configurable": {"thread_id": uuid.uuid4()}}

    response = graph.invoke({}, config)
    print(response["author"])
    print("===" * 10)
    print(response["joke"])

    # 获取所有的历史状态
    states = list(graph.get_state_history(config))
    selected_state = states[1]
    # 更新选定的状态
    new_config = graph.update_state(selected_state.config, values={"author": "莫言"})
    print(new_config)
    result = graph.invoke(None, new_config)
    print(result)


def get_states(graph):
    states = list(graph.get_state_history(config))
    for state in states:
        print(state.next)
        print(state.config["configurable"]["checkpoint_id"])


"""
多智能体
    通过supervisor节点，对用户的输入进行分类，然后根据分类结果，选择不同的agent节点进行处理
    每个agent节点，选择不同的工具进行处理，最后将处理结果汇总，返回给supervisor节点
    supervisor节点再将结果返回给用户
"""

if __name__ == "__main__":
    # interrupt_demo()
    time_travel_demo()