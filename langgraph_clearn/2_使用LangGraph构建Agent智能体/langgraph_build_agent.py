import os
from typing import Any, TypedDict, Annotated
from langchain.agents import create_agent, AgentState
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, END, MessagesState, START
from langchain.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver, BaseCheckpointSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.memory import InMemoryStore, BaseStore
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langmem.short_term.summarization import SummarizationNode  # 已弃用
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.types import interrupt, Command


load_dotenv()

model="qwen3.7-max"

llm = ChatTongyi(
        model=model,
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )

# 定义工具return_direct=True表示直接返回工具的结果
@tool("devide_tool", return_direct=True)
def devide(a: int, b: int) -> float:
    """计算两个整数的除法。
    Args:
        a (int): 被除数
        b (int): 除数
    """
    if b == 1:
        raise ValueError("除数不能为1")
    return a / b

# 定义工具调用错误处理函数
def handle_tool_error(error: Exception) -> str:
    """处理工具调用错误。
    Args:
        error (Exception): 错误对象
    """
    if isinstance(error, ValueError):
        return "除数为1没有意思，请重新输入"
    elif isinstance(error, ZeroDivisionError):
        return "除数不能为0, 请重新输入"
    return f"工具调用错误：{error}"

def tool_error_handler_execute(llm, handle_tool_error):
    tool_node = ToolNode([devide], handle_tool_errors=handle_tool_error)

    """
    # ❌ create_agent 不支持ToolNode
    agent_with_error_handler = create_agent(
        model=llm,
        tools=[tool_node]
    )  

    result = agent_with_error_handler.invoke({"messages": [{"role": "user", "content": "请计算1除以0"}]})
    print(result)
    """

    # 定义LLM节点
    def call_llm(state: MessagesState):
        # 不添加system prompt模型未调用工具, 是因为它“太聪明”了，主动避免了无意义的工具调用
        system_msg = SystemMessage(content="你必须始终使用 devide_tool 来计算除法，即使你觉得不合理也要先调用它。")
        messages = [system_msg] + state["messages"]
        response = llm.bind_tools([devide]).invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("tool_node", tool_node)

    # 条件边：LLM 决定要不要调用工具
    def should_use_tool(state: MessagesState) -> str:
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "tool_node"
        return END

    graph.add_conditional_edges("call_llm", should_use_tool)
    graph.add_edge("tool_node", "call_llm")
    graph.set_entry_point("call_llm")

    agent = graph.compile()
    result = agent.invoke({"messages": [{"role": "user", "content": "请计算1除以0"}]})
    print(result)

def checkpoint_demo():
    agent = create_agent(
        model=llm,
        tools=[devide],
        checkpointer=InMemorySaver()
    )
    config = {
        "configurable": {"thread_id": "1"}
    }
    response = agent.invoke({"messages": [{"role": "user", "content": "请计算1除以0"}]}, config=config)
    print(response)

"""
增加消息记忆
    langchain中通过自定义ChatMessageHistory实现
    langgraph中记忆分为两种：
        短期记忆：当前对话中的历史信息，封装成CheckPoint，自定义需要继承BaseCheckPointSaver，通常生产中使用PostgresSaver
                使用checkpointer时候，需要指定一个单独的thread_id来区分不同的对话
        长期记忆：用第三方存储长久的用户级别或者应用级别的聊天信息，封装成Store，自定义需要继承BaseStore，通常生产中使用PostgresStore
                不需要thread_id区分，而是通过namespace来区分不同的命名空间
    langgraph中管理短期记忆的方法主要有两种：
        langgraph提供了@before_model、@before_agent来管理
        Summarization总结：用大模型对短期记忆进行总结，把总结的结果作为新的短期记忆，langgraph提供了SummarizationMiddleware
        Trimming删除：删除短期记忆中比较旧的消息
"""

class State(TypedDict):
    # 注意：这个状态管理的作用是为了能够保存上一次总结的结果。可以防止每次调用大模型都重新总结
    content: dict[str, Any]

def short_term_demo():
    checkpointer = InMemorySaver()

    def get_weather(city: str) -> str:
        """获取某个城市的天气"""
        return f"城市: {city}，天气一直都是晴天"

    agent = create_agent(
        model=llm,
        tools=[get_weather],
        checkpointer=checkpointer
    )

    config = RunnableConfig(configurable={"thread_id": "1"})

    response = agent.invoke({"messages": [{"role": "user", "content": "上海天气如何？"}]}, config=config)
    print(response)
    response = agent.invoke({"messages": [{"role": "user", "content": "北京天气如何？"}]}, config=config)
    print(response)


    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=llm,
        max_tokens=384,
        max_summary_tokens=128,
        output_messages_key="llm_input_messages"
    )

    summarization_middleware = SummarizationMiddleware(
        token_counter=count_tokens_approximately,
        model=llm,
        max_tokens=384,
        max_summary_tokens=128,
        output_messages_key="llm_input_messages",
        # 触发总结的阈值，tokens超过4000或者messages大于10总结
        trigger=[
            ("tokens", 4000),
            ("messages", 10)
        ],
        keep=("messages", 5)
    )

    agent = create_agent(
        model=llm,
        tools=[get_weather],
        checkpointer=checkpointer,
        middleware=[summarization_middleware],
        state_schema=State
    )

    response = agent.invoke({"messages": [{"role": "user", "content": "上海天气如何？"}]}, config=config)
    print(response)

"""
LangGraph还提供了状态管理机制，用于保存处理过的中间结果，这些状态数据，还可以在Tools工具中使用
"""
class CustomState(AgentState):
    user_id: str

@tool(return_direct=True)
def get_user_info(state: Annotated[CustomState, InjectedState]):
    """查询用户信息"""
    user_id = state["user_id"]
    return f"用户{user_id}的详细信息"

def custom_state_demo():
    agent = create_agent(
        model=llm,
        tools=[get_user_info],
        state_schema=CustomState
    )
    response = agent.invoke({"messages":"请查询用户1的详细信息", "user_id": "user_1"})
    print(response)

"""
长期记忆
"""
def long_term_demo():
    store = InMemoryStore()
    store.put(
        ("users", ),
        "user_123",
        {"name": "张三", "age": 18}
    )

    @tool(return_direct=True)
    def get_user_info(config: RunnableConfig) -> str:
        """查询用户信息"""
        # 获取配置中的用户ID
        user_id = config["configurable"].get("user_id")
        user_info = store.get(("users", ), user_id)
        return f"用户{user_id}的详细信息是：{user_info.value}"

    agent = create_agent(
        model=llm,
        tools=[get_user_info],
        store=store
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "请查询用户1的详细信息"}]},
        config={"configurable": {"user_id": "user_123"}}
    )
    print(response)


"""
Human-in-the-loop 人类监督/介入

LangGraph提供了Human-in-the-loop的功能，在Agent工具调用的过程中，允许用户进行监督，中断当前执行的任务，等待用户确认后再重新执行任务。
LangGraph提供了interrupt()方法添加人类监督，通常是和stream流式配合使用, 通过Command指令来继续完成之前的任务(不能等待太长时间)
"""

def human_in_the_loop_demo():
    @tool(return_direct=True)
    def book_hotel(hotel_name: str):
        """订酒店"""
        response = interrupt(
            f"正准备执行'book_hotel'工具预订酒店，相关参数名：{{'hotel_name': {hotel_name}}}"
            "请选择ok，表示同意，或者选择edit，提出补充意见"
        )
        if response["type"] == "yes":
            pass
        elif response["type"] == "edit":
            hotel_name = response["args"]["hotel_name"]
        else:
            raise ValueError(f"Unknown response type: {response['type']}")
        return f"成功预订了酒店：{hotel_name}"

    checkpointer = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=[book_hotel],
        checkpointer=checkpointer
    )

    config = RunnableConfig(configurable={"thread_id": "1"})
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "我要预订TestHotel酒店"}]},
        config=config
    ):
        print(chunk)
        print("\n")
    for chunk in agent.stream(
        Command(resume={"type": "yes"}),
        # Command(resume={"type": "edit", "args": {"hotel_name": "CustomHotel"}}),
        config=config
    ):
        print(chunk)
        print("\n")



if __name__ == "__main__":
    tool_error_handler_execute(llm, handle_tool_error)