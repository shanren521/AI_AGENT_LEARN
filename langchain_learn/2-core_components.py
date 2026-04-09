from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, before_model, after_model, SummarizationMiddleware
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.types import Command
from typing import Any
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, SecretStr
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
import ast

basic_model = ChatOpenAI(model="gemma4:e4b", base_url="http://localhost:11434/v1", api_key=SecretStr("ollama"))
advanced_model = ChatOpenAI(model="gemma4:e4b", base_url="http://localhost:11434/v1", api_key=SecretStr("ollama"))

system_msg = SystemMessage(
    "You are a helpful assistant that responds to questions with three exclamation marks."
)
human_msg = HumanMessage("What is the capital of France?")


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))


def one_demo():
    agent = create_agent(
        model=basic_model,
        tools=[],
        middleware=[dynamic_model_selection]
    )
    basic_model.invoke([system_msg, human_msg])


"""
Tools工具
"""

# Context
USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}


@dataclass
class UserContext:
    user_id: str


@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information"""
    user_id = runtime.context.user_id
    if user_id in UserContext:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"


# Short-Term memory
def short_term_demo():
    model = ChatOpenAI(model="gtp-4.1")


# memory(state)
def memory_state_demo():
    # access state 访问状态
    @tool
    def get_last_user_message(runtime: ToolRuntime) -> str:
        """Get the most recent message from the user."""
        messages = runtime.state["messages"]

        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
        return "No user messages found"

    @tool
    def get_user_preference(
            pref_name: str,
            runtime: ToolRuntime
    ) -> str:
        """Get a user preference value."""
        preferences = runtime.state.get("user_preferences", {})
        return preferences.get(pref_name, "Not set")

    # Command更新状态
    @tool
    def set_user_name(new_name: str) -> Command:
        """Set the user's name in the conversation state."""
        return Command(update={"user_name": new_name})


# Long-Term Memory
def long_term_demo():
    @tool
    def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
        """Look up user info"""
        store = runtime.store
        user_info = store.get(("users",), user_id)
        return str(user_info.value) if user_info else "Unknown user"

    @tool
    def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
        """Save user info"""
        store = runtime.store
        store.put(("users",), user_id, user_info)
        return "Successfully saved user info"

    model = ChatOpenAI(
        model="gemma4:e4b",
        base_url="http://localhost:11434/v1",
        api_key=SecretStr("ollama"),
        temperature=0.7
    )
    store = InMemoryStore()
    agent = create_agent(
        model,
        tools=[get_user_info, save_user_info],
        store=store
    )
    response1 = agent.invoke({
        "messages": [{"role": "user",
                      "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
    })
    print(f"response1: {response1}")

    # Second session: get user info
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
    })
    print(f"response2: {response2}")


# Stream writer 流媒体撰稿人
@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}"


# Execution info 执行信息
@tool
def log_execution_context(runtime: ToolRuntime) -> str:
    """Log execution identity information."""
    info = runtime.execution_info
    print(f"Thread: {info.thread_id}, Run: {info.run_id}")
    print(f"Attempt: {info.node_attempt}")
    return "done"


# Server info 服务信息
@tool
def get_assistant_scoped_data(runtime: ToolRuntime) -> str:
    """Fetch data scoped to the current assistant."""
    server = runtime.server_info
    if server is not None:
        print(f"Assistant: {server.assistant_id}, Graph: {server.graph_id}")
        if server.user is not None:
            print(f"User: {server.user.identity}")
    return "done"


"""
ToolNode 工具节点
"""


# Basic usage 基本用法
def basic_usage():
    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return str(ast.literal_eval(expression))

    tool_node = ToolNode([search, calculator])
    builder = StateGraph(MessagesState)
    builder.add_node("tools", tool_node)


# Tool return values 工具返回值
def tool_ret_values():
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"It is currently sunny in {city}."

    @tool
    def get_weather_data(city: str) -> dict:
        """Get structured weather data for a city."""
        return {
            "city": city,
            "temperature_c": 22,
            "conditions": "sunny",
        }

    @tool
    def set_language(language: str, runtime: ToolRuntime) -> Command:
        """Set the preferred response language."""
        return Command(
            update={
                "preferred_language": language,
                "messages": [
                    ToolMessage(
                        content=f"Language set to {language}",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )


# Error handing 错误处理
def error_handing(tools):
    # Default: catch invocation errors, re-raise execution errors
    tool_node = ToolNode(tools)

    # Catch all errors and return error message to LLM
    tool_node = ToolNode(tools, handle_tool_errors=True)

    # Custom error message
    tool_node = ToolNode(tools, handle_tool_errors="Something went wrong, please try again.")

    # Custom error handler
    def handle_error(e: ValueError) -> str:
        return f"Invalid input: {e}"

    tool_node = ToolNode(tools, handle_tool_errors=handle_error)

    # Only catch specific exception types
    tool_node = ToolNode(tools, handle_tool_errors=(ValueError, TypeError))


# Route with tools_condition 使用tools_condition进行路由
def route_with_tools(call_llm, tools):
    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")

    graph = builder.compile()


# State injection 状态注入
def state_injection():
    @tool
    def get_message_count(runtime: ToolRuntime) -> str:
        """Get the number of messages in the conversation."""
        messages = runtime.state["messages"]
        return f"There are {len(messages)} messages."

    tool_node = ToolNode([get_message_count])


"""
Short-term memory 短期记忆
"""


# Usage 用法
def use_func(get_user_info):
    agent = create_agent(
        "gemma4:e4b",
        tools=[get_user_info],
        checkpointer=InMemorySaver()
    )

    agent.invoke({"messages": [{{"role": "user", "content": "Hi! My name is Bob."}}]},
                 {"configurable": {"thread_id": "1"}})

    # In production在生产中
    DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()  # auto create tables in PostgreSQL
        agent = create_agent(
            "gemma4:e4b",
            tools=[get_user_info],
            checkpointer=checkpointer
        )


# Customizing agent memory  自定义代理内存
def custom_agent_memory():
    class CustomAgentState(AgentState):
        user_id: str
        preferences: dict

    agent = create_agent(
        "gemma4:e4b",
        tools=[get_account_info],
        state_schema=CustomAgentState,
        checkpointer=InMemorySaver()
    )
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "user_123",
            "preferences": {"theme": "dark"}
            },
        {"configurable": {"thread_id": "1"}})


# Common patterns  常见模式
def common_patterns():

    # 修剪消息
    @before_model
    def trim_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Keep only the last few messages to fit context window."""
        messages = state["messages"]

        if len(messages) <= 3:
            return None

        first_msg = messages[0]
        recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
        new_messages = [first_msg] + recent_messages

        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]
        }

    agent = create_agent(
        "gemma4:e4b",
        tools=[get_account_info],
        middleware=[trim_message],
        checkpointer=InMemorySaver()
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": "hi, my name is bob"}, config)
    agent.invoke({"messages": "write a short poem about cats"}, config)
    agent.invoke({"messages": "now do the same but for dogs"}, config)
    final_response = agent.invoke({"messages": "what's my name?"}, config)

    final_response["messages"][-1].pretty_print()


    # 删除消息
    def delete_message(state):
        messages = state["messages"]
        if len(messages) > 2:
            # 删除最早的两条消息
            return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
        # 删除所有消息
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}


    # 消息摘要

    agent = create_agent(
        model="gemma4:e4b",
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4.1",
                trigger=("tokens", 4000),
                keep=("messages", 20)

            )
        ],
        checkpointer=InMemorySaver()
    )

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    agent.invoke({"messages": "hi, my name is bob"}, config)
    agent.invoke({"messages": "write a short poem about cats"}, config)
    agent.invoke({"messages": "now do the same but for dogs"}, config)
    final_response = agent.invoke({"messages": "what's my name?"}, config)

    final_response["messages"][-1].pretty_print()

# Access memory 访问内存
def access_memory():
    # Tools 工具
    class CustomState(AgentState):
        user_id: str

    @tool
    def get_user_info(
            runtime: ToolRuntime
    ) -> str:
        """Look up user info."""
        user_id = runtime.state["user_id"]
        return "User is John Smith" if user_id == "user_123" else "Unknown user"

    agent = create_agent(
        model="gpt-5-nano",
        tools=[get_user_info],
        state_schema=CustomState,
    )

    result = agent.invoke({
        "messages": "look up user information",
        "user_id": "user_123"
    })
    print(result["messages"][-1].content)


    #


if __name__ == "__main__":
    basic_usage()
