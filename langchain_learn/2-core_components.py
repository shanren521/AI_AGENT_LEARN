from langchain_openai import ChatOpenAI, OpenAI
from dataclasses import dataclass
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import (wrap_model_call, ModelRequest, ModelResponse, before_model, after_model,
                                         SummarizationMiddleware, dynamic_prompt, after_agent, HumanInTheLoopMiddleware,
                                         PIIMiddleware, TodoListMiddleware, LLMToolSelectorMiddleware, ToolRetryMiddleware,
                                         LLMToolEmulator, ClearToolUsesEdit, ContextEditingMiddleware, ShellToolMiddleware,
                                         HostExecutionPolicy, FilesystemFileSearchMiddleware)
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AIMessageChunk, AnyMessage
from langgraph.types import Command, Interrupt
from typing import Any, TypedDict, Literal, Union
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, SecretStr, Field
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig, Runnable
from langgraph.config import get_stream_writer
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy
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
    # 使用工具读取短期记忆
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


    # 从工具写入短期记忆
    class CustomState2(AgentState):
        user_name: str
    class CustomContext(BaseModel):
        user_id: str
    @tool
    def update_user_info(
            runtime: ToolRuntime[CustomContext, CustomState]
    ) -> Command:
        """Look up and update user info."""
        user_id = runtime.context.user_id
        name = "John Smith" if user_id == "user_123" else "Unknown user"
        return Command(update={
            "user_name": name,
            "messages": [
                ToolMessage("Successfully looked up user information", tool_call_id=runtime.tool_call_id)
            ]
        })

    @tool
    def greet(
            runtime: ToolRuntime[CustomContext, CustomState]
    ) -> str | Command:
        """Use this to greet the user once you found their info."""
        user_name = runtime.state.get("user_name", None)
        if user_name is None:
            return Command(update={
                "messages": [
                    ToolMessage(
                        "Please call the 'update_user_info' tool it will get and update the user's name.",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            })
        return f"Hello {user_name}!"

    agent = create_agent(
        model="gpt-5-nano",
        tools=[update_user_info, greet],
        state_schema=CustomState,
        context_schema=CustomContext,
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "greet the user"}]},
        context=CustomContext(user_id="user_123"),
    )

# Prompts 提示词
def dynamic_prompts():
    class CustomContext(TypedDict):
        user_name: str

    def get_weather(city: str) -> str:
        """Get the weather in a city."""
        return f"The weather in {city} is always sunny!"

    @dynamic_prompt
    def dynamic_system_prompt(request: ModelRequest) -> str:
        user_name = request.runtime.context["user_name"]
        system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
        return system_prompt

    agent = create_agent(
        model="gpt-5-nano",
        tools=[get_weather],
        middleware=[dynamic_system_prompt],
        context_schema=CustomContext,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        context=CustomContext(user_name="John Smith"),
    )
    for msg in result["messages"]:
        msg.pretty_print()

# Before model
def before_model_demo():
    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Keep only the last few messages to fit context window."""
        messages = state["messages"]
        if len(messages) <= 3:
            return None  # No changes needed

        first_msg = messages[0]

        recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
        new_messages = [first_msg] + recent_messages

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }

    agent = create_agent(
        "gpt-5-nano",
        tools=[],
        middleware=[trim_messages],
        checkpointer=InMemorySaver()
    )

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": "hi, my name is bob"}, config)
    agent.invoke({"messages": "write a short poem about cats"}, config)
    agent.invoke({"messages": "now do the same but for dogs"}, config)
    final_response = agent.invoke({"messages": "what's my name?"}, config)

    final_response["messages"][-1].pretty_print()


# After model
def after_model_demo():
    @after_model
    def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
        """Remove messages containing sensitive words."""
        STOP_WORDS = ["password", "secret"]
        last_message = state["messages"][-1]
        if any(word in last_message.content for word in STOP_WORDS):
            return {"messages": [RemoveMessage(id=last_message.id)]}
        return None

    agent = create_agent(
        model="gpt-5-nano",
        tools=[],
        middleware=[validate_response],
        checkpointer=InMemorySaver(),
    )

"""
Streaming 流媒体
"""

# **Agent progress  代理进度**
def agent_progress():
    def get_weather(city: str) -> str:
        """Get weather for a given city."""

        return f"It's always sunny in {city}!"

    agent = create_agent(
        model="gpt-5-nano",
        tools=[get_weather],
    )
    for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
            stream_mode="updates",
            version="v2",
    ):
        # API 接口需要进行json.loads
        if chunk["type"] == "updates":
            for step, data in chunk["data"].items():
                print(f"step: {step}")
                print(f"content: {data['messages'][-1].content_blocks}")


# **LLM tokens**
def llm_tokens():
    def get_weather(city: str) -> str:
        """Get weather for a given city."""

        return f"It's always sunny in {city}!"

    agent = create_agent(
        model="gpt-5-nano",
        tools=[get_weather],
    )
    for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
            stream_mode="messages",
            version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            print(f"node: {metadata['langgraph_node']}")
            print(f"content: {token.content_blocks}")
            print("\n")


# **Custom updates  自定义更新**
def custom_updates():
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        writer = get_stream_writer()  # ******
        # stream any arbitrary data
        writer(f"Looking up data for city: {city}")
        writer(f"Acquired data for city: {city}")
        return f"It's always sunny in {city}!"

    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=[get_weather],
    )

    for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
            stream_mode="custom",
            version="v2",
    ):
        if chunk["type"] == "custom":
            print(chunk["data"])


# **Stream multiple modes  流式多模式**
def stream_multi_mode():
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        writer = get_stream_writer()
        writer(f"Looking up data for city: {city}")
        writer(f"Acquired data for city: {city}")
        return f"It's always sunny in {city}!"

    agent = create_agent(
        model="gpt-5-nano",
        tools=[get_weather],
    )
    messages = {"messages": [{"role": "user", "content": "What is the weather in SF?"}]}
    for chunk in agent.stream(
            messages,
            stream_mode=["updates", "custom"],  # 顺序执行
            version="v2",
    ):
        print(f"stream_mode: {chunk['type']}")
        print(f"content: {chunk['data']}")
        print("\n")


# **Common patterns  常见模式**
def streaming_common_patterns():
    # 流式思维/推理tokens
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    model = ChatOpenAI(
        model="gpt-4.1",
        timeout=None,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True, "budget_tokens": 5000}  # vLLM 开启 thinking
        }
    )

    agent: Runnable = create_agent(
        model=model,
        tools=[get_weather],
    )

    for token, metadata in agent.stream(
            {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
            stream_mode="messages",
    ):
        if not isinstance(token, AIMessageChunk):
            continue
        reasoning = [b for b in token.content_blocks if b["type"] == "reasoning"]
        text = [b for b in token.content_blocks if b["type"] == "text"]
        if reasoning:
            print(f"[thinking] {reasoning[0]['reasoning']}", end="")
        if text:
            print(text[0]["text"], end="")

    # 流式工具调用
    agent = create_agent("openai:gpt-5.2", tools=[get_weather])

    def _render_message_chunk(token: AIMessageChunk) -> None:
        if token.text:
            print(token.text, end="|")
        if token.tool_call_chunks:
            print(token.tool_call_chunks)
        # N.B. all content is available through token.content_blocks

    def _render_completed_message(message: AnyMessage) -> None:
        if isinstance(message, AIMessage) and message.tool_calls:
            print(f"Tool calls: {message.tool_calls}")
        if isinstance(message, ToolMessage):
            print(f"Tool response: {message.content_blocks}")

    input_message = {"role": "user", "content": "What is the weather in Boston?"}
    for chunk in agent.stream(
            {"messages": [input_message]},
            stream_mode=["messages", "updates"],
            version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if isinstance(token, AIMessageChunk):
                _render_message_chunk(token)
        elif chunk["type"] == "updates":
            for source, update in chunk["data"].items():
                if source in ("model", "tools"):  # `source` captures node name
                    _render_completed_message(update["messages"][-1])

    # 访问已完成的消息
    class ResponseSafety(BaseModel):
        """Evaluate a response as safe or unsafe."""
        evaluation: Literal["safe", "unsafe"]

    safety_model = init_chat_model("openai:gpt-5.2")

    @after_agent
    def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Model-based guardrail: Use an LLM to evaluate response safety."""
        stream_writer = get_stream_writer()
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        model_with_tools = safety_model.bind_tools([ResponseSafety], tool_choice="any")
        result = model_with_tools.invoke(
            [
                {
                    "role": "system",
                    "content": "Evaluate this AI response as generally safe or unsafe."
                },
                {
                    "role": "user",
                    "content": f"AI response: {last_message.text}"
                }
            ]
        )

        stream_writer(result)
        tool_call = result.tool_calls[0]
        if tool_call["args"]["evaluation"] == "unsafe":
            last_message.content = "I cannot provide that response. Please rephrase your request."

        return None

    agent = create_agent("openai:gpt-5.2", tools=[get_weather], middleware=[safety_guardrail])

    for chunk in agent.stream(
            {"messages": [input_message]},
            stream_mode=["messages", "updates", "custom"],
            version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if isinstance(token, AIMessageChunk):
                _render_message_chunk(token)
        elif chunk["type"] == "updates":
            for source, update in chunk["data"].items():
                if source in ("model", "tools"):
                    _render_completed_message(update["messages"][-1])
        elif chunk["type"] == "custom":
            # access completed message in stream
            print(f"Tool calls: {chunk['data'].tool_calls}")

    # 人机交互式流媒体
    agent = create_agent(
        "openai:gpt-5.2",
        tools=[get_weather],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})],
        checkpointer=InMemorySaver()
    )
    def _render_interrupt(interrupt: Interrupt) -> None:
        interrupts = interrupt.value
        for request in interrupts["action_requests"]:
            print(request["description"])

    input_message = {
        "role": "user",
        "content": (
            "Can you look up the weather in Boston and San Francisco?"
        ),
    }
    config = {"configurable": {"thread_id": "some_id"}}
    interrupts = []
    for chunk in agent.stream(
            {"messages": [input_message]},
        config=config,
        stream_mode=["messages", "updates"],
        version="v2"
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if isinstance(token, AIMessageChunk):
                _render_message_chunk(token)
        elif chunk["type"] == "updates":
            for source, update in chunk["data"].items():
                if source in ("model", "tools"):
                    _render_completed_message(update["messages"][-1])
                if source == "__interrupt__":
                    interrupts.extend(update)
                    _render_interrupt(update[0])

    # 流式子代理
    weather_model = init_chat_model("openai:gpt-5.2")
    weather_agent = create_agent(
        model=weather_model,
        tools=[get_weather],
        name="weather_agent",
    )

    def call_weather_agent(query: str) -> str:
        """Query the weather agent."""
        result = weather_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        return result["messages"][-1].text

    supervisor_model = init_chat_model("openai:gpt-5.2")
    agent = create_agent(
        model=supervisor_model,
        tools=[call_weather_agent],
        name="supervisor",
    )

    current_agent = None
    for chunk in agent.stream(
            {"messages": [input_message]},
        stream_mode=["messages", "updates"],
        subgraphs=True,
        version="v2"
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if agent_name := metadata.get("lc_agent_name"):
                if agent_name != current_agent:
                    current_agent = agent_name
            if isinstance(token, AIMessage):
                _render_message_chunk(token)
        elif chunk["type"] == "updates":
            for source, update in chunk["data"].items():
                if source in ("model", "tools"):
                    _render_completed_message(update["messages"][-1])


# Disable streaming  禁用流媒体
def disable_streaming():
    model = ChatOpenAI(
        model="gtp-4.1",
        streaming=False
    )


"""
Structured output  结构化输出
"""

# **Provider strategy 供应商策略**
def provider_strategy():
    class ContactInfo(BaseModel):
        """Contact information for a person."""
        name: str = Field(description="The name of the person")
        email: str = Field(description="The email address of the person")
        phone: str = Field(description="The phone number of the person")

    agent = create_agent(
        model="gpt-4.1",
        response_format=ContactInfo
    )

    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
        }
    )
    print(result["structured_response"])


# **Tool calling strategy  工具调用策略**
def tool_calling_strategy(tools):
    class ProductReview(BaseModel):
        """Analysis of a product review."""
        rating: int | None = Field(description="The rating of the product", ge=1, le=5)
        sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
        key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

    agent = create_agent(
        model="gpt-5",
        tools=tools,
        response_format=ToolStrategy(ProductReview)
    )

    result = agent.invoke({
        "messages": [{"role": "user",
                      "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
    })

    response = result["structured_response"]

    # 自定义工具消息内容
    class MeetingAction(BaseModel):
        """Action items extracted from a meeting transcript."""
        task: str = Field(description="The specific task to be completed")
        assignee: str = Field(description="Person responsible for the task")
        priority: Literal["low", "medium", "high"] = Field(description="Priority level")
    agent = create_agent(
        model="gpt-5",
        tools=[],
        response_format=ToolStrategy(
            schema=MeetingAction,
            tool_message_content="Action item captured and added to meeting notes!"
        )
    )

    # Error handling  错误处理
    # 1. 多重结构化输出错误
    class ContactInfo(BaseModel):
        name: str = Field(description="Person's name")
        email: str = Field(description="Email address")

    class EventDetails(BaseModel):
        event_name: str = Field(description="Name of the event")
        date: str = Field(description="Event date")

    agent = create_agent(
        model="gpt-5",
        tools=[],
        response_format=ToolStrategy(Union[ContactInfo, EventDetails])  # Default: handle_errors=True
    )

    # 2. Schema validation error  架构验证错误
    class ProductRating(BaseModel):
        rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
        comment: str = Field(description="Review comment")

    agent = create_agent(
        model="gpt-5",
        tools=[],
        response_format=ToolStrategy(ProductRating),  # Default: handle_errors=True
        system_prompt="You are a helpful assistant that parses product reviews. Do not make any field or value up."
    )









if __name__ == "__main__":
    basic_usage()
