from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (ModelCallLimitMiddleware, ToolCallLimitMiddleware, ModelFallbackMiddleware, PIIMiddleware)
from langchain.agents.middleware.types import ResponseT, StateT
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph.state import StateGraph
from langchain.tools import tool, ToolRuntime
from langgraph.runtime import Runtime
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import Command
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (wrap_model_call, ModelRequest, ModelResponse, before_model, after_model,
                                         SummarizationMiddleware, dynamic_prompt, after_agent, HumanInTheLoopMiddleware,
                                         PIIMiddleware, TodoListMiddleware, LLMToolSelectorMiddleware, ToolRetryMiddleware,
                                         LLMToolEmulator, ClearToolUsesEdit, ContextEditingMiddleware, ShellToolMiddleware,
                                         HostExecutionPolicy, FilesystemFileSearchMiddleware, wrap_tool_call, ExtendedModelResponse,
                                         AgentMiddleware, hook_config)
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware, CompiledSubAgent, SubAgent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
from typing import Any, Callable, Annotated

from langgraph.typing import ContextT
from typing_extensions import NotRequired
import re


"""
middleware 中间件
"""


"""
Prebuilt middleware 预构建中间件
"""

# **Model call limit  模型呼叫限制**
def model_call_limit():
    agent = create_agent(
        model="gpt-4.1",
        checkpointer=InMemorySaver(),
        tools=[],
        middleware=[
            ModelCallLimitMiddleware(
                thread_limit=10,
                run_limit=5,
                exit_behavior="end"
            )
        ]
    )

# **Tool call limit  工具调用限制**
def tool_call_limit(search_tool, database_tool):
    agent = create_agent(
        model="gpt-4.1",
        tools=[search_tool, database_tool],
        middleware=[
            ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
            ToolCallLimitMiddleware(
                tool_name="search",
                thread_limit=5,
                run_limit=3
            )
        ]
    )

# **Model fallback  模型的备选**
def model_fallback():
    agent = create_agent(
        model="gpt-4.1",
        tools=[],
        middleware=[
            ModelFallbackMiddleware(
                "gpt-5",
                "claude-3-7-sonnet"
            )
        ]
    )


# **PII detection  PII 检测**
def pii_detection():
    agent = create_agent(
        model="",
        tools=[],
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True)
        ]
    )

    # 自定义功能 - 带有验证功能的复杂检测逻辑
    # Method 1: Regex pattern string
    agent1 = create_agent(
        model="",
        tools=[],
        middleware=[
            PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block")
        ]
    )

    # Method 2: Compiled regex pattern
    agent2 = create_agent(
        model="",
        tools=[],
        middleware=[
            PIIMiddleware("phone_number", detector=re.compile(r'\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}'), strategy='mask')

        ]
    )

    # Method 3: Custom detector function
    def detect_ssn(content: str) -> list[dict[str, str | int]]:
        """
        Detect SSN with validation.
        :param content:
        :return: a list of dictionaries with 'text', 'start', and 'end' keys
        """
        matches = []
        pattern = r'\d{3}-\d{2}-\d{4}'
        for match in re.finditer(pattern, content):
            ssn = match.group(0)
            # Validate: first 3 digits shouldn't be 000, 666, or 900-999
            first_three = int(ssn[:3])
            if first_three not in [0, 666] and not(900 <= first_three <=999):
                matches.append({'text': ssn, 'start': match.start(), 'end': match.end()})

    agent3 = create_agent(
        model='',
        tools=[],
        middleware=[
            PIIMiddleware('ssn', detector=detect_ssn, strategy='hash')
        ]
    )


# **LLM tool selector  LLM 工具选择器**
def llm_tool_selector(tool1, tool2, tool3):
    agent = create_agent(
        model='',
        tools=[tool1, tool2, tool3],
        middleware=[
            LLMToolSelectorMiddleware(model='', max_tools=3, always_include=['search'])
        ]
    )

# **Tool retry  工具重试**
def tool_retry(search_tool, database_tool):
    agent = create_agent(
        model='',
        tools=[search_tool, database_tool],
        middleware=[
            ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0)
        ]
    )

# **LLM tool emulator  LLM 工具模拟器**
def tool_emulator(get_weather, search_database, send_email):
    agent = create_agent(
        model='',
        tools=[get_weather, search_database, send_email],
        middleware=[
            LLMToolEmulator()
        ]
    )

# **Context editing  上下文编辑**
def context_editing():
    agent = create_agent(
        model='',
        tools=[],
        middleware=[
            # trigger设置触发编辑的token数，keep保留最新工具的数量
            ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100000, keep=3)])
        ]
    )

# **Shell tool  Shell 工具**
def shell_tool(search_tool):
    agent = create_agent(
        model='',
        tools=[search_tool],
        middleware=[
            ShellToolMiddleware(
                workspace_root='/workspace',
                execution_policy=HostExecutionPolicy()
            )
        ]
    )

# **File search文件搜索**
def file_search():
    agent = create_agent(
        model='',
        tools=[],
        middleware=[
            FilesystemFileSearchMiddleware(
                root_path='/workspace',
                use_ripgrep=True
            )
        ]
    )

# **Filesystem middleware  文件系统中间件**
def file_system():
    agent = create_agent(
        model='',
        middleware=[
            FilesystemMiddleware(
                backend=None,
                system_prompt='Write to the filesystem when...',
                custom_tool_descriptions={
                    'ls': 'Use the ls tool when...',
                    'read_file': 'Use the read_file tool to...'
                }
            )
        ]
    )

# **Filesystem middleware  文件系统中间件**
def filesystem_middleware():
    store = InMemoryStore()
    agent = create_agent(
        model='',
        store=store,
        middleware=[
            FilesystemMiddleware(
                backend=CompositeBackend(
                    default=StateBackend(),
                    routes={'/memories/': StoreBackend()}
                ),
                custom_tool_descriptions={
                    'ls': 'Use the ls tool when...',
                    'read_file': 'Use the read_file tool to...'
                }
            )
        ]
    )

# **Subagent  次级代理人**
def subagent():
    @tool
    def get_weather(city: str) -> str:
        """Get the weather in a city"""
        return f"The weather in {city} is sunny."

    def make_backend(runtime: ToolRuntime):
        # 可以基于 runtime 动态决定目录、配置等
        return FilesystemBackend(
            root_dir="/workspace",
            virtual_mode=True,
        )
    sub_agent = SubAgent(
            name="weather",
            description="This subagent can get weather in cities.",
            system_prompt="Use the get_weather tool to get the weather in a city.",
            tools=[get_weather],
            model="gpt-4.1",
            middleware=[],
        )
    subagent_middleware = SubAgentMiddleware(
        backend=make_backend,
        subagents=[sub_agent]
    )
    agent = create_agent(
        model='',
        middleware=[subagent_middleware]
    )

    # 构建图作为子代理
    def create_weather_graph():
        workflow = StateGraph(...)
        return workflow.compile()

    weather_graph = create_weather_graph()

    weather_subagent = CompiledSubAgent(
        name='weather',
        description='This subagent can get weather in cities.',
        runnable=weather_graph
    )
    subagent_middleware = SubAgentMiddleware(
        subagents=[weather_subagent],
        backend=make_backend
    )
    agent = create_agent(
        model='',
        middleware=[subagent_middleware]
    )


"""
Custom middleware  自定义中间件
"""

# **Hooks 钩子**
def hooks_demo():
    # 1 node-style hooks
    @before_model(can_jump_to=['end'])
    def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state['messages']) >= 50:
            return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
        return None

    @after_model
    def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None


    # 2 wrap-style hooks
    @wrap_model_call
    def retry_model(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        for attempt in range(3):
            try:
                return handler(request)
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Retry {attempt + 1}/3 after error: {e}")

# **State updates 状态更新**
def state_updates():
    # node-style hooks
    class TrackingState(AgentState):
        model_call_count: NotRequired[int]

    @after_model(state_schema=TrackingState)
    def increment_after_model(state: TrackingState, runtime: Runtime) -> dict[str, Any] | None:
        return {'model_call_count': state.get('model_call_count', 0) + 1}

    # wrap-style hooks
    class UsageTrackingState(AgentState):
        """Agent State with token usage tracking"""
        last_model_call_tokens: NotRequired[int]

    @wrap_model_call(state_schema=UsageTrackingState)
    def track_usage(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ExtendedModelResponse:
        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={'last_model_call_tokens': 150})
        )

    # 多个中间件的组合
    def _last_wins(_a: str, b: str) -> str:
        """Reducer: last writer wins (outer overwrites inner)."""
        return b

    class CustomMiddlewareState(AgentState):
        """Agent state: trace_layer uses last-wins (outer wins), messages use additive reducer."""

        # Non-reducer field with last-wins: both middleware write; outermost value wins
        trace_layer: NotRequired[Annotated[str, _last_wins]]

    class OuterMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
        ) -> ExtendedModelResponse:
            response = handler(request)
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={
                    'trace_layer': 'outer',
                    'messages': [SystemMessage(content="outer ran")]
                })
            )

    class InnerMiddleware(AgentMiddleware):
        """Adds trace_layer and message. Outer adds to same keys; trace_layer: outer wins, messages: additive."""
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ExtendedModelResponse:
            response = handler(request)
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={
                    'trace_layer': 'inner',
                    'messages': [SystemMessage(content='inner ran')]
                })
            )

# **Create middleware 创建中间件**
def create_middleware():
    # 1 Decorator-based middleware 基于装饰器的中间件
    @before_model
    def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    @wrap_model_call
    def retry_model(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(3):
            try:
                return handler(request)
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Retry {attempt + 1}/3 after error: {e}")

    agent = create_agent(
        model='',
        middleware=[log_before_model, retry_model],
        tools=[...]
    )

    # 2 Class-based middleware  基于类的中间件
    class LoggingMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"About to call model with {len(state['messages'])} messages")
            return None

        def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"Model returned: {state['messages'][-1].content}")
            return None

        async def abefore_model(
                self, state: AgentState, runtime: Runtime
        ) -> dict[str, Any] | None:
            # Async version of before_model
            return None

        async def aafter_model(
                self, state: AgentState, runtime: Runtime
        ) -> dict[str, Any] | None:
            # Async version of after_model
            print(f"Model returned: {state['messages'][-1].content}")
            return None

    agent = create_agent(
        model='',
        middleware=[LoggingMiddleware()],
        tools=[...]
    )

# **Custom state schema  自定义状态模式**

def custom_state_schema():
    class CustomState(AgentState):
        model_call_count: NotRequired[int]
        user_id: NotRequired[str]

    @before_model(state_schema=CustomState, can_jump_to=['end'])
    def check_call_limit(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
        count = state.get('model_call_count', 0)
        if count > 10:
            return {'jump_to': 'end'}
        return None

    @after_model(state_schema=CustomState)
    def increment_counter(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
        return {'model_call_count': state.get('model_call_count', 0) + 1}

    agent = create_agent(
        model='',
        middleware=[check_call_limit, increment_counter],
        tools=[]
    )
    agent.invoke({
        'messages': [HumanMessage('hello')],
        'model_call_count': 0,
        'user_id': 'user_123'
    })

# **Agent jumps 代理跳转**
def agent_jumps():
    @after_model
    @hook_config(can_jump_to=['end'])
    def check_for_blocked(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state['messages'][-1]
        if 'BLOCKED' in last_message.content:
            return {
                'messages': [AIMessage('I cannot respond to that request.')],
                'jump_to': 'end'
            }

        return None

## **Examples 示例**

def examples():
    # 1 Dynamic prompt 动态prompt
    @wrap_model_call
    def add_context(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        new_context = list(request.system_message.content_blocks) + [
            {'type': 'text', 'text': 'Additional context'}
        ]
        new_system_message = SystemMessage(content=new_context)
        return handler(request.override(system_message=new_system_message))

    # 2 Dynamic model selection  动态模型选择
    complex_model = init_chat_model("claude-sonnet-4-6")
    simple_model = init_chat_model("claude-haiku-4-5-20251001")

    @wrap_model_call
    def dynamic_model(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        if len(request.messages) > 10:
            model = complex_model
        else:
            model = simple_model
        return handler(request.override(model=model))

    # 3 Dynamically selecting tools 动态选择工具
    @wrap_model_call
    def select_tools(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Middleware to select relevant tools based on state/context."""
        # Select a small, relevant subset of tools based on state/context
        relevant_tools = select_relevant_tools(request.state, request.runtime)
        return handler(request.override(tools=relevant_tools))

    agent = create_agent(
        model='',
        tools=all_tools,
        middleware=[select_tools]
    )













