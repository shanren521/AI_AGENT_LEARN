"""Advanced usage 高级用法"""
import asyncio
import json
from typing import Any, Callable
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.messages import AIMessage, AnyMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import (PIIMiddleware, HumanInTheLoopMiddleware, AgentMiddleware, AgentState,
                                         hook_config, SummarizationMiddleware,
                                         dynamic_prompt, ModelResponse, ModelRequest, after_model, before_model,
                                         wrap_model_call)
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from mcp.types import TextContent, LoggingMessageNotificationParams, ElicitRequestParams, ElicitResult
from mcp.shared.context import RequestContext
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext



"""
Built-in guardrails  内置防护机制
"""

# PII detection PII检测
def pii_detection(customer_service_tool, email_tool):
    agent = create_agent(
        model='',
        tools=[customer_service_tool, email_tool],
        middleware=[
            PIIMiddleware(
                'email',
                strategy='redact',
                apply_to_input=True
            ),
            PIIMiddleware(
                'credit_card',
                strategy='mask',
                apply_to_input=True
            ),
            PIIMiddleware(
                'api_key',
                detector=r'sk-[a-zA-Z0-9]{32}',
                strategy='block',
                apply_to_input=True
            )
        ]
    )

# Human-in-the-loop  人机参与
def human_in_loop(search_tool, send_email_tool, delete_database_tool):
    agent = create_agent(
        model='',
        tools=[search_tool, send_email_tool, delete_database_tool],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    'send_email': True,
                    'delete_database': True,
                    'search': False
                }
            )
        ],
        checkpointer=InMemorySaver()
    )
    config = RunnableConfig(configurable={"thread_id": "some_id"})
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Send an email to the team"}]},
        config=config
    )

    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config  # Same thread ID to resume the paused conversation
    )


"""
Custom guardrails  自定义防护机制
"""

# Before agent guardrails  在代理之前防护
class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""
    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=['end'])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state['messages']:
            return None

        first_message = state['messages'][0]
        if first_message.type != 'human':
            return None

        content = first_message.content.lower()
        for keyword in self.banned_keywords:
            if keyword in content:
                return {
                    'messages': [{
                        'role': 'assistant',
                        'content': 'I cannot process requests containing inappropriate content. Please rephrase your request.'
                    }],
                    'jump_to': 'end'
                }

        return None

def before_agent_guardrails(search_tool, calculator_tool):
    agent = create_agent(
        model='',
        tools=[search_tool, calculator_tool],
        middleware=[
            ContentFilterMiddleware(
                banned_keywords=['hack', 'exploit', 'malware']
            )
        ]
    )

# after agent guardrails 在代理之后防护
class SafetyGuardrailMiddleware(AgentMiddleware):
    """Model-based guardrail: Use an LLM to evaluate response safety."""

    def __init__(self):
        super().__init__()
        self.safety_model = init_chat_model('gpt-4.1')

    @hook_config(can_jump_to=['end'])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state['messages']:
            return None

        last_message = state['messages'][-1]
        if not isinstance(last_message, AIMessage):
            return None

        safety_prompt = f"""Evaluate if this response is safe and appropriate.
        Respond with only 'SAFE' or 'UNSAFE'.

        Response: {last_message.content}"""

        result = self.safety_model.invoke([{'role': 'user', 'content': safety_prompt}])
        if "UNSAFE" in result.content:
            last_message.content = "I cannot provide that response. Please rephrase your request."

        return None

def after_agent_guardrails(search_tool, calculator_tool):
    agent = create_agent(
        model='',
        tools=[search_tool, calculator_tool],
        middleware=[SafetyGuardrailMiddleware()]
    )

# Combine multiple guardrails 组合多个防护
def combine_multi_guard(search_tool, send_email_tool):
    agent = create_agent(
        model='',
        tools=[search_tool, send_email_tool],
        middleware=[
            ContentFilterMiddleware(banned_keywords=['hack', 'exploit']),
            PIIMiddleware('email', strategy='redact', apply_to_input=True),
            PIIMiddleware('email', strategy='redact', apply_to_output=True),
            HumanInTheLoopMiddleware(interrupt_on={'send_email': True}),
            SafetyGuardrailMiddleware()
        ]
    )

"""
Runtime 运行时
"""

# Access 使用权
def access_demo():
    @dataclass
    class Context:
        user_name: str
        user_id: str

    agent = create_agent(
        model='',
        tools=[...],
        context_schema=Context
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        context=Context(user_name="John Smith")
    )

    # 1 Inside tools  内部工具
    @tool
    def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
        """Fetch the user's email preferences from the store."""
        user_id = runtime.context.user_id
        preferences: str = "The user prefers you to write a brief and polite email."
        if runtime.store:
            if memory := runtime.store.get(('users', ), user_id):
                preferences = memory.value['preferences']
        return preferences

    # 2 Execution info and server info inside tools 工具内部包含执行信息和服务器信息
    @tool
    def context_aware_tool(runtime: ToolRuntime) -> str:
        """A tool that uses execution and server info."""
        info = runtime.execution_info
        print(f"Thread: {info.thread_id}, Run: {info.run_id}")

        server = runtime.server_info
        if server is not None:
            print(f"Assistant: {server.assistant_id}")
            if server.user is not None:
                print(f"User: {server.user.identity}")

        return "done"

    # 3 Inside middleware  中间件内部
    @dynamic_prompt
    def dynamic_system_prompt(request: ModelRequest) -> str:
        user_name = request.runtime.context.user_name
        system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
        return system_prompt

    @before_model
    def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
        print(f"Processing request for user: {runtime.context.user_name}")
        return None

    @after_model
    def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
        print(f"Completed request for user: {runtime.context.user_name}")
        return None

    agent = create_agent(
        model="gpt-5-nano",
        tools=[...],
        middleware=[dynamic_system_prompt, log_before_model, log_after_model],
        context_schema=Context
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        context=Context(user_name="John Smith")
    )

    # 4 Execution info and server info inside middleware 中间件内部的执行信息和服务器信息
    @before_model
    def auth_gate(state: AgentState, runtime: Runtime) -> dict | None:
        """Block unauthenticated users when running on LangGraph Server."""
        server = runtime.server_info
        if server is not None and server.user is None:
            raise ValueError("Authentication required")
        print(f"Thread: {runtime.execution_info.thread_id}")
        return None

"""
Model context  模型上下文
"""


@dataclass
class CustomContext:
    user_id: str
    user_role: str
    deployment_env: str
    user_jurisdiction: str
    industry: str
    compliance_frameworks: list[str]
    api_key: str

# System Prompt系统提示
def system_prompt_demo():
    # 1.1 从状态中获取消息计数或对话上下文
    @dynamic_prompt
    def state_aware_prompt(request: ModelRequest) -> str:
        message_count = len(request.messages)

        base = "You are a helpful assistant."
        if message_count > 10:
            base += '\nThis is a long conversation - be extra concise.'
        return base
    state_agent = create_agent(
        model='',
        tools=[...],
        middleware=[state_aware_prompt]
    )

    # 1.2 从长期记忆中获取用户偏好设置

    @dynamic_prompt
    def store_aware_prompt(request: ModelRequest) -> str:
        user_id = request.runtime.context.user_id

        store = request.runtime.store
        user_prefs = store.get(('preferences',), user_id)

        base = 'You are a helpful assistant.'

        if user_prefs:
            style = user_prefs.value.get('communication_style', 'balanced')
            base += f'\nUser prefers {style} responses'
        return base

    store_agent = create_agent(
        model='',
        tools=[...],
        middleware=[store_aware_prompt],
        context_schema=CustomContext,
        store=InMemoryStore()
    )

    # 1.3 从运行时上下文中访问用户 ID 或配置

    @dynamic_prompt
    def context_aware_prompt(request: ModelRequest) -> str:
        user_role = request.runtime.context.user_role
        env = request.runtime.context.deployment_env

        base = "You are a helpful assistant."

        if user_role == 'admin':
            base += '\nYou have admin access. You can perform all operations.'
        elif user_role == 'viewer':
            base += "\nYou have read-only access. Guide users to read operations only."

        if env == 'production':
            base += "\nBe extra careful with any data modifications."

        return base

    context_agent = create_agent(
        model='',
        tools=[...],
        middleware=[context_aware_prompt],
        context_schema=CustomContext
    )

# Message 消息
def message_context_demo():
    # 1 当与当前查询相关时，从状态中注入已上传的文件上下文
    @wrap_model_call
    def inject_file_context(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Inject context about files user has uploaded this session."""
        uploaded_files = request.state.get('uploaded_files', [])
        if uploaded_files:
            file_descriptions = []
            for file in uploaded_files:
                file_descriptions.append(
                    f'- {file["name"]} ({file["type"]}): {file["summary"]}'
                )

            file_context = f"""Files you have access to in this conversation:
{chr(10).join(file_descriptions)}
Reference these files when answering questions."""

            messages = [
                *request.messages,
                {'role': 'user', 'content': file_context}
            ]
            request = request.override(messages=messages)
        return handler(request)

    agent = create_agent(
        model='',
        tools=[...],
        middleware=[inject_file_context]
    )

    # 2 将用户在store的邮件写作风格导入到邮件草稿指导中
    @wrap_model_call
    def inject_writing_style(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Inject user's email writing style from Store."""
        user_id = request.runtime.context.user_id

        store = request.runtime.store

        writing_style = store.get(('writing_style', ), user_id)

        if writing_style:
            style = writing_style.value
            style_context = f"""Your writing style:
- Tone: {style.get('tone', 'professional')}
- Typical greeting: "{style.get('greeting', 'Hi')}"
- Typical sign-off: "{style.get('sign_off', 'Best')}"
- Example email you've written:
{style.get('example_email', '')}"""
            messages = [
                *request.messages,
                {'role': 'user', 'content': style_context}
            ]
            request = request.override(messages=messages)
        return handler(request)

    agent = create_agent(
        model='',
        tools=[...],
        middleware=[inject_writing_style],
        context_schema=CustomContext,
        store=InMemoryStore()
    )

    # 3 根据用户管辖范围，从运行时上下文注入合规性规则
    @wrap_model_call
    def inject_compliance_rules(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Inject compliance constraints from Runtime Context."""
        jurisdiction = request.runtime.context.user_jurisdiction
        industry = request.runtime.context.industry
        frameworks = request.runtime.context.compliance_frameworks

        rules = []
        if 'GDPR' in frameworks:
            rules.append("- Must obtain explicit consent before processing personal data")
            rules.append("- Users have right to data deletion")
        if "HIPAA" in frameworks:
            rules.append("- Cannot share patient health information without authorization")
            rules.append("- Must use secure, encrypted communication")
        if industry == "finance":
            rules.append("- Cannot provide financial advice without proper disclaimers")

        if rules:
            compliance_context = f"""Compliance requirements for {jurisdiction}:
{chr(10).join(rules)}
"""
            messages = [
                *request.messages,
                {'role': 'user', 'content': compliance_context}
            ]
            request = request.override(messages=messages)

        return handler(request)

    agent = create_agent(
        model='',
        tools=[...],
        middleware=[inject_compliance_rules],
        context_schema=CustomContext
    )

# Tools 工具
def context_tools_demo(public_search, private_search, advanced_search,
                       search_tool, analysis_tool, export_tool):
    # 1 Defining tools  定义工具
    @tool(parse_docstring=True)
    def search_orders(
            user_id: str,
            status: str,
            limit: int = 10
    ) -> str:
        """Search for user orders by status.

            Use this when the user asks about order history or wants to check
            order status. Always filter by the provided status.

            Args:
                user_id: Unique identifier for the user
                status: Order status: 'pending', 'shipped', or 'delivered'
                limit: Maximum number of results to return
        """
        pass

    # 2 Selecting tools  选择工具
    # 2.1 仅在完成特定对话里程碑后才启用高级工具
    @wrap_model_call
    def state_based_tools(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Filter tools based on conversation State."""
        state = request.state
        is_authenticated = state.get('authenticated', False)
        message_count = len(state['messages'])

        if not is_authenticated:
            tools = [t for t in request.tools if t.name.startswith('public_')]
            request = request.override(tools=tools)
        elif message_count < 5:
            tools = [t for t in request.tools if t.name != 'advanced_search']
            request = request.override(tools=tools)
        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[public_search, private_search, advanced_search],
        middleware=[state_based_tools]
    )

    # 2.2 根据用户或store中的功能标志筛选工具
    @wrap_model_call
    def store_based_tools(
            reqeust: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Filter tools based on Store preferences."""
        user_id = reqeust.runtime.context.user_id

        store = reqeust.runtime.store
        feature_flags = store.get(('features', ), user_id)

        if feature_flags:
            enabled_features = feature_flags.value.get('enabled_tools', [])
            tools = [t for t in reqeust.tools if t.name in enabled_features]
            request = reqeust.override(tools=tools)
        return handler(reqeust)

    agent = create_agent(
        model='',
        tools=[search_tool, analysis_tool, export_tool],
        middleware=[store_based_tools],
        context_schema=CustomContext,
        store=InMemoryStore()
    )

# Model 模型
def context_model_demo():
    # 1 根据对话时长使用不同的模型
    large_model = init_chat_model("claude-sonnet-4-6")
    standard_model = init_chat_model("gpt-4.1")
    efficient_model = init_chat_model("gpt-4.1-mini")

    @wrap_model_call
    def state_based_model(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Select model based on State conversation length."""
        message_count = len(request.messages)

        if message_count > 20:
            model = large_model
        elif message_count > 10:
            model = standard_model
        else:
            model = efficient_model

        request = request.override(model=model)

        return handler(request)

    agent = create_agent(
        model='',
        tools=[...],
        middleware=[state_based_model]
    )

# Response format  回复格式
def response_format():
    # 1 自定义格式
    class CustomerSupportTicket(BaseModel):
        """Structured ticket information extracted from customer message."""

        category: str = Field(
            description="Issue category: 'billing', 'technical', 'account', or 'product'"
        )
        priority: str = Field(
            description="Urgency level: 'low', 'medium', 'high', or 'critical'"
        )
        summary: str = Field(
            description="One-sentence summary of the customer's issue"
        )
        customer_sentiment: str = Field(
            description="Customer's emotional tone: 'frustrated', 'neutral', or 'satisfied'"
        )

    # 2 Selecting formats  选择格式
    # 2.1 根据对话状态配置结构化输出：
    class SimpleResponse(BaseModel):
        """Simple response for early conversation."""
        answer: str = Field(description="A brief answer")

    class DetailedResponse(BaseModel):
        """Detailed response for established conversation."""
        answer: str = Field(description="A detailed answer")
        reasoning: str = Field(description="Explanation of reasoning")
        confidence: float = Field(description="Confidence score 0-1")

    @wrap_model_call
    def state_based_output(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Select output format based on State."""
        message_count = len(request.messages)

        if message_count < 3:
            request = request.override(response_format=SimpleResponse)
        else:
            request = request.override(response_format=DetailedResponse)

        return handler(request)

    agent = create_agent(
        model='',
        tools=[...],
        middleware=[state_based_output]
    )

"""
Tool context 工具上下文
"""

# Reads  阅读
def tool_context_reads():
    # 1 从“状态”查看当前会话信息
    @tool
    def check_authentication(
            runtime: ToolRuntime
    ) -> str:
        """Check if user is authenticated."""
        current_state = runtime.state
        is_authenticated = current_state.get('authenticated', False)

        if is_authenticated:
            return 'User is authenticated'
        else:
            return 'User is not authenticated'

    agent = create_agent(
        model='',
        tools=[check_authentication]
    )

# Writes 写
def tool_context_write():
    # 1 使用命令写入状态以跟踪会话特定信息
    @tool
    def authenticate_user(
            password: str,
            runtime: ToolRuntime
    ) -> Command:
        """Authenticate user and update State."""
        # Perform authentication (simplified)
        if password == "correct":
            # Write to State: mark as authenticated using Command
            return Command(
                update={"authenticated": True},
            )
        else:
            return Command(update={"authenticated": False})

    agent = create_agent(
        model="gpt-4.1",
        tools=[authenticate_user]
    )

    # 2 从store读取以访问已保存的用户偏好设置
    @tool
    def get_preference(
            preference_key: str,
            runtime: ToolRuntime[CustomContext]
    ) -> str:
        """Get user preference from Store."""
        user_id = runtime.context.user_id

        # Read from Store: get existing preferences
        store = runtime.store
        existing_prefs = store.get(("preferences",), user_id)

        if existing_prefs:
            value = existing_prefs.value.get(preference_key)
            return f"{preference_key}: {value}" if value else f"No preference set for {preference_key}"
        else:
            return "No preferences found"

    agent = create_agent(
        model="gpt-4.1",
        tools=[get_preference],
        context_schema=CustomContext,
        store=InMemoryStore()
    )

"""
Life-cycle context  生命周期背景
"""

# Example: Summarization  示例：总结
def example_demo():
    agent = create_agent(
        model='',
        tools=[...],
        middleware=[
            SummarizationMiddleware(
                model='',
                trigger={'tokens': 4000},
                keep={'messages': 20}
            )
        ]
    )


"""
Model Context Protocol (MCP)
模型上下文协议（MCP）
"""

"""
Quickstart  快速入门
"""

def quickstart_demo():
    async def main():
        client = MultiServerMCPClient(
            {
                'math': {
                    'transport': 'stdio',
                    'command': 'python',
                    'args': ['/path/to/math_server.py']
                },
                'weather': {
                    'transport': 'http',
                    'url': 'http://localhost:8000/mcp'
                }
            }
        )
        tools = await client.get_tools()
        agent = create_agent(
            model='',
            tools=tools
        )
        math_response = await agent.ainvoke(
            {'messages': [{'role': 'user', 'content': "what's (3 + 5) x 12?"}]}
        )
        weather_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
        )
        print(math_response)
        print(weather_response)
    asyncio.run(main())

"""
Custom servers  自定义服务器
"""

math_mcp = FastMCP('Math')
def custom_example_demo():
    @math_mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    @math_mcp.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    math_mcp.run(transport='stdio')

"""
Transports 传输方式
"""

# HTTP
async def http_transport_demo():
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
            }
        }
    )

    # 1.Passing headers  传递信息头
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
                "headers": {
                    "Authorization": "Bearer YOUR_TOKEN",
                    "X-Custom-Header": "custom-value"
                },
            }
        }
    )
    tools = await client.get_tools()
    agent = create_agent('gpt-4.1', tools)
    response = await agent.ainvoke({
        'messages': {'role': 'user', 'content': 'what is the weather in nyc?'}})

    # 2.Authentication 认证
    auth = None  # 自定义的认证方法/类
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
                "auth": auth,
            }
        }
    )

# Stdio 标准输入输出
async def stdio_transport_demo():
    client = MultiServerMCPClient(
        {
            'math': {
                'transport': 'stdio',
                'command': 'python',
                'args': ['/path/to/math_server.py']
            }
        }
    )


# Stateful sessions 状态会话
async def stateful_session_demo():
    client = MultiServerMCPClient(
        {}
    )
    async with client.session('server_name') as session:
        tools = await load_mcp_tools(session)
        agent = create_agent(
            'gpt-4.1',
            tools
        )

"""
Core features 核心功能
"""

# Tools
async def core_tools_demo():
    # 1 加载工具
    client = MultiServerMCPClient()
    tools = await client.get_tools()
    agent = create_agent('gpt-4.1', tools)
    result = await agent.ainvoke(
        {'messages': [{'role': 'user', 'content': 'Get data from the server'}]}
    )
    # 2 结构化内容
    # 2.1 从工件中提取结构化内容
    for message in result['messages']:
        if isinstance(message, ToolMessage) and message.artifact:
            structured_content = message.artifact['structured_content']

    # 2.2 通过拦截器追加结构化内容
    async def append_structured_content(request: MCPToolCallRequest, handler):
        """Append structured content from artifact to tool message."""
        result = await handler(request)
        if result.structuredContent:
            result.content += [
                TextContent(type='text', text=json.dumps(result.structuredContent))
            ]
        return result
    client = MultiServerMCPClient({}, tool_interceptors=[append_structured_content])

    # 2.3Multimodal tool content  多模态工具内容
    for message in result['messages']:
        if message.type == 'tool':
            for block in message.content_blocks:
                if block['type'] == 'text':
                    pass
                elif block['type'] == 'image':
                    pass
# Resources 资源
async def core_resource_demo():
    # 加载资源
    client = MultiServerMCPClient({})
    blobs = await client.get_resources('server_name')

    # or await client.get_resources('server_name', uris=['file:///path/to/file.txt'])

    for blob in blobs:
        print(f"URI: {blob.metadata['uri']}, MIME type: {blob.mimetype}")
        print(blob.as_string())  # For text content

    # load_mcp_resources

    async with client.session('server_name') as session:
        blobs = await load_mcp_resources(session)
        # blobs = await load_mcp_resources(session, uris=["file:///path/to/file.txt"])

# Prompts 提示词
async def core_prompts_demo():
    client = MultiServerMCPClient({})
    messages = await client.get_prompt('server_name', 'summarize')

    messages = await client.get_prompt(
        'server_name',
        'code_review',
        arguments={'language': 'python', 'focus': 'security'}
    )
    for message in messages:
        print(f'{message.type}: {message.content}')

    async with client.session('server_name') as session:
        message = await load_mcp_prompt(session, 'summarize')

        message = await load_mcp_prompt(session,
                                        'code_review',
                                        arguments={'language': 'python', 'focus': 'security'})

"""
Advanced features 高阶功能
"""

# Tool interceptors 工具拦截器
async def tools_interceptors_demo(get_token_for_tool):
    # 1 访问运行时上下文
    async def inject_user_context(
            request: MCPToolCallRequest,
            handler,
    ):
        """Inject user credentials into MCP tool calls"""
        runtime = request.runtime
        # 1.1 运行时上下文
        user_id = runtime.context.user_id
        api_key = runtime.context.api_key

        # 1.2 存储
        store = runtime.store
        prefs = store.get(('preferences', ), user_id)

        # 1.3 状态
        state = runtime.state
        is_authenticated = state.get('is_authenticated', False)
        sensitive_tools = ["delete_file", "update_settings", "export_data"]

        # 1.4 工具调用ID
        tool_call_id = runtime.tool_call_id


        if request.name in sensitive_tools and not is_authenticated:
            return ToolMessage(
                content='Authentication required. Please log in first.',
                tool_call_id=tool_call_id
            )


        modified_request = request.override(
            args={**request.args, 'user_id': user_id,
                  'language': prefs.value.get('language', 'en')}
        )
        return await handler(modified_request)

    client = MultiServerMCPClient({}, tool_interceptors=[inject_user_context])

    tools = await client.get_tools()
    agent = create_agent('gpt-4.1',
                         tools,
                         context_schema=CustomContext,
                         store=InMemoryStore())  # 需要store的时候传

    result = await agent.ainvoke({'messages': []})


    # 2 状态更新和命令
    async def handle_task_completion(
            request: MCPToolCallRequest,
            handler,
    ):
        """Mark task complete and hand off to summary agent."""
        result = await handler(request)
        if request.name == "submit_order":
            return Command(
                update={
                    'messages': [result] if isinstance(result, ToolMessage) else [],
                    'task_status': 'completed',
                },
                goto='summary_agent'
            )
        return result

    # 3 自定义拦截器
    async def logging_interceptor(
            request: MCPToolCallRequest,
            handler
    ):
        """Log tool calls before and after execution."""

        # 3.1 基本模式
        if request.name == "logging":
            print(f"Calling tool: {request.name} with args: {request.args}")
            result = await handler(request)
            print(f"Tool {request.name} returned: {result}")
            return result
        # 3.2 修改请求
        elif request.name == "modified":
            modified_args = {k: v * 2 for k, v in request.args.items()}
            modified_request = request.override(args=modified_args)
            return await handler(modified_request)
        # 3.3 在运行时修改标题
        elif request.name == 'modify_title':
            token = get_token_for_tool(request.name)
            modified_request = request.override(
                headers={"Authorization": f"Bearer {token}"}
            )
            return await handler(modified_request)

    # 3.4 组合拦截器
    """
    Execution order:
    outer: before -> inner: before -> tool execution -> inner: after -> outer: after
    """
    async def outer_interceptor(request, handler):
        print("outer: before")
        result = await handler(request)
        print("outer: after")
        return result

    async def inner_interceptor(request, handler):
        print("inner: before")
        result = await handler(request)
        print("inner: after")
        return result

    # 3.5 错误处理
    async def retry_interceptor(
            request: MCPToolCallRequest,
            handler,
            max_retries: int = 3,
            delay: float = 1.0,
    ):
        """Retry failed tool calls with exponential backoff."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await handler(request)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    print(f"Tool {request.name} failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
        raise last_error

    client = MultiServerMCPClient({}, tool_interceptors=[logging_interceptor, outer_interceptor,
                                                         inner_interceptor, retry_interceptor])

# Progress notifications 进度通知
def progress_notifications():
    async def on_progress(
            progress: float,
            total: float | None,
            message: str | None,
            context: CallbackContext
    ):
        """Handle progress updates from MCP servers."""
        percent = (progress / total * 100) if total else progress
        tool_info = f" ({context.tool_name})" if context.tool_name else ""
        print(f"[{context.server_name}{tool_info}] Progress: {percent:.1f}% - {message}")

    client = MultiServerMCPClient(
        {},
        callbacks=Callbacks(on_progress=on_progress)
    )

# 日志记录
def log_info():
    async def on_logging_message(
            params: LoggingMessageNotificationParams,
            context: CallbackContext
    ):
        """Handle log messages from MCP servers."""
        print(f"[{context.server_name}] {params.level}: {params.data}")

    client = MultiServerMCPClient(
        {},
        callbacks=Callbacks(on_logging_message=on_logging_message)
    )

# Elicitation 提取
server = FastMCP('Profile')

class UserDetails(BaseModel):
    email: str
    age: int

def elicitation_demo():
    # 1 服务器设置
    @server.tool
    async def create_profile(name: str, ctx: Context) -> str:
        """Create a user profile, requesting details via elicitation."""
        result = await ctx.elicit(
            message=f"Please provide details for {name}'s profile:",
            response_type=UserDetails,
        )
        if result.action == "accept" and result.data:
            return f"Created profile for {name}: email={result.data.email}, age={result.data.age}"
        if result.action == "decline":
            return f"User declined. Created minimal profile for {name}."
        return "Profile creation cancelled."
    server.run(transport='http')

    # 2 Client setup 客户端设置
    async def on_elicitation(
            mcp_context: RequestContext,
            params: ElicitRequestParams,
            context: CallbackContext
    ) -> ElicitResult:
        """Handle elicitation requests from MCP servers."""
        # In a real application, you would prompt the user for input
        # based on params.message and params.requestedSchema
        return ElicitResult(
            action='accept',
            content={'email': 'user@example.com', 'age': 25}
        )

    client = MultiServerMCPClient(
        {
            "profile": {
                "url": "http://localhost:8000/mcp",
                "transport": "http",
            }
        },
        callbacks=Callbacks(on_elicitation=on_elicitation),
    )

    # 3 响应操作
    # Accept with data
    ElicitResult(action="accept", content={"email": "user@example.com", "age": 25})

    # Decline (user doesn't want to provide info)
    ElicitResult(action="decline")

    # Cancel (abort the operation)
    ElicitResult(action="cancel")

if __name__ == "__main__":
    access_demo()




