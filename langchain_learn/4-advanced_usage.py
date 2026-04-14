"""Advanced usage 高级用法"""
from typing import Any
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.messages import AIMessage, AnyMessage
from langchain.agents.middleware.types import StateT
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import (PIIMiddleware, HumanInTheLoopMiddleware, AgentMiddleware, AgentState, hook_config,
                                         dynamic_prompt, ModelResponse, ModelRequest, after_model, before_model)
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

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



if __name__ == "__main__":
    access_demo()




