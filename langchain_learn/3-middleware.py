from langchain.agents import create_agent
from langchain.agents.middleware import (ModelCallLimitMiddleware, ToolCallLimitMiddleware, ModelFallbackMiddleware, PIIMiddleware)
from langgraph.checkpoint.memory import InMemorySaver




"""
middleware 中间件
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
        model="gpt-4.1",
        tools=[],
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        ]
    )






