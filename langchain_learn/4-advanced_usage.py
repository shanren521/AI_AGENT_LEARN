"""Advanced usage 高级用法"""
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import (PIIMiddleware, HumanInTheLoopMiddleware)

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






