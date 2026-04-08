import os
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, SecretStr
from typing import TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}"

def create_agent_tools():
    agent = create_agent(
        model="",
        tools=[get_weather],
        system_prompt="You are a helpful assistant"
    )
    result = agent.invoke(
        AgentInput(
            messages=[HumanMessage(content="what is the weather in sf")]
        )
    )
    print(result)


def system_prompt():
    SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""
    return SYSTEM_PROMPT

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return get_weather(city)


class Context(BaseModel):
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.get("user_id")
    return "Florida" if user_id == "1" else "SF"

def configure_model(model_name):
    model = init_chat_model(
        model_name,
        temperature=0.7,
        timeout=60,
        max_tokens=1024
    )
    return model

class AgentInput(BaseModel):
    messages: list[BaseMessage]

@dataclass
class ResponseFormat:
    """Response schema for the agent"""
    punny_response: str
    weather_conditions: str | None = None

# 添加内存
checkpointer = InMemorySaver()

def create_run_agent():
    model = configure_model("")
    agent = create_agent(
        model=model,
        system_prompt=system_prompt(),
        tools=[get_user_location, get_weather_for_location],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer
    )
    config = RunnableConfig(configurable={"thread_id": "1"})
    response = agent.invoke(
        AgentInput(
            messages=[HumanMessage(content="what is the weather outside?")]
        ),
        config=config,
        context=Context(user_id="1")
    )
    print(response["structured_response"])

    response = agent.invoke(
        AgentInput(
            messages=[HumanMessage(content="thank you")]
        ),
        config=config,
        context=Context(user_id="1")
    )
    print(response["structured_response"])


def openai_model(model_name: str = "gpt-4o-mini"):
    os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxx"  # 或提前设置环境变量

    model = init_chat_model(
        model_name,
        model_provider="openai",  # ✅ 指定 provider
        temperature=0.7,
        timeout=60,
        max_tokens=1024
    )
    return model

def openai_type_model(model_name: str = "deepseek-chat"):
    model = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        timeout=60,
        max_tokens=1024,
        base_url="https://api.deepseek.com/v1",  # 替换为对应平台地址
        api_key=SecretStr(os.getenv("OPENAI_API_KEY")),          # 对应平台的 API Key

    )
    return model

def ollama_model(model_name: str = "llama3"):
    model = ChatOllama(
        model=model_name,
        temperature=0.7,
        num_predict=1024,           # 对应 max_tokens
        base_url="http://localhost:11434",  # Ollama 默认地址
    )
    return model

if __name__ == "__main__":
    create_agent_tools()











