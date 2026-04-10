from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain.messages import ToolMessage, SystemMessage, HumanMessage

basic_model = ChatOpenAI(model='gpt-4o-mini')
advanced_model = ChatOpenAI(model='gpt-4o')


# Dynamic model
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity"""
    message_count = len(request.state["message"])

    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model
    return handler(request.override(model=model))

tools = None

def dynamic_model():
    agent = create_agent(
        model=basic_model,
        tools=tools,
        middleware=[dynamic_model_selection]
    )

# Define Tools
@tool
def search(query: str) -> str:
    """Search for information"""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"Weather in {location}: Sunny, 72°F"

# Tool error handing
@wrap_model_call
def handle_tool_error(request, handler):
    """Handle tool execution errors with custom messages"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

# System prompt


def define_tools():
    agent = create_agent(basic_model,
                         tools=[search,get_weather],
                         middleware=[handle_tool_error],
                         system_prompt="You are a helpful assistant. Be concise and accurate.")

def system_prompt():
    literary_agent = create_agent(
        model="anthropic:claude-sonnet-4-5",
        system_prompt=SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with analyzing literary works.",
                },
                {
                    "type": "text",
                    "text": "<the entire contents of 'Pride and Prejudice'>",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        )
    )
    result = literary_agent.invoke({"messages": [HumanMessage("Analyze the major themes in 'Pride and Prejudice'")]})

if __name__ == '__main__':


