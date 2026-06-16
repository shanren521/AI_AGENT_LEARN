import os

from dns.e164 import query
from dotenv import load_dotenv
from datetime import datetime

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.prebuilt import create_react_agent


load_dotenv()

model="qwen3.7-max"

llm = ChatTongyi(
        model=model,
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )

def demo_one():
    tools = []
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate"
    )


    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""
        You are a helpful assistant.
        """,
    )

    agent.invoke({"messages": [{"role": "user", "content": "What is the meaning of life?"}]})


# 自定义辅助工具
def custom_tool_demo():
    @tool
    def get_current_date():
        """获取当前日期"""
        return datetime.today().strftime("%Y-%m-%d")

    query = "今天是几月几号？"

    def llm_chain(prompt):
        llm_with_tools = llm.bind_tools([get_current_date])
        all_tools = {"get_current_date": get_current_date}
        messages = [prompt]
        ai_msg = llm_with_tools.invoke(messages)
        print(ai_msg)
        messages.append(ai_msg)
        print(ai_msg.tool_calls)
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                selected_tool = all_tools[tool_call["name"].lower()]
                tool_msg = selected_tool.invoke(tool_call)
                messages.append(tool_msg)
        content = llm_with_tools.invoke(messages).content
        print(content)

    def langgraph_chain(prompt):
        agent = create_agent(
            model=llm,
            tools=[get_current_date],
            system_prompt="You are a helpful assistant"
        )
        content = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        print(content)

    langgraph_chain(query)



if __name__ == "__main__":
    custom_tool_demo()
