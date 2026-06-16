import os
import asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
"""
两种方式：
    sse：是一种基于HTTP协议实现的长连接协议，服务端单向主动推送。
    stdio：本地进程间通信，通过标准输入输出传递消息。

"""
model="qwen3.7-max"

llm = ChatOpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    model=model,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

async def agent_mcp():
    client = MultiServerMCPClient(
        {
            "amap-maps-streamableHTTP": {
                "url": "https://mcp.amap.com/mcp?key=8d47c3e9f26995ce04e3896d1131c32a",
                "transport": "streamable-http",
            }
        }
        # {
        #     "amap-maps": {
        #         "command": "npx",
        #         "args": ["-y", "@amap/amap-maps-mcp-server"],
        #         "env": {
        #             "AMAP_MAPS_API_KEY": "8d47c3e9f26995ce04e3896d1131c32a"
        #         },
        #         "transport": "stdio"
        #     }
        # }
    )
    tools = await client.get_tools()
    agent = create_agent(
        model=llm,
        tools=tools
    )
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "帮我规划一条从北京到上海的自驾游路线"}]}
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(agent_mcp())
