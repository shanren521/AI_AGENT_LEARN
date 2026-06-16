from mcp import StdioServerParameters, stdio_client, ClientSession
import mcp

server_params = StdioServerParameters(
    command="python",
    args=["./mcp_server.py"],
    env=None
)

async def handle_sampling_message(message: mcp.types.CreateMessageRequestParams) -> mcp.types.CreateMessageResult:
    return mcp.types.CreateMessageResult(
        role="assistant",
        content=mcp.types.TextContent(
            type="text",
            text="Hello, World!"
        ),
        model="qwen-3.7-max",
        stopReason="endTurn"
    )

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=handle_sampling_message) as session:
            await session.initialize()

            prompts = await session.list_prompts()
            print(f"prompts: {prompts}")

            tools = await session.list_tools()
            print(f"tools: {tools}")

            resources = await session.list_resources()
            print(f"resources: {resources}")

            result = await session.call_tool("weather", {"city": "北京"})
            print(f"result: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())






