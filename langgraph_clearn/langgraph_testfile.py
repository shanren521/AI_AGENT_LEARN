from langchain.tools import tool
from langgraph_sdk import get_client
import asyncio


@tool
def divide(a: int, b: int) -> float:
    """
    Divide `a` and `b`

    Args:
        a: First int
        b: Second int
    """
    return a / b

def sub(a, b):
    return a - b

client = get_client(url='http://localhost:2024')

async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph?",
            }],
        },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

if __name__ == "__main__":
    print(divide.get_name())
    print(sub.__name__)
    asyncio.run(main())


