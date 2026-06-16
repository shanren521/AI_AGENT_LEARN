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


async def main():
    client = get_client(url='http://localhost:2024')
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

# 2块一瓶啤酒，2个空瓶换一瓶啤酒，4个瓶盖换一瓶啤酒，N块最多能买几瓶啤酒
def buy_beer(n):
    total = n // 2
    a = total  # 空瓶
    b = total  # 瓶盖
    while a >= 2 or b >= 4:
        new_total_a = a // 2
        new_total_b = b // 4
        new_total = new_total_a + new_total_b
        if new_total == 0:
            break
        total += new_total
        a = a % 2 + new_total
        b = b % 4 + new_total
    print(total)


if __name__ == "__main__":
    print(divide.get_name())
    print(sub.__name__)
    buy_beer(10)
    # asyncio.run(main())


