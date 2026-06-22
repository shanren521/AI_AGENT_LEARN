import asyncio
from operator import add
from typing import TypedDict, Annotated

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

redis_url = "redis://localhost:6379"

load_dotenv()

model="qwen3.7-max"

llm = ChatOpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    model=model,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

NODES = ["travel", "couplet", "joke", "other"]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str

def other_node(state: State):
    print(">>> Other Node")
    writer = get_stream_writer()
    writer({"node", ">>> Other Node"})
    return {"messages": [HumanMessage(content="我暂时无法回答这个问题")], "type": "other"}

def supervisor_node(state: State):
    print(">>> Supervisor Node")
    writer = get_stream_writer()
    writer({"node", ">>> supervisor node"})
    system_prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
    如果用户的问题是和旅游路线规划相关的，就返回travel，
    如果用户的问题是希望讲一个笑话，就返回joke，
    如果用户的问题是对一个对联，就返回couplet，
    如果其他问题，返回other，
    除了这几个选项外，不要返回任何其他内容
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]}
    ]
    if "type" in state:
        writer({"supervisor_step": f"已获得{state['type']} 智能体处理结果"})
        return {"type": END}
    else:
        response = llm.invoke(messages)
        type_node = response.content
        writer({"supervisor_step": f"问题分类结果{type_node}"})
        if type_node in NODES:
            return {"type": type_node}
        else:
            raise ValueError("type is not in (travel, joke, other, couplet)")

def travel_node(state: State):
    print(">>> Travel Node")
    writer = get_stream_writer()
    writer({"node", ">>> Travel Node"})
    system_prompt = "你是一个专业的旅行规划助手，根据用户的问题，生成一个旅游路线规划。不要超过200字"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]}
    ]
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
    tools = asyncio.run(client.get_tools())
    agent = create_agent(
        model=llm,
        tools=tools
    )
    response = agent.invoke({"messages": messages})
    writer({"travel_result": response["messages"][-1].content})
    return {"messages": [AIMessage(content=response["messages"][-1].content)], "type": "travel"}

def joke_node(state: State):
    print(">>> Joke Node")
    writer = get_stream_writer()
    writer({"node", ">>> Joke Node"})
    system_prompt = "你是一个笑话大师，根据用户的问题，写一个不超过100字的笑话"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]}
    ]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)], "type": "joke"}

def couplet_node(state: State):
    print(">>> Couplet Node")
    writer = get_stream_writer()
    writer({"node", ">>> Couplet Node"})
    couplet_template = ChatPromptTemplate.from_messages([
        ("system", """
        你是一个对联大师，你的任务是根据用户给出的上联，设计出一个下联。
        回答时，可以参考下面的对联：
        {samples}
        请用中文回答问题
        """),
        ("user", "{user_input}")
    ])

    query = state["messages"][0]
    embedding_model = DashScopeEmbeddings(model="text-embedding-v3")
    config = RedisConfig(
        index_name="couplet",
        redis_url=redis_url
    )
    vector_store = RedisVectorStore(embedding_model, config=config)
    template_samples = []
    scored_results = vector_store.similarity_search_with_score(query, k=3)
    for doc, score in scored_results:
        template_samples.append(doc.page_content)
    prompt = couplet_template.invoke({"samples": template_samples, "user_input": query})
    writer({"couplet_prompt": f"{prompt}"})
    response = llm.invoke(prompt)
    writer({"couplet_result": f"{response.content}"})
    return {"messages": [HumanMessage(content=response.content)], "type": "couplet"}

def routing_func(state: State):
    node_type = state["type"]
    if node_type == "travel":
        return "travel_node"
    elif node_type == "joke":
        return "joke_node"
    elif node_type == "couplet":
        return "couplet_node"
    elif node_type == END:
        return END
    else:
        return "other_node"

builder = StateGraph(State)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)

builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", routing_func, ["travel_node", "joke_node", "couplet_node", "other_node", END])
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")
builder.add_edge("supervisor_node", END)

checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    for chunk in graph.stream({"messages": ["最新的AI Agent的技术"]},
                 config,
                 stream_mode="values"):
        print(chunk)