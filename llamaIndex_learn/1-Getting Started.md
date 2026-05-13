https://developers.llamaindex.ai/python/framework/getting_started/

# 入门指南

LlamaIndex 的核心运作逻辑遵循一个清晰的五步流程：加载 (Load) -> 解析 (Parse) -> 索引 (Index) -> 存储 (Store) -> 查询 (Query)

# 1 高级概念

## 1.1 大型语言模型

LLMs 是启动 LlamaIndex 的基本创新。它们是一种人工智能（AI）计算机系统，能够理解、生成和操作自然语言，包括根据其训练数据或在查询时提供的数据回答问题。

## 1.2 代理应用

当 LLM 在应用程序中使用时，通常用于做决策、采取行动和/或与世界交互。这是代理应用程序的核心定义。

代理应用的关键特征:
+ LLM 增强：LLM 通过工具（即代码中的任意可调用函数）、内存和/或动态提示进行增强。
+ 提示链：使用多个相互依赖的 LLM 调用，其中一个 LLM 的输出作为下一个 LLM 的输入。
+ 路由：使用 LLM 将应用程序路由到应用程序中的下一个适当步骤或状态。
+ 并行性：应用程序可以并行执行多个步骤或操作。
+ 编排：使用 LLMs 的分层结构来编排较低级别的操作和 LLMs。
+ 反思：LLM 用于反思和验证先前步骤或 LLM 调用的输出，这可用于指导应用程序进入下一个适当的步骤或状态。

## 1.3 代理

我们将代理定义为“代理应用程序”的一个特定实例。代理是一种软件，通过结合 LLMs 与其他工具和记忆，在推理循环中半自主地执行任务，该推理循环决定下一步使用哪个工具（如果有的话）。

## 1.4 RAG检索增强生成

检索增强生成（RAG）是使用 LlamaIndex 构建数据支持型 LLM 应用的核心技术。它允许 LLM 在查询时接收您的私有数据来回答问题，而不是在训练时使用您的数据。为了避免每次都将所有数据发送给 LLM，RAG 会对您的数据进行索引，并选择性地将相关部分与您的查询一同发送。

## 1.5 应用场景

数据支持型 LLM 应用有无尽的使用场景，但它们大致可分为五类:
+ 代理：代理是由 LLM 驱动的自动化决策者，通过一组工具与世界交互。代理可以执行任意数量的步骤来完成给定任务，动态决定最佳行动方案，而不是遵循预定的步骤。这使其在处理更复杂的任务时具有额外的灵活性。
+ 工作流：LlamaIndex 中的工作流是一种特定的基于事件驱动的抽象，允许你编排一系列步骤和 LLMs 调用。工作流可用于实现任何代理式应用，并且是 LlamaIndex 的核心组件。
+ 结构化数据提取 Pydantic 提取器允许你指定从数据中提取的精确数据结构，并使用 LLMs 以类型安全的方式填充缺失部分。这对于从非结构化来源（如 PDF、网站等）提取结构化数据非常有用，并且是自动化工作流的关键。
+ 查询引擎：查询引擎是一个端到端的流程，允许你对自己的数据进行提问。它接收一个自然语言查询，并返回一个响应，以及检索并传递给 LLM 的参考上下文。
+ 聊天引擎：聊天引擎是与你的数据端到端进行对话的流程（多次来回互动，而不是单一的问答）。

# 2 安装与设置

## 2.1 从pip快速开始安装

```bash
pip install llama-index

pip install llama-index-embeddings-openai

pip install llama-index-llms-ollama llama-index-embeddings-huggingface
```

默认情况下，我们使用 OpenAI 的 gpt-3.5-turbo 模型进行文本生成，使用 text-embedding-ada-002 进行检索和嵌入。要使用这些功能，您必须在环境变量中设置 OPENAI_API_KEY。您可以通过登录您的 OpenAI 账户并创建一个新的 API 密钥来获取 API 密钥。

## 2.2 从pip自定义安装

```bash
pip install llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-huggingface
```

## 2.3 从源代码安装

```bash
poetry shell
poetry self add poetry-plugin-shell
poetry install --with dev,docs
pip install -e llama-index-integrations/readers/llama-index-readers-file
pip install -e llama-index-integrations/llms/llama-index-llms-ollama
```

# 3 入门教程

## 3.1 基础代理示例

```python
import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers"""
    return a * b

agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4.1"),
    system_prompt="You are a helpful assistant that can multiply two numbers."
)

async def main():
    response = await agent.run("What is 1234 * 4567")
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())
```

本地模型的使用:
```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(
        model="llama3.1",
        request_timeout=360.0,
        context_window=8000
    ),
    system_prompt="You are a helpful assistant that can multiply two numbers."
)

async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```


## 3.2 添加聊天历史

AgentWorkflow 也能够记住之前的消息。这包含在 Context 的 AgentWorkflow 中。

```python
from llama_index.core.workflow import Context

ctx = Context(agent)

response = await agent.run("My name is John", ctx=ctx)
response = await agent.run("What is my name?", ctx=ctx)
```

## 3.3 添加RAG功能

现在让我们通过添加文档搜索功能来增强我们的代理。首先，让我们使用终端获取一些示例数据

```bash
mkdir data
wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt -O data/paul_graham_essay.txt
```

现在我们可以使用 LlamaIndex 创建一个用于搜索文档的工具。默认情况下，我们的 VectorStoreIndex 将使用 text-embedding-ada-002 来自 OpenAI 的嵌入来嵌入和检索文本。

```python
import os
import asyncio

from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(
    model="llama3.1",
    request_timeout=360.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers"""
    return a * b

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)

agent = FunctionAgent(
    tools=[multiply, search_documents],
    llm=OpenAI(model="gpt-4.1"),
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)

agent_local = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[multiply, search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions."""
)

async def main():
    response = await agent.run(
        "What did the author do in college? Also, what's 7 * 8?"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```

## 3.4 存储RAG索引

为了避免每次都重新处理文档，你可以将索引持久化到磁盘
```python
index.storage_context.persist("./storage")
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
```

# 4 探索LlamaIndex视频系列

## 4.1 自下而上开发(Llama文档机器人)

github地址: https://github.com/run-llama/llama_docs_bot

[第 1 部分] LLMs 和提示 [第 2 部分] 文档和元数据 [第 3 部分] 评估 [第 4 部分] 嵌入 [第 5 部分] 检索器和后处理器

## 4.2 子问题查询引擎

将通过复杂查询分解为更简单的子查询来探索回答复杂查询的方法
```python
# nest_asyncio用于解决python异步编程中事件循环嵌套冲突的代码
# nest_asyncio 是一个第三方库，它的作用是给 Python 标准库的 asyncio 打 “补丁”，使其允许在已经存在的事件循环（Event Loop）中再次嵌套运行新的事件循环。
import os
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import Settings

# 配置LLM服务
Settings.llm = OpenAI(model="gpt-4.1", temperature=0.2)  # 全局变量，后续可以直接使用不需要赋值

"""
下载数据
!mkdir -p 'data/10k/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'
"""

# 加载数据
lyft_docs = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()


# 构建索引
lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

# 构建查询引擎
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

# lyft_index.as_retriever(similarity_top_k=3)构建一个纯检索器，不调用LLM，只负责找资料

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021"
            ),
        )
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021"
            ),
        )
    )
]

s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)

# 运行查询 (query内部会自动执行检索Retrieve(从索引中找出最相关的文档块)、增强Augment(将检索的文档块和问题拼接成新的提示词)、生成Generate(用新的提示词输入给模型)，返回Response对象，最终答案字符串.response、参考文档.source_nodes)
response = s_engine.query(
    "Compare and contrast the customer segments and geographies that grew the"
    " fastest"
)

print(response)

```
https://developers.llamaindex.ai/python/examples/usecases/10k_sub_question/

## 4.3 文档管理

涵盖了如何管理来自不断更新的源（例如 Discord）的文档，以及如何避免文档重复并节省嵌入令牌。

```python
import os
import json
import subprocess

from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage

subprocess.run(["python", "group_conversations.py", "./data/discord_dumps/help_channel_dump_05_25_23.json"])

with open("conversation_docs.json", "r") as f:
    threads = json.load(f)
documents = []
for thread in threads:
    thread_text = thread["thread"]
    thread_id = thread["metadata"]["id"]
    timestamp = thread["metadata"]["timestamp"]
    documents.append(
        Document(text=thread_text, id_=thread_id, metadata={"date": timestamp})
    )

index = VectorStoreIndex.from_documents(documents)
print("ref_docs ingested: ", len(index.ref_doc_info))
print("number of input documents: ", len(documents))

# 查看特定线程的文档信息
thread_id = threads[0]["metadata"]["id"]
print(index.ref_doc_info[thread_id])

# 保存索引到磁盘
index.storage_context.persist(persist_dir="./storage")

index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))


# 用新数据刷新索引
subprocess.run(["python", "group_conversations.py", "./data/discord_dumps/help_channel_dump_06_02_23.json"])

with open("conversation_docs.json", "r") as f:
    new_threads = json.load(f)
    
new_documents = []
for new_thread in new_threads:
    thread_text = new_thread["thread"]
    thread_id = new_thread["metadata"]["id"]
    timestamp = new_thread["metadata"]["timestamp"]
    new_documents.append(
        Document(text=thread_text, id_=thread_id, metadata={"date": timestamp})
    )
    
print("Number of new documents: ", len(new_documents) - len(documents))

# refresh 刷新索引
refreshed_docs = index.refresh(new_documents, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}})

```

https://github.com/run-llama/llama_index/tree/main/docs/examples/discover_llamaindex/document_management

## 4.4 联合文本到SQL和语义搜索

介绍了 LlamaIndex 内置的工具，用于将 SQL 和语义搜索结合到一个统一的查询界面中。
SQLAutoVectorQueryEngine该查询引擎允许将结构化表中的洞察与非结构化数据结合起来。

llama-index-vector-stores-pinecone是 LlamaIndex 官方推出的 Pinecone 向量数据库原生集成扩展包, 核心作用是为你的 LlamaIndex RAG 应用提供生产级、可持久化、高性能的向量存储后端

pinecone可替代方案:
+ 本地轻量零成本替代
  + chroma(专为本地 RAG 开发优化): 自动把向量和元数据存在本地磁盘，程序重启不丢失，彻底解决 LlamaIndex 默认内存存储的痛点
  + faiss: 工业级向量检索引擎, 本地检索性能天花板, 仅为检索引擎，无完整数据库能力，复杂元数据过滤需要自行封装, 对本地检索速度有高要求，数据量较大的本地测试场景
+ 小团队 / 个人生产首选
  + qdrant(通用生产首选): 高性能、低内存占用、极简部署, 中小规模生产场景的标杆方案
  + milvus(大规模企业级首选): 专为超大规模向量场景设计, 支持万亿级向量分布式存储、标量 + 向量混合检索、多模态向量、GPU 加速、时间旅行等
  + pgvector(已有PostgreSQL首选): 复用 PostgreSQL 的全部能力（ACID 事务、权限管理、备份、高可用），SQL 原生支持标量 + 向量混合检索，过滤能力极强, 纯向量检索性能略低于 Milvus/Qdrant，适合千万级以内向量规模

pip install llama-index-vector-stores-pinecone
pip install llama-index-vector-stores-chroma
pip install llama-index-readers-wikipedia
pip install llama-index-llms-openai

```python
import os
import nest_asyncio
nest_asyncio.apply()
import logging
import sys
import chromadb
import pinecone

from llama_index.core import StorageContext, VectorStoreIndex, Settings  # pinecone
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine, RetrieverQueryEngine, SQLAutoVectorQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from sqlalchemy import (
    Column,
    Integer,
    String,
    create_engine,
    MetaData,
    Table,
    select,
    column,
    insert
)
from llama_index.readers.wikipedia import WikipediaReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 创建通用对象

# pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
pinecone_index = pinecone.Index("quickstart")

pinecone_index.delete(deleteAll=True)

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace="wiki_cities"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_store_index = VectorStoreIndex([], storage_context=storage_context)

# chroma
db = chromadb.PersistentClient(path="./chroma_db")
collection_name = "wiki_cities"
if collection_name in [col.name for col in db.list_collections()]:
    db.delete_collection(collection_name)
    
chroma_collection = db.create_collection(collection_name)
chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
chroma_storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)

chroma_vector_store_index = VectorStoreIndex([], storage_context=chroma_storage_context)

# 创建数据库模式 + 测试数据
engine = create_engine("sqlite:///:memory:", future=True)
metadata_obj = MetaData()

table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False)
)

metadata_obj.create_all(engine)
print(metadata_obj.tables.keys())

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
]
for row in rows:
    insert_stmt = insert(city_stats_table).values(row)
    with engine.begin() as connection:
        cursor = connection.execute(insert_stmt)

with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM city_stats")
    print(cursor.fetchall())


# 加载数据
cities = ["Toronto", "Berlin", "Tokyo"]
wiki_docs = WikipediaReader().load_data(cities)


# 构建SQL索引
sql_database = SQLDatabase(engine, include_tables=["city_stats"])
# # 创建自然语言转 SQL 的查询引擎
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"]
)

# 构建向量索引
for city, wiki_doc in zip(cities, wiki_docs):
    nodes = Settings.node_parser.get_nodes_from_documents([wiki_doc])
    for node in nodes:
        node.metadata = {"title": city}
    chroma_vector_store_index.insert_nodes(nodes)



"""
定义查询引擎，设为工具
"""
vector_store_info = VectorStoreInfo(
    content_info="articles about different cities",
    metadata_info=[
        MetadataInfo(name="title", type="str", description="The name of the city")
    ]
)

# 创建自动检索器（能根据问题自动决定是否用元数据过滤）
vector_auto_retriever = VectorIndexAutoRetriever(
    chroma_vector_store_index, vector_store_info=vector_store_info
)

# 封装为查询引擎
retriever_query_engine = RetrieverQueryEngine.from_args(
    vector_auto_retriever, llm=OpenAI(model="gpt-4.1")
)

# 将两个引擎转为 Agent 可用的 Tool
sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing: city_stats, containing the population/country of"
        " each city"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    description=(
        f"Useful for answering semantic questions about different cities"
    ),
)

# SQL + 向量 混合查询引擎
# 定义SQLAutoVectorQueryEngine: 先查 SQL 找结构化数据，再查向量找非结构化细节
query_engine = SQLAutoVectorQueryEngine(
    sql_tool, vector_tool, llm=OpenAI(model="gpt-4")
)

# 1. 先通过 SQL 找到人口最多的城市 (Tokyo)；2. 再通过向量查 Tokyo 的艺术文化
response = query_engine.query(
    "Tell me about the arts and culture of the city with the highest"
    " population"
)
print(str(response))

```

https://developers.llamaindex.ai/python/examples/query_engine/sqlautovectorqueryengine

# 5 常见问题

## 5.1 将文档解析成更小的片段

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

Settings.chunk_size = 512
Settings.chunk_overlap = 5

documents = []
index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=512)])
```

## 5.2 使用不同的向量存储

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, SimpleDirectoryReader, VectorStoreIndex


chroma_client = chromadb.PersistentClient("./data/chroma")
chrom_collection = chroma_client.create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chrom_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# StorageContext 定义了文档、嵌入和索引的存储后端
documents = SimpleDirectoryReader("./data/paul_graham").load_data()
index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```

## 5.3 查询时检索更多上下文

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)  # default 2
response = query_engine.query("What did the author do growing up?")
print(response)
```

## 5.4 使用不同的LLM

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(
    model="mistral",
    request_timeout=60.0,
    context_window=8000
)
documents = SimpleDirectoryReader("./data/paul_graham").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    llm=Ollama(
    model="mistral",
    request_timeout=60.0,
    context_window=8000
    )
)
```

## 5.5 使用不同的响应模式

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("What did the author do growing up?")
```

response_mode的七种模式:
+ default/refine: 
  + 工作方式: 先拿第一个文本块生成答案 → 再用后面的文本块迭代优化、补充答案。
  + 优点: 答案精准、连贯。
  + 缺点: 慢，调用 LLM 次数多。
  + 适合: 需要精准、详细回答的场景。
+ tree_summarize:
  + 工作方式: 把所有检索到的文本全部一次性丢给 LLM，让它总结 / 合并成一个答案。
  + 优点: 快，适合长文本总结。
  + 缺点: 受 LLM 上下文长度限制。
  + 适合: 总结整篇文章\汇总多个文档信息
+ compact:
  + 工作方式: 把文本尽量塞满上下文窗口，减少 LLM 调用次数，比 default 快。
  + 优点: 速度 + 效果平衡。
  + 适合: 普通问答，追求性价比。
+ simple_summarize:
  + 工作方式: 强制把所有文本拼在一起，让 LLM 直接总结。 （比 tree_summarize 更简单粗暴）
  + 适合: 极短快速总结。
+ accumulate:
  + 工作方式: 对每一段文本分别生成答案，最后把所有答案拼接起来。
  + 适合: 需要逐条列出信息，不做合并。
+ compact_accumulate:
  + 先塞满文本块，再逐条生成答案。
  + 速度更快的 accumulate。
+ no_text:
  + 工作方式: 不生成自然语言答案
  + 只返回检索到的文档节点（原始文本）。
  + 适合: 你只想拿检索结果，自己处理，不让 LLM 生成回答。

## 5.6 将响应流式传输回来

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True)  # 无记忆
response = query_engine.query("What did the author do growing up?")
print(response.print_response_stream())
```

## 5.7 用聊天机器人代替问答

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_chat_engine()  # 有记忆
response = query_engine.chat("What did the author do growing up?")
print(response)

response = query_engine.chat("Oh interesting, tell me more.")
print(response)
```

# 6 入门工具

## 6.1 入门工具

### 6.1.1 create-llama:全栈web应用生成器

create-llama 工具是一个命令行工具, 可以帮助你创建一个全栈的 Web 应用程序，你可以选择前端和后端，它会对你的文档进行索引，并允许你与它们进行聊天。

```bash
npx create-llama@latest
```

### 6.1.2 SEC insights: 高级查询技术

索引和查询财务申报文件是生成式 AI 的一个非常常见的用例

https://github.com/run-llama/sec-insights

### 6.1.3 Chat LLamaIndex:全栈聊天应用

Chat LlamaIndex 是另一个全栈、开源的应用程序，具有多种交互模式，包括流式聊天和多模态图像查询

https://github.com/run-llama/chat-llamaindex
### 6.1.4 LlamaBot: Slack和Discord应用

LlamaBot 是另一个开源应用程序，这次用于构建一个 Slack 机器人，该机器人可以监听组织内的消息并回答有关发生的事情的问题

https://github.com/run-llama/llamabot
