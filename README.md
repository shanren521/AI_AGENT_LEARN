
# 官方资源

| 资源                     | 链接                                                                  | 说明               |
| :--------------------- | :------------------------------------------------------------------ | :--------------- |
| **LangChain 官方文档**     | [python.langchain.com](https://python.langchain.com)                | 最权威的API参考和教程     |
| **LangChain Cookbook** | [官方Cookbook](https://python.langchain.com/docs/cookbook)            | 生产级代码模式示例        |
| **GitHub 仓库**          | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 70k+ stars，源码和示例 |


📚 优质入门教程
1. 综合入门指南（推荐）
《LangChain: A Comprehensive Beginner's Guide》 
覆盖核心概念：Prompts、Models、Chains、Tools、Agents、Memory
包含完整安装步骤和第一个Agent构建
适合零基础入门
2. Python实战教程
《LangChain Python Tutorial: Complete Guide 2025》 
提供完整GitHub代码仓库（可直接运行）
包含Django/FastAPI集成
详细讲解Memory策略（Buffer、Window、Summary、Token Buffer）
工具使用（Function Calling）完整示例
3. 中文入门教程
《LangChain 中文入门教程》 
GitHub开源项目，包含完整实战代码
覆盖：基础问答、Google搜索、超长文本总结、本地知识库、YouTube问答机器人
提供Google Colab笔记本可直接运行
《LangChain从入门到实战：8000字详解》 
CSDN上的详细中文教程
详解Agent的观察-思考-行动（Observation-Thought-Action）机制

| 工具            | 用途          | 适用场景                    |
| :------------ | :---------- | :---------------------- |
| **LangChain** | 基础框架，链式调用   | 线性工作流、简单应用              |
| **LangGraph** | 状态机驱动的循环工作流 | 复杂Agent、多轮决策、需要循环和条件的场景 |
| **LangSmith** | 监控和调试平台     | 生产环境追踪、评估               |

# Agent 模块深度教程

| 资源                  | 链接                                                                                                                                | 重点内容                             |
| :------------------ | :-------------------------------------------------------------------------------------------------------------------------------- | :------------------------------- |
| **ReAct Agent官方文档** | [python.langchain.com/docs/modules/agents/agent\_types/react](https://python.langchain.com/docs/modules/agents/agent_types/react) | ReAct框架原理与实现                     |
| **Tools使用指南**       | [官方Tools教程](https://python.langchain.com/docs/modules/agents/tools/)                                                              | @tool装饰器、StructuredTool、BaseTool |
| **LangGraph Agent** | [LangGraph Agent教程](https://langchain-ai.github.io/langgraph/tutorials/introduction/)                                             | 状态机驱动的循环Agent                    |

优质实战教程
《LangChain Agent 完整开发指南》

关键学习要点：
ReAct循环：Thought → Action → Observation → ... → Final Answer
工具描述：description的质量直接决定Agent调用准确性
Agent类型选择：
create_react_agent：通用场景（推荐）
create_openai_functions_agent：OpenAI模型专用（更稳定）
create_structured_chat_agent：需要多参数工具时

# RAG (Retrieval-Augmented Generation) 深度教程

| 阶段      | 资源                                                                                            | 内容               |
| :------ | :-------------------------------------------------------------------------------------------- | :--------------- |
| **基础**  | [官方RAG Quickstart](https://python.langchain.com/docs/use_cases/question_answering/quickstart) | 5分钟上手基础RAG       |
| **进阶**  | [RAG From Scratch系列](https://github.com/langchain-ai/rag-from-scratch)                        | 18集视频+代码，从0构建RAG |
| **生产级** | [LangChain RAG Cookbook](https://python.langchain.com/docs/cookbook/retrieval)                | 高级检索策略、评估方法      |

高级RAG技术（生产必备）
| 技术                            | 解决的问题      | 代码参考                                                                                                   |
| :---------------------------- | :--------- | :----------------------------------------------------------------------------------------------------- |
| **Multi-Query检索**             | 用户问题表述不准确  | [教程](https://python.langchain.com/docs/templates/rag-chroma-multi-query)                               |
| **Rerank重排序**                 | 初步检索结果质量不高 | Cohere Rerank或Cross-Encoder                                                                            |
| **Parent Document Retrieval** | 块太小丢失上下文   | [官方指南](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever) |
| **Self-RAG/Adaptive RAG**     | 判断是否需要检索   | LangGraph实现                                                                                            |
| **HyDE (假设文档嵌入)**             | 查询-文档语义鸿沟  | Hypothetical Document Embeddings                                                                       |

Memory 模块深度教程

| 资源                             | 说明                                                                                           |
| :----------------------------- | :------------------------------------------------------------------------------------------- |
| **官方Memory指南**                 | [python.langchain.com/docs/modules/memory](https://python.langchain.com/docs/modules/memory) |
| **ConversationBufferMemory详解** | [教程](https://python.langchain.com/docs/modules/memory/types/buffer)                          |
| **ConversationSummaryMemory**  | [长对话优化方案](https://python.langchain.com/docs/modules/memory/types/summary)                    |


三个模块的整合实战
推荐项目：GitHub langchain-ai/rag-from-scratch 
这个项目包含18个递进式notebook，覆盖：
+ 基础RAG → 多查询检索 → Rerank → Agentic RAG
+ 每个阶段都对比Memory策略的影响
+ 最终构建带记忆的自主Agent

学习检查清单

| 模块         | 必会技能                                                        | 自测标准                     |
| :--------- | :---------------------------------------------------------- | :----------------------- |
| **Agent**  | ① 创建自定义Tool<br>② 实现ReAct循环<br>③ 调试Agent轨迹（verbose=True）     | 能构建一个调用2个以上工具的数学/搜索Agent |
| **RAG**    | ① 文档加载与分割<br>② 5种检索策略<br>③ 评估检索质量（Context Precision/Recall） | 能在100页PDF上实现问答，且能指出答案来源  |
| **Memory** | ① 4种Memory类型的选择<br>② 在Agent中持久化记忆<br>③ 记忆与向量存储结合            | 实现一个记住用户偏好的长期对话Agent     |


# Agent - 自定义Tool开发（生产级）
基础Tool（@tool装饰器）
```python
from langchain.tools import tool
from pydantic import BaseModel, Field

# 定义输入参数结构（LLM会自动填充）
class WeatherInput(BaseModel):
    location: str = Field(description="城市名称，如'北京'")
    date: str = Field(description="日期，格式'2024-01-01'，默认今天")

@tool(args_schema=WeatherInput)
def get_weather(location: str, date: str) -> str:
    """获取指定城市的天气信息"""
    # 实际调用天气API
    return f"{location} {date} 天气：晴，25°C"

# 测试
print(get_weather.invoke({"location": "上海", "date": "2024-04-07"}))
```

高级Tool（StructuredTool - 多参数复杂逻辑）
```python
from langchain.tools import StructuredTool
from typing import Optional, List

class OrderQueryInput(BaseModel):
    order_id: str = Field(description="订单号")
    include_items: bool = Field(default=True, description="是否包含商品明细")
    fields: Optional[List[str]] = Field(default=None, description="指定返回字段")

def query_order_system(order_id: str, include_items: bool, fields: Optional[List[str]]) -> dict:
    """查询ERP订单系统"""
    # 模拟数据库查询
    result = {
        "order_id": order_id,
        "status": "已发货",
        "total": 299.00,
        "items": ["商品A", "商品B"] if include_items else None
    }
    if fields:
        return {k: v for k, v in result.items() if k in fields}
    return result

order_tool = StructuredTool.from_function(
    func=query_order_system,
    name="query_order",
    description="查询订单详情，支持指定返回字段",
    args_schema=OrderQueryInput,
    return_direct=False  # 结果给LLM继续处理
)
```

Tool错误处理与重试（生产必备）
```python
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

class ToolException(Exception):
    """Tool业务异常"""
    pass

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_database(query: str) -> str:
    """搜索内部知识库"""
    try:
        # 模拟API调用
        result = api.search(query)
        if not result:
            return "未找到相关信息，建议换个关键词试试"
        return result
    except ConnectionError:
        raise ToolException("数据库连接失败，正在重试...")
    except Exception as e:
        # 返回给LLM，让Agent决定如何处理
        return f"查询出错：{str(e)}，请尝试简化查询条件"

# 在Agent中配置错误处理
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # 处理LLM输出格式错误
    max_iterations=5,            # 防止无限循环
    early_stopping_method="generate"  # 达到max_iter时生成最终答案
)
```

# RAG - 评估指标与优化
RAGAS自动化评估（行业标准）

```python
# 安装：pip install ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,      # 忠实度：答案是否基于检索内容
    answer_relevancy, # 答案相关性：是否回答问题
    context_precision, # 上下文精确率：检索块中有用的比例
    context_recall     # 上下文召回率：需要的信息是否被检索
)

# 准备评估数据（questions + contexts + answers + ground_truths）
eval_data = {
    "question": ["什么是RAG？", "LangChain如何安装？"],
    "contexts": [["RAG是检索增强生成..."], ["pip install langchain..."]],
    "answer": ["RAG结合检索和生成...", "使用pip安装..."],
    "ground_truth": ["检索增强生成技术", "pip install langchain"]
}

result = evaluate(eval_data, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)  # 查看各指标得分
```

高级检索策略对比

| 策略                | 适用场景            | 代码实现                                                 |
| :---------------- | :-------------- | :--------------------------------------------------- |
| **MMR (最大边际相关性)** | 需要多样性结果，避免重复    | `search_type="mmr"`, `fetch_k=20`, `lambda_mult=0.5` |
| **Self-Query**    | 结构化数据，带元数据过滤    | `SelfQueryRetriever.from_llm()`                      |
| **Multi-Vector**  | 长文档，需要小块检索+大块生成 | `ParentDocumentRetriever`                            |
| **HyDE**          | 查询语义不匹配文档       | 生成假设答案再嵌入检索                                          |
| **Step-back**     | 问题太具体，需要抽象后检索   | LLM生成step-back问题再检索                                  |

# Memory - 持久化与跨会话管理
方案1：数据库存储（SQLite/PostgreSQL）
```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

# 每个用户独立的session_id
def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///memory.db"  # 或 PostgreSQL
    )

# 在LCEL中使用
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 调用时指定session_id
response = chain_with_history.invoke(
    {"input": "你好，我是张三"},
    config={"configurable": {"session_id": "user_123"}}
)
# 下次调用相同session_id会自动加载历史
```

方案2：Redis缓存（高性能）
```python
from langchain_redis import RedisChatMessageHistory

history = RedisChatMessageHistory(
    session_id="user_456",
    redis_url="redis://localhost:6379/0",
    ttl=3600  # 1小时过期
)
```

方案3：向量存储长期记忆（跨会话回忆）
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.memory import VectorStoreRetrieverMemory

# 构建长期记忆库
embeddings = OpenAIEmbeddings()
long_term_memory = FAISS.from_texts(
    ["用户张三喜欢Python编程", "用户李四关注AI安全"],
    embeddings
)

# 创建检索式记忆
memory = VectorStoreRetrieverMemory(
    retriever=long_term_memory.as_retriever(search_kwargs={"k": 2}),
    memory_key="long_term_context"
)

# 在Prompt中使用
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手。相关背景：{long_term_context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
```

混合记忆架构（短期+长期）
```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def combine_memories(input_dict):
    # 加载短期记忆（最近对话）
    short_term = short_term_memory.load_memory_variables(input_dict)
    # 检索长期记忆（相关事实）
    long_term = long_term_memory.load_memory_variables(input_dict)
    return {
        "short_term": short_term["chat_history"],
        "long_term": long_term["long_term_context"],
        "input": input_dict["input"]
    }

chain = (
    combine_memories
    | prompt
    | llm
)
```

# 生产部署 checklist
| 组件         | 关键配置                                            | 监控指标           |
| :--------- | :---------------------------------------------- | :------------- |
| **Agent**  | max\_iterations=5, handle\_parsing\_errors=True | 工具调用成功率、平均迭代次数 |
| **RAG**    | 检索top-k=5-10, 重排序rerank\_top=3                  | 检索延迟、答案忠实度分数   |
| **Memory** | token限制2000-4000, 数据库连接池                        | 存储成本、查询延迟      |



# 开源自托管类向量数据库
名称	语言	特点
Milvus	Go/Python	支持海量数据（亿级），分布式架构，适合机器学习推荐系统
Weaviate	Go	图数据结构组织，支持混合搜索（向量+关键词），可自动将文本转为向量
Qdrant	Rust	专注高效向量搜索，支持全文搜索且不影响向量搜索性能，性能极佳
Chroma	Python	轻量级，专为 LLM 应用设计，适合本地开发和小型项目
FAISS	C++/Python	Facebook 出品，本地库（非数据库），适合离线批量检索

# 向量类型对比

向量类型	位宽	特点	适用场景
单精度浮点（float32）	32-bit	精度高，通用性强	BERT、OpenAI Embeddings
半精度浮点（float16）	16-bit	内存减半，适合GPU	大规模向量库
二进制（Binary）	1-bit	极致压缩，比float小32倍	哈希指纹、SimHash
稀疏向量（Sparse）	可变	仅存非零值，节省空间	TF-IDF、BM25

# 向量数据库集成对比
| 数据库               | 最佳场景          | 特点                |
| :---------------- | :------------ | :---------------- |
| **Chroma**        | 本地开发、小型项目     | 零配置、内存/本地文件、免费    |
| **FAISS**         | 内存检索、高频查询     | Meta开源、极速、无持久化    |
| **Pinecone**      | 生产环境、无需运维     | 全托管、自动扩缩容、按量付费    |
| **Milvus/Zilliz** | 大规模、企业级       | 十亿级向量、分布式、GPU加速   |
| **Weaviate**      | 混合搜索（向量+BM25） | GraphQL接口、模块化AI集成 |
| **Qdrant**        | 过滤查询复杂        | Rust编写、高性能过滤、开源友好 |


```
用户输入
    │
    ▼
┌─────────────────────────────────────────┐
│  Prompts  ← 格式化输入，注入上下文        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Memory   ← 注入历史对话/长期记忆         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Indexes  ← 检索相关外部知识（RAG）       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Chains   ← 串联多步骤处理流程            │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Models   ← 调用大语言模型生成回答        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Agents   ← 需要工具？自主调用外部工具    │
└────────────────┬────────────────────────┘
                 │
                 ▼
            最终输出
```

```
┌──────────────────────────────────────────────┐
│           LangChain 六大核心模块               │
├─────────────────┬────────────────────────────┤
│ 1. Model I/O    │ 模型输入输出标准化           │
│ 2. Retrieval    │ 外部数据检索（RAG核心）       │
│ 3. Chains       │ 组件链式编排                 │
│ 4. Memory       │ 对话记忆管理                 │
│ 5. Agents       │ 自主决策与工具调用            │
│ 6. Callbacks    │ 全链路监控与钩子              │
└─────────────────┴────────────────────────────┘
```

# 性能优化清单
| 场景       | 优化策略     | 代码要点                                   |
| :------- | :------- | :------------------------------------- |
| **高并发**  | 异步 + 连接池 | `agent_executor.ainvoke()` + `aiohttp` |
| **长链路**  | 流式输出     | `agent_executor.stream()`              |
| **成本控制** | 模型路由     | 简单任务用`gpt-3.5`，复杂用`gpt-4`              |
| **延迟敏感** | 缓存 + 预检索 | `LangChainCache` + 向量索引预热              |
| **可观测性** | 全链路追踪    | `LangSmith` + 自定义Metadata              |


# 当前版本（1.x）的变化总结

| 组件 | 原始定位 | 当前状态 | 替代/演进方案 |
|------|----------|----------|---------------|
| **Model I/O** | 模型交互 | ✅ 核心保留 | 拆分为独立包 + LCEL |
| **Memory** | 对话记忆 | ❌ 已废弃 | LangGraph `MemorySaver` |
| **Chains** | 流程串联 | ⚠️ 逐步废弃 | LCEL 管道 `\|` 写法 |
| **Retrieval** | 外部检索 | ✅ 核心保留 | 四步 RAG 流程更清晰 |
| **Agents** | 自主决策 | ⚠️ 重构 | LangGraph `create_react_agent` |
| **Callbacks** | 监控追踪 | ✅ 核心保留 | 新增 LangSmith 集成 |


LCEL 统一了链式调用方式
LangGraph 接管了状态管理和复杂 Agent 编排
各模型集成拆包独立，按需安装，避免臃肿
整体架构从"黑盒封装"走向"透明可控" 2

