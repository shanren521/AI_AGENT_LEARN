# LangChain 面试知识点大全（Python 后端 / AI Agent 方向）

## 1. LangChain 是什么

### 核心定义

LangChain 是一个用于构建 LLM 应用与 AI Agent 的开发框架。

目标：

* 统一大模型调用
* 封装 Prompt
* 管理上下文
* 集成外部工具
* 构建 Agent
* 实现 RAG
* 支持工作流编排

---

# 2. LangChain 核心架构（高频）

面试必问：

> LangChain 的核心模块有哪些？

---

## 2.1 Models（模型层）

统一封装各种 LLM。

### 包括：

| 类型             | 说明     |
| -------------- | ------ |
| LLM            | 文本补全模型 |
| ChatModel      | 聊天模型   |
| EmbeddingModel | 向量模型   |

---

### 常见模型

| 模型         | 用途   |
| ---------- | ---- |
| OpenAI GPT | 通用对话 |
| Claude     | 长上下文 |
| Gemini     | 多模态  |
| Ollama     | 本地模型 |
| Qwen       | 国产模型 |

---

### 面试重点

#### Q：LLM 和 ChatModel 区别？

LLM：

```python
llm.invoke("介绍一下Python")
```

ChatModel：

```python
chat.invoke([
    HumanMessage(content="你好")
])
```

区别：

| LLM              | ChatModel |
| ---------------- | --------- |
| 纯文本输入            | Message输入 |
| 早期Completion API | Chat API  |
| 不适合Agent         | Agent标准   |

---

## 2.2 Prompt Templates

用于动态拼接 Prompt。

---

### PromptTemplate

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "请介绍一下{topic}"
)

prompt.format(topic="LangChain")
```

---

### ChatPromptTemplate

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是AI助手"),
    ("human", "{question}")
])
```

---

### 面试重点

#### Q：为什么不能直接字符串拼接？

原因：

* 可维护性差
* Prompt难管理
* 无法复用
* 不方便Few-shot
* 不利于链式调用

---

## 2.3 Output Parser

用于解析模型输出。

---

### 常见 Parser

| Parser               | 用途    |
| -------------------- | ----- |
| StrOutputParser      | 文本    |
| JsonOutputParser     | JSON  |
| PydanticOutputParser | 结构化输出 |

---

### 示例

```python
parser = JsonOutputParser()

chain = prompt | llm | parser
```

---

### 面试高频

#### Q：为什么需要 OutputParser？

因为：

LLM 输出不稳定。

Parser 用于：

* 保证格式统一
* 降低后处理复杂度
* 结构化输出
* Agent Tool调用

---

# 3. LCEL（LangChain Expression Language）

这是 LangChain 新版核心。

高频。

---

## 3.1 LCEL 是什么

一种链式表达式语言。

核心操作符：

```python
|
```

---

### 示例

```python
chain = prompt | llm | parser
```

---

## 3.2 Runnable

LCEL 本质：

所有组件都实现 Runnable 接口。

---

### Runnable 常见能力

| 方法      | 作用   |
| ------- | ---- |
| invoke  | 单次调用 |
| batch   | 批量   |
| stream  | 流式   |
| ainvoke | 异步   |
| astream | 异步流  |

---

### 面试重点

#### Q：为什么 LangChain 后来强调 LCEL？

旧版 Chain：

* 黑盒
* 扩展差
* 调试困难

LCEL：

* 可组合
* 声明式
* 易并行
* 易流式
* 易监控

---

# 4. Chains（链）

---

## 4.1 什么是 Chain

多个组件串联。

例如：

```text
用户问题
 -> Prompt
 -> LLM
 -> Parser
 -> 输出
```

---

## 4.2 常见 Chain

### LLMChain（旧）

```python
LLMChain(llm=llm, prompt=prompt)
```

新版已弱化。

---

### SequentialChain

多个步骤串联。

---

### RouterChain

动态路由。

例如：

* 数学问题 -> 数学模型
* 代码问题 -> Code模型

---

# 5. Memory（记忆）

Agent 高频。

---

## 5.1 为什么需要 Memory

LLM 默认无状态。

Memory 用于：

* 保存上下文
* 多轮对话
* 用户历史

---

## 5.2 常见 Memory

| 类型                         | 特点   |
| -------------------------- | ---- |
| ConversationBufferMemory   | 全量历史 |
| ConversationSummaryMemory  | 总结历史 |
| ConversationWindowMemory   | 窗口   |
| VectorStoreRetrieverMemory | 向量记忆 |

---

## 5.3 面试重点

#### Q：为什么不能无限保存聊天记录？

因为：

* Token爆炸
* 成本上升
* 注意力衰减
* 上下文长度限制

---

# 6. RAG（检索增强生成）

最核心面试内容之一。

---

# 6.1 RAG 工作流

```text
用户问题
 -> embedding
 -> 向量检索
 -> 召回文档
 -> 拼接Prompt
 -> LLM生成
```

---

## 6.2 LangChain 中的 RAG 组件

| 组件             | 作用   |
| -------------- | ---- |
| DocumentLoader | 文档加载 |
| TextSplitter   | 文本切分 |
| Embedding      | 向量化  |
| VectorStore    | 向量存储 |
| Retriever      | 检索器  |

---

# 6.3 DocumentLoader

### 常见 Loader

| Loader        | 用途  |
| ------------- | --- |
| PyPDFLoader   | PDF |
| TextLoader    | TXT |
| CSVLoader     | CSV |
| WebBaseLoader | 网页  |

---

# 6.4 Text Splitter（极高频）

## 为什么要切分

LLM Context 有限制。

Embedding 也不适合太长文本。

---

## 常见切分器

| Splitter                       | 特点        |
| ------------------------------ | --------- |
| CharacterTextSplitter          | 固定字符      |
| RecursiveCharacterTextSplitter | 递归切分（最常用） |
| TokenTextSplitter              | Token级    |

---

## chunk_size / overlap

面试高频。

### chunk_size

每块长度。

### chunk_overlap

块之间重叠。

作用：

* 保持语义连续
* 防止信息断裂

---

# 6.5 Embedding

## Embedding 是什么

把文本转换为向量。

---

## 常见 Embedding 模型

| 模型               | 特点   |
| ---------------- | ---- |
| OpenAI Embedding | 通用   |
| BGE              | 中文强  |
| E5               | 检索优秀 |
| Instructor       | 指令型  |

---

## 面试重点

#### Q：Embedding 为什么能检索相似文本？

因为：

语义相近文本：

在高维空间距离更近。

---

# 6.6 VectorStore（高频）

## 常见向量库

| 向量库      | 特点           |
| -------- | ------------ |
| FAISS    | 本地           |
| Chroma   | 轻量           |
| Milvus   | 分布式          |
| pgvector | PostgreSQL扩展 |
| Weaviate | 云原生          |

---

## 面试重点

#### Q：FAISS 和 Milvus 区别？

| FAISS | Milvus |
| ----- | ------ |
| 单机    | 分布式    |
| 内存型   | 企业级    |
| 简单    | 高可用    |
| 本地开发  | 生产系统   |

---

# 6.7 Retriever（高频）

## Retriever vs VectorStore

很多人答错。

### VectorStore

负责：

* 存储向量
* ANN搜索

### Retriever

负责：

* 召回逻辑
* top-k
* MMR
* rerank
* 混合检索

---

## 检索策略

| 方法         | 特点   |
| ---------- | ---- |
| similarity | 相似度  |
| mmr        | 多样性  |
| hybrid     | 混合检索 |

---

# 6.8 Rerank（高频）

## 为什么需要 rerank

向量召回：

可能不精准。

---

## 工作流

```text
粗召回
 -> rerank模型
 -> 精排序
```

---

## 常见 rerank

| 模型            | 特点  |
| ------------- | --- |
| bge-reranker  | 中文强 |
| Cohere Rerank | 商业  |
| CrossEncoder  | 精度高 |

---

# 7. Agent（最核心）

AI Agent 面试重点。

---

# 7.1 Agent 是什么

Agent =

```text
LLM + Planning + Tool Use + Memory
```

---

# 7.2 Agent 工作流

```text
用户问题
 -> 思考
 -> 选择工具
 -> 执行工具
 -> 观察结果
 -> 再推理
 -> 输出
```

---

# 7.3 Tool（极高频）

## Tool 是什么

Agent 可调用函数。

### 示例

```python
@tool
def search_weather(city: str):
    return "25度"
```

---

## Tool 本质

LLM Function Calling。

---

# 7.4 ReAct Agent（高频）

经典 Agent 架构。

## 思维流程

```text
Thought
Action
Observation
```

循环。

---

## 面试重点

#### Q：ReAct 为什么重要？

因为：

让模型：

* 推理
* 调用工具
* 根据结果继续推理

---

# 7.5 Function Calling

现代 Agent 核心。

## OpenAI Function Calling

模型输出：

```json
{
  "tool": "search_weather",
  "args": {
    "city": "东京"
  }
}
```

---

## 面试重点

#### Q：Function Calling 为什么比 Prompt Tool 更稳定？

因为：

* JSON约束
* Schema约束
* 减少幻觉
* 可程序化执行

---

# 7.6 Multi-Agent

高阶问题。

## 架构

```text
Supervisor
  -> Research Agent
  -> Coding Agent
  -> Writing Agent
```

---

## 面试重点

#### Q：为什么需要 Multi-Agent？

单Agent：

* 上下文混乱
* Tool过多
* 任务复杂度高

多Agent：

* 专业分工
* 可扩展
* 易控制

---

# 8. LangGraph（现在非常重要）

## LangGraph 是什么

LangGraph 是 LangChain 推出的状态化 Agent 工作流框架。

---

## 为什么出现 LangGraph

传统 Agent：

* 不稳定
* 不可控
* 难恢复

LangGraph：

* 状态机
* 可中断
* 可恢复
* 可循环
* DAG工作流

---

## 核心概念

| 概念               | 作用   |
| ---------------- | ---- |
| State            | 全局状态 |
| Node             | 节点   |
| Edge             | 边    |
| Conditional Edge | 条件路由 |

---

## 面试重点

#### Q：LangChain 和 LangGraph 区别？

| LangChain | LangGraph |
| --------- | --------- |
| 简单链式      | 状态图       |
| 线性        | DAG       |
| Agent封装   | 可控工作流     |
| 简单应用      | 复杂Agent系统 |

---

# 9. Streaming（流式输出）

## 为什么重要

提升用户体验。

---

## 实现

```python
for chunk in llm.stream():
    print(chunk)
```

---

## SSE

后端高频。

```text
LLM
 -> FastAPI
 -> SSE
 -> 前端
```

---

# 10. Callback & Observability

## Callback

用于：

* 日志
* Token统计
* 链路监控

---

## LangSmith（高频）

LangSmith 用于：

* Trace
* 调试
* Prompt管理
* 评估

---

## 面试重点

#### Q：为什么 AI 系统更需要可观测性？

因为：

LLM：

* 非确定性
* Prompt敏感
* Tool调用复杂
* Agent链路长

---

# 11. Prompt Engineering（高频）

## 常见 Prompt 技术

| 技术               | 说明    |
| ---------------- | ----- |
| Zero-shot        | 零样本   |
| Few-shot         | 少样本   |
| Chain of Thought | 思维链   |
| Self-Consistency | 多路径   |
| ReAct            | 推理+行动 |

---

# 12. Token 管理（高频）

## 为什么重要

直接影响：

* 成本
* 性能
* 延迟

---

## 优化方法

| 方法       | 作用      |
| -------- | ------- |
| Prompt压缩 | 减Token  |
| 摘要记忆     | 历史压缩    |
| RAG召回优化  | 减少无关上下文 |
| 小模型路由    | 降低成本    |

---

# 13. 缓存（高频）

## LangChain Cache

减少重复调用。

---

## 常见缓存

| 类型            | 场景  |
| ------------- | --- |
| InMemoryCache | 本地  |
| RedisCache    | 分布式 |

---

# 14. 异步（后端高频）

## 为什么异步重要

LLM：

* IO密集
* 延迟高

---

## 异步调用

```python
await chain.ainvoke()
```

---

## 面试重点

#### Q：为什么 AI 系统必须大量异步化？

因为：

* Tool调用慢
* 网络IO多
* Agent步骤长
* 并发量大

---

# 15. MCP（2025+ 高频）

## MCP 是什么

Anthropic 提出的：

Model Context Protocol。

---

## 目标

统一：

* Tool
* Context
* 外部系统

---

## 类似：

AI 世界的：

```text
HTTP协议
```

---

# 16. 面试高频场景题

# 16.1 如何设计企业知识库？

标准答案：

```text
文档加载
 -> 清洗
 -> chunk
 -> embedding
 -> vector db
 -> retriever
 -> rerank
 -> prompt
 -> llm
```

---

# 16.2 如何降低幻觉？

## 标准答案

### 方法：

* RAG
* Function Calling
* JSON输出
* 限制Prompt
* 引用来源
* 降低temperature
* rerank

---

# 16.3 如何优化 RAG 效果？

## 高频答案

### 数据层

* 清洗
* 去重
* chunk优化

### 检索层

* hybrid search
* rerank
* metadata filtering

### 生成层

* Prompt优化
* 引用约束

---

# 16.4 Agent 为什么容易失控？

## 原因

* 无限循环
* Tool hallucination
* 上下文污染
* 长链误差累积

## 解决方案

* step限制
* human-in-the-loop
* tool白名单
* 状态机控制
* LangGraph

---

# 17. 面试官最喜欢深挖的问题

## Q1：为什么 RAG 不等于 Fine-tuning？

| RAG  | Fine-tuning |
| ---- | ----------- |
| 外部知识 | 参数内知识       |
| 更新快  | 更新慢         |
| 成本低  | 成本高         |
| 可溯源  | 不可溯源        |

---

## Q2：为什么 Agent 比 Workflow 更难？

因为：

Agent：

* 非确定性
* 动态决策
* 路径不可预测

Workflow：

* 固定流程
* 易测试
* 易监控

---

## Q3：为什么很多 AI 项目最终会弱化 Agent？

因为：

复杂 Agent：

* 不稳定
* 成本高
* 难维护

工业界更偏向：

```text
Workflow + Limited Agent
```

---

# 18. LangChain 项目实战方向（面试加分）

## 初级项目

* PDF聊天
* 企业知识库
* AI客服

---

## 中级项目

* SQL Agent
* 浏览器Agent
* 多Tool Agent

---

## 高级项目

* Multi-Agent
* Deep Research
* Coding Agent
* Autonomous Agent

---

# 19. 面试中的错误回答（避坑）

## 错误1

> “LangChain 就是调 OpenAI API”

问题：

太浅。

---

## 错误2

> “Agent 就是自动调用函数”

缺少：

* planning
* reasoning
* memory

---

## 错误3

> “RAG 就是向量搜索”

缺少：

* chunk
* rerank
* prompt
* retrieval strategy

---

# 20. 建议的学习顺序（非常重要）

## 第一阶段

基础：

* Prompt
* ChatModel
* LCEL
* OutputParser

---

## 第二阶段

RAG：

* Embedding
* VectorDB
* Retriever
* Rerank

---

## 第三阶段

Agent：

* Tool
* Function Calling
* ReAct
* Memory

---

## 第四阶段

工程化：

* FastAPI
* Streaming
* Async
* LangSmith

---

## 第五阶段

高级：

* LangGraph
* Multi-Agent
* MCP
* 长上下文
* AI Infra

---

# 21. 面试一句话总结模板

## LangChain

> LangChain 本质是一个用于构建 LLM 应用的编排框架，核心能力包括 Prompt 管理、RAG、Tool 调用、Agent 工作流以及模型上下文管理。

---

## RAG

> RAG 本质是“检索 + 生成”，通过外部知识增强 LLM，解决模型知识时效性与幻觉问题。

---

## Agent

> Agent 本质是让 LLM 具备推理、规划和调用工具的能力。

---

## LangGraph

> LangGraph 本质是面向复杂 Agent 的状态机工作流框架，用于解决 Agent 不稳定和不可控问题。

