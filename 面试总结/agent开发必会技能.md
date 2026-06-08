# Python 后端转 AI Agent 开发：完整技术栈地图

你已经有 Python 后端基础（Flask/Django/Numpy/Pandas），因此重点不是“学 Python”，而是补齐：

* LLM 基础
* Agent 工程化
* 推理编排
* RAG
* 多智能体
* 工具调用
* 推理优化
* AI Infra
* AI 产品工程

下面按「必须掌握 → 进阶 → 高阶」分层。

---

# 一、基础层（必须）

这是 AI Agent 工程师的核心地基。

---

# 1. Python 工程能力（强化）

你已经有基础，但 AI Agent 对工程能力要求更高。

## 必学

* async / await
* asyncio
* 多线程 / 多进程
* queue
* context manager
* typing
* pydantic
* dataclass
* decorator
* generator
* logging
* retry
* cache

## 必会库

* `pydantic`
* `tenacity`
* `httpx`
* `aiohttp`
* `orjson`
* `uvloop`
* `rich`
* `typer`

---

# 2. API / Backend

AI Agent 本质仍然是后端系统。

## 必学

* REST API
* WebSocket
* SSE（Streaming）
* JWT
* OAuth2
* OpenAPI
* gRPC

## 框架

* FastAPI（核心）
* Flask（了解即可）
* Django（非重点）

## 为什么 FastAPI 极重要

Agent：

* 流式输出
* 长连接
* async
* 并发调用 LLM
* tool call

都依赖 FastAPI。

---

# 3. 数据库

## SQL

必须：

* PostgreSQL

需要掌握：

* JSONB
* 向量扩展
* 全文检索

---

## NoSQL

* Redis（核心）
* MongoDB（可选）

Redis 在 Agent 里大量用于：

* memory
* cache
* queue
* session
* state

---

# 4. Linux / DevOps

AI Agent 开发非常依赖部署。

## 必学

* Linux
* shell
* systemd
* nginx
* docker
* docker compose

## 进阶

* Kubernetes
* Helm
* GPU deployment

---

# 二、LLM 基础（核心）

这是 AI Agent 的根。

---

# 5. Transformer / LLM 原理

不用研究论文级别，但必须理解。

## 必学

* Token
* Embedding
* Attention
* Transformer
* Context Window
* KV Cache
* Temperature
* Top-p
* Hallucination
* Prompt Injection
* Function Calling
* Structured Output

---

# 6. Prompt Engineering

## 必学

* Zero-shot
* Few-shot
* Chain of Thought
* ReAct
* Self-Consistency
* Tree of Thoughts
* Reflection
* Plan & Execute

## 必须理解

Prompt 已经不是“写提示词”：

而是：

* agent behavior design
* workflow design
* reasoning control

---

# 7. Embedding

## 必学

* embedding 原理
* semantic similarity
* cosine similarity
* chunking
* rerank

## 常用模型

* bge
* e5
* jina embeddings
* OpenAI embeddings

---

# 8. 向量数据库（RAG 核心）

## 必学

* 向量检索
* ANN
* HNSW
* Hybrid Search

## 主流

* Milvus
* pgvector
* Weaviate
* Qdrant
* Chroma

## 推荐路线

先：

* pgvector

再：

* Milvus / Qdrant

---

# 三、Agent Framework（核心）

这是 AI Agent 开发最重要部分。

---

# 9. LangChain（必须）

## 必学

* PromptTemplate
* LCEL
* Runnable
* Chain
* Memory
* Tool
* Retriever
* OutputParser
* AgentExecutor

## 必须掌握

* async chain
* streaming
* callback
* tracing

---

# 10. LangGraph（极重要）

这是现在 Agent 核心框架。

## 必学

* StateGraph
* Node
* Edge
* Conditional Edge
* Reducer
* Checkpoint
* Memory
* Human-in-the-loop

## 高阶

* Multi-agent
* Supervisor
* Swarm
* Reflection Agent
* Planning Agent

---

# 11. LlamaIndex

重点：

* RAG
* document pipeline
* retrieval pipeline

## 必学

* Document
* Node
* Index
* Retriever
* Query Engine
* Router
* Agent

---

# 12. MCP（Model Context Protocol）

2025~2026 已经非常重要。

## 必学

* MCP Server
* MCP Client
* Tool Discovery
* Tool Registry
* Context Sharing

## 必须理解

未来 Agent 工具生态标准化：

MCP 很可能成为核心协议。

---

# 四、RAG（核心）

90% 企业 AI 项目都需要。

---

# 13. RAG 全链路

## 必学

### 文档处理

* PDF parsing
* OCR
* chunking
* metadata

### Retrieval

* dense retrieval
* sparse retrieval
* hybrid retrieval

### Rerank

* cross encoder
* reranker

### Generation

* grounded generation
* citation

---

# 14. 文档解析

## 必学

* unstructured
* pymupdf
* marker
* docling

---

# 15. OCR

## 主流

* PaddleOCR
* GOT-OCR
* MinerU

---

# 五、Tool Calling（Agent 核心）

Agent 的本质：

> LLM + Tool Use

---

# 16. Tool Calling

## 必学

* function calling
* structured output
* JSON schema
* tool routing
* tool planning

---

# 17. Browser Agent

## 必学

* Playwright
* browser-use
* stagehand

## 重要

大量 AI Agent 岗位：

其实是：

* Browser Agent
* Web Agent

---

# 18. Computer Use Agent

## 方向

* GUI Agent
* Desktop Agent

## 技术

* computer-use
* pyautogui
* vision agent

---

# 六、多智能体（高级）

2025 以后开始大量出现。

---

# 19. Multi-Agent

## 必学架构

* Supervisor
* Planner
* Executor
* Critic
* Reflection
* Debate

---

# 20. 多 Agent 框架

## 主流

* CrewAI
* AutoGen
* LangGraph
* OpenAI Swarm

---

# 七、模型服务（重要）

---

# 21. 模型推理部署

## 必学

* vLLM
* SGLang
* Ollama

## 高阶

* TensorRT-LLM
* llama.cpp
* TGI

---

# 22. 模型 API

## 必学

* OpenAI API
* Anthropic API
* Gemini API

## 国产

* DeepSeek
* Qwen
* Kimi
* GLM

---

# 八、AI Infra（高阶）

---

# 23. GPU 基础

## 必学

* CUDA 基础
* 显存
* tensor parallel
* kv cache
* quantization

---

# 24. 推理优化

## 必学

* batching
* speculative decoding
* flash attention
* quantization

---

# 九、AI Workflow（非常重要）

---

# 25. Workflow Engine

## 必学

* DAG
* State Machine
* Event Driven

## 相关

* LangGraph
* Temporal
* Prefect

---

# 十、评估与观测（企业级）

---

# 26. AI Evaluation

## 必学

* hallucination evaluation
* RAG evaluation
* trajectory evaluation

## 工具

* Ragas
* DeepEval
* LangSmith

---

# 27. Observability

## 必学

* tracing
* prompt logging
* token logging

## 工具

* LangSmith
* LangFuse
* Helicone

---

# 十一、缓存与性能

---

# 28. Cache

## 必学

* semantic cache
* redis cache
* embedding cache

---

# 29. Queue

## 必学

* Celery
* RabbitMQ
* Kafka

---

# 十二、前端（至少会一点）

很多 Agent 工程师死在：

不会流式前端。

---

# 30. 前端基础

## 必学

* React
* Next.js
* SSE streaming
* websocket

---

# 十三、云与部署

---

# 31. 云平台

## 必学

* AWS
* 阿里云
* GCP

---

# 32. Serverless

## 必学

* Modal
* RunPod
* Vercel AI

---

# 十四、安全（非常重要）

---

# 33. AI Security

## 必学

* Prompt Injection
* Jailbreak
* Tool Abuse
* Data Leakage

---

# 十五、真正企业项目会用到的技术

---

# 34. 企业 Agent 系统

## 常见架构

```text
Frontend
    ↓
API Gateway
    ↓
Agent Service
    ↓
LLM Router
    ↓
Tool Layer
    ↓
RAG Layer
    ↓
Vector DB
    ↓
Observability
```

---

# 十六、必须做的项目（核心）

没有项目经验，很难进入 AI Agent。

---

# 35. 必做项目路线

## 初级

### 1. ChatPDF

技术：

* FastAPI
* pgvector
* LangChain

---

## 中级

### 2. RAG QA System

加入：

* rerank
* hybrid retrieval
* streaming

---

## 中级

### 3. Browser Agent

技术：

* Playwright
* LangGraph
* tool calling

---

## 高级

### 4. Multi-Agent System

例如：

* Research Agent
* Coding Agent
* Report Agent

---

## 高级

### 5. Deep Research Agent

类似：

* OpenAI Deep Research
* Manus

核心：

* planning
* browser
* memory
* reflection

---

# 十七、最推荐学习路线（现实版）

---

# 第一阶段（1个月）

## 目标

掌握 LLM + RAG

## 学习

* FastAPI
* OpenAI API
* Prompt
* Embedding
* Vector DB
* LangChain
* pgvector

---

# 第二阶段（1~2个月）

## 目标

掌握 Agent

## 学习

* LangGraph
* Tool Calling
* Memory
* Planning
* Browser Agent

---

# 第三阶段（2个月）

## 目标

工程化

## 学习

* vLLM
* Redis
* Celery
* LangSmith
* Docker
* Kubernetes

---

# 第四阶段（长期）

## 目标

高级 Agent

## 学习

* Multi-agent
* MCP
* Deep Research
* AI Infra
* 推理优化

---

# 十八、现在企业最缺的人

目前市场最缺的不是：

* 会调 prompt 的人

而是：

* 能做 Agent 工程化的人

即：

* 能写后端
* 能做 workflow
* 能做 tool use
* 能做 RAG
* 能做部署
* 能做 observability
* 能解决 hallucination
* 能处理生产问题

这类人非常少。

---

# 十九、GitHub 必看项目

## Agent

* LangChain
* LangGraph
* LlamaIndex
* CrewAI
* AutoGen

## Browser Agent

* browser-use
* stagehand

## RAG

* Haystack
* RAGFlow

## 推理服务

* vLLM
* SGLang
* Ollama

## Observability

* LangSmith
* LangFuse

## MCP

* Model Context Protocol
