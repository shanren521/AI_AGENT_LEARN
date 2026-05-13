from llama_index.core import VectorStoreIndexfrom pywin.framework.toolmenu import toolsfrom llama_index.core.agent import FunctionAgentfrom langchain_openai import OpenAI

# 0 概念

| 层级      | 组件                                              | 作用             |
| ------- | ----------------------------------------------- | -------------- |
| **数据层** | `SimpleDirectoryReader`                         | 加载文档           |
| **处理层** | `SentenceSplitter`                              | 文档分块（Chunking） |
| **索引层** | `VectorStoreIndex`                              | 构建向量索引         |
| **检索层** | `VectorIndexRetriever` / `QueryFusionRetriever` | 检索相关片段         |
| **生成层** | `QueryEngine` / `ChatEngine`                    | 生成最终回答         |
| **记忆层** | `ChatMemoryBuffer`                              | 维护对话历史         |
| **路由层** | `RouterQueryEngine` / `LLMSingleSelector`       | 智能路由查询         |

## 1 基础文档回答

**SimpleDirectoryReader**
+ 作用：从目录加载文档，自动检测 .txt, .pdf, .docx, .md 等格式 
+ load_data()：返回 Document 对象列表
+ 常用参数：
  + input_dir：输入目录路径
  + input_files：指定具体文件列表（替代 input_dir）
  + recursive=True：递归包含子目录
  + required_exts=[".pdf", ".txt"]：按扩展名过滤文件
  + file_metadata=func：自定义元数据生成函数，接收文件路径返回字典

**VectorStoreIndex**
+ 作用：将文档转为向量嵌入并存储，是最常用的索引类型 
+ from_documents(documents, **kwargs)：从文档构建索引
  + documents：Document 列表
  + transformations：转换管道（如 [SentenceSplitter()]）
  + storage_context：存储上下文（用于持久化或外部向量库）
  + show_progress=True：显示索引进度
  + docStoreStrategy=DocStoreStrategy.UPSERTS：避免重复文档 

**as_query_engine**(**kwargs)：创建查询引擎
+ similarity_top_k：检索最相似的 K 个片段（默认 2）
+ response_mode：响应合成模式（"compact", "refine", "tree_summarize" 等）
+ node_postprocessors：后处理器列表（如重排序器）
+ query_engine.query(query_str)
  + 返回 Response 对象，包含 response（文本）和 source_nodes（来源片段）

## 2 文档分块与自定义处理

**SentenceSplitter**
+ 作用：基于句子边界智能分块，避免切断语义 
+ from_defaults() / 直接初始化：
  + chunk_size：每块最大长度（默认 1024 tokens）
  + chunk_overlap：块间重叠（默认 20 tokens），保持上下文连续性 
  + separator：主分隔符（默认空格）
  + paragraph_separator：段落分隔符（默认 \n\n\n）
  + tokenizer：自定义分词函数
  + secondary_chunking_regex：二次切分正则
  + include_metadata=True：包含元数据
  + include_prev_next_rel=True：包含前后节点关系 

## 3 带记忆的对话系统

**ChatMemoryBuffer**
+ 作用：管理对话历史的内存缓冲区，token 感知 
+ from_defaults()：
  + token_limit：记忆的最大 token 数（如 3000, 4000），超限后旧消息会被丢弃 
  + chat_store：外部存储（如 RedisChatStore, PostgresChatStore），实现跨会话持久化 
  + chat_store_key：存储键（如用户 ID），区分不同用户的记忆 
+ 方法：
  + chat_engine.chat(message)：发送消息并获取回复
  + chat_engine.reset()：清空对话历史 
  + chat_engine.chat_history：查看当前历史记录
  + chat_mode 选项 ：
    + "context"：每轮注入检索上下文，简单直接
    + "condense_plus_context"：推荐模式。先将"当前问题+历史"重写为独立查询，再检索。例如将"它们有什么区别？"重写为"LlamaIndex 的不同索引类型有什么区别？" 
    + "condense_question"：仅重写问题，不注入上下文
    + "simple"：纯对话，不进行检索（适用于 SimpleChatEngine）

## 4 混合检索

**QueryFusionRetriever**
+ 作用：融合多个检索器的结果，提升召回率 
+ 参数：
  + retrievers：检索器列表（如 [vector_retriever, bm25_retriever]）
  + similarity_top_k：最终返回的 Top-K 结果
  + num_queries：LLM 生成的查询变体数（默认 1）。例如原始查询"量化技术优缺点"可能生成"量化交易好处"、"量化投资缺点"等变体，对每个变体都执行检索 
  + mode：融合模式 
    + "reciprocal_rerank"（RRF）：倒数排名融合，最常用
    + "relative_score"：相对得分融合（Min-Max 归一化 + 加权求和）
    + "dist_based_score"：基于分布的得分融合（考虑均值和标准差）
    + "simple"：简单重排序
  + use_async=True：异步并行检索，降低延迟

**BM25Retriever**
+ 作用：基于 BM25 算法的关键词检索器
+ from_defaults()：
  + index：索引对象
  + similarity_top_k：检索数量
  + nodes：也可直接从节点构建（BM25Retriever.from_defaults(nodes=nodes)）

## 5 查询路由

**RouterQueryEngine**
+ 作用：根据查询内容路由到最合适的查询引擎 
+ 参数： 
  + selector：选择器，决定使用哪个引擎
  + query_engine_tools：QueryEngineTool 列表，每个工具需包含 query_engine 和 description
  + summarizer：可选的摘要合成器

**LLMSingleSelector**
+ 作用：使用 LLM 从多个选项中选择一个最合适的 
+ from_defaults()：
  + llm：指定 LLM（默认使用全局 Settings.llm）

**QueryEngineTool**
+ 作用：将查询引擎包装为工具，供路由或 Agent 使用
+ from_defaults()：
  + query_engine：查询引擎实例
  + description：关键参数。描述该工具适用场景，LLM 根据此描述做路由决策。描述越具体，路由越准确 

## 6 自定义查询引擎

**VectorIndexRetriever**
+ 作用：从向量索引中检索相似节点
+ 参数：
  + index：向量索引实例
  + similarity_top_k：返回最相似的 K 个节点

**SimilarityPostprocessor**
+ 作用：基于相似度阈值过滤结果
+ 参数：
  + similarity_cutoff：相似度截断值（0-1），低于此值的节点被过滤 

**get_response_synthesizer()**
+ 作用：获取响应合成器，控制如何整合检索结果生成答案
+ response_mode：
  + "compact"：紧凑模式，将检索结果压缩后一次性输入 LLM
  + "refine"：精炼模式，逐块迭代优化答案
  + "tree_summarize"：树形摘要，适合长文档总结
  + "simple_summarize"：简单摘要

## 7 Agent智能体

**FunctionCallingAgent**
+ 作用：基于 LLM 函数调用能力的 Agent，自动决定使用哪些工具 
+ from_tools()：
  + tools：BaseTool 列表
  + llm：LLM 实例
  + memory：记忆缓冲区（可选，默认无记忆）
  + verbose=True：打印执行过程
+ 底层：实际创建 FunctionCallingAgentWorker 并包装为 AgentRunner 

**AgentRunner**
+ 作用：管理 Agent 的执行循环（run_step, create_task 等）
+ 方法：
  + chat(message)：对话式交互
  + run_task(task)：运行任务

QueryEngine: 无记忆的查询引擎
ChatEngine: 带有记忆的查询引擎

LlamaIndex中四个最基础、最重要的类:
+ Document: 数据容器, 代表任何来源的原始数据（如一个PDF文件、一个网页、一个数据库行）的轻量级容器。
  + text: 存储文档的文本内容
  + metadata: 存储文档的元数据，如文件名、作者、创建时间、页数、页码、页内容、页链接、页图片、页音频、页视频、页音频链接、页视频链接、页图片链接、页音频描述、页视频描述、页图片描述、页音频时间轴、页视频时间轴、页图片时间轴、页音频时间轴、页视频时间轴、页图片时间轴、页音频时间轴、页视频时间轴、页图片时间轴、页音频时间轴、页视频时间轴、页图片时间轴、页音频时间轴、页视频时间轴、页图片时间轴、页音频
  + doc_id: 存储文档的ID
+ Node: 最小处理单元，是 Document 被解析/分割后产生的“块”（chunk），是索引和检索操作的基础单元。
  + text: 节点包含的文本块
  + metadata: 继承自源文档的元数据
  + relationships: 定义了与其他节点的关系
+ Index: 索引，是数据结构，用于存储和检索节点。
  + from_documents: 从文档构建索引
  + as_query_engine: 将索引转换为查询引擎
  + insert(doc): 向现有索引中动态插入新文档
+ Response: 查询引擎的返回结果，包含查询结果和元数据。
  + response: 查询结果
  + source_nodes: 用于生成答案的源节点列表
  + metadata: 响应元数据
  + extra_info: 额外的信息
  + node_ids: 节点ID列表
  + print_response_source: 用于打印溯源信息

# 1 构建LLM应用

分为三个主要部分：构建 RAG 管道、构建代理和构建工作流

构建代理：代理是可以通过一组工具与世界交互的 LLM 驱动的知识工作者。

工作流：工作流是一种较低级别的、事件驱动的抽象，用于构建智能体应用程序。它们是构建任何高级智能体应用程序的基础层。

为您的智能体添加 RAG：检索增强生成（RAG）是将数据传递给 LLM 的关键技术，也是更复杂智能体系统的一个组成部分。

# 2 使用LLMs

```bash
pip install llama-index-llms-openai
```

```python
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import ChatMessage

# stream_complete\acomplete
response = OpenAI().complete("William Shakespeare is ")
print(response)
```

## 2.1 聊天接口

```python
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a joke."),
]

# 流式响应stream_chat和astream_chat 异步需要使用await
chat_response = llm.chat(messages)
```

## 2.2 指定模型

```python
llm = OpenAI(model="gpt-4o-mini")
response = llm.complete("Who is Laurie Voss?")
print(response)
```

## 2.3 多模态LLMs

```python
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")

messages = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="image.png"),
            TextBlock(text="Describe the image in a few sentences."),
        ],
    )
]

resp = llm.chat(messages)
print(resp.message.content)
```

## 2.4 工具调用

```python
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


def generate_song(name: str, artist: str) -> Song:
    """Generate a song with provided name and artist"""
    return {"name": name, "artist": artist}

tool = FunctionTool.from_defaults(fn=generate_song)
llm = OpenAI(model="gpt-4o")

response = llm.predict_and_call(
    [tool],
    "Pick a random song for me",
)
```

# 3 构建代理

## 3.1 使用现有工具

### 3.1.1 从LlamaHub使用现有工具

```bash
pip install llama-index-tools-yahoo-finance
```

```python
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI

def multiply(a: int, b: int) -> int:
    return a * b

def add(a: int, b: int) -> int:
    return a + b

finance_tools = YahooFinanceToolSpec().to_tool_list()

finance_tools.extend([multiply, add])

workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    llm=OpenAI(model="gpt-4.1"),
    tools=finance_tools,
    system_prompt="You are a helpful assistant."
)

async def main():
    response = await workflow.run(
        user_msg="What's the current stock price of NVIDIA?"
    )
    print(response)
```

## 3.2 维护状态

默认情况下， AgentWorkflow 在运行之间是无状态的。这意味着代理不会保留任何先前运行的记忆

要在运行之间保持状态，我们将创建一个新的 Context，名为 ctx。我们将工作流传递给它，以正确配置这个 Context 对象，供将要使用它的工作流使用。

```python
from llama_index.core.workflow import Context
from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI


workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    llm=OpenAI(model="gpt-4.1"),
    tools=[],
    system_prompt="You are a helpful assistant."
)
ctx = Context(workflow=workflow)

response = await workflow.run(user_msg="Hi, my name is Laurie!", ctx=ctx)
```

### 3.2.1 长时间维持状态

Context 是可序列化的，因此可以保存到数据库、文件等，并在稍后重新加载。

JsonSerializer只能序列化普通数据（str、int、dict、list），可以跨语言使用

JsonPickleSerializer能序列化Python 对象、类、函数，只能 Python 用
```python
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer

ctx_dict = ctx.to_dict(serializer=JsonSerializer())

restored_ctx = Context.from_dict(
    workflow, ctx_dict, serializer=JsonSerializer()
)
response3 = await workflow.run(user_msg="What's my name?", ctx=restored_ctx)
```

### 3.2.2 工具和状态

工具也可以定义为可以访问工作流上下文的工具。这意味着你可以从上下文中设置和检索变量，并在工具中使用它们，或者在不同的工具之间传递信息。

AgentWorkflow 使用了一个对所有代理都可用的上下文变量 state 。你可以依赖 state 中的信息，而无需显式传递它。

```python
from llama_index.core.agent import AgentWorkflow

async def set_name(ctx: Context, name: str) -> str:
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["name"] = name

    return f"Name set to {name}"


workflow = AgentWorkflow.from_tools_or_functions(
    [set_name],
    llm=llm,
    system_prompt="You are a helpful assistant that can set a name.",
    initial_state={"name": "unset"}
)

ctx = Context(workflow=workflow)

response = await workflow.run(user_msg="What's my name?", ctx=ctx)
print(response)  # name is unset

# 显示设置名称
response2 = await workflow.run(user_msg="My name is Laurie", ctx=ctx)
print(str(response2))

state = await ctx.store.get("state")
print("Name as stored in state: ", state["name"])
```

## 3.3 流式输出和事件

AgentWorkflow 提供了一套预构建的事件，你可以使用它们将输出流式传输给用户。

将介绍一个执行时间较长的网络搜索工具Tavily

```bash
pip install llama-index-tools-tavily-research
```

```python
from llama_index.tools.tavily_research import TavilyToolSpec
import os
from llama_index.core.agent.workflow import AgentStream

tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))

workflow = FunctionAgent(
    tools=tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information.",
)

handler = workflow.run(user_msg="What's the weather like in San Francisco?")
async for event in handler.stream_events():
    if isinstance(event, AgentStream):
        print(event.delta, end="", flush=True)
```

## 3.4 Human in the loop 人工介入

要实现人工介入，我们将让工具发出一个在流程中其他步骤都不会接收的事件。然后我们会指示工具等待直到它接收到一个特定的“回复”事件。

wait_for_event 用于等待 HumanResponseEvent。
waiter_event 是写入事件流的事件，用于通知调用者我们正在等待响应。
waiter_id 是针对此特定等待调用的唯一标识符。它有助于确保我们为每个 waiter_id 只发送一个 waiter_event 。
requirements 参数用于指定我们想要等待具有特定 user_name 的人类响应事件。
```python
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent
from llama_index.core.workflow import Context
from llama_index.core.agent import AgentWorkflow, FunctionAgent

async def dangerous_task(ctx: Context) -> str:
    """A dangerous task that requires human confirmation."""
    # emit a waiter event (InputRequiredEvent here)
    # and wait until we see a HumanResponseEvent
    
    question =  "Are you sure you want to proceed? "
    response = await ctx.wait_for_event(
        HumanResponseEvent,
        waiter_id=question,
        waiter_event=InputRequiredEvent(
            prefix=question,
            user_name="Human",
        ),
        requirements={"user_name": "Human"}
    )
    
    if response.response.strip().lower() == "yes":
        return "Dangerous task completed successfully."
    else:
        return "Dangerous task aborted."

workflow = FunctionAgent(
    tools=[dangerous_task],
    llm=llm,
    system_prompt="You are a helpful assistant that can perform dangerous tasks."
)

handler = workflow.run(user_msg="I want to proceed with the dangerous task.")

async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        response = input(event.prefix)
        handler.ctx.send_event(
            HumanResponseEvent(
                response=response,
                user_name=event.user_name
            )
        )
        
response = await handler
print(str(response))

```

## 3.5 LlamaIndex中的多智能体模式

当需要多个专家来解决一个任务时，LlamaIndex 提供了几种选择，每种选择都在便利性和灵活性之间进行权衡。

### 3.5.1 模式一: AgentWorkflow 代理工作流(线性集群模式)

声明一组智能体，并让 AgentWorkflow 管理交接。

何时使用——你希望开箱即用多代理行为，几乎不需要额外代码，并且你满意 AgentWorkflow 随附的默认交接启发式算法。

AgentWorkflow 本身就是一个预配置的工作流，用于理解代理、状态和工具调用。你提供一个包含一个或多个代理的数组，告诉它哪个应该开始，它将：

+ 给根代理用户消息。

+ 执行该代理选择的所有工具。

+ 允许代理在决定时将控制权交给另一个代理。

+ 重复直到某个代理返回最终答案。

**can_handoff_to**: 允许当前智能体把任务交接/转交给其他智能体。

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

research_agent = FunctionAgent(
    name="ReSearchAgent",
    description="Search the web and record notes.",
    system_prompt="You are a researcher… hand off to WriteAgent when ready.",
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"]
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Writes a markdown report from the notes.",
    system_prompt="You are a writer… ask ReviewAgent for feedback when done.",
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"]
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Reviews a report and gives feedback.",
    system_prompt="You are a reviewer…",  # etc.
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"]
)

agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required."
    }
)

resp = await agent_workflow.run(
    user_msg="Write me a report on the history of the web …"
)
```

AgentWorkflow 负责所有编排工作，边进行边流式传输事件，以便您可以随时向用户通报进度。

### 3.5.2 模式二: Orchestrator agent 指挥代理(子代理作为工具)

一个“协调者”智能体选择下一个要调用的子智能体；这些子智能体作为工具暴露给它。

何时使用——你需要一个单一的地方来决定每一步，以便注入自定义逻辑，但你仍然倾向于使用声明式代理作为工具体验，而不是自己编写计划器。

在这种模式中，你仍然构建专业代理（ ResearchAgent ， WriteAgent ， ReviewAgent ），但你不需要它们相互传递。相反，你将每个代理的 run 方法作为工具公开，并将这些工具交给一个新的顶层代理——协调器。

```python
import re
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context

async def call_research_agent(ctx: Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    result = await research_agent.run(
        user_msg=f"Write some notes about the following: {prompt}"
    )

    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["research_notes"].append(str(result))

    return str(result)

async def call_write_agent(ctx: Context) -> str:
    """Useful for writing a report based on the research notes or revising the report based on feedback."""
    async with ctx.store.edit_state() as ctx_state:
        notes = ctx_state["state"].get("research_notes", None)
        if not notes:
            return "No research notes to write from."

        user_msg = f"Write a markdown report from the following notes. Be sure to output the report in the following format: <report>...</report>:\n\n"

        # Add the feedback to the user message if it exists
        feedback = ctx_state["state"].get("review", None)
        if feedback:
            user_msg += f"<feedback>{feedback}</feedback>\n\n"

        # Add the research notes to the user message
        notes = "\n\n".join(notes)
        user_msg += f"<research_notes>{notes}</research_notes>\n\n"

        # Run the write agent
        result = await write_agent.run(user_msg=user_msg)
        report = re.search(
            r"<report>(.*)</report>", str(result), re.DOTALL
        ).group(1)
        ctx_state["state"]["report_content"] = str(report)

    return str(report)


async def call_review_agent(ctx: Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    async with ctx.store.edit_state() as ctx_state:
        report = ctx_state["state"].get("report_content", None)
        if not report:
            return "No report content to review."

        result = await review_agent.run(
            user_msg=f"Review the following report: {report}"
        )
        ctx_state["state"]["review"] = result

    return result


orchestrator = FunctionAgent(
    system_prompt=(
        "You are an expert in the field of report writing. "
        "You are given a user request and a list of tools that can help with the request. "
        "You are to orchestrate the tools to research, write, and review a report on the given topic. "
        "Once the review is positive, you should notify the user that the report is ready to be accessed."
    ),
    llm=orchestrator_llm,
    tools=[
        call_research_agent,
        call_write_agent,
        call_review_agent,
    ],
    initial_state={
        "research_notes": [],
        "report_content": None,
        "review": None,
    },
)

response = await orchestrator.run(
    user_msg="Write me a report on the history of the web …"
)
print(response)
```

### 3.5.3 模式三: Custom planner 自定义规划器(DIY提示+解析)

需要编写 LLM 提示（通常是 XML / JSON），自己规划序列并在代码中强制调用代理。

何时使用——极致灵活性。你需要强制执行非常特定的计划格式，与外部调度器集成，或收集先前模式无法开箱即用的额外元数据。

这里的意思是，你编写一个提示，指示 LLM 输出一个结构化计划（XML / JSON / YAML）。你自己的 Python 代码解析该计划并强制执行它。从属代理可以是任何东西—— FunctionAgent s、RAG 管道或其他服务。

```python
import re
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field
from typing_extensions import Any, Optional

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step
)

PLANNER_PROMPT = """You are a planner chatbot.

Given a user request and the current state, break the solution into ordered <step> blocks.  Each step must specify the agent to call and the message to send, e.g.
<plan>
  <step agent=\"ResearchAgent\">search for …</step>
  <step agent=\"WriteAgent\">draft a report …</step>
  ...
</plan>

<state>
{state}
</state>

<available_agents>
{available_agents}
</available_agents>

The general flow should be:
- Record research notes
- Write a report
- Review the report
- Write the report again if the review is not positive enough

If the user request does not require any steps, you can skip the <plan> block and respond directly.
"""

class InputEvent(StartEvent):
    user_msg: Optional[str] = Field(default=None)
    chat_history: list[ChatMessage]
    state: Optional[dict[str, Any]] = Field(default=None)
    
class OutputEvent(StopEvent):
    response: str
    chat_history: list[ChatMessage]
    state: dict[str, Any]

class StreamEvent(Event):
    delta: str
    
class PlanEvent(Event):
    step_info: str
    
class PlanStep(BaseModel):
    agent_name: str
    agent_input: str
    
class Plan(BaseModel):
    steps: list[PlanStep]
    
class ExecuteEvent(Event):
    plan: Plan
    chat_history: list[ChatMessage]
    
class PlannerWorkflow(Workflow):
    llm: OpenAI = OpenAI(
        model="o3-mini",
        api_key="sk-...."
    )
    agents: dict[str, FunctionAgent] = {
        "ResearchAgent": research_agent,
        "WriteAgent": write_agent,
        "ReviewAgent": review_agent,
    }
    
    @step
    async def plan(
            self, ctx: Context, ev: InputEvent
    ) -> ExecuteEvent | OutputEvent:
        if ev.state:
            await ctx.store.set("state", ev.state)
            
        chat_history = ev.chat_history
        
        if ev.user_msg:
            user_msg = ChatMessage(
                role="user",
                content=ev.user_msg,
            )
            chat_history.append(user_msg)
    
        state = await ctx.store.get("state")
        available_agents_str = "\n".join(
            [
                f'<agent name="{agent.name}">{agent.description}</agent>'
                for agent in self.agents.values()
            ]
        )
        system_prompt = ChatMessage(
            role="system",
            content=PLANNER_PROMPT.format(
                state=str(state),
                available_agents=available_agents_str
            )
        )

        response = await self.llm.astream_chat(
            messages=[system_prompt] + chat_history
        )
        
        full_response = ""
        async for chunk in response:
            full_response += chunk.delta or ""
            if chunk.delta:
                ctx.write_event_to_stream(
                    StreamEvent(delta=chunk.delta)
                )

        xml_match = re.search(r"(<plan>.*</plan>)", full_response, re.DOTALL)
        if not xml_match:
            chat_history.append(ChatMessage(
                role="assistant",
                content=full_response,
            ))
            return OutputEvent(
                response=full_response,
                chat_history=chat_history,
                state=state,
            )
        else:
            xml_str = xml_match.group(1)
            root = ET.fromstring(xml_str)
            plan = Plan(steps=[])
            for step in root.findall("step"):
                plan.steps.append(
                    PlanStep(
                        agent_name=step.attrib["agent"],
                        agent_input=step.text.strip() if step.text else "",
                    )
                )

            return ExecuteEvent(plan=plan, chat_history=chat_history)

    @step
    async def execute(self, ctx: Context, ev: ExecuteEvent) -> InputEvent:
        chat_history = ev.chat_history
        plan = ev.plan

        for step in plan.steps:
            agent = self.agents[step.agent_name]
            agent_input = step.agent_input
            ctx.write_event_to_stream(
                PlanEvent(
                    step_info=f'<step agent="{step.agent_name}">{step.agent_input}</step>'
                )
            )
            if step.agent_name == "ResearchAgent":
                await call_research_agent(ctx, agent_input)
            elif step.agent_name == "WriteAgent":
                # Note: we aren't passing the input from the plan since
                # we're using the state to drive the write agent
                await call_write_agent(ctx)
            elif step.agent_name == "ReviewAgent":
                await call_review_agent(ctx)

        state = await ctx.store.get("state")
        chat_history.append(
            ChatMessage(
                role="user",
                content=f"I've completed the previous steps, here's the updated state:\n\n<state>\n{state}\n</state>\n\nDo you need to continue and plan more steps?, If not, write a final response.",
            )
        )
        return InputEvent(
            chat_history=chat_history
        )
```

## 3.6 使用结构化输出

大多数情况下，你需要以特定格式获取代理的结果。代理的结果可以通过两种方式返回结构化的 JSON:
+ output_cls – 一个用作输出模式的 Pydantic 模型
+ structured_output_fn – 对于更高级的使用场景，可以提供一个自定义函数来验证或将代理的对话重写为你想要的任何模型。

### 3.6.1 使用output_cls

```python
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field

llm = OpenAI(model="gpt-4.1")

class MathResult(BaseModel):
    operation: str = Field(description="the performed operation")
    result: int = Field(description="the result of the operation")

def multiply(x: int, y: int):
    """Multiply two numbers"""
    return x * y

agent = FunctionAgent(
    tools=[multiply],
    name="calculator",
    system_prompt="You are a calculator agent who can multiply two numbers using the `multiply` tool.",
    output_cls=MathResult,
    llm=llm
)

response = await agent.run("What is 3415 * 43144?")
print(response.structured_response)
print(response.get_pydantic_model(MathResult))


# 多代理工作流
class Weather(BaseModel):
    location: str = Field(description="The location")
    weather: str = Field(description="The weather")


def get_weather(location: str):
    """Get the weather for a given location"""
    return f"The weather in {location} is sunny"

agent = FunctionAgent(
    llm=llm,
    tools=[get_weather],
    system_prompt="You are a weather agent that can get the weather for a given location",
    name="WeatherAgent",
    description="The weather forecaster agent.",
)

main_agent = FunctionAgent(
    name="MainAgent",
    tools=[],
    description="The main agent",
    system_prompt="You are the main agent, your task is to dispatch tasks to secondary agents, specifically to WeatherAgent",
    can_handoff_to=["WeatherAgent"],
    llm=llm,
)

workflow = AgentWorkflow(
    agents=[main_agent, agent],
    root_agent=main_agent.name,
    output_cls=Weather,
)

response = await workflow.run("What is the weather in Tokyo?")
print(response.structured_response)
print(response.get_pydantic_model(Weather))
```

### 3.6.2 使用structured_output_fn

自定义函数应输入由代理工作流产生的 ChatMessage 对象序列，并返回一个字典（该字典可以转换为 BaseModel 子类）

```python
import json
from llama_index.core.llms import ChatMessage
from typing import List, Dict, Any


class Flavor(BaseModel):
    flavor: str
    with_sugar: bool

async def structured_output_parsing(
        messages: List[ChatMessage],
) -> Dict[str, Any]:
    sllm = llm.as_structured_llm(Flavor)
    messages.append(
        ChatMessage(
            role="user",
            content="Given the previous message history, structure the output based on the provided format.",
        )
    )
    response = await sllm.achat(messages)
    return json.loads(response.messages.content)

def get_flavor(ice_cream_shop: str):
    return "Strawberry with no extra sugar"

agent = FunctionAgent(
    tools=[get_flavor],
    name="ice_cream_shopper",
    system_prompt="You are an agent that knows the ice cream flavors in various shops.",
    structured_output_fn=structured_output_parsing,
    llm=llm,
)

response = await agent.run(
    "What strawberry flavor is available at Gelato Italia?"
)
print(response.structured_response)
print(response.get_pydantic_model(Flavor))
```

### 3.6.3 流式传输结构化输出

```python
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStreamStructuredOutput,
)

handler = agent.run("What strawberry flavor is available at Gelato Italia?")

async for event in handler.stream_events():
    if isinstance(event, AgentInput):
        print(event)
    elif isinstance(event, AgentStreamStructuredOutput):
        print(event.output)
        print(event.get_pydantic_model(Weather))
    elif isinstance(event, ToolCallResult):
        print(event)
    elif isinstance(event, ToolCall):
        print(event)
    elif isinstance(event, AgentOutput):
        print(event)
    else:
        pass

response = await handler
```

# 4 构建RAG管道

## 4.1 Indexing 索引

### 4.1.1 索引是什么？

在 LlamaIndex 术语中，一个 Index 是一个由 Document 对象组成的数据结构，旨在通过 LLM 进行查询。

### 4.1.2 向量存储索引

VectorStoreIndex 是目前您最常遇到的索引类型。向量存储索引将您的文档分割成节点，然后为每个节点创建 vector embeddings 文本，以便 LLM 进行查询。

### 4.1.3 什么是embeddings？

Vector embeddings 是 LLM 应用程序功能的核心。

一个 vector embedding ，通常简称为嵌入，是文本语义或含义的数值表示。即使实际文本差异很大，具有相似含义的两段文本也会有数学上相似的嵌入。

### 4.1.4 Vector Store Index 嵌入文档

使用LLM的API将所有文本转换为嵌入

### 4.1.5 检索

VectorStoreIndex 会返回最相似的嵌入及其对应的文本片段。它返回的嵌入数量被称为 k ，因此控制返回多少嵌入的参数被称为 top_k 。由于这个原因，这种类型的搜索通常被称为“top-k 语义检索”。

### 4.1.6 Vector Store Index 的使用

from_documents 还接受一个可选参数 show_progress 。将其设置为 True 可在索引构建过程中显示进度条。

```python
from llama_index.core import VectorStoreIndex

documents = []
index = VectorStoreIndex.from_documents(documents)
```

### 4.1.7 摘要索引

摘要索引是一种更简单的索引形式，最适合于需要生成文档中文字摘要的查询。它简单地存储所有文档，并将它们全部返回给查询引擎。

## 4.2 Loading 加载

通常由三个阶段组成:
+ 加载数据
+ 转换数据
+ 索引并存储数据

### 4.2.1 加载器

LlamaIndex 通过数据连接器（也称为 Reader ）来完成这一过程。数据连接器从不同的数据源中摄取数据，并将数据格式化为 Document 对象。一个 Document 是一个数据集合（目前是文本，未来还将包括图像和音频）以及关于这些数据的元数据。

#### 4.2.1.1 SimpleDirectoryReader 加载

从指定目录中的每个文件中创建文档。它内置在 LlamaIndex 中，可以读取多种格式，包括 Markdown、PDF 文件、Word 文档、PowerPoint 演示文稿、图像、音频和视频。

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
```

#### 4.2.1.2 从数据库中加载

```bash
pip install llama-index-readers-database
pip install sqlalchemy pymysql psycopg2-binary
```

```python
import os
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    schema=os.getenv("DB_SCHEME"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    dbname=os.getenv("DB_NAME"),
)

query = "SELECT * FROM users"
documents = reader.load_data(query=query)
```

#### 4.2.1.3 直接创建文档

```python
from llama_index.core import Document

doc = Document(text="text")
```

### 4.2.2 转换

在数据加载后，你需要处理和转换你的数据，然后再将其放入存储系统中。这些转换包括分块、提取元数据以及嵌入每个块。

#### 4.2.2.1 高级转换API

```python
from llama_index.core import VectorStoreIndex

documents = []
index = VectorStoreIndex.from_documents(documents)
index.as_query_engine()

from llama_index.core.node_parser import SentenceSplitter

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

from llama_index.core import Settings

Settings.text_splitter = text_splitter
index = VectorStoreIndex.from_documents(
    documents, transformations=[text_splitter]
)
```

#### 4.2.2.2 低级转换API

你可以通过将我们的转换模块（文本分割器、元数据提取器等）作为独立组件使用，或者在我们的声明式转换管道接口中组合它们来实现这一点。

**将文档分割成节点**

处理你的文档的一个关键步骤是将其分割成“块”/节点对象。关键思想是将你的数据处理成可检索/输入 LLM 的小块。

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

documents = SimpleDirectoryReader("./data").load_data()
pipeline = IngestionPipeline(transformations=[TokenTextSplitter(), ...])
nodes = pipeline.run(documents=documents)
```

### 4.2.3 添加元数据

```python
document = Document(
    text="text",
    metadata={"filename": "<doc_file_name>", "category": "<category>"},
)
```

### 4.2.4 添加嵌入

```python
from llama_index.core.schema import TextNode

node1 = TextNode(text="<text_chunk>", id_="<node_id>")
node2 = TextNode(text="<text_chunk>", id_="<node_id>")
index = VectorStoreIndex([node1, node2])
```

## 4.3 Querying 查询

所有查询的基础是 QueryEngine 。获取 QueryEngine 最简单的方法是让索引为你创建一个

```python
query_engine = index.as_query_engine()
response = query_engine.query(
    "Write an email to the user given their background information."
)
print(response)
```

### 4.3.1 查询阶段

查询由三个不同的阶段组成:
+ 检索是指从你的 Index 中找到并返回与你的查询最相关的文档。
+ 后处理是指对检索到的 Node 进行可选的重新排序、转换或过滤，例如要求它们具有特定的元数据，如关键词。
+ 响应合成是指将你的查询、最相关的数据和提示结合，发送给你的 LLM 以返回响应。

### 4.3.2 自定义查询阶段

```python
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

index = VectorStoreIndex.from_documents(documents)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

response_synthesizer = get_response_synthesizer()
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
)

response = query_engine.query("What did the author do growing up?")
print(response)
```

#### 4.3.2.1 配置检索器

```python
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)
```

#### 4.3.2.2 配置节点后处理器

KeywordNodePostprocessor : 通过 required_keywords 和 exclude_keywords 过滤节点。

SimilarityPostprocessor : 通过设置相似度分数阈值（因此仅适用于基于嵌入的检索器）过滤节点。

PrevNextNodePostprocessor : 基于 Node 关系，为检索到的 Node 对象补充额外的相关上下文。


```python
node_postprocessors = [
    KeywordNodePostprocessor(
        required_keywords=["Combinator"], exclude_keywords=["Italy"]
    )
]
query_engine = RetrieverQueryEngine.from_args(
    retriever, node_postprocessors=node_postprocessors
)
response = query_engine.query("What did the author do growing up?")
```

#### 4.3.2.3 配置响应合成

在检索器获取相关节点后，一个 BaseSynthesizer 通过组合信息来合成最终响应。

```python
query_engine = RetrieverQueryEngine.from_args(
    retriever, response_mode=response_mode
)
```

response_mode支持的选项:
+ default : 通过依次遍历每个检索到的 Node “创建和细化”答案；这为每个节点进行单独的 LLM 调用。适用于更详细的答案。
+ compact : 在每次 LLM 调用时通过尽可能多地填充能适应最大提示大小的 Node 文本块来“压缩”提示。如果提示中太多块无法一次性填充，则通过多个提示“创建和优化”答案。
+ tree_summarize : 给定一组 Node 对象和查询，递归地构建一棵树并返回根节点作为响应。适用于摘要目的。
+ no_text : 仅运行检索器以获取将要发送到 LLM 的节点，而实际上并不发送它们。然后可以通过检查 response.source_nodes 进行检验。
+ accumulate : 给定一组 Node 对象和查询，将查询应用于每个 Node 文本块，同时将响应累积到数组中。返回所有响应的连接字符串。适用于当你需要分别对每个文本块运行相同查询的情况

## 4.4 Storing 存储

### 4.4.1 持久化到磁盘

```python
index.storage_context.persist(persist_dir="<persist_dir>")

graph.root_index.storage_context.persist(persist_dir="<persist_dir>")


from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")

# load index
index = load_index_from_storage(storage_context)
```

### 4.4.1.1 使用向量数据库

```python
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


documents = SimpleDirectoryReader("./data").load_data()

db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)


query_engine = index.as_query_engine()
response = query_engine.query("What is the meaning of life?")
print(response)
```

**从磁盘加载索引**

```python
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)
query_engine = index.as_query_engine()
response = query_engine.query("What is llama2?")
print(response)
```

### 4.4.2 插入文档或节点

```python
from llama_index.core import VectorStoreIndex


index = VectorStoreIndex([])
for doc in documents:
    index.insert(doc)
```

# 5 结构化数据提取

## 5.1 简介

### 5.1.1 什么是Pydantic

Pydantic 是一个广泛使用的数据验证和转换库。它严重依赖于 Python 类型声明。

```python
from pydantic import BaseModel
from typing import List, Optional

class User(BaseModel):
    id: int
    name: str = "JS"

class Foo(BaseModel):
    count: int
    size: Optional[float] = None

class Bar(BaseModel):
    apple: str = "x"
    banana: str = "y"

class Spam(BaseModel):
    foo: Foo
    bars: List[Bar]
```

### 5.1.2 将Pydantic对象转换为JSON结构

```python
schema = User.model_json_schema()
print(f"schema: {schema}")
"""
{
  "properties": {
    "id": {
      "title": "Id",
      "type": "integer"
    },
    "name": {
      "default": "Jane Doe",
      "title": "Name",
      "type": "string"
    }
  },
  "required": [
    "id"
  ],
  "title": "User",
  "type": "object"
}
"""
```

### 5.1.3 使用注解

```python
from datetime import datetime


class LineItem(BaseModel):
    """A line item in an invoice."""

    item_name: str = Field(description="The name of this item")
    price: float = Field(description="The price of this item")


class Invoice(BaseModel):
    """A representation of information from an invoice."""

    invoice_id: str = Field(
        description="A unique identifier for this invoice, often a number"
    )
    date: datetime = Field(description="The date this invoice was created")
    line_items: list[LineItem] = Field(
        description="A list of all the items in this invoice"
    )

"""
{
  "$defs": {
    "LineItem": {
      "description": "A line item in an invoice.",
      "properties": {
        "item_name": {
          "description": "The name of this item",
          "title": "Item Name",
          "type": "string"
        },
        "price": {
          "description": "The price of this item",
          "title": "Price",
          "type": "number"
        }
      },
      "required": ["item_name", "price"],
      "title": "LineItem",
      "type": "object"
    }
  },
  "description": "A representation of information from an invoice.",
  "properties": {
    "invoice_id": {
      "description": "A unique identifier for this invoice, often a number",
      "title": "Invoice Id",
      "type": "string"
    },
    "date": {
      "description": "The date this invoice was created",
      "format": "date-time",
      "title": "Date",
      "type": "string"
    },
    "line_items": {
      "description": "A list of all the items in this invoice",
      "items": {
        "$ref": "#/$defs/LineItem"
      },
      "title": "Line Items",
      "type": "array"
    }
  },
  "required": ["invoice_id", "date", "line_items"],
  "title": "Invoice",
  "type": "object"
}
"""
```

## 5.2 使用结构化LLM

```bash
pip install llama-index-readers-file
```

```python
from llama_index.readers.file import PDFReader
from pathlib import Path
from llama_index.llms.openai import OpenAI

pdf_reader = PDFReader()

documents = pdf_reader.load_data(file=Path("./data/example.pdf"))
text = documents[0].text

llm = OpenAI(model="gpt-4o")
sllm = llm.as_structured_llm(Invoice)
response = sllm.complete(text)
json_response = json.loads(response.text)
print(json.dumps(json_response, indent=2))
"""
{
    "invoice_id": "Visa \u2022\u2022\u2022\u20224469",
    "date": "2024-10-10T19:49:00",
    "line_items": [
        {"item_name": "Trip fare", "price": 12.18},
        {"item_name": "Access for All Fee", "price": 0.1},
        {"item_name": "CA Driver Benefits", "price": 0.32},
        {"item_name": "Booking Fee", "price": 2.0},
        {"item_name": "San Francisco City Tax", "price": 0.21},
    ],
}
"""
```

## 5.3 结构化预测

结构化预测让你对应用程序如何调用 LLM 和使用 Pydantic 有更细粒度的控制。

```python
from llama_index.core.prompts import PromptTemplate

prompt = PromptTemplate(
    "Extract an invoice from the following text. If you cannot find an invoice ID, use the company name '{company_name}' and the date as the invoice ID: {text}"
)
response = llm.structured_predict(
    Invoice, prompt, text=text, company_name="Google"
)
json_output = response.model_dump_json()
print(json.dumps(json.loads(json_output), indent=2))
```

### 5.3.1 底层原理

根据你使用的 LLM， structured_predict 使用两种不同的类之一来处理调用 LLM 和解析输出。

+ FunctionCallingProgram: LLM 具有函数调用 API
  + 将 Pydantic 对象转换为工具
  + 提示 LLM 并强制其使用该工具
  + 返回生成的 Pydantic 对象
+ LLMTextCompletionProgram:  LLMs 仅支持文本
  + 以 JSON 格式输出 Pydantic 模式
  + 将模式和数据发送给 LLM，并使用提示指令让其在符合模式的形式中响应
  + 在 Pydantic 对象上调用 model_validate_json() ，传入从 LLM 返回的原始文本

### 5.3.2 直接调用预测类

```python
from llama_index.core.program import LLMTextCompletionProgram, FunctionCallingProgram
textCompletion = LLMTextCompletionProgram.from_defaults(
    output_cls=Invoice,
    llm=llm,
    prompt=PromptTemplate(
        "Extract an invoice from the following text. If you cannot find an invoice ID, use the company name '{company_name}' and the date as the invoice ID: {text}"
    ),
)

output = textCompletion(company_name="Uber", text=text)

# 通过PydanticOutputParser自定义输出解析方式
from llama_index.core.output_parsers import PydanticOutputParser

class MyOutputParser(PydanticOutputParser):
    def get_pydantic_object(self, text: str):
        # do something more clever than this
        return self.output_parser.model_validate_json(text)


textCompletion = LLMTextCompletionProgram.from_defaults(
    llm=llm,
    prompt=PromptTemplate(
        "Extract an invoice from the following text. If you cannot find an invoice ID, use the company name '{company_name}' and the date as the invoice ID: {text}"
    ),
    output_parser=MyOutputParser(output_cls=Invoice),
)
```

## 5.4 低级结构化数据提取

### 5.4.1 直接调用工具

```python
from llama_index.core.program.function_program import get_function_tool

tool = get_function_tool(Invoice)

resp = llm.chat_with_tools(
    [tool],
    user_msg="Extract an invoice from the following text: " + text,
    tool_required=True,  # can optionally force the tool call
    allow_parallel_tool_calls=True,
)

tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)
outputs = []
for tool_call in tool_calls:
    if tool_call.tool_name == "Invoice":
        outputs.append(Invoice(**tool_call.tool_kwargs))

print(outputs[0])
```

### 5.4.2 直接提示

```python
schema = Invoice.model_json_schema()
prompt = "Here is a JSON schema for an invoice: " + json.dumps(
    schema, indent=2
)
prompt += (
    """
  Extract an invoice from the following text.
  Format your output as a JSON object according to the schema above.
  Do not include any other text than the JSON object.
  Omit any markdown formatting. Do not include any preamble or explanation.
"""
    + text
)

response = llm.complete(prompt)

print(response)

invoice = Invoice.model_validate_json(response.text)

pprint(invoice)
```

## 5.5 结构化输入

### 5.5.1 单独使用结构化输入

```python
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai import OpenAI
from typing import Dict
from pydantic import BaseModel

template_str = "Please extract from the following XML code the contact details of the user:\n\n```xml\n{{ user | to_xml }}\n```\n\n"
prompt = RichPromptTemplate(template_str)

class User(BaseModel):
    name: str
    surname: str
    age: int
    email: str
    phone: str
    social_accounts: Dict[str, str]

user = User(
    name="John",
    surname="Doe",
    age=30,
    email="john.doe@example.com",
    phone="123-456-7890",
    social_accounts={"bluesky": "john.doe", "instagram": "johndoe1234"},
)

prompt.format(user=user)

llm = OpenAI(model="gpt-4o")
response = llm.chat(prompt.format_messages(user=user))
print(response.message.content)
```

### 5.5.2 将结构化输入与结构化输出结合使用

```python
from pydantic import Field
from typing import Optional


class SocialAccounts(BaseModel):
    instagram: Optional[str] = Field(default=None)
    bluesky: Optional[str] = Field(default=None)
    x: Optional[str] = Field(default=None)
    mastodon: Optional[str] = Field(default=None)


class ContactDetails(BaseModel):
    email: str
    phone: str
    social_accounts: SocialAccounts


sllm = llm.as_structured_llm(ContactDetails)

structured_response = await sllm.achat(prompt.format_messages(user=user))

print(structured_response.raw.email)
print(structured_response.raw.phone)
print(structured_response.raw.social_accounts.instagram)
print(structured_response.raw.social_accounts.bluesky)
```

# 6 跟踪与调试

## 6.1 基础日志记录

```python
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
```

## 6.2 回调处理器

```python
import llama_index.core

llama_index.core.set_global_handler("simple")
```

# 7 评估

# 8 整合所有内容

# 9 隐私和安全
