from torch.utils.checkpoint import checkpointfrom langgraph_runtime_inmem.checkpoint import InMemorySaverfrom torch.utils.checkpoint import checkpointfrom langchain_core.messages import AIMessageChunkfrom anyio.lowlevel import checkpointfrom win32comext.shell.demos.servers.folder_view import tasksfrom langgraph.graph import add_messagesfrom win32comext.shell.demos.servers.folder_view import tasks

# LangGraph

# 1 Get Started 

## 1.1 Quickstart 快速入门

### 1.1.1 定义工具和模型

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model


model = init_chat_model(
    'gpt-4.1',
    temperature=0
)

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply `a` and `b`
    
    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """
    Adds `a` and `b`
    
    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """
    Divide `a` and `b`
    
    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


```

### 1.1.2 定义模型节点

@task 装饰器将函数标记为可以作为代理一部分执行的任务。任务可以在你的入口函数中同步或异步调用。

```python
from langgraph.graph import add_messages
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall
)
from langchain_core.messages import BaseMessage
from langgraph.func import task, entrypoint


@task
def call_llm(messages: list[BaseMessage]):
    """LLM decides whether to call a tool or not"""
    return model_with_tools.invoke(
        [
            SystemMessage(content='You are a helpful assistant tasked with performing arithmetic on a set of inputs.')
        ]
        + messages
    )

```

### 1.1.3 定义工具节点

```python
@task
def call_tool(tool_call: ToolCall):
    """Performs the tool call"""
    tool = tools_by_name[tool_call['name']]
    return tool.invoke(tool)
```

### 1.1.4 定义代理

在 Functional API 中，您无需显式定义节点和边，而是在单个函数内编写标准控制流逻辑（循环、条件语句）。

```python
@entrypoint
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()
    
    while True:
        if not model_response.tool_calls:
            break

        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()
    messages = add_messages(messages, model_response)
    return messages

messages = [HumanMessage(content="Add 3 and 4.")]
for chunk in agent.stream(messages, stream_mode="updates"):
    print(chunk)
    print("\n")
```

## 1.2 Local Server 本地服务器

### 1.2.1 安装LangGraph CLI

```bash
# Python >= 3.11 is required.
pip install -U "langgraph-cli[inmem]"
```

### 1.2.2 创建一个LangGraph应用

从 new-langgraph-project-python 模板创建一个新应用。该模板展示了一个单节点应用，你可以通过自己的逻辑进行扩展。

```bash
langgraph new path/to/your/app --template new-langgraph-project-python
```

### 1.2.3 安装依赖

```bash
cd path/to/your/app
pip install -e .
```

### 1.2.4 创建.env 文件

```
LANGSMITH_API_KEY=lsv2...
```

### 1.2.5 启动Agent服务器

langgraph dev 命令以内存模式启动 Agent Server。此模式适用于开发和测试目的。在生产环境中使用时，请部署具有持久化存储后端访问权限的 Agent Server。

```bash
langgraph dev
```
使用 --tunnel 标志与您的命令一起创建安全隧道，因为 Safari 在连接到 localhost 服务器时有限制

### 1.2.6 在Studio中测试应用程序

Studio 是一个专门的 UI，您可以通过它连接到 LangGraph API 服务器，以在本地可视化、交互和调试您的应用程序。通过访问 langgraph dev 命令输出中提供的 URL 在 Studio 中测试您的图

对于在自定义主机/端口上运行的 Agent Server，请更新 URL 中的 baseUrl 查询参数。例如，如果您的服务器运行在 http://myhost:3000 上

### 1.2.7 测试API

#### 1.2.7.1 安装 LangGraph Python SDK

```bash
pip install langgraph-sdk
```

#### 1.2.7.2 向助手发送消息（无线程运行）

```python
from langgraph_sdk import get_client
import asyncio

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
    
asyncio.run(main())
```

## 1.3 Thinking in LangGraph 在LangGraph中思考

从你想要的自动化流程开始

当你使用 LangGraph 构建一个智能体时，你首先需要将其分解为称为节点的独立步骤。然后，你需要描述每个节点中的不同决策和转换。最后，通过一个共享状态将节点连接起来，每个节点都可以从中读取和写入。

### 1.3.1 将流程分解为独立的步骤

首先，确定你流程中的不同步骤。每个步骤将变成一个节点（一个执行特定任务的函数）。然后，勾勒出这些步骤如何相互连接。

### 1.3.2 确定每个步骤需要完成什么

对于图中的每个节点，确定它代表什么类型的操作以及它需要什么上下文才能正常工作。

### 1.3.3 设计状态

状态是所有节点在代理中可访问的共享内存。你可以把它想象成代理在处理过程中用来记录所有学习和决策的笔记本。

保持状态原始，按需格式化提示
```python
from typing import TypedDict, Literal

class EmailClassification(TypedDict):
    intent: Literal['question', 'bug', 'billing', 'feature', 'complex']
    urgency: Literal['low', 'medium', 'high', 'critical']
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str
    
    # Classification result
    classification: EmailClassification | None

    # Raw search/API results
    search_results: list[str] | None  # List of raw document chunks
    customer_history: dict | None  # Raw customer data from CRM

    # Generated content
    draft_response: str | None
    messages: list[str] | None
```

### 1.3.4 构建节点

现在我们将每一步实现为一个函数。在 LangGraph 中，一个节点就是一个 Python 函数，它接受当前状态并返回对它的更新。

### 1.3.5 连接起来

现在我们将节点连接成一个工作图。由于节点自行处理路由决策，我们只需要少数几个基本边。

为了启用带有 interrupt() 的人机交互，我们需要使用检查点来在运行之间保存状态

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END


workflow = StateGraph(EmailAgentState)

workflow.add_node('read_email', read_email)
workflow.add_node("classify_intent", classify_intent)

# Add retry policy for nodes that might have transient failures
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3)
)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

# Add only the essential edges
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# Compile with checkpointer for persistence, in case run graph with Local_Server --> Please compile without checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

## 1.4 Workflows + agents 工作流 + 代理

+ 工作流具有预定的代码路径，并设计为按特定顺序运行。
+ 代理是动态的，它们定义自己的流程和工具使用。

LangGraph 在构建代理和工作流时提供了多种优势，包括持久化、流式处理以及支持调试和部署。

### 1.4.1 LLMs and augmentation

工作流和代理系统基于 LLMs 以及你为它们添加的各种增强功能。工具调用、结构化输出和短期记忆是定制 LLMs 以满足你需求的几个选项。

```python
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    search_query: str = Field(None, description='Query that is optimized web search.')
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )

structured_llm = llm.with_structured_output(SearchQuery)

def multiply(a: int, b: int) -> int:
    return a * b

llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke('What is 2 times 3?')

msg.tool_calls
```

### 1.4.2 Prompt chaining

提示链式是指每个 LLM 调用处理前一个调用的输出。它通常用于执行可以分解为更小、可验证步骤的明确任务。一些例子包括：
+ 将文档翻译成不同的语言
+ 验证生成内容的一致性

**Graph API**
```python
from io import BytesIO
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from PIL import Image


class GraphState(TypedDict):
    topic: str
    joke: str
    improved_joke: str 
    final_joke: str 

def generate_joke(state: GraphState):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f'Write a short joke about {state["topic"]}')
    return {'joke': msg.content}

def check_punchline(state: GraphState):
    """Gate function to check if the joke has a punchline"""
    if '?' in state['joke'] or '!' in state['joke']:
        return 'Pass'
    return 'Fail'

def improve_joke(state: GraphState):
    """Second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {'improved_joke': msg.content}

def polish_joke(state: GraphState):
    """Third LLM call to polish the joke"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {'final_joke': msg.content}

workflow = StateGraph(GraphState)
workflow.add_node(generate_joke)
workflow.add_node(improve_joke)
workflow.add_node(polish_joke)
workflow.add_edge(START, 'generate_joke')
workflow.add_conditional_edges('generate_joke', check_punchline, {'Fail': 'improve_joke', 'Pass': END})
workflow.add_edge('improve_joke', 'polish_joke')
workflow.add_edge('polish_joke', END)

chain = workflow.compile()
img = Image.open(BytesIO(chain.get_graph().draw_mermaid_png()))
img.show()

state = chain.invoke({'topic': 'cats'})
print("Initial joke:")
print(state["joke"])
print("\n--- --- ---\n")
if "improved_joke" in state:
    print("Improved joke:")
    print(state["improved_joke"])
    print("\n--- --- ---\n")

    print("Final joke:")
    print(state["final_joke"])
else:
    print("Final joke:")
    print(state["joke"])
```

**Function API**
```python
from langgraph.func import entrypoint, task

@task
def generate_joke(topic: str):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a short joke about {topic}")
    return msg.content


def check_punchline(joke: str):
    """Gate function to check if the joke has a punchline"""
    # Simple check - does the joke contain "?" or "!"
    if "?" in joke or "!" in joke:
        return "Fail"

    return "Pass"


@task
def improve_joke(joke: str):
    """Second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {joke}")
    return msg.content


@task
def polish_joke(joke: str):
    """Third LLM call for final polish"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {joke}")
    return msg.content

@entrypoint()
def prompt_chaining_workflow(topic: str):
    original_joke = generate_joke(topic).result()
    if check_punchline(original_joke) == "Pass":
        return original_joke

    improved_joke = improve_joke(original_joke).result()
    return polish_joke(improved_joke).result()

for step in prompt_chaining_workflow.stream("cats", stream_mode="updates"):
    print(step)
    print("\n")
```

#### 1.4.2.1 Parallelization  并行化

通过并行化，LLMs 同时处理一个任务。这可以通过同时运行多个独立的子任务，或者多次运行同一任务以检查不同的输出来实现。并行化通常用于
+ 将子任务拆分并行运行，可提高速度
+ 多次运行任务以检查不同输出，可增强信心

例子:
+ 运行一个子任务处理文档中的关键词，以及第二个子任务检查格式错误
+ 运行一个任务多次，根据不同标准（如引用数量、使用来源数量和来源质量）对文档进行准确性评分

**Graph API**
```python
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


def call_llm_1(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def call_llm_2(state: State):
    """Second LLM call to generate story"""

    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}


def call_llm_3(state: State):
    """Third LLM call to generate poem"""

    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}


def aggregator(state: State):
    """Combine the joke, story and poem into a single output"""

    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
img = Image.open(BytesIO(parallel_workflow.get_graph().draw_mermaid_png()))

# Invoke
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])
```

**Function API**
```python
@task
def call_llm_1(topic: str):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a joke about {topic}")
    return msg.content


@task
def call_llm_2(topic: str):
    """Second LLM call to generate story"""
    msg = llm.invoke(f"Write a story about {topic}")
    return msg.content


@task
def call_llm_3(topic):
    """Third LLM call to generate poem"""
    msg = llm.invoke(f"Write a poem about {topic}")
    return msg.content


@task
def aggregator(topic, joke, story, poem):
    """Combine the joke and story into a single output"""

    combined = f"Here's a story, joke, and poem about {topic}!\n\n"
    combined += f"STORY:\n{story}\n\n"
    combined += f"JOKE:\n{joke}\n\n"
    combined += f"POEM:\n{poem}"
    return combined


# Build workflow
@entrypoint()
def parallel_workflow(topic: str):
    joke_fut = call_llm_1(topic)
    story_fut = call_llm_2(topic)
    poem_fut = call_llm_3(topic)
    return aggregator(
        topic, joke_fut.result(), story_fut.result(), poem_fut.result()
    ).result()

# Invoke
for step in parallel_workflow.stream("cats", stream_mode="updates"):
    print(step)
    print("\n")
```

### 1.4.3 Routing 路由

路由工作流处理输入，然后将它们引导至特定上下文的任务。这使你能够为复杂任务定义专门的工作流。例如，一个用于回答产品相关问题的构建工作流可能会首先处理问题的类型，然后将请求路由至定价、退款、退货等特定流程。

**Graph API**
```python
from typing_extensions import Literal
from langchain.messages import HumanMessage, SystemMessage


# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="The next step in the routing process"
    )


# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)


# State
class State(TypedDict):
    input: str
    decision: str
    output: str


# Nodes
def llm_call_1(state: State):
    """Write a story"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_2(state: State):
    """Write a joke"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_3(state: State):
    """Write a poem"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to story, joke, or poem based on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}


# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"


# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

# Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# Compile workflow
router_workflow = router_builder.compile()

# Show the workflow
img = Image.open(BytesIO(router_workflow.get_graph().draw_mermaid_png()))

# Invoke
state = router_workflow.invoke({"input": "Write me a joke about cats"})
print(state["output"])
```

**Function API**
```python
from typing_extensions import Literal
from pydantic import BaseModel
from langchain.messages import HumanMessage, SystemMessage

class Route(BaseModel):
    step: Literal['poem', 'story', 'joke'] = Field(
        None, description='The next step in the routing process'
    )

@task
def llm_call_1(input_: str):
    """Write a story"""
    result = llm.invoke(input_)
    return result.content

@task 
def llm_call_2(input_: str):
    """Write a joke"""
    result = llm.invoke(input_)
    return result.content

@task 
def llm_call_3(input_: str):
    """Write a poem"""
    result = llm.invoke(input_)
    return result.content

def llm_call_router(input_: str):
    """Route the input to the appropriate node"""
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to story, joke, or poem based on the user's request."
            ),
            HumanMessage(content=input_)
        ]
    )
    return decision.step

@entrypoint()
def router_workflow(input_: str):
    next_step = llm_call_router(input_)

    if next_step == 'story':
        llm_call = llm_call_1
    elif next_step == 'joke':
        llm_call = llm_call_2
    elif next_step == 'poem':
        llm_call = llm_call_3

    return llm_call(input_).result()

for step in router_workflow.stream("Write me a joke about cats", stream_mode="updates"):
    print(step)
    print("\n")
```

### 1.4.4 协调器-工作器

在一个协调器-工作器配置中，协调器:
+ 将任务分解为子任务
+ 将子任务分配给工作者
+ 将工作者输出综合为最终结果

Orchestrator-worker 工作流提供了更高的灵活性，通常用于无法像并行化那样预定义子任务的情况。这在编写代码或需要跨多个文件更新内容的工作流中很常见。

### 1.4.5 评估器-优化器

在评估器-优化器工作流程中，一个 LLM 调用会生成一个响应，而另一个 LLM 则评估该响应。如果评估器或人工参与环节确定响应需要改进，就会提供反馈并重新生成响应。
这个循环会持续进行，直到生成一个可接受的响应。

评估器-优化器工作流通常用于当任务有特定的成功标准，但需要迭代才能达到该标准时。例如，在两种语言之间翻译文本时，并不总是能找到完美的匹配。
可能需要几次迭代才能生成在两种语言中具有相同含义的翻译。

**Graph API**
```python
# Graph state
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )


# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)


# Nodes
def llm_call_generator(state: State):
    """LLM generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""

    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}


# Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


# Build workflow
optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # Name returned by route_joke : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Show the workflow
display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

# Invoke
state = optimizer_workflow.invoke({"topic": "Cats"})
print(state["joke"])
```

**Function API**
```python
class Feedback(BaseModel):
    grade: Literal['funny', 'not funny'] = Field(
        description='Decide if the joke is funny or not.'
    )
    feedback: str = Field(
        description='If the joke is not funny, provide feedback on how to improve it.'
    )

evaluator = llm.with_structured_output(Feedback)

@task 
def llm_call_generator(topic: str, feedback: Feedback):
    """LLM generates a joke"""
    if feedback:
        msg = llm.invoke(
            f"Write a joke about {topic} but take into account the feedback: {feedback}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {topic}")
    return msg.content

@task
def llm_call_evaluator(joke: str):
    """LLM evaluates the joke"""
    feedback = evaluator.invoke(f"Grade the joke {joke}")
    return feedback

@entrypoint()
def optimizer_workflow(topic: str):
    feedback = None
    while True:
        joke = llm_call_generator(topic, feedback).result()
        feedback = llm_call_evaluator(joke).result()
        if feedback.grade == "funny":
            break

    return joke

for step in optimizer_workflow.stream("Cats", stream_mode="updates"):
    print(step)
    print("\n")
```

### 1.4.6 Agent 代理

代理通常被实现为一个使用工具执行操作的 LLM。它们在持续反馈循环中运行，并用于问题与解决方案不可预测的情况。代理比工作流拥有更高的自主性，
能够就其使用的工具以及如何解决问题做出决策。你仍然可以定义可用的工具集以及代理的行为指南。

# 2 Capabilities 功能

## 2.1 Persistence 持久化

LangGraph 具有内置的持久化层，用于将图状态保存为检查点。当你使用检查点器编译图时，图状态在每个执行步骤的快照都会被保存，并组织成线程。这支持人机交互工作流、对话记忆、时间旅行调试和容错执行。

### 2.1.1 Core concepts 核心概念

#### 2.1.1.1 Threads 线程

线程是每个检查点保存时分配的唯一 ID 或线程标识符。它包含一系列运行的累积状态。当运行被执行时，助手底层图的状态将被持久化到线程中。

调用带检查点的图时，必须在配置的 configurable 部分指定 thread_id
```python
{"configurable": {"thread_id": "1"}}
```

可以检索线程的当前和历史状态。为了持久化状态，在执行 run 之前必须先创建线程。LangSmith API 提供了多个用于创建和管理线程及线程状态的端点。

检查点使用 thread_id 作为存储和检索检查点的主键。没有它，检查点无法保存状态或在中断后恢复执行，因为检查点使用 thread_id 来加载保存的状态。

#### 2.1.1.2 Checkpoints 检查点

一个线程在特定时间点的状态称为检查点。检查点是每个超级步骤时保存的图状态的快照，由 StateSnapshot 对象表示（有关完整字段参考，请参阅 StateSnapshot 字段）。

**超级步骤**
LangGraph 在每个超级步骤边界创建检查点。超级步骤是图的单个“滴答”，在该步骤中所有计划执行的节点（可能并行）执行。对于 START -> A -> B -> END 这样的顺序图，输入、节点 A 和节点 B 都有各自的超级步骤——每个步骤后都会生成一个检查点。
理解超级步骤边界对于时间旅行很重要，因为你可以只能从检查点（即超级步骤边界）恢复执行。

检查点是持久化的，并且可以用于在稍后时间恢复线程的状态。

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar":["a"]}

def node_b(state: State):
    return {"foo": "b", "bar":["b"]}

workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar": []}, config)

# 检查点 START -> node_a -> node_b -> END
```

**检查点命名空间**
每个检查点都有一个 checkpoint_ns （检查点命名空间）字段，用于标识它属于哪个图或子图(空字符串表示该检查点属于父图（根图）)

"node_name:uuid" ：该检查点属于作为给定节点调用的子图。对于嵌套子图，命名空间使用 | 分隔符连接（例如， "outer_node:uuid|inner_node:uuid"

### 2.1.2 获取和更新状态

#### 2.1.2.1 获取状态

在与保存的图状态交互时，你必须指定一个线程标识符。你可以通过调用 graph.get_state(config) 来查看图的最新状态。这将返回一个 StateSnapshot 对象，
该对象对应于配置中提供的线程 ID 关联的最新检查点，或者如果提供了检查点 ID，则关联于线程的检查点 ID 的检查点。
```python
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)
```

#### 2.1.2.2 获取历史状态

您可以通过调用 graph.get_state_history(config) 获取给定线程的图执行完整历史记录。这将返回与配置中提供的线程 ID 关联的 StateSnapshot 对象列表。
重要的是，检查点将按时间顺序排列，最新的检查点 / StateSnapshot 将位于列表的第一位。
```python
list(graph.get_state_history(config))
```

#### 2.1.2.3 查找特定检查点
您可以过滤状态历史记录以查找符合特定条件的检查点

#### 2.1.2.4 重放
回放会重新执行先前检查点中的步骤。使用先前的 checkpoint_id 调用图来重新运行检查点后的节点。检查点前的节点会被跳过（它们的结果已经被保存）。
检查点后的节点会重新执行，包括任何 LLM 调用、API 请求或中断——这些在回放期间总是会被重新触发。

#### 2.1.2.5 更新状态

你可以使用 update_state 编辑图状态。这会创建一个新的检查点，其中包含更新后的值——它不会修改原始检查点。更新被视为与节点更新相同：
当定义了 reducer 函数时，值会通过这些函数传递，因此带有 reducer 的通道会累积值而不是覆盖它们。

你可以选择性地指定 as_node 来控制更新被视为来自哪个节点，这会影响下一个执行的节点。

### 2.1.3 内存存储

状态模式指定了一组在图执行时被填充的键。如上所述，状态可以被检查点器在每个图步骤中写入到线程中，从而实现状态持久化。

仅使用检查点器，我们无法跨线程共享信息。这促使我们需要 Store 接口。以示例说明，我们可以定义一个 InMemoryStore 来跨线程存储用户信息。我们只需像之前一样编译图，并传递存储器。

InMemoryStore 适用于开发和测试。在生产环境中，请使用持久化存储，如 PostgresStore 或 RedisStore 。所有实现都扩展了 BaseStore，它是节点函数签名中使用的类型注解。

#### 2.1.3.1 基本用法

```bash
pip install langgraph-checkpoint-sqlite
pip install langgraph-checkpoint-postgres
```

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore, AsyncPostgresStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver, ShallowPostgresSaver


store = InMemoryStore()  # 长期记忆存储(跨线程共享数据)
saver = InMemorySaver()  # 短期记忆存储(保存对话状态快照，实现多轮对话上下文)
```

记忆由 tuple 命名空间管理，在这个特定示例中，命名空间将是 (<user_id>, "memories") 。命名空间可以是任意长度，并代表任何事物，不必特定于用户。
```python
user_id = "1"
namespace_for_memory = (user_id, "memories")
```

我们使用 store.put 方法将记忆保存到存储中的命名空间。当我们这样做时，我们会指定上述定义的命名空间，并为记忆指定一个键值对：键是记忆的唯一标识符（ memory_id ），值（一个字典）是记忆本身。
```python
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
store.put(namespace_for_memory, memory_id, memory)
```

我们可以使用 store.search 方法读取我们命名空间中的记忆，它将返回给定用户的所有记忆列表。最新的记忆是列表中的最后一个。
```python
memories = store.search(namespace_for_memory)
memories[-1].dict()
```

#### 2.1.3.2 语义搜索
除了简单的检索，存储库还支持语义搜索，允许您根据含义而非精确匹配来查找记忆。要启用此功能，请使用嵌入模型配置存储库
```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        'embed': init_embeddings('openai:text-embedding-3-small'),  # 嵌入模型
        'dims': 1536,  # 嵌入维度
        'fields': ['food_preference', '$']  # 索引字段
    }
)

memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3  # Return top 3 matches
)
```

你可以通过配置 fields 参数或在使用存储记忆时指定 index 参数来控制哪些部分的记忆被嵌入
```python
# Store with specific fields to embed
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "food_preference": "I love Italian cuisine",
        "context": "Discussing dinner plans"
    },
    index=["food_preference"]  # Only embed "food_preferences" field
)

# Store without embedding (still retrievable, but not searchable)
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False
)
```

### 2.1.4 检查点库

在底层，检查点功能由符合 BaseCheckpointSaver 接口的检查点对象提供。
LangGraph 提供了多个检查点实现，所有这些都通过独立的、可安装的库实现。

#### 2.1.4.1 检查点器接口

每个检查点器都遵循 BaseCheckpointSaver 接口，并实现以下方法
+ put: 存储带有其配置和元数据的检查点
+ put_writes: 存储与配置和元数据相关的写入(待处理写入)
+ get_tuple: 使用给定的配置(thread_id和checkpoint_id)获取一个检查点元组。用于在graph.get_state()中填充StateSnapshot
+ list: 列出符合给定配置和过滤标准的检查点。用于在graph.get_state_history()中填充状态历史记录

如果检查点器与异步图执行一起使用（即通过 .ainvoke 、 .astream 、 .abatch 执行图），将使用上述方法的异步版本（ .aput 、 .aput_writes 、 .aget_tuple 、 .alist ）。

#### 2.1.4.2 序列化器

当检查点保存图状态时，需要序列化状态中的通道值。这通过序列化对象来完成。

langgraph_checkpoint 定义了实现序列化器的协议，并提供了一个默认实现（ JsonPlusSerializer ），该实现处理多种类型，包括 LangChain 和 LangGraph 基本类型、日期时间、枚举等。

**使用pickle进行序列化**

默认序列化器 JsonPlusSerializer 在底层使用 ormsgpack 和 JSON，这并不适用于所有类型的对象。

如果你希望对于当前不被我们的 msgpack 编码器支持的对象（例如 Pandas 数据框）回退使用 pickle，你可以使用 JsonPlusSerializer 的 pickle_fallback 参数：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

graph.compile(
    checkpointer=InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
)
```

**加密**

Checkpointers 可以选择性地加密所有持久化状态。要启用此功能，将 EncryptedSerializer 的实例传递给任何 BaseCheckpointSaver 实现的 serde 参数。
创建加密序列化器的最简单方法是通过 from_pycryptodome_aes ，它从 LANGGRAPH_AES_KEY 环境变量中读取 AES 密钥（或接受 key 参数）

```python
import sqlite3
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

serde = EncryptedSerializer.from_pycryptodome_aes()  # 读取 LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect('checkpoints.db'), serde=serde)

checkpointer = PostgresSaver.from_conn_string('postgresql://user:password@host:port/dbname', serde=serde)
checkpointer.setup()

```

在 LangSmith 上运行时，只要存在 LANGGRAPH_AES_KEY ，加密就会自动启用，因此您只需提供环境变量。其他加密方案可以通过实现 CipherProtocol 并将其提供给 EncryptedSerializer 来使用。

## 2.2 Durable execution 持久执行

持久化执行是一种技术，其中进程或工作流在关键点保存其进度，允许它暂停并在稍后从它停止的地方继续。这在需要人工参与的场景中特别有用，用户可以在继续之前检查、验证或修改流程，
以及在可能遇到中断或错误的长任务中（例如，对 LLM 的调用超时）。通过保留已完成的工作，持久化执行使进程能够在不重新处理先前步骤的情况下恢复——即使在显著延迟后（例如，一周后）。

### 2.2.1 Requirements 要求

为了在 LangGraph 中利用持久化执行，你需要:
+ 通过指定一个将保存工作流进度的检查点（checkpointer）来在你的工作流中启用持久化。
+ 执行工作流时指定线程标识符。这将跟踪特定工作流实例的执行历史。
+ 将任何非确定性操作（例如，随机数生成）或具有副作用的操作（例如，文件写入、API 调用）包裹在任务中，以确保当工作流恢复时，这些操作不会针对特定运行重复执行，而是从持久化层检索其结果。

### 2.2.2 确定性和一致重播

当您恢复工作流运行时，代码不会从执行停止的同一行代码处恢复；相反，它将识别一个合适的起点，从中继续未完成的任务。这意味着工作流将从起点重播所有步骤，直到它到达停止点。

因此，在编写用于持久化执行的流程时，你必须将任何非确定性操作（例如随机数生成）和任何具有副作用的操作（例如文件写入、API 调用）包裹在任务或节点内部。

为确保你的流程是确定性的并且可以一致地重放，请遵循以下指南:
+ 避免重复工作：如果一个节点包含多个具有副作用的操作（例如日志记录、文件写入或网络调用），请将每个操作包裹在单独的任务中。这确保了当流程被恢复时，操作不会被重复执行，其结果将从持久化层中检索。
+ 封装非确定性操作：将任何可能产生非确定性结果的代码（例如随机数生成）包裹在任务或节点内部。这确保了在恢复时，流程将按照记录的确切步骤序列以相同的结果执行。
+ 使用幂等操作：在可能的情况下，确保副作用（例如 API 调用、文件写入）是幂等的。这意味着如果工作流中的操作在失败后重试，它将产生与第一次执行时相同的效果。这对于导致数据写入的操作尤为重要。
  如果任务开始但未能成功完成，工作流的恢复将重新运行该任务，依靠记录的结果来保持一致性。使用幂等键或验证现有结果以避免无意中的重复，确保工作流执行的顺畅和可预测性。

### 2.2.3 持久性模式

LangGraph 支持三种持久性模式，允许您根据应用程序的需求平衡性能和数据一致性。更高的持久性模式会增加工作流执行的额外开销。您可以在调用任何图执行方法时指定持久性模式:
```python
graph.stream(
    {'input': 'test'},
    durability='sync'
)
```
持久化模式，从最不持久到最持久:
+ "exit" : LangGraph 仅在图执行成功、出现错误或由于人为中断时才持久化更改。这为长时间运行的图提供了最佳性能，但意味着中间状态不会被保存，因此您无法从执行中途发生的系统故障（如进程崩溃）中恢复。
+ "async" : LangGraph 在执行下一步时异步持久化更改。这提供了良好的性能和持久性，但如果进程在执行过程中崩溃，存在 LangGraph 不会写入检查点的风险。
+ "sync" : LangGraph 在下一步开始前同步持久化变更。这确保了 LangGraph 在继续执行前会写入每个检查点，以性能开销为代价提供了高持久性。

### 2.2.4 节点中使用任务

如果一个节点包含多个操作，你可能发现将每个操作转换为任务比将操作重构为单独的节点更容易

**Original原始**
```python
import requests
from typing_extensions import TypedDict, NotRequired
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

class MyState(TypedDict):
    url: str
    result: NotRequired[str]

def call_api(state: MyState):
    """Example node that makes an API request."""
    result = requests.get(state['url']).text[:100]
    return {'result': result}

builder = StateGraph(MyState)
builder.add_node('call_api', call_api)
builder.add_edge(START, 'call_api')
builder.add_edge('call_api', END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}
graph.invoke({"url": "https://www.example.com"}, config)
```

**Using Tasks 使用任务**
```python
from langgraph.func import task

class MyState(TypedDict):
    urls: list[str]
    result: NotRequired[str]

@task
def _make_request(url: str):
    return requests.get(url).text[:100]

def call_api(state: MyState):
    """Example node that makes an API request."""
    requests_ = [_make_request(url) for url in state['urls']]
    results = [request_.result() for request_ in requests_]
    return {'results': results}

builder = StateGraph(MyState)
builder.add_node("call_api", call_api)

# Connect the start and end nodes to the call_api node
builder.add_edge(START, "call_api")
builder.add_edge("call_api", END)

# Specify a checkpointer
checkpointer = InMemorySaver()

# Compile the graph with the checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Define a config with a thread ID.
thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

graph.invoke({"urls": ["https://www.example.com"]}, config)
```

### 2.2.5 恢复工作流

一旦您在您的流程中启用了持久执行，您可以在以下场景中恢复执行:
+ 暂停和恢复流程：使用中断函数在特定点暂停流程，并使用 Command 原语使用更新后的状态恢复它。有关详细信息，请参阅中断。
+ 从故障中恢复：在异常后（例如 LLM 提供商中断）自动从最后一个成功的检查点恢复流程。这涉及通过将其作为输入值提供 None 来使用相同的线程标识符执行流程（请参阅此使用功能 API 的示例）。

### 2.2.6 恢复工作流起点

+ 如果你使用的是 StateGraph（图 API），起点是执行停止的节点的开始处。
+ 如果你在节点内进行子图调用，起点将是调用被停止的子图的父节点。在子图中，起点将是执行停止的具体节点。
+ 如果你使用的是 Functional API，起点是执行停止的入口点开始。

## 2.3 Streaming 流式传输

LangGraph 实现了一个流式系统来展示实时更新。流式传输对于增强基于 LLMs 构建的应用程序的响应性至关重要。通过逐步显示输出，即使完整的响应尚未准备好，
流式传输也能显著改善用户体验（UX），尤其是在处理 LLMs 的延迟时。

### 2.3.1 Get started 开始使用

#### 2.3.1.1 基本用法

LangGraph 图表暴露了 stream （同步）和 astream （异步）方法来作为迭代器生成流式输出。传递一个或多个流式模式来控制您接收的数据。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Status: {chunk['data']['status']}")
```

#### 2.3.1.2 流输出格式

将 version="v2" 传递给 stream() 或 astream() 以获取统一的输出格式。每个块都是一个 StreamPart 字典，具有一致的形状——无论流模式、模式数量或子图设置如何

```bash
{
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),           # namespace tuple, populated for subgraph events
    "data": ...,        # the actual payload (type varies by stream mode)
}
```

每种流模式都有一个对应的 TypedDict ，包含 ValuesStreamPart 、 UpdatesStreamPart 、 MessagesStreamPart 、 CustomStreamPart 、 CheckpointStreamPart 、 TasksStreamPart 、 DebugStreamPart 。
可以从 langgraph.types 中导入这些类型。联合类型 StreamPart 在 part["type"] 上是一个不相交的联合，使编辑器和类型检查器能够实现完全的类型缩小。

v2 格式还支持类型缩小，这意味着您可以通过 chunk["type"] 过滤数据块并获取正确的有效载荷类型。每个分支都会将 part["data"] 缩小为该模式的具体类型

```python
for part in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["values", "updates", "messages", "custom"],
    version="v2",
):
    if part["type"] == "values":
        # ValuesStreamPart — full state snapshot after each step
        print(f"State: topic={part['data']['topic']}")
    elif part["type"] == "updates":
        # UpdatesStreamPart — only the changed keys from each node
        for node_name, state in part["data"].items():
            print(f"Node `{node_name}` updated: {state}")
    elif part["type"] == "messages":
        # MessagesStreamPart — (message_chunk, metadata) from LLM calls
        msg, metadata = part["data"]
        print(msg.content, end="", flush=True)
    elif part["type"] == "custom":
        # CustomStreamPart — arbitrary data from get_stream_writer()
        print(f"Progress: {part['data']['progress']}%")
```

### 2.3.2 Stream modes 流模式

将以下一个或多个流模式作为列表传递给 stream 或 astream 方法:
+ values: 获取每个步骤的完整状态快照。
+ updates: 获取每个步骤的更新状态
+ messages: 获取每个步骤的 LLM 输出
+ custom: 来自节点通过get_stream_writer发出的自定义数据
+ checkpoints: 检查点事件(与get_state())相同的格式，需要一个检查点器
+ tasks: 任务开始/结束事件，包含结果和错误。需要检查点器
+ debug: 所有可用信息-结合checkpoints和tasks并附加额外元数据

#### 2.3.2.1 图状态

使用 updates 和 values 流模式来流式传输图在执行过程中的状态。
+ updates 在图的每一步之后流式传输状态的更新。
+ values 在图的每一步之后流式传输状态的全部值。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class MyState(TypedDict):
    topic: str
    joke: str

def refine_topic(state: MyState):
    return {'topic': state['topic'] + ' and cats'}

def generate_joke(state: MyState):
    return {'joke': f"This is a joke about {state['topic']}"}

graph = (
    StateGraph(MyState)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, 'refine_topic')
    .add_edge('refine_topic', 'generate_joke')
    .add_edge('generate_joke', END)
    .compile()
)

for chunk in graph.stream(
        {'topic': 'ice cream'},
        stream_mode=["updates", "values"],
        version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
    elif chunk["type"] == "values":
        print(f"State: topic={chunk['data']['topic']}, joke={chunk['data']['joke']}")
```

#### 2.3.2.2 LLM tokens

使用 messages 流式模式可以从你的图中的任何部分（包括节点、工具、子图或任务）逐个流式传输大型语言模型（LLM）的输出。

messages 模式流式传输的输出是一个 (message_chunk, metadata) 元组，其中:
+ message_chunk:  来自 LLM 的标记或消息片段。
+ metadata : 一个包含有关图节点和 LLM 调用详情的字典。

可以将 tags 与 LLM 调用关联起来，通过 LLM 调用过滤流式传输的标记。

**tags=\['nostream'\]**
使用 nostream 标签完全从流中排除 LLM 输出。带有 nostream 标签的调用仍然运行并产生输出；它们的 token 仅在 messages 模式下不被发出。

仅从特定节点流式传输标记，使用 stream_mode="messages" 并按流式传输元数据中的 langgraph_node 字段过滤输出

```python
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START


@dataclass
class MyState:
    topic: str
    joke: str = ""


model = init_chat_model(model="gpt-5.4-mini", tags=['joke']) # or poem

def call_model(state: MyState):
    """Call the LLM to generate a joke about a topic"""
    # Note that message events are emitted even when the LLM is run using .invoke rather than .stream
    model_response = model.invoke(
        [
            {"role": "user", "content": f"Generate a joke about {state.topic}"}
        ]
    )
    return {"joke": model_response.content}

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

# The "messages" stream mode streams LLM tokens with metadata
# Use version="v2" for a unified StreamPart format
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        if message_chunk.content and metadata["langgraph_node"] == "node_name":
            print(message_chunk.content, end="|", flush=True)
        if metadata["tags"] == ["joke"]:
            print(message_chunk.content, end="|", flush=True)
        if message_chunk.content:
            print(message_chunk.content, end="|", flush=True)
```

#### 2.3.2.3 自定义数据

要在 LangGraph 节点或工具内部发送自定义用户定义数据，请按照以下步骤操作:
+ 使用 get_stream_writer 访问流写入器并发出自定义数据。
+ 调用 .stream() 或 .astream() 时设置 stream_mode="custom" 以获取流中的自定义数据。您可以组合多种模式（例如 ["updates", "custom"] ），但至少必须有一个是 "custom" 。

```python
from typing import TypedDict
from langchain.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    # Get the stream writer to send custom data
    writer = get_stream_writer()
    # Emit a custom key-value pair (e.g., progress update)
    writer({"custom_key": "Generating custom data inside node"})
    return {"answer": "some data"}

@tool
def query_database(query: str) -> str:
    """Query the database."""
    # Access the stream writer to send custom data
    writer = get_stream_writer()
    # Emit a custom key-value pair (e.g., progress update)
    writer({"data": "Retrieved 0/100 records", "type": "progress"})
    # perform query
    # Emit another custom key-value pair
    writer({"data": "Retrieved 100/100 records", "type": "progress"})
    return "some-answer"

graph = (
    StateGraph(State)
    .add_node(node)
    .add_edge(START, "node")
    .compile()
)

inputs = {"query": "example"}

# Set stream_mode="custom" to receive the custom data in the stream
for chunk in graph.stream(inputs, stream_mode="custom", version="v2"):
    if chunk["type"] == "custom":
        print(f"Custom event: {chunk['data']['custom_key']}")

```

#### 2.3.2.4 子图输出

要在流式输出中包含子图的输出，您可以在父图的 .stream() 方法中设置 subgraphs=True 。这将同时流式传输父图和任何子图的输出。

输出将作为 (namespace, data) 元组流式传输，其中 namespace 是一个包含子图调用节点的路径的元组，例如 ("parent_node:<task_id>", "child_node:<task_id>") 。

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode="updates",
    version="v2",
):
    print(chunk["type"])  # "updates"
    print(chunk["ns"])    # () for root, ("node_name:<task_id>",) for subgraph
    print(chunk["data"])  # {"node_name": {"key": "value"}}
```

#### 2.3.2.5 检查点

使用 checkpoints 流式模式在图执行时接收检查点事件。每个检查点事件与 get_state() 的输出格式相同。需要检查点器。

```python
from langgraph.checkpoint.memory import MemorySaver

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile(checkpointer=MemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {"topic": "ice cream"},
    config=config,
    stream_mode="checkpoints",
    version="v2",
):
    if chunk["type"] == "checkpoints":
        print(chunk["data"])
```

#### 2.3.2.6 任务

使用 tasks 流式模式来接收任务开始和结束事件，当图执行时。任务事件包括关于正在运行的节点、其结果以及任何错误的信息。需要检查点器。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    config=config,
    stream_mode="tasks",
    version="v2",
):
    if chunk["type"] == "tasks":
        print(chunk["data"])
```

#### 2.3.2.7 调试Debug

使用 debug 流式模式，在整个图执行过程中尽可能多地流式传输信息。流式传输的输出包括节点名称以及完整状态。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="debug",
    version="v2",
):
    if chunk["type"] == "debug":
        print(chunk["data"])
```

#### 2.3.2.8 同时使用多种模式

使用 version="v2" 时，每个块都是一个 StreamPart 字典。使用 chunk["type"] 来区分不同的模式：
```python
for chunk in graph.stream(inputs, stream_mode=["updates", "custom"], version="v2"):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node `{node_name}` updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Custom event: {chunk['data']}")
```

### 2.3.3 Advanced 高级

#### 2.3.3.1 与任何LLM使用

可以使用 stream_mode="custom" 从任何 LLM API 流式传输数据——即使该 API 没有实现 LangChain 聊天模型接口。

```python
from langgraph.config import get_stream_writer

def call_arbitrary_model(state):
    """Example node that calls an arbitrary model and streams the output"""
    # Get the stream writer to send custom data
    writer = get_stream_writer()
    # Assume you have a streaming client that yields chunks
    # Generate LLM tokens using your custom streaming client
    for chunk in your_custom_streaming_client(state["topic"]):
        # Use the writer to send custom data to the stream
        writer({"custom_llm_chunk": chunk})
    return {"result": "completed"}

graph = (
    StateGraph(State)
    .add_node(call_arbitrary_model)
    # Add other nodes and edges as needed
    .compile()
)
# Set stream_mode="custom" to receive the custom data in the stream
for chunk in graph.stream(
    {"topic": "cats"},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        # The chunk data will contain the custom data streamed from the llm
        print(chunk["data"])
```

#### 2.3.3.2 特定模型禁用流式输出

如果你的应用程序混合了支持流式传输的模型和不支持的模型，你可能需要显式禁用不支持的模型的流式传输。

在初始化模型时设置 streaming=False 。

```python

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-6",
    # Set streaming=False to disable streaming for the chat model
    streaming=False
)

from langchain_openai import ChatOpenAI

# Set streaming=False to disable streaming for the chat model
model = ChatOpenAI(model="o1-preview", streaming=False)

```

## 2.4 Interrupts 中断

中断允许你在特定点暂停图执行，并在继续之前等待外部输入。这使你能够实现需要外部输入才能继续的人机交互模式。当触发中断时，LangGraph 会使用其持久化层保存图状态，并无限期等待你恢复执行。

中断通过在任何图节点中调用 interrupt() 函数来工作。该函数接受任何 JSON 可序列化值，并将其展示给调用者。当你准备好继续时，通过重新调用图使用 Command 来恢复执行，这将成为节点内部 interrupt() 调用的返回值。

与静态断点（在特定节点之前或之后暂停）不同，中断是动态的：它们可以放置在代码的任何位置，并且可以根据你的应用逻辑进行条件性设置。

+ 检查点会保持你的进度：检查点会写入确切的图状态，以便你稍后可以恢复，即使处于错误状态。
+ thread_id 是你的指针：设置 config={"configurable": {"thread_id": ...}} 来告诉检查点器加载哪个状态。
+ 中断负载通过 chunk["interrupts"] 暴露出来：当你使用 version="v2" 进行流式传输时，传递给 interrupt() 的值会出现在 values 流区块的 interrupts 字段中，这样你就能知道图正在等待什么。

### 2.4.1 使用interrupt暂停

interrupt 函数会暂停图执行并返回一个值给调用者。当你在节点中调用 interrupt 时，LangGraph 会保存当前的图状态并等待你输入以继续执行。

要使用 interrupt ，你需要:
+ 一个检查点来持久化图状态（在生产环境中使用一个持久化的检查点）
+ 在配置中的一个线程 ID，以便运行时知道从哪个状态继续
+ 要在需要暂停的地方调用 interrupt() （payload 必须为 JSON 序列化格式）

```python
from langgraph.types import interrupt

def approval_node(state: State):
    # Pause and ask for approval
    approved = interrupt("Do you approve this action?")

    # When you resume, Command(resume=...) returns that value here
    return {"approved": approved}
```

### 2.4.2 恢复中断

中断执行后，您通过调用带有恢复值的 Command 来重新启动图。恢复值会传递回 interrupt 调用，使节点能够使用外部输入继续执行。
```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"input": "data"}, config=config, version="v2")

print(result.interrupts)

graph.invoke(Command(resume=True), config=config, version="v2")
```

关于恢复的关键点:
+ 恢复时必须使用与中断发生时相同的线程 ID
+ 传递给 Command(resume=...) 的值成为 interrupt 调用的返回值
+ 节点在恢复时将从调用 interrupt 的节点开始重新启动，因此 interrupt 之前的任何代码(节点内部)都会再次运行
+ 你可以将任何 JSON 序列化值作为恢复值传递

Command(resume=...) 是唯一一个作为 invoke() / stream() 输入的 Command 模式。其他 Command 参数（ update ， goto ， graph ）是为从节点函数返回而设计的。
不要将 Command(update=...) 作为输入来继续多轮对话——而是传递一个普通的输入字典。

Command(resume=...) 是给图的外部调用者在中中断后使用的；Command(update=...) / Command(goto=...) 是给图的内部节点做流程控制用的。
多轮对话继续执行时，直接传普通字典即可，千万不要把内部控制指令当外部输入传进去。

### 2.4.3 常见模式

中断解锁的关键在于能够暂停执行并等待外部输入。这对于多种用例非常有用，包括:
+ 审批工作流：在执行关键操作（API 调用、数据库更改、金融交易）前暂停
+ 处理多个中断：在单个调用中恢复多个中断时，将中断 ID 与恢复值配对
+ 审阅和编辑：在继续之前，让人类审阅和修改 LLM 输出或工具调用
+ 中断工具调用：在执行工具调用之前暂停，以审阅和编辑工具调用
+ 验证人类输入：在进行下一步之前暂停，以验证人类输入

#### 2.4.3.1 带有人工介入的中断流

在构建具有人工介入工作流程的交互式代理时，你可以同时流式传输消息块和节点更新，以便在处理中断的同时提供实时反馈。

使用多种流模式（ "messages" 和 "updates" ）与 subgraphs=True （如果存在子图）来实现:
+ 实时流式传输 AI 响应，随着它们的生成而生成
+ 检测图遇到中断时
+ 处理用户输入并无缝继续执行

```python
async for chunk in graph.astream(
    initial_input,
    stream_mode=['messages', 'updates'],
    subgraphs=True,
    config=config,
    version="v2"
):
    if chunk["type"] == "messages":
        msg, _ = chunk['data']
        if isinstance(msg, AIMessageChunk) and msg.content:
            display_streaming_content(msg.content)
        elif chunk["type"] == "updates":
        # Check for interrupts in the updates data
            if "__interrupt__" in chunk["data"]:
                interrupt_info = chunk["data"]["__interrupt__"][0].value
                user_response = get_user_input(interrupt_info)
                initial_input = Command(resume=user_response)
                break
            else:
                current_node = list(chunk["data"].keys())[0]
```

#### 2.4.3.2 处理多个中断

当多个并行分支同时中断时（例如，分支到多个节点，每个节点都调用 interrupt() ），您可能需要在单个调用中恢复多个中断。在单个调用中恢复多个中断时，
将每个中断 ID 映射到其恢复值。这确保了每个响应在运行时与正确的中断配对。

```python
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt

class MyState(TypedDict):
    vals: Annotated[list[str], operator.add]

def node_a(state: MyState):
    answer = interrupt('question_a')
    return {'vals': [f'a:{answer}']}

def node_b(state: MyState):
    answer = interrupt('question_b')
    return {'vals': [f'b:{answer}']}

graph = (
    StateGraph(MyState)
    .add_node('a', node_a)
    .add_node('b', node_b)
    .add_edge(START, 'a')
    .add_edge(START, 'b')
    .add_edge('a', END)
    .add_edge('b', END)
    .compile(checkpointer=InMemorySaver())
)
config = {"configurable": {"thread_id": "1"}}

interrupted_result = graph.invoke({"vals": []}, config)
print(interrupted_result)
"""
{
    'vals': [],
    '__interrupt__': [
        Interrupt(value='question_a', id='bd4f3183600f2c41dddafbf8f0f7be7b'),
        Interrupt(value='question_b', id='29963e3d3585f0cef025dd0f14323f55')
    ]
}
"""
resume_map = {
    i.id: f"answer for {i.value}"
    for i in interrupted_result["__interrupt__"]
}

result = graph.invoke(Command(resume=resume_map), config)

print("Final state:", result)
#> Final state: {'vals': ['a:answer for question_a', 'b:answer for question_b']}

```

#### 2.4.3.3 批准或拒绝

中断最常用的用途之一是在执行关键操作前暂停并请求批准。例如，你可能需要请求人工批准 API 调用、数据库更改或其他重要决策。

```python
from typing import Literal
from langgraph.types import interrupt, Command

def approval_node(state: State) -> Command[Literal["proceed", "cancel"]]:
    # Pause execution; payload shows up under result["__interrupt__"]
    is_approved = interrupt({
        "question": "Do you want to proceed with this action?",
        "details": state["action_details"]
    })

    # Route based on the response
    if is_approved:
        return Command(goto="proceed")  # Runs after the resume payload is provided
    else:
        return Command(goto="cancel")
```

#### 2.4.3.4 审核和编辑状态

有时您希望在继续之前让人类审查和编辑图状态的一部分。这对于纠正 LLMs、添加缺失信息或进行调整非常有用。

```python
from langgraph.types import interrupt

def review_node(state: State):
    # Pause and show the current content for review (surfaces in result["__interrupt__"])
    edited_content = interrupt({
        "instruction": "Review and edit this content",
        "content": state["generated_text"]
    })

    # Update the state with the edited version
    return {"generated_text": edited_content}

# 恢复时，请提供编辑后的内容
graph.invoke(
    Command(resume="The edited and improved text"),  # Value becomes the return from interrupt()
    config=config
)
```

#### 2.4.3.5 工具中的中断

也可以直接在工具函数内部放置中断。这会使工具在每次被调用时暂停以等待批准，并允许在执行之前对工具调用进行人工审查和编辑。

```python
import sqlite3

from typing import TypedDict
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt


class AgentState(TypedDict):
    messages: list[dict]

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a given recipient."""
    response = interrupt({
        'action': 'send_email',
        'to': to,
        'subject': subject,
        'body': body,
        'messages': 'Approve sending this email?'
    })
    
    if response.get('action') == 'approve':
        final_to = response.get('to', to)
        final_subject = response.get('subject', subject)
        final_body = response.get('body', body)
        print(f"[send_email] to={final_to} subject={final_subject} body={final_body}")
        return f"Email sent to {final_to}"
    return 'Email cancelled by user'


model = ChatOpenAI(model='gpt-4.1').bind_tools([send_email])

def agent_node(state: AgentState):
    result = model.invoke(state['messages'])
    return {'messages': state['messages'] + [result]}

builder = StateGraph(AgentState)
builder.add_node('agent', agent_node)
builder.add_edge(START, 'agent')
builder.add_edge('agent', END)

checkpointer = SqliteSaver(sqlite3.connect('tool-approval.db'))
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "email-workflow"}}
initial_ = graph.invoke(
    {
        'messages': [
            {'role': 'user', 'content': 'Send an email to <EMAIL> about the project status.'}
        ]
    },
    config=config
)

print(initial_["__interrupt__"]) 

resumed = graph.invoke(
    # resume的值会传递给response
    Command(resume={'action': 'approve', 'subject': 'Updated subject'}),
    config=config
)
print(resumed["messages"][-1])

```

#### 2.4.3.6 验证人类输入

有时你需要验证来自人类的输入，并在输入无效时再次询问。你可以使用循环中的多个 interrupt 调用来实现这一点。

```python
def get_age_node(state: FormState):
    prompt = "What is your age?"

    while True:
        answer = interrupt(prompt)  # payload surfaces in result["__interrupt__"]

        if isinstance(answer, int) and answer > 0:
            return {"age": answer}

        prompt = f"'{answer}' is not a valid age. Please enter a positive number."
```

### 2.4.4 中断规则

当你在节点中调用 interrupt 时，LangGraph 会通过抛出异常来暂停执行，这个异常会向上传递到调用栈，并由运行时捕获。运行时会通知图保存当前状态并等待外部输入。

当执行恢复时（在你提供所需输入之后），运行时会从头开始重新启动整个节点——它不会从 interrupt 被调用时的确切行继续执行。
这意味着在 interrupt 之前运行的任何代码都会再次执行。由于这个原因，在使用中断时需要遵循一些重要的规则，以确保它们按预期行为。

#### 2.4.4.1 不要将中断包裹在try/except中

interrupt 在调用点暂停执行的方式是通过抛出一个特殊异常。如果你将 interrupt 调用包裹在 try/except 块中，你将捕获这个异常，中断将不会被传递回图。

**逻辑分离**

```python
def node_a(state: State):
    # ✅ Good: interrupting first, then handling
    # error conditions separately
    interrupt("What's your name?")
    try:
        fetch_data()  # This can fail
    except Exception as e:
        print(e)
    return state
```

#### 2.4.4.2 不要在节点内重新排序中断调用

在单个节点中使用多个中断是很常见的，但如果处理不当，可能会导致意外行为。

当一个节点包含多个中断调用时，LangGraph 会为执行该节点的任务维护一个特定的恢复值列表。每当执行恢复时，它会从节点的开头开始。对于遇到的中断，
LangGraph 会检查任务恢复列表中是否存在匹配的值。匹配是严格基于索引的，因此节点内中断调用的顺序非常重要。

+ 不要使用非确定性跨执行逻辑来循环 interrupt 调用
+ 在节点内不要有条件地跳过 interrupt 调用

```python
def node_a(state: State):
    # ✅ Good: interrupt calls happen in the same order every time
    name = interrupt("What's your name?")
    age = interrupt("What's your age?")
    city = interrupt("What's your city?")

    return {
        "name": name,
        "age": age,
        "city": city
    }
```

#### 2.4.4.3 不要在中断调用中返回复杂值

根据所使用的检查点器，复杂值可能无法序列化（例如，你不能序列化一个函数）。为了让你的图适应任何部署，最佳实践是只使用可以合理序列化的值。

+ 将简单的、JSON 可序列化类型传递给 interrupt
+ 传递具有简单值的字典/对象
+ 不要将函数、类实例或其他复杂对象传递给 interrupt

```python
def node_a(state: State):
    # ✅ Good: passing simple types that are serializable
    name = interrupt("What's your name?")
    count = interrupt(42)
    approved = interrupt(True)

    return {"name": name, "count": count, "approved": approved}
```

#### 2.4.4.4 中断调用前的副作用必须是幂等的

由于中断是通过重新运行它们被调用的节点来工作的，所以在 interrupt 之前被调用的副作用（side effects）应该（理想情况下）是幂等的。
为了理解，幂等性意味着相同的操作可以多次应用，而不会在初始执行之外改变结果。

例如，您可能有一个 API 调用用于在节点内部更新记录。如果在调用该 API 后调用 interrupt ，当节点恢复时，它可能会多次重新运行，可能会覆盖初始更新或创建重复记录。

+ 在 interrupt 之前使用幂等操作
+ 在 interrupt 调用之后放置副作用
+ 在可能的情况下将副作用分离到单独的节点中
+ 在 interrupt 之前不要执行非幂等操作
+ 在创建新记录前请先检查是否已存在

```python
def node_a(state: State):
    # ✅ Good: using upsert operation which is idempotent
    # Running this multiple times will have the same result
    db.upsert_user(
        user_id=state["user_id"],
        status="pending_approval"
    )

    approved = interrupt("Approve this change?")

    return {"approved": approved}
```

### 2.4.5 与作为函数调用的子图一起使用

在一个节点内调用子图时，父图将从子图被调用的节点中 interrupt 触发的位置开始继续执行。同样，子图也将从 interrupt 被调用的节点开始继续执行。

```python
def node_in_parent_graph(state: State):
    some_code()  # <-- This will re-execute when resumed
    # Invoke a subgraph as a function.
    # The subgraph contains an `interrupt` call.
    subgraph_result = subgraph.invoke(some_input)
    # ...

def node_in_subgraph(state: State):
    some_other_code()  # <-- This will also re-execute when resumed
    result = interrupt("What's your name?")
    # ...
```

### 2.4.6 使用中断进行调试

要调试和测试一个图，你可以使用静态断点作为断点来逐个节点地执行图。静态断点在节点执行之前或之后定义的点被触发。你可以在编译图时通过指定 interrupt_before 和 interrupt_after 来设置这些断点。

**在编译时**
+ 断点是在 compile 时间设置的。
+ interrupt_before 指定执行应在节点执行前暂停的节点。
+ interrupt_after 指定节点执行后应暂停的节点。
+ 需要设置检查点才能启用断点。
+ 图运行到第一个断点时停止。
+ 通过输入 None 来恢复图执行。这将运行图直到下一个断点。
```python
graph = builder.compile(
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"],
    checkpointer=checkpointer,
)

# Pass a thread ID to the graph
config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}

# Run the graph until the breakpoint
graph.invoke(inputs, config=config)

# Resume the graph
graph.invoke(None, config=config)
```

**在运行时**
+ 同上编译时
```python
config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}

# Run the graph until the breakpoint
graph.invoke(
    inputs,
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"],
    config=config,
)

# Resume the graph
graph.invoke(None, config=config)
```

## 2.5 Time Travel 时间旅行

### 2.5.1 概述

LangGraph 通过检查点支持时间旅行:
+ 重播: 从先前的检查点重新尝试。
+ 分支: 从先前的检查点分支，使用修改后的状态探索替代路径。

两者都通过从先前的检查点恢复来工作。检查点之前的节点不会重新执行（结果已经保存）。检查点之后的节点会重新执行，包括任何 LLM 调用、API 请求和中断（这可能产生不同的结果）。

### 2.5.2 Replay 重播

使用先前的检查点的配置来调用图，以便从该点重新播放

重新播放会重新执行节点——它不仅仅是读取缓存。LLM 调用、API 请求和中断会再次触发，并可能返回不同的结果。从最终检查点（没有 next 节点）重新播放是无操作的。

使用 get_state_history 找到你想重新播放的检查点，然后用该检查点的配置调用 invoke:
```python
from typing_extensions import TypedDict, NotRequired

from langchain_core.utils.uuid import uuid7
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class MyState(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


def generate_topic(state: MyState):
    return {'topic': 'socks in the dryer'}

def write_joke(state: MyState):
    return {'joke': f'Why do {state["topic"]} disappear? They elope!'}

checkpointer = InMemorySaver()
graph = (
    StateGraph(MyState)
    .add_node('generate_topic', generate_topic)
    .add_node('write_joke', write_joke)
    .add_edge(START, 'generate_topic')
    .add_edge('generate_topic', 'write_joke')
    .compile(checkpointer=checkpointer)
)

config = {'configurable': {'thread_id': str(uuid7())}}

result = graph.invoke({}, config=config)

history = list(graph.get_state_history(config=config))
for state in history:
    print(f"next={state.next}, checkpoint_id={state.config['configurable']['checkpoint_id']}")

before_joke = next(s for s in history if s.next == ("write_joke",))
replay_result = graph.invoke(None, before_joke.config)
```

### 2.5.3 Fork 分支

分支是从过去的检查点创建一个新的分支，带有修改的状态。在先前的检查点上调用 update_state 来创建分支，然后使用 invoke 和 None 来继续执行。

```python
# Find checkpoint before write_joke
history = list(graph.get_state_history(config))
before_joke = next(s for s in history if s.next == ("write_joke",))

# Fork: update state to change the topic
fork_config = graph.update_state(
    before_joke.config,
    values={"topic": "chickens"},
)

# Resume from the fork — write_joke re-executes with the new topic
fork_result = graph.invoke(None, fork_config)
print(fork_result["joke"])  # A joke about chickens, not socks
```

#### 2.5.3.1 从特定节点开始

当你调用 update_state 时，值会使用指定节点的写作者（包括归约器）来应用。检查点记录该节点已生成更新，然后从该节点的后继节点继续执行。

默认情况下，LangGraph 从检查点的版本历史中推断 as_node 。从特定检查点分叉时，这种推断几乎总是正确的。

在以下情况下显式指定 as_node:
+ 并行分支：多个节点在同一步骤中更新了状态，而 LangGraph 无法确定哪个是最后一个（ InvalidUpdateError ）。
+ 没有执行历史：在全新线程上设置状态（测试中常见）。
+ 跳过节点：将 as_node 设置为后续节点，让图认为该节点已运行。

```python
# graph: generate_topic -> write_joke

# Treat this update as if generate_topic produced it.
# Execution resumes at write_joke (the successor of generate_topic).
fork_config = graph.update_state(
    before_joke.config,
    values={"topic": "chickens"},
    as_node="generate_topic",
)
```

### 2.5.4 Interrupts 中断

如果您的图使用 interrupt 进行人机交互工作流，在时间旅行期间中断总是会被重新触发。包含中断的节点重新执行， interrupt() 暂停等待新的 Command(resume=...) 。

```python
from typing_extensions import TypedDict
from langgraph.types import Command, interrupt, Interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph

class MyState(TypedDict):
    value: list[str]

def ask_human(state: MyState):
    answer = interrupt("What is your name?")
    return {"value": [f"Hello, {answer}!"]}

def final_step(state: MyState):
    return {"value": ["Done"]}

graph = (
    StateGraph(MyState)
    .add_node("ask_human", ask_human)
    .add_node("final_step", final_step)
    .add_edge(START, "ask_human")
    .add_edge("ask_human", "final_step")
    .add_edge("final_step", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

graph. invoke({"value": []}, config=config)
graph.invoke(Command(resume='Alice'), config=config)

history = list(graph.get_state_history(config))
before_ask = [s for s in history if s.next == ("ask_human",)][-1]

replay_result = graph.invoke(None, before_ask.config)
fork_config = graph.update_state(before_ask.config, {"value": ["forked"]})
fork_result = graph.invoke(None, fork_config)

graph.invoke(Command(resume='Bob'), fork_config)

```

#### 2.5.4.1 多个中断

如果你的图在多个点收集输入（例如，多步骤表单），你可以在中断之间分叉来更改后续答案，而无需重新询问早期问题。

```python
def ask_name(state):
    name = interrupt("What is your name?")
    return {"value": [f"name:{name}"]}

def ask_age(state):
    age = interrupt("How old are you?")
    return {"value": [f"age:{age}"]}

# Graph: ask_name -> ask_age -> final

history = list(graph.get_state_history(config))
between = [s for s in history if s.next == ("ask_age", )][-1]
fork_config = graph.update_state(between.config, {"value": ["modified"]})
result = graph.invoke(None, fork_config)

```

### 2.5.5 Subgraphs 子图

子图的时旅行取决于子图是否拥有自己的检查点器。这决定了你可以从哪个粒度进行时旅行。

```python

subgraph = (
    StateGraph(MyState)
    .add_node("step_a", step_a)
    .add_node("step_b", step_b)
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .compile()
)

graph = (
    StateGraph(MyState)
    .add_node("subgraph", subgraph)
    .add_edge(START, "subgraph")
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"value": []}, config)  # Hits step_a interrupt
graph.invoke(Command(resume='Alice'), config)  # Hits step_b interrupt
graph.invoke(Command(resume="30"), config) # Completes
history = list(graph.get_state_history(config))
before_sub = [s for s in history if s.next == ("subgraph_node",)][-1]

fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
result = graph.invoke(None, fork_config)

```

## 2.6 Memory 记忆

AI 应用需要内存来在多个交互中共享上下文。在 LangGraph 中，你可以添加两种类型的内存:
+ 将短期记忆作为你代理状态的一部分，以启用多轮对话。
+ 添加长期记忆来存储跨会话的用户特定或应用级数据。

### 2.6.1 添加短期记忆

短期记忆（线程级持久化）使代理能够跟踪多轮对话。要添加短期记忆：
```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

checkpoint = InMemorySaver()

builder = StateGraph(...)
graph = builder.compile(checkpointer=checkpoint)
config = {"configurable": {"thread_id": "1"}}
graph.invoke(
    {"messages": [{"role": "user", "content": "hello"}]},
    config=config
)
```

#### 2.6.1.1 用于生产环境

在生产环境中，使用一个由数据库支持的检查点器

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    builder = StateGraph(...)
    graph = builder.compile(checkpointer=checkpointer)
```

#### 2.6.1.2 在子图中使用

如果你的图包含子图，你只需要在编译父图时提供检查点器。LangGraph 将自动将检查点器传播到子子图中。

```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

class State(TypedDict):
    foo: str

# Subgraph

def subgraph_node_1(state: State):
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph

builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 2.6.2 添加长期记忆

使用长期记忆来存储跨对话的用户特定或应用特定的数据。

```python
from langgraph.store.memory import InMemoryStore  
from langgraph.graph import StateGraph

store = InMemoryStore()

builder = StateGraph(...)
graph = builder.compile(store=store)
```

#### 2.6.2.1 在节点中访问存储

一旦你编译了一个带有存储的图，LangGraph 会自动将存储注入到你的节点函数中。访问存储的推荐方式是通过 Runtime 对象。

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_core.utils.uuid import uuid7

@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    
    memories = await runtime.store.asearch(
        nmaespace, query=state["messages"][-1].content, limit=3
    )
    
    info = "\n".join([d.value["data"] for d in memories])
    
    await  runtime.store.aput(
        namespace, str(uuid7()), {"data": "User prefers dark mode"}
    )

buidler = StateGraph(MessagesState, context_schema=Context)
buidler.add_node(call_model)
buidler.add_edge(START, "call_model")
graph = buidler.compile(store=store)

graph.invoke(
    {"messages": [{"role": "user", "content": "hello"}]},
    config={"configurable": {"thread_id": "1"}},
    context=Context(user_id="1")
)

```

#### 2.6.2.2 用于生产环境

在生产环境中，使用一个由数据库支持的存储:

```python
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresStore.from_conn_string(DB_URI) as store:
    builder = StateGraph(...)
    graph = builder.compile(store=store)
```

#### 2.6.2.3 使用语义搜索

在您的图记忆存储中启用语义搜索，以使图代理能够通过语义相似性搜索存储中的项目。

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore


embeddings = init_embeddings("openai:text-embedding-3-small")

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536
    }
)

store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})
items = store.search(
    ("user_123", "memories"), query="I'm hungry", limit=1
)

```

### 2.6.3 管理短期记忆

启用短期记忆后，长对话可能会超出 LLM 的上下文窗口。常见解决方案有:
+ 修剪消息：在调用 LLM 之前移除前 N 条或后 N 条消息
+ 永久删除 LangGraph 状态中的消息
+ 总结消息：总结历史记录中的早期消息，并用摘要替换它们
+ 管理检查点以存储和检索消息历史
+ 自定义策略（例如，消息过滤等）

#### 2.6.3.1 修剪消息

大多数 LLMs 都有一个最大支持上下文窗口（以 token 为单位）。决定何时截断消息的一种方法是计算消息历史中的 token 数量，并在接近该限制时截断。如果你使用 LangChain，可以使用 trim messages 工具，并指定要保留的 token 数量，以及使用 strategy （例如，保留最后一个 max_tokens ）来处理边界。

```python
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately  
)

def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )
    response = model.invoke(messages)
    return {"messages": [response]}
```

#### 2.6.3.2 删除消息

可以从图状态中删除消息以管理消息历史

要从图状态中删除消息，您可以使用 RemoveMessage 。要使 RemoveMessage 正常工作，您需要使用带有 add_messages 简化器的状态键，例如 MessagesState

```python
from langchain.messages import RemoveMessage  

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

# 删除所有消息
from langgraph.graph.message import REMOVE_ALL_MESSAGES  

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

```

#### 2.6.3.3 总结消息

修剪或删除消息的问题在于，你可能会因筛选消息队列而丢失信息。由于这一点，一些应用程序受益于使用聊天模型来更复杂地总结消息历史的方法。

可以使用提示和编排逻辑来总结消息历史。例如，在 LangGraph 中，你可以扩展 MessagesState 以包含一个 summary 键:

```python
from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str
```

#### 2.6.3.4 管理检查点

可以查看和删除检查点存储的信息

**查看线程状态**

```python
config = {
    "configurable": {
        "thread_id": "1",
        # optionally provide an ID for a specific checkpoint,
        # otherwise the latest checkpoint is shown
        # "checkpoint_id": "1f029ca3-1f5b-6704-8004-820c16b69a5a"  #

    }
}
graph.get_state(config)

checkpointer.get_tuple(config)

```

**查看线程历史**

```python
config = {
    "configurable": {
        "thread_id": "1"
    }
}
list(graph.get_state_history(config))

list(checkpointer.list(config))
```

**删除线程的所有检查点**

```python
thread_id = "1"
checkpointer.delete_thread(thread_id)
```

#### 2.6.3.5 数据库管理

如果你使用任何基于数据库的持久化实现（如 Postgres 或 Redis）来存储短期和/或长期记忆，你需要在将其用于数据库之前运行迁移来设置所需的模式。

按照惯例，大多数特定于数据库的库会在检查点或存储实例上定义一个 setup() 方法来运行所需的迁移。然而，你应该检查你使用的 BaseCheckpointSaver 或 BaseStore 的具体实现，以确认确切的方法名称和使用方式。

## 2.7 Subgraphs 子图

子图的作用包括:
+ 构建多智能体系统
+ 在多个图中重用一组节点
+ 分布式开发：当你希望不同的团队独立地工作在图的不同部分时，你可以将每个部分定义为一个子图，只要子图接口（输入和输出模式）得到尊重，父图可以在不知道子图任何细节的情况下构建

### 2.7.1 Setup 设置

```bash
pip install -U langgraph
```

### 2.7.2 定义子图通信

#### 2.7.2.1 在节点内调用子图

当父图和子图具有不同的状态模式（没有共享键）时，在节点函数内调用子图。在多代理系统中为每个代理保留私有消息历史时很常见。

节点函数在调用子图之前将父状态转换为子图状态，并在返回之前将结果转换回父状态

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START

class SubgraphState(TypedDict):
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "hi! " + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

class MyState(TypedDict):
    foo: str

def call_subgraph(state: MyState):
    subgraph_output = subgraph.invoke({"bar": state["foo"]})
    return {"foo": subgraph_output["bar"]}

builder = StateGraph(MyState)
builder.add_node(call_subgraph)
builder.add_edge(START, "call_subgraph")
graph = builder.compile()
```

#### 2.7.2.2 将子图作为节点添加

当父图和子图共享状态键时，可以直接将编译好的子图传递给 add_node 。无需包装函数——子图会自动从父图的状态通道中读取和写入。例如，在多智能体系统中，智能体通常通过共享消息键进行通信。

如果你的子图与父图共享状态键，你可以按照以下步骤将其添加到你的图中:
+ 定义子图工作流（示例中的 subgraph_builder ），并编译它
+ 在定义父图工作流时，将编译后的子图传递给 add_node 方法

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START


class State(TypedDict):
    foo: str

def subgraph_node_1(state: State):
    return {"foo": "hi! " + state["foo"]}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph

builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()
```

### 2.7.3 子图持久化

当你使用子图时，你需要决定在调用之间其内部数据会发生什么。考虑一个将任务委托给专业子代理的客户支持机器人：应该让“账单专家”子代理记住客户之前的问题，还是每次被调用时都重新开始？

checkpointer 参数控制 .compile() 子图的持久化

#### 2.7.3.1 有状态

有状态子图继承父图的检查点器，这支持中断、持久执行和状态检查。这两种有状态模式在状态保留时间上有所不同。

**按调用（默认）**

在每次调用子图时都是独立的，且子代理不需要记住任何先前调用的信息时，使用每次调用的持久化。这是最常见的模式，特别是对于多代理系统，其中子代理处理一次性请求，例如“查找此客户的订单”或“总结此文档”。

忽略 checkpointer 或将其设置为 None 。每次调用都是全新的，但在单个调用内，子图会继承父级的检查点，并可以使用 interrupt() 来暂停和继续。

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt


@tool
def fruit_info(fruit_name: str) -> str:
    """Look up fruit info"""
    return f"Info about {fruit_name}"

@tool
def veggie_info(veggie_name: str) -> str:
    """Look up veggie info"""
    return f"Info about {veggie_name}"

fruit_agent = create_agent(
    model='gpt-4.1',
    tools=[fruit_info],
    system_prompt="You are a fruit expert. Use the fruit_info tool. Respond in one sentence."
)

veggie_agent = create_agent(
    model="gpt-4.1",
    tools=[veggie_info],
    system_prompt="You are a veggie expert. Use the veggie_info tool. Respond in one sentence."
)

@tool
def ask_fruit_expert(question: str) -> str:
    """Ask the fruit expert. Use for ALL fruit questions."""
    response = fruit_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return response["messages"][-1].content

@tool
def ask_veggie_expert(question: str) -> str:
    """Ask the veggie expert. Use for ALL veggie questions."""
    response = veggie_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return response["messages"][-1].content

agent = create_agent(
    model="gpt-4.1",
    tools=[ask_fruit_expert, ask_veggie_expert],
    system_prompt=(
        "You have two experts: ask_fruit_expert and ask_veggie_expert. "
        "ALWAYS delegate questions to the appropriate expert."
    ),
    checkpointer=MemorySaver()
)


```

**每个线程**

当子代理需要记住之前的交互时，使用线程级别的持久化。例如，一个在多次交流中逐步构建上下文的研究助理，或一个跟踪已编辑文件编码助理。子代理的对话历史和数据在同一线程的多次调用中累积。每次调用都会从上次调用的地方继续。

使用 checkpointer=True 编译以启用此功能

每个线程的子图不支持并行工具调用。当 LLM 将每个线程的子代理作为工具使用时，它可能会尝试并行调用该工具多次（例如，同时询问水果专家关于苹果和香蕉的情况）。这会导致检查点冲突，因为两个调用都写入相同的命名空间。

使用 LangChain 的 ToolCallLimitMiddleware 来防止这种情况。如果你使用纯 LangGraph StateGraph 进行构建，需要自己防止并行工具调用——例如，通过配置你的模型禁用并行工具调用，或添加逻辑确保同一子图不会并行多次被调用。

```python
fruit_agent = create_agent(
    model="gpt-5.4-mini",
    tools=[fruit_info],
    prompt="You are a fruit expert. Use the fruit_info tool. Respond in one sentence.",
    checkpointer=True,
)

agent = create_agent(
    model="gpt-5.4-mini",
    tools=[ask_fruit_expert],
    prompt="You have a fruit expert. ALWAYS delegate fruit questions to ask_fruit_expert.",
    middleware=[
        ToolCallLimitMiddleware(tool_name="ask_fruit_expert", run_limit=1),
    ],
    checkpointer=MemorySaver(),
)
```

#### 2.7.3.2 无状态

当你想以普通函数调用的方式运行子代理且不希望有检查点开销时，请使用此方法。子图无法暂停/恢复，并且无法受益于持久化执行。使用 checkpointer=False 编译。

```python
subgraph_builder = StateGraph(...)
subgraph = subgraph_builder.compile(checkpointer=False)
```

#### 2.7.3.3 检查点引用

![子图检查点引用.png](./images/子图检查点引用.png)

+ 中断（HITL）：子图可以使用 interrupt()来暂停执行并等待用户输入，然后从停止的地方继续执行。
+ 多轮记忆：子图可以在同一线程内的多次调用中保持其状态。每次调用都会从上一次调用的地方继续，而不是重新开始。
+ 多次调用（不同的子图）：多个不同的子图实例可以在单个节点内被调用，而不会出现检查点命名空间冲突。
+ 多次调用（相同的子图）：相同的子图实例可以在单个节点内被多次调用。在有状态持久化的情况下，这些调用会写入相同的检查点命名空间并发生冲突——请使用每次调用持久化代替。
+ 状态检查：子图的状通过 get_state(config, subgraphs=True) 可用于调试和监控。

### 2.7.4 查看子图状态

当你启用持久化时，可以使用子图选项来检查子图状态。在无状态检查点（ checkpointer=False ）的情况下，不会保存子图检查点，因此子图状态不可用。

**每次调用**

仅返回当前调用中的子图状态。每次调用都是全新的

```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

class State(TypedDict):
    foo: str

# Subgraph
def subgraph_node_1(state: State):
    value = interrupt("Provide value:")
    return {"foo": state["foo"] + value}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()  # inherits parent checkpointer

# Parent graph
builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"foo": ""}, config)

# View subgraph state for the current invocation
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state  # ******** 

# Resume the subgraph
graph.invoke(Command(resume="bar"), config)
```

**每个线程**

返回在此线程上所有调用中累积的子图状态。

```python
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Subgraph with its own persistent state
subgraph_builder = StateGraph(MessagesState)
# ... add nodes and edges
subgraph = subgraph_builder.compile(checkpointer=True)  # ********

# Parent graph
builder = StateGraph(MessagesState)
builder.add_node("agent", subgraph)
builder.add_edge(START, "agent")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "what did I say?"}]}, config)

# View accumulated subgraph state (includes messages from both invocations)
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state  # *******
```

### 2.7.5 流式传输子图输出

要在流式输出中包含子图的输出，您可以在父图的流式方法中设置子图选项。这将同时流式传输父图和任何子图的输出。

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode="updates",
    version="v2",
):
    print(chunk["type"])  # "updates"
    print(chunk["ns"])    # () for root, ("node_2:<task_id>",) for subgraph
    print(chunk["data"])  # {"node_name": {"key": "value"}}
```

# 3 Production 生产

## 3.1 Application structure 应用结构

一个 LangGraph 应用程序由一个或多个图、一个配置文件（ langgraph.json ）、一个指定依赖关系的文件，以及一个可选的 .env 文件（指定环境变量）组成。

### 3.1.1 关键概念

使用 LangSmith 进行部署时，应提供以下信息:
+ 一个 LangGraph 配置文件（ langgraph.json ），用于指定应用程序所需的依赖项、图和环境变量。
+ 实现应用程序逻辑的图。
+ 一个指定运行应用程序所需依赖项的文件。
+ 应用程序运行所需的环境变量。

### 3.1.2 文件结构

```
my-app/
├── my_agent # all project code lies within here
│   ├── utils # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py # tools for your graph
│   │   ├── nodes.py # node functions for your graph
│   │   └── state.py # state definition of your graph
│   ├── __init__.py
│   └── agent.py # code for constructing your graph
├── .env # environment variables
├── requirements.txt # package dependencies
└── langgraph.json # configuration file for LangGraph
└── pyproject.toml # dependencies for your project
```

### 3.1.3 配置文件

langgraph.json 文件是一个 JSON 文件，用于指定部署 LangGraph 应用所需的依赖项、图、环境变量和其他设置。

```json
{
  "dependencies": ["langchain_openai", "./your_package"],
  "graphs": {
    "my_agent": "./your_package/your_file.py:agent"
  },
  "env": "./.env"
}
```

### 3.1.4 依赖项

LangGraph 应用程序可能依赖于其他 Python 包。

通常需要指定以下信息来正确设置依赖项:
+ 一个位于指定依赖项的文件（例如 requirements.txt 、 pyproject.toml 或 package.json ）。
+ 在 LangGraph 配置文件中指定运行 LangGraph 应用所需的依赖项的 dependencies 键
+ 任何额外的二进制文件或系统库都可以使用 LangGraph 配置文件中的 dockerfile_lines 键来指定。

### 3.1.5 图

在 LangGraph 配置文件中使用 graphs 键来指定部署的 LangGraph 应用程序中将有哪些图可用。

可以在配置文件中指定一个或多个图。每个图由一个名称（应唯一）和路径标识，路径可以是：(1) 编译后的图或(2) 定义图的函数。

### 3.1.6 环境变量

在本地使用部署的 LangGraph 应用程序，可以在 LangGraph 配置文件的 env 键中配置环境变量。

对于生产环境部署，你通常需要在部署环境中配置环境变量。

## 3.2 Test 测试

### 3.2.1 前提条件

```bash
pip install -U pytest
```

### 3.2.2 入门指南

由于许多 LangGraph 代理依赖于状态，一个有用的模式是在每次使用它的测试之前创建你的图，然后在测试中用一个新的 checkpointer 实例编译它。

下面的示例展示了如何使用一个简单的线性图来演示这个过程，该图通过 node1 和 node2 进行进展。每个节点都会更新单个状态键 my_key 

```python
import pytest 
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

def create_graph() -> StateGraph:
    class MyState(TypedDict):
        my_key: str
    
    graph = StateGraph(MyState)
    graph.add_node("node1", lambda state: {"my_key": "hello from node1"})
    graph.add_node("node2", lambda state: {"my_key": "hello from node2"})

    graph.add_edge(START, "node1")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", END)
    return graph


def test_basic_agent_execution() -> None:
    checkpointer = InMemorySaver()
    graph = create_graph()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    result = compiled_graph.invoke(
        {"my_key": "initial_value"},
        config={"configurable": {"thread_id": "1"}}
    )
    assert result["my_key"] == "hello from node2"
```

### 3.2.3 测试单个节点和边

编译的 LangGraph 代理会暴露每个独立节点的引用，形式为 graph.nodes 。你可以利用这一点来测试代理中的单个节点。请注意，这将绕过编译图时传递的任何检查点器:

```python
import pytest
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


def create_graph() -> StateGraph:
    class MyState(TypedDict):
        my_key: str

    graph = StateGraph(MyState)
    graph.add_node("node1", lambda state: {"my_key": "hello from node1"})
    graph.add_node("node2", lambda state: {"my_key": "hello from node2"})
    graph.add_edge(START, "node1")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", END)
    return graph

def test_individual_node_execution() -> None:
    # Will be ignored in this example
    checkpointer = InMemorySaver()
    graph = create_graph()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    result = compiled_graph.nodes["node1"].invoke(
        {"my_key": "initial_value"},
    )
    assert result["my_key"] == "hello from node1"
```

### 3.2.4 部分执行

对于由较大图组成的代理，你可能希望测试代理中的部分执行路径，而不是端到端地执行整个流程。在某些情况下，将这些部分重构为子图可能具有语义意义，你可以像平常一样独立调用这些子图。

如果你不希望更改代理图的整体结构，可以使用 LangGraph 的持久化机制来模拟一个状态，使代理在期望部分的开始之前暂停，并在期望部分的结束处再次暂停。步骤如下:
+ 使用检查点（内存检查点 InMemorySaver 足够用于测试）来编译你的代理。
+ 调用你的代理的 update_state 方法，并将 as_node 参数设置为你要开始测试的前一个节点的名称。
+ 使用与更新状态时相同的 thread_id 调用你的代理，并将 interrupt_after 参数设置为你要停止的那个节点的名称。

```python
import pytest
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


def create_graph() -> StateGraph:
    class MyState(TypedDict):
        my_key: str
        
    graph = StateGraph(MyState)
    graph.add_node("node1", lambda state: {"my_key": "hello from node1"})
    graph.add_node("node2", lambda state: {"my_key": "hello from node2"})
    graph.add_node("node3", lambda state: {"my_key": "hello from node3"})
    graph.add_node("node4", lambda state: {"my_key": "hello from node4"})
    graph.add_edge(START, "node1")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", "node3")
    graph.add_edge("node3", "node4")
    graph.add_edge("node4", END)

    return graph

def test_partial_execution_from_node2_to_node3() -> None:
    checkpointer = InMemorySaver()
    graph = create_graph()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    compiled_graph.update_state(
        config={"configurable": {"thread_id": "1"}},
        values={"my_key": "initial_value"},
        as_node="node1"  # 要测试节点的前一个节点名称
    )

    result = compiled_graph.invoke(
        None,
        config={"configurable": {"thread_id": "1"}},
        interrupt_after="node3"  # 要停止的节点名称
    )
    assert result["my_key"] == "hello from node3"
```

## 3.3 LangSmith Studio

在本地使用 LangChain 构建代理时，可视化代理内部的操作、实时交互以及调试问题会很有帮助。LangSmith Studio 是一个免费的视觉界面，用于从本地机器开发和测试你的 LangChain 代理。

Studio 连接到本地运行的代理，展示代理执行的每一步：发送给模型的提示、工具调用及其结果，以及最终输出。你可以测试不同的输入、检查中间状态，并在无需额外代码或部署的情况下迭代代理的行为。

### 3.3.1 前提条件

开始之前，请确保你拥有以下内容:
+ 一个 LangSmith 账户：在 smith.langchain.com 上注册（免费）或登录。
+ 一个 LangSmith API 密钥：按照创建 API 密钥指南操作。
+ 如果您不想让数据追踪到 LangSmith，请在应用程序的 .env 文件中设置 LANGSMITH_TRACING=false 。禁用追踪后，没有数据会离开您的本地服务器。

### 3.3.2 设置本地Agent服务器

#### 3.3.2.1 安装langgraph CLI

```bash
# Python >= 3.11 is required.
pip install --upgrade "langgraph-cli[inmem]"
```

#### 3.3.2.2 准备代理

```python
from langchain.agents import create_agent

def seed_email(to: str, subject: str, body: str):
    """Seed an email"""
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    return f"Email sent to {to}"

agent = create_agent(
    "gpt-4.1",
    tools=[seed_email],
    system_prompt="You are an email assistant.Always use the seed_email tool to send emails."
)
```

#### 3.3.2.3 环境变量
Studio 需要一个 LangSmith API 密钥来连接你的本地代理。在你的项目根目录中创建一个 .env 文件，并将你的 LangSmith API 密钥添加到该文件中。
```bash
LANGSMITH_API_KEY=<your-api-key>
```

#### 3.3.2.4 创建langgraph配置文件

LangGraph CLI 使用配置文件来定位您的代理和管理依赖项。在您的应用程序目录中创建一个 langgraph.json 文件:
```bash
# langgraph.json
{
  "dependencies": ["."],
  "graph": {
    "agent": "./src/agent.py:agent"
  },
  "env": ".env"
}
```

create_agent 函数会自动返回一个编译好的 LangGraph 图，这是 graphs 配置文件中的键所期望的。

#### 3.3.2.5 安装依赖

```bash
pip install langchain langchain-openai
```

#### 3.3.2.6 在Studio中查看Agent

启动开发服务器以连接您的代理到工作室:
```bash
langgraph dev
```
服务器运行后，您的代理可通过 http://127.0.0.1:2024 的 API 访问，也可通过 https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024 的 Studio UI 访问

## 3.4 LangSmith Deployment LangSmith 部署

本指南将向您展示如何将您的代理部署到 LangSmith Cloud，这是一个专为代理工作负载设计的完全托管的主机平台。通过云部署，您可以直接从您的 GitHub 仓库进行部署——LangSmith 负责基础设施、扩展和运营问题。

传统的主机平台是为无状态、短生命周期的 Web 应用程序构建的。LangSmith Cloud 是专为有状态、长时间运行的代理设计的，这些代理需要持久状态和后台执行。

### 3.4.1 前提条件

+ GitHub 账户
+ LangSmith 账户

### 3.4.2 部署Agent

#### 3.4.2.1 在github上创建一个仓库

你的应用程序代码必须存储在 GitHub 仓库中才能在 LangSmith 上部署。支持公开和私有仓库。对于快速入门，首先确保你的应用兼容 LangGraph，方法是按照本地服务器设置指南进行操作。然后，将你的代码推送到仓库。

#### 3.4.2.2 部署到langsmith

+ 导航至 LangSmith 部署，登录 LangSmith。在左侧边栏中，选择部署。
+ 点击“+ 新建部署”按钮。将打开一个面板，您可以在其中填写所需字段。
+ 如果您是首次使用或添加尚未连接的私有仓库，请点击“添加新账户”按钮，并按照说明连接您的 GitHub 账户。
+ 选择您的应用程序的存储库。点击提交以部署。这可能需要大约 15 分钟才能完成。您可以在部署详情视图中查看状态。

#### 3.4.2.3 在工作室中测试应用程序

应用程序部署完成: 
+ 选择您刚刚创建的部署以查看更多详细信息。
+ 点击右上角的 Studio 按钮。Studio 将打开以显示您的图。

#### 3.4.2.4 获取部署API URL

+ 在 LangGraph 的部署详情视图中，点击 API URL 将其复制到剪贴板。
+ 点击 URL 将其复制到剪贴板。

#### 3.4.2.5 测试API

```bash
pip install langgraph-sdk
```

向代理发送消息:
```python
from langgraph_sdk import get_sync_client # or get_client for async

client = get_sync_client(url="your-deployment-url", api_key="your-langsmith-api-key")

for chunk in client.runs.stream(
    None,    # Threadless run
    "agent", # Name of agent. Defined in langgraph.json.
    input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph?",
        }],
    },
    stream_mode="updates",
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
```

## 3.5 LangSmith Observability LangSmith 可观测性

跟踪是一系列应用程序从输入到输出的步骤。每个单独的步骤由一个运行表示。您可以使用 LangSmith 来可视化这些执行步骤。要使用它，请为您的应用程序启用跟踪。

### 3.5.1 前提条件

+ 在 smith.langchain.com 上注册（免费）或登录。
+ 一个 LangSmith API 密钥：请遵循创建 API 密钥指南。

### 3.5.2 启用跟踪

设置环境变量:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
```

默认情况下，跟踪将记录到名为 default 的项目中。要配置自定义项目名称，请参阅记录到项目

### 3.5.3 选择性跟踪

您可以使用 LangSmith 的 tracing_context 上下文管理器选择性地跟踪特定的调用或应用程序的一部分
```python
import langsmith as ls

# This WILL be traced
with ls.tracing_context(enabled=True):
    agent.invoke({"messages": [{"role": "user", "content": "Send a test email to alice@example.com"}]})

# This will NOT be traced (if LANGSMITH_TRACING is not set)
agent.invoke({"messages": [{"role": "user", "content": "Send another email"}]})
```

### 3.5.4 将日志记录到项目

**静态地**
```bash
export LANGSMITH_PROJECT=my-agent-project
```

**动态地**
```python
import langsmith as ls

with ls.tracing_context(project_name="email-agent-test", enabled=True):
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Send a welcome email"}]
    })
```

### 3.5.5 向跟踪添加元数据

**可以向跟踪添加自定义元数据和标签**
```python
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Send a welcome email"}]},
    config={
        "tags": ["production", "email-assistant", "v1.0"],
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456",
            "environment": "production"
        }
    }
)
```

**tracing_context 也接受标签和元数据以实现细粒度控制**
````python
with ls.tracing_context(
    project_name="email-agent-test",
    enabled=True,
    tags=["production", "email-assistant", "v1.0"],
    metadata={"user_id": "user_123", "session_id": "session_456", "environment": "production"}):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Send a welcome email"}]}
    )
````

### 3.5.6 使用匿名化器来防止在跟踪记录中记录敏感数据

可能希望遮盖敏感数据，以防止其被记录到 LangSmith 中。你可以创建匿名化器，并使用配置将其应用于你的图。此示例将从发送到 LangSmith 的跟踪中删除与社会保障号码格式 XXX-XX-XXXX 匹配的内容。

```python
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.graph import StateGraph, MessagesState
from langsmith import Client
from langsmith.anonymizer import create_anonymizer

anonymizer = create_anonymizer([
    # Matches SSNs
    { "pattern": r"\b\d{3}-?\d{2}-?\d{4}\b", "replace": "<ssn>" }
])

tracer_client = Client(anonymizer=anonymizer)
tracer = LangChainTracer(client=tracer_client)
# Define the graph
graph = (
    StateGraph(MessagesState)
    ...
    .compile()
    .with_config({'callbacks': [tracer]})
)
```

# 4 LangGraph APIs

## 4.1 Graph API 图API

### 4.1.1 选择APIs

LangGraph 提供两种不同的 API 来构建代理工作流：图 API 和函数 API。这两种 API 共享相同的底层运行时，可以在同一个应用程序中一起使用，但它们是为不同的用例和开发偏好而设计的。

#### 4.1.1.1 快速决策指南

+ **使用图API**:
  + 复杂的流程可视化，用于调试和文档
  + 显式状态管理，跨多个节点共享数据
  + 条件分支，包含多个决策点
  + 需要后期合并的并行执行路径
  + 团队协作，可视化表示有助于理解
+ **使用Function API**:
  + 对现有过程式代码进行最小化代码更改
  + 标准控制流（if/else、循环、函数调用）
  + 函数作用域状态，无需显式状态管理
  + 使用更少的模板代码进行快速原型开发
  + 具有简单分支逻辑的线性工作流

#### 4.1.1.2 详细对比

##### 4.1.1.2.1 何时使用图API

**复杂的决策树和分支逻辑**

当你的工作流中有多个依赖于不同条件的决策点时，Graph API 可以使这些分支显式化并易于可视化
```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    messages: list
    current_tool: str
    retry_count: int

def should_continue(state: AgentState):
    if state["retry_count"] > 3:
        return "end"
    elif state["current_tool"] == "search":
        return "process_search"
    else:
        return "call_llm"

workflow = StateGraph(AgentState)
workflow.add_node("call_llm", call_llm_node)
workflow.add_node("process_search", process_search_node)
workflow.add_conditional_edges("call_llm", should_continue)
```

**跨多个组件的状态管理**

当你需要在工作流的不同部分之间共享和协调状态时，Graph API 的显式状态管理是有益的
```python
# Multiple nodes can access and modify shared state
class WorkflowState(TypedDict):
    user_input: str
    search_results: list
    generated_response: str
    validation_status: str

def search_node(state):
    # Access shared state
    results = search(state["user_input"])
    return {"search_results": results}

def validation_node(state):
    # Access results from previous node
    is_valid = validate(state["generated_response"])
    return {"validation_status": "valid" if is_valid else "invalid"}
```

**并行处理与同步**

当你需要并行运行多个操作并组合它们的结果时，Graph API 可以自然地处理这种情况
```python
# Parallel processing of multiple data sources
workflow.add_node("fetch_news", fetch_news)
workflow.add_node("fetch_weather", fetch_weather)
workflow.add_node("fetch_stocks", fetch_stocks)
workflow.add_node("combine_data", combine_all_data)

# All fetch operations run in parallel
workflow.add_edge(START, "fetch_news")
workflow.add_edge(START, "fetch_weather")
workflow.add_edge(START, "fetch_stocks")

# Combine waits for all parallel operations to complete
workflow.add_edge("fetch_news", "combine_data")
workflow.add_edge("fetch_weather", "combine_data")
workflow.add_edge("fetch_stocks", "combine_data")
```

**团队开发与文档**

Graph API 的可视化特性使团队能够更轻松地理解、记录和维护复杂的流程。
```python
# Clear separation of concerns - each team member can work on different nodes
workflow.add_node("data_ingestion", data_team_function)
workflow.add_node("ml_processing", ml_team_function)
workflow.add_node("business_logic", product_team_function)
workflow.add_node("output_formatting", frontend_team_function)
```

##### 4.1.1.2.2 何时使用 Functional API

**现有过程式代码**

当你有使用标准控制流的现有代码，并希望以最小的重构添加 LangGraph 功能时。
```python
from langgraph.func import entrypoint, task

@task
def process_user_input(user_input: str) -> dict:
    # Existing function with minimal changes
    return {"processed": user_input.lower().strip()}

@entrypoint(checkpointer=checkpointer)
def workflow(user_input: str) -> str:
    # Standard Python control flow
    processed = process_user_input(user_input).result()

    if "urgent" in processed["processed"]:
        response = handle_urgent_request(processed).result()
    else:
        response = handle_normal_request(processed).result()

    return respons
```

**简单逻辑的线性工作流**

当你的工作流主要是顺序执行且具有简单的条件逻辑时。
```python
@entrypoint(checkpointer=checkpointer)
def essay_workflow(topic: str) -> dict:
    # Linear flow with simple branching
    outline = create_outline(topic).result()

    if len(outline["points"]) < 3:
        outline = expand_outline(outline).result()

    draft = write_draft(outline).result()

    # Human review checkpoint
    feedback = interrupt({"draft": draft, "action": "Please review"})

    if feedback == "approve":
        final_essay = draft
    else:
        final_essay = revise_essay(draft, feedback).result()

    return {"essay": final_essay}
```

**快速原型设计**

当你想快速测试想法，而不必承担定义状态模式和图结构的开销时。
```python
@entrypoint(checkpointer=checkpointer)
def quick_prototype(data: dict) -> dict:
    # Fast iteration - no state schema needed
    step1_result = process_step1(data).result()
    step2_result = process_step2(step1_result).result()

    return {"final_result": step2_result}
```

**函数范围的状态管理**

当你的状态自然地局限于单个函数，并且不需要广泛共享时。
```python
@task
def analyze_document(document: str) -> dict:
    # Local state management within function
    sections = extract_sections(document)
    summaries = [summarize(section) for section in sections]
    key_points = extract_key_points(summaries)

    return {
        "sections": len(sections),
        "summaries": summaries,
        "key_points": key_points
    }

@entrypoint(checkpointer=checkpointer)
def document_processor(document: str) -> dict:
    analysis = analyze_document(document).result()
    # State is passed between functions as needed
    return generate_report(analysis).result()
```

#### 4.1.1.3 结合两个API

可以在同一应用程序中使用这两个 API。
```python
from langgraph.graph import StateGraph
from langgraph.func import entrypoint


coordination_graph = StateGraph(CoordinationState)
coordination_graph.add_node("orchestrator", orchestrator_node)
coordination_graph.add_node("agent_a", agent_a_node)
coordination_graph.add_node("agent_b", agent_b_node)

@entrypoint()
def data_processor(raw_data: dict) -> dict:
    cleaned = clean_data(raw_data).result()
    transformed = transform_data(cleaned).result()
    return  transformed

# Use the functional API result in the graph
def orchestrator_node(state):
    processed_data = data_processor.invoke(state["raw_data"])
    return {"processed_data": processed_data}
```

#### 4.1.1.4 API之间的迁移

当您的功能工作流变得复杂时，您可以迁移到图 API

**从Function API到图API**
```python
# Before: Functional API
@entrypoint(checkpointer=checkpointer)
def complex_workflow(input_data: dict) -> dict:
    step1 = process_step1(input_data).result()

    if step1["needs_analysis"]:
        analysis = analyze_data(step1).result()
        if analysis["confidence"] > 0.8:
            result = high_confidence_path(analysis).result()
        else:
            result = low_confidence_path(analysis).result()
    else:
        result = simple_path(step1).result()

    return result

# After: Graph API
class WorkflowState(TypedDict):
    input_data: dict
    step1_result: dict
    analysis: dict
    final_result: dict

def should_analyze(state):
    return "analyze" if state["step1_result"]["needs_analysis"] else "simple_path"

def confidence_check(state):
    return "high_confidence" if state["analysis"]["confidence"] > 0.8 else "low_confidence"

workflow = StateGraph(WorkflowState)
workflow.add_node("step1", process_step1_node)
workflow.add_conditional_edges("step1", should_analyze)
workflow.add_node("analyze", analyze_data_node)
workflow.add_conditional_edges("analyze", confidence_check)
# ... add remaining nodes and edges
```

**从图API到Function API**
```python
# Before: Over-engineered Graph API
class SimpleState(TypedDict):
    input: str
    step1: str
    step2: str
    result: str

# After: Simplified Functional API
@entrypoint(checkpointer=checkpointer)
def simple_workflow(input_data: str) -> str:
    step1 = process_step1(input_data).result()
    step2 = process_step2(step1).result()
    return finalize_result(step2).result()
```

### 4.1.2 图API

#### 4.1.2.1 图

在核心层面，LangGraph 将代理工作流建模为图。您通过三个关键组件定义代理的行为:
+ State: 一个共享数据结构，表示您应用程序的当前快照。它可以任何数据类型，但通常使用共享状态模式定义。
+ Nodes: 编码你代理逻辑的函数。它们接收当前状态作为输入，执行某些计算或副作用，并返回更新后的状态。
+ Edges: 根据当前状态决定要执行哪个 Node 。它们可以是条件分支或固定转换。

#### 4.1.2.2 状态

指定图架构的主要文档化方式是使用 TypedDict 。如果你想在状态中提供默认值，使用 dataclass 。我们还支持使用 Pydantic BaseModel 作为你的图状态，
如果你需要递归数据验证（但请注意，Pydantic 的性能不如 TypedDict 或 dataclass ）。

```python
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # Read from OverallState, write to PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # Read from PrivateState, write to OutputState
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})
# {'graph_output': 'My name is Lance'}
```

#### 4.1.2.3 节点

在 LangGraph 中，节点是 Python 函数（可以是同步的或异步的），它们接受以下参数:
+ state: 图的当前状态
+ config: 一个 RunnableConfig 对象，其中包含配置信息（如 thread_id ）和跟踪信息（如 tags ）
+ runtime: 一个 Runtime 对象，其中包含运行时 context 以及其他信息（如 store 、 stream_writer 、 execution_info 和 server_info ）

```python
from dataclasses import dataclass
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

class State(TypedDict):
    input: str
    results: str

@dataclass
class Context:
    user_id: str

builder = StateGraph(State)

def plain_node(state: State):
    return state

def node_with_runtime(state: State, runtime: Runtime[Context]):
    print("In node: ", runtime.context.user_id)
    return {"results": f"Hello, {state['input']}!"}

def node_with_execution_info(state: State, runtime: Runtime):
    print("In node with thread_id: ", runtime.execution_info.thread_id)
    return {"results": f"Hello, {state['input']}!"}


builder.add_node("plain_node", plain_node)
builder.add_node("node_with_runtime", node_with_runtime)
builder.add_node("node_with_execution_info", node_with_execution_info)
...
```

#### 4.1.2.4 边

边定义了逻辑如何路由以及图如何决定停止。

+ 普通边：直接从一个节点跳转到下一个节点。
+ 条件边：调用一个函数来决定下一步跳转到哪个节点。
+ 入口点：当用户输入到达时，第一个要调用的节点。
+ 条件入口点：调用一个函数来决定当用户输入到达时，第一个要调用的节点。

```python
# Normal edges  普通边
graph.add_edge("node_a", "node_b")


# Conditional edges  条件边缘
graph.add_conditional_edge("node_a", "node_b", condition_function)


# Entry point  入口点
graph.add_edge(START, "node_a")


# Conditional entry point  条件入口点
graph.add_conditional_edges(START, routing_function)
```

#### 4.1.2.5 Send

为了支持这种设计模式，LangGraph 支持从条件式边返回 Send 对象。 Send 接收两个参数：第一个是节点的名称，第二个是要传递给该节点的状态。

```python
from langgraph.types import Send

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

graph.add_conditional_edges("node_a", continue_to_jokes)
```

#### 4.1.2.6 Command

Command 是一个用于控制图执行的通用原语。参数:
+ update: 应用状态更新（类似于从节点返回更新）。
+ goto: 导航到特定节点（类似于条件边）。
+ graph: 在从子图导航时，目标父图。
+ resume: 在中断后提供值以恢复执行。

Command 在三种情境中使用:
+ 从节点返回：使用 update 、 goto 和 graph 将状态更新与控制流结合。
+ invoke 或 stream 的输入：使用 resume 在中断后继续执行。
+ 从工具返回：与从节点返回类似，在工具内部结合状态更新和控制流。

```python

# 节点返回
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```

**运行时上下文**

```python
@dataclass
class ContextSchema:
    llm_provider: str = "openai"
# 可以指定 context_schema 作为传递给节点的运行时上下文。这对于传递不属于图状态的信息给节点很有用。例如，传递依赖项，如模型名称或数据库连接。
graph = StateGraph(State, context_schema=ContextSchema)
# 可以使用 invoke 方法的 context 参数将此上下文传递给图。
graph.invoke(inputs, context={"llm_provider": "anthropic"})


```

### 4.1.3 使用图API

## 4.2 Functional API 函数调用API














