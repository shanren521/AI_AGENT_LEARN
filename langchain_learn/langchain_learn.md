# langchain 学习笔记

## 1 核心组件

### 1.1 Agents

智能体将语言模型与工具相结合，创建能够对任务进行推理、决定使用哪些工具并迭代地寻求解决方案的系统。
create_agent提供了一个可用于生产环境的代理实现

### 1.2 Models

#### 1.2.1基本用法

+ 使用代理时 ，可以动态指定模型。
+ 独立运行 - 可以直接调用模型（在代理循环之外）来执行文本生成、分类或提取等任务，而无需代理框架。

**初始化模型**

在LangChain中使用独立模型的最简单方法是使用init_chat_model(openai标准格式), 如果是第三方格式(包含vllm)使用ChatOpenAI，ollama本地部署使用ChatOllam

**关键方法**

invoke/stream/batch/

invoke create_agent创建的模型和ChatOpenAI创建的模型调用invoke输入的数据格式不一样,agent更严格

stream可以通过循环获取数据块

batch并行处理请求,会返回整个批次的最终输出，可以通过batch_as_completed()在生成完成后立即接收输出

**参数**

```
model string required  也可用:分隔，例如openai:o1
api_key string 模型密钥
temperature number 温度，越大随机性越强，越小越确定
max_tokens number 最大token数
timeout number  超时时间
max_retries number  最大重试次数(默认6)
```

**工具调用**

使用bind_tools绑定tool

**结构化输出**
LangChain 支持多种模式类型和方法来强制输出结构化数据。通过with_structured_output方法来指定

#### 1.2.2 Advanced topics高级主题

**Model profiles**
LangChain 聊天模型可以通过 profile 属性公开一个支持的功能和特性字典

**多模态**
某些模型可以处理并返回非文本数据，例如图像、音频和视频。也可以返回多模态数据作为响应的一部分.

**推理**
通过判断模型返回的字段是否有reasoning，可以将这个推理过程展现出来，以便更好地理解模型是如何得出最终答案的。

### 1.3 Messages
在 LangChain 中，消息是模型的基本上下文单元。代表模型的输入和输出，携带与 LLM 交互时表示对话状态所需的内容和元数据。

**包含的对象有**
+ role: system\user
+ content: 文本、图像、音频、文档等
+ metadata：可选字段，例如响应消息、消息ID和令牌使用情况

#### 1.3.1 基本用法

HumanMessage, AIMessage, SystemMessage一般使用这三个构造输入message

**文本prompt**
最简单的生成任务

**Message prompt**
通过特定的消息对象列表传给模型

以下情况使用消息prompt
+ 管理多轮对话
+ 处理多模态内容
+ 包括系统说明

**字典格式**

**Message types消息类型**
+ System message: 告诉模型如何运行，并为交互提供上下文
+ Human message: 用户输入以及与模型的交互
+ AI message: 模型生成的响应，包括文本内容、工具调用和元数据
+ Tool message: 表示工具调用的输出


**工具调用**
response.tool_calls可以获取模型返回的函数调用

**token usage**
response.usage_metadata获取token使用情况，API模型可能方法不一样

**stream and chunks**
for循环获取模型流式返回，chunk.text

### 1.4 Tools 工具

**创建工具**

+ 基本工具定义: 最简单的方法就是使用@tool装饰器
+ 自定义工具属性: @tool(xxx)
+ 高级模式定义: 使用 Pydantic 模型或 JSON 模式定义复杂输入@tool(args_schema=WeatherInput)

**访问上下文**

短期记忆（会话级）：近几轮摘要、当前任务状态（常放 Redis）
长期记忆（用户级）：偏好、画像、历史关键事实（常放向量库+结构化库）
所有的runtime都是ToolRuntime对象

+ 短期记忆: state 访问状态，Command更新代理的状态
+ memory(state):
  + Access state 访问状态：工具可以通过 runtime.state 访问当前对话状态
  + Update state 更新状态：使用 Command 更新代理的状态。这对于需要更新自定义状态字段的工具非常有用
+ Context: 上下文提供在调用时传递的不可变配置数据。可用于用户 ID、会话详细信息或在对话过程中不应更改的应用程序特定设置。runtime.context
+ 长期记忆：BaseStore 提供持久存储，可在会话之间保留数据。与状态（短期记忆）不同，保存到存储中的数据在以后的会话中仍然可用。
  + 通过 runtime.store 访问数据存储。该数据存储使用命名空间/键模式来组织数据
+ Stream writer: 在工具执行过程中，实时传输工具的更新信息。使用 ToolRuntime().stream_writer 发出自定义更新
+ Execution info 执行信息：通过 runtime.execution_info 从工具内部访问线程 ID、运行 ID 和重试状态
+ Server info 服务信息：当您的工具在 LangGraph Server 上运行时，可通过 runtime.server_info 访问助手 ID、图 ID 和已验证用户


**ToolNode工具节点**

+ Basic usage 基本用法：参考代码
+ Tool return values 工具返回值
  + 返回一个string，返回值被转为ToolMessage，不会更改任何状态，模型识别后会决定下一步做什么
  + 返回一个object，将对象系列化并作为工具输出发送回去，不会直接更改图状态，模型可以读取特定字段进一步处理
  + 返回一个带有可选消息的Command，使用update更新状态，对于可能被并行工具调用更新的字段，使用reducer
    + 当工具需要更新图状态（例如，设置用户偏好或应用状态）时，返回一个 Command 。您可以返回一个包含或不包含 ToolMessage Command 。
    + 如果模型需要确认工具操作成功（例如，确认偏好设置更改），请在更新中包含 ToolMessage ，并使用 runtime.tool_call_id 作为 tool_call_id 参数。
+ Error handing 错误处理：配置工具错误的处理方式。
+ Route with tools_condition 使用tools_condition进行路由：使用 tools_condition 根据 LLM 是否发出工具调用进行条件路由
+ State injection 状态注入: 工具可以通过 ToolRuntime 访问当前图状态

**Prebuild tools 预构建工具**
+ LangChain 提供了一系列预构建的工具和工具包，用于执行诸如网页搜索、代码解析、数据库访问等常见任务。这些即用型工具可以直接集成到您的代理中，无需编写自定义代码。

**Server-side tool use  服务器端工具的使用**
+ 某些聊天模型内置了一些工具，这些工具由模型提供商在服务器端执行。这些工具包括网页搜索和代码解释器等功能，无需您定义或托管工具逻辑。

### 1.5 Short-term memory 短期记忆

**Usage 用法**
要向代理添加短期记忆（线程级持久性），需要在创建代理时指定 checkpointer

+ In production在生产中: 使用由数据库支持的检查点

**Customizing agent memory  自定义代理内存**

代理使用 AgentState 来管理短期记忆，特别是通过 messages 键来管理对话历史记录。自定义状态模式通过 state_schema 参数传递给 create_agent 函数。

**Common patterns  常见模式**

启用短期记忆后，长时间的对话可能会超出 LLM 的上下文窗口

+ 修剪消息：移除前N条或后N条消息
+ 删除消息：永久删除LangGraph状态中的消息, 要使 RemoveMessage 生效，您需要将状态键与 add_messages reducer 一起使用。
+ 消息摘要：将历史记录中的早期消息进行总结，并用摘要替换他们, 修剪或删除消息的问题在于，您可能会因消息队列的清理而丢失信息。
  因此，一些应用程序会受益于更复杂的方法，例如使用聊天模型来汇总消息历史记录。SummarizationMiddleware 
+ 自定义策略：如消息过滤

**Access memory 访问内存**

+ Tools工具
  + 使用工具读取短期记忆：使用 runtime 参数（类型为 ToolRuntime ）访问工具中的短期记忆（状态）。
  + 从工具写入短期记忆：要在执行期间修改代理的短期记忆（状态），您可以直接从工具返回状态更新。

**Prompt**
在中间件中访问短期记忆（状态），以根据对话历史记录或自定义状态字段创建动态prompts。

**Before model**
在 @before_model 中间件中访问短期记忆（状态）以在模型调用之前处理消息。

**After model**
在 @after_model 中间件中访问短期记忆（状态），以便在模型调用后处理消息。


### 1.6 Streaming 流媒体

能实现的功能：
+ 流式传输代理进度：在代理的每个步骤之后获取状态更新。
+ 流式LLM tokens：在生成时流式输出tokens
+ 流式思考/推理：作为模型推理的表面生成
+ 流式自定义更新：发出用户定义的信号
+ 流式传输多种模式：可选择updates(代理进度)、messages(LLM tokens + metadata)、自定义(任意用户数据)

**Supported stream modes  支持的流模式**
将以下一种或多种流模式作为列表传递给 stream 或 astream 方法(updates\messages\custom)

**Agent progress  代理进度**
要流式传输代理进度，请使用 stream 或 astream 方法，并将 stream_mode="updates" 。这样会在代理执行每个步骤后发出一个事件。

**LLM tokens**
要实时传输 LLM 生成的令牌，请使用 stream_mode="messages", 和API 的stream=True相同

**Custom updates  自定义更新**
要从正在执行的工具中流式传输更新，可以使用 get_stream_writer 。

**Stream multiple modes  流式多模式**
你可以通过传递流模式列表来指定多个流模式： stream_mode=["updates", "custom"] 。
每个流式传输的数据块都是一个 StreamPart 字典，包含 type 、 ns 和 data 三个键。使用 chunk["type"] 确定流模式，使用 chunk["data"] 访问有效负载。

**Common patterns  常见模式**

+ Streaming thinking/reasoning tokens 流式思维/推理tokens:通过筛选 type 为 "reasoning"获取内容(stream_mode="messages")
+ Streaming tool calls 流式工具调用: 指定 stream_mode="messages" 将流式传输代理中所有 LLM 调用生成的增量消息块 。要访问已解析工具调用的完整消息
+ Accessing completed messages 访问已完成的消息: 在某些情况下，已完成的消息不会反映在状态更新中。如果您有权访问代理内部机制，则可以使用自定义更新在流式传输期间访问这些消息。否则，您可以在流式传输循环中聚合消息块
+ Streaming with human-in-the-loop 人机交互式流媒体: 处理人机交互中断
+ Streaming from sub-agents 流式子代理: 当代理中存在多个 LLM 时，通常需要在生成消息时消除消息来源的歧义。为此，在创建每个代理时，需要为其指定一个 name 。然后，在 "messages" 模式下进行流式传输时，可以通过元数据中的 lc_agent_name 键获取此名称。

**Disable streaming  禁用流媒体**
在某些应用场景中，您可能需要禁用特定模型的单个令牌流式传输。这在以下情况下非常有用：
+ 利用多智能体系统来控制哪些智能体输出其内容
+ 将支持流媒体的模型与不支持流媒体的模型混合使用。
+ 部署到 LangSmith 平台 ，并希望阻止某些模型输出流式传输到客户端

**v2 streaming format  v2 流媒体格式**
将 version="v2" 传递给 stream() 或 astream() 可获得统一的输出格式。每个数据块都是一个 StreamPart 字典，包含 type 、 ns 和 data 三个键——无论流模式或模式数量如何，其结构都相同

v2 格式还改进了 invoke() 方法——它返回一个带有 .value 和 .interrupts 属性的 GraphOutput 对象，将状态与中断元数据清晰地分离


### 1.7 Structured output  结构化输出
结构化输出允许代理以特定且可预测的格式返回数据。

LangChain 的 create_agent 可以自动处理结构化输出。用户设置所需的结构化输出模式，当模型生成结构化数据时，这些数据会被捕获、验证，然后返回到 agent 状态的 'structured_response' 键中。

**Response format  回复格式**
使用 response_format 控制代理返回结构化数据的方式：
+ ToolStrategy[StructuredResponseT] ：使用工具调用进行结构化输出
+ ProviderStrategy[StructuredResponseT] ：使用提供商原生结构化输出
+ type[StructuredResponseT] ：模式类型 - 根据模型功能自动选择最佳策略
+ None ：未明确请求结构化输出

**Provider strategy  供应商策略**
要使用此策略，请配置 ProviderStrategy
```python
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    strict: bool | None = None
```

**Tool calling strategy  工具调用策略**
对于不支持原生结构化输出的模型，LangChain 使用工具调用来实现相同的结果。这适用于所有支持工具调用的模型（大多数现代模型）。
```python
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]
```

+ Custom tool message content 自定义工具消息内容: tool_message_content 参数允许您自定义生成结构化输出时显示在对话历史记录中的消息
+ Error handling  错误处理: 模型在通过工具调用生成结构化输出时可能会出错。LangChain 提供智能重试机制来自动处理这些错误。
  + 多重结构化输出错误: 当模型错误地调用多个结构化输出工具时，代理会在 ToolMessage 中提供错误反馈，并提示模型重试
  + Schema validation error  架构验证错误: 当结构化输出与预期模式不符时，代理会提供具体的错误反馈
  + Error handling strategies 错误处理策略: 可以使用 handle_errors 参数自定义错误处理方式
    + ```python 
      # Custom error message:  自定义错误信息
      ToolStrategy(
          schema=ProductRating,
          handle_errors="Please provide a valid rating between 1-5 and include a comment.")
      
      # Handle specific exceptions only: 仅处理特定异常情况
      ToolStrategy(
          schema=ProductRating,
          handle_errors=ValueError  # Only retry on ValueError, raise others)
      
      # Handle multiple exception types:处理多种异常类型
      ToolStrategy(
          schema=ProductRating,
          handle_errors=(ValueError, TypeError)  # Retry on ValueError and TypeError)
      
      # No error handling:  无错误处理
      response_format = ToolStrategy(
          schema=ProductRating,
          handle_errors=False  # All errors raised)
      ```

## 2 middleware 中间件
中间件提供了一种更严格控制代理内部发生事情的方法。中间件适用于以下用途: 
+ 通过日志记录、分析和调试跟踪代理行为。
+ 转换prompts、 工具选择和输出格式。
+ 增加了重试 、 后备和提前终止逻辑。
+ 应用速率限制 、保护栏和个人身份识别（PII）检测 。

通过传递给 create_agent 添加中间件
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gtp-4.1",
    tools=[...],
    middleware=[
        SummarizationMiddleware(...),
        HumanInTheLoopMiddleware(...)
    ]
)
```

### 2.1 Overview 概述

**The agent loop  智能体循环**
核心代理循环包括调用模型，让模型选择执行工具，然后在调用工具不复存在时完成

中间件在每一步之前和之后都暴露了钩子

**Additional resources  附加资源**
+ 内置中间件
+ 自定义中间件

### 2.2 内置中间件
LangChain 和 Deep Agents 为常见场景提供了预构建的中间件

**与提供者无关的中间件**
+ Summarization  摘要: 当接近token限制时，自动总结历史对话
+ Human-in-the-loop  人机参与: 暂停执行以供人工批准工具调用。
+ Model call limit  模型呼叫限制: 限制模型调用次数，以防止过高成本。
+ Tool call limit  工具调用限制: 通过限制呼叫次数来控制工具执行。
+ Model fallback  模型的备选: 当主模式失败时，会自动回退到其他模式。
+ PII detection  PII 检测: 检测并处理个人身份信息（PII）。
+ To-do list  待办事项列表: 为客服人员配备任务规划和跟踪能力。
+ LLM tool selector  LLM 工具选择器: 在调用主模型之前，先用 LLM 选择相关工具。
+ Tool retry  工具重试: 用指数回撤自动重试失败的工具调用。
+ Model retry  模型重试: 自动用指数退回方式重试失败的模型调用。
+ LLM tool emulator  LLM 工具仿真器: 用 LLM 模拟工具执行以进行测试。
+ Context editing  上下文编辑: 通过修剪或清理工具使用来管理对话上下文。
+ Shell tool  壳体工具: 向代理开放一个持久的壳会话以执行命令。
+ File search  文件搜索: 在文件系统文件上提供 Glob 和 Grep 搜索工具。
+ Filesystem  文件系统: 为代理提供存储上下文和长期记忆的文件系统。
+ Subagent  副代理人: 增加生成子代理人的能力。

**Summarization  摘要**
当接近token限制时自动总结对话历史，保留近期消息同时压缩旧上下文。摘要适用于以下情况:
+ 长时间的对话超出上下文窗口。
+ 多回合对话，历史悠久。
+ 在保持完整对话上下文的重要应用中。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)
```

**Human-in-the-loop  人机参与**
在执行工具调用前，暂停执行，以便人工批准、编辑或拒绝。 人机介入有助于以下用途:
+ 需要人工批准的高风险操作（例如数据库写入、金融交易）。
+ 必须有人监督的合规工作流程。
+ 长期对话，人工反馈引导经纪人。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

def your_read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def your_send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model="gpt-4.1",
    tools=[your_read_email_tool, your_send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "your_send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "your_read_email_tool": False
            }
        )
    ]
)
```

**Model call limit  模型调用限制**
限制模型调用次数，以防止无限循环或过高成本。模型调用限制适用于以下情况：
+ 防止失控代理调用过多 API。
+ 对生产部署实施成本控制。
+ 在特定调用预算内测试agent行为。

**Tool call limit  工具调用限制**
通过限制工具调用次数来控制代理执行，无论是全局调用所有工具还是针对特定工具。工具调用限制适用于以下情况: 
+ 防止对昂贵外部 API 的过度调用。
+ 限制网络搜索或数据库查询。
+ 对特定工具使用强制执行速率限制。
+ 防止失控的代理循环。

**Model fallback  模型的备选**
当主模型失败时，自动回退到其他模型。模型的备援适用于以下情况:
+ 构建能够处理模型故障的韧性代理。
+ 通过回归更便宜的模型来优化成本。
+ OpenAI、Anthropic 等平台的提供者冗余。

**PII detection  PII 检测**
利用可配置策略检测并处理对话中的个人身份信息（PII）。PII 检测适用于以下情况:
+ 医疗保健和金融应用，要求合规。
+ 需要清理日志的客服人员。
+ 任何处理敏感用户数据的应用程序。















使用结构化输出时，不支持预绑定模型（已调用 bind_tools 的模型）。如果需要使用结构化输出进行动态模型选择，请确保传递给中间件的模型未预先绑定。
使用wrap_model_call实现


