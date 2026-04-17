# langchain 学习笔记

# 1 核心组件

## 1.1 Agents

智能体将语言模型与工具相结合，创建能够对任务进行推理、决定使用哪些工具并迭代地寻求解决方案的系统。
create_agent提供了一个可用于生产环境的代理实现

## 1.2 Models

### 1.2.1基本用法

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

### 1.2.2 Advanced topics高级主题

**Model profiles**

LangChain 聊天模型可以通过 profile 属性公开一个支持的功能和特性字典

**多模态**

某些模型可以处理并返回非文本数据，例如图像、音频和视频。也可以返回多模态数据作为响应的一部分.

**推理**

通过判断模型返回的字段是否有reasoning，可以将这个推理过程展现出来，以便更好地理解模型是如何得出最终答案的。

## 1.3 Messages
在 LangChain 中，消息是模型的基本上下文单元。代表模型的输入和输出，携带与 LLM 交互时表示对话状态所需的内容和元数据。

**包含的对象有**
+ role: system\user
+ content: 文本、图像、音频、文档等
+ metadata：可选字段，例如响应消息、消息ID和令牌使用情况

### 1.3.1 基本用法

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

## 1.4 Tools 工具

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

## 1.5 Short-term memory 短期记忆

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


## 1.6 Streaming 流媒体

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


## 1.7 Structured output  结构化输出
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

# 2 middleware 中间件
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

## 2.1 Overview 概述

**The agent loop  智能体循环**

核心代理循环包括调用模型，让模型选择执行工具，然后在调用工具不复存在时完成

中间件在每一步之前和之后都暴露了钩子

**Additional resources  附加资源**
+ 内置中间件
+ 自定义中间件

## 2.2 内置中间件
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

**PII detection  PII 检测****PII detection  PII 检测**

利用可配置策略检测并处理对话中的个人身份信息（PII）。PII 检测适用于以下情况:
+ 医疗保健和金融应用，要求合规。
+ 需要清理日志的客服人员。
+ 任何处理敏感用户数据的应用程序。
+ **Custom PII types  自定义 PII 类型**
  + 您可以通过提供 detector 参数来创建自定义 PII 类型。这样，您可以检测超出内置类型范围的特定于您使用场景的模式。
  + 自定义检测器的三种方法：
    + 正则表达式模式字符串 - 简单模式匹配
    + 自定义功能 - 带有验证功能的复杂检测逻辑
    + 自定义检测器函数签名

**To-do list  待办事项清单**

为代理配备任务规划和跟踪功能，以处理复杂的多步骤任务。待办事项清单在以下方面非常有用：
+ 需要协调使用多种工具的复杂多步骤任务。
+ 需要长期关注项目进展情况的项目。

**LLM tool selector  LLM 工具选择器**

使用 LLM 在调用主模型之前智能地选择相关工具。LLM 工具选择器可用于以下用途：
+ 拥有众多工具（10 个以上）的代理，但大多数工具与每次查询都不相关。
+ 通过过滤不相关的工具来减少令牌使用量。
+ 提高模型聚焦性和准确性。

该中间件使用结构化输出向 LLM 询问哪些工具与当前查询最相关。结构化输出模式定义了可用工具的名称和描述。模型提供者通常会在后台将这些结构化输出信息添加到系统提示中。

**Tool retry  工具重试**

使用可配置的指数退避算法自动重试失败的工具调用。工具重试功能适用于以下情况：
+ 处理外部 API 调用中的瞬态故障。
+ 提高网络依赖型工具的可靠性。
+ 构建能够优雅地处理临时错误的弹性代理。

**LLM tool emulator  LLM 工具模拟器**

使用 LLM 模拟工具执行过程进行测试，用 AI 生成的响应替换实际的工具调用。LLM 工具模拟器可用于以下用途：
+ 无需使用实际工具即可测试代理行为。
+ 当外部工具不可用或成本高昂时，开发代理。
+ 在实际工具实施之前，先对代理工作流程进行原型设计。

**Context editing  上下文编辑**

当达到令牌限制时，通过清除较旧的工具调用输出来管理对话上下文，同时保留最近的结果。这有助于在包含大量工具调用的长时间对话中保持上下文窗口的可控性。上下文编辑在以下情况下非常有用：
+ 长时间的对话，期间调用了许多超出令牌限制的工具。
+ 通过移除不再相关的旧工具输出来降低代币成本.
+ 仅保留上下文中最新的 N 个工具结果

**Shell tool  Shell 工具**

向代理公开持久化的 shell 会话以执行命令。Shell 工具中间件可用于以下用途：
+ 需要执行系统命令的agent
+ 开发和部署自动化任务
+ 测试和验证工作流程
+ 文件系统操作和脚本执行

**File search文件搜索**

在文件系统上提供 Glob 和 Grep 搜索工具。文件搜索中间件可用于以下用途：
+ 代码探索与分析
+ 按名称模式查找文件
+ 使用正则表达式搜索代码内容
+ 需要进行文件发现的大型代码库

**Filesystem middleware  文件系统中间件**

上下文工程是构建高效智能体的主要挑战之一。当使用返回可变长度结果的工具（例如 web_search 和 RAG）时，这一点尤其困难，因为过长的结果会迅速填满上下文窗口。

Deep Agents 的 FilesystemMiddleware 提供了四个用于与短期记忆和长期记忆交互的工具：
+ ls ：列出文件系统中的文件
+ read_file ：读取整个文件或文件中指定数量的行
+ write_file ：向文件系统写入新文件
+ edit_file ：编辑文件系统中已存在的文件

**Short-term vs long-term filesystem 短期文件系统与长期文件系统**

默认情况下，这些工具会将数据写入图状态中的本地“文件系统”。要启用跨线程的持久存储，请配置一个 CompositeBackend ，将特定路径（例如 /memories/ ）路由到 StoreBackend 。

当您为 /memories/ 配置带有 StoreBackend ` CompositeBackend 时，所有以 `/memories/` 为前缀的文件都会保存到持久存储中，并在不同的线程之间保留。不带此前缀的文件则保留在临时状态存储中。


**Subagent  次级代理人**

将任务交给子代理可以隔离上下文，保持主（主管）代理的上下文窗口干净，同时还能深入执行任务。

Deep Agents 的子代理中间件允许您通过 task 工具提供子代理。

子代理由名称 、 描述 、 系统提示和工具定义。您还可以为子代理提供自定义模型或额外的中间件 。当您希望为子代理提供一个与主代理共享的额外状态键时，这将特别有用。

## 2.3 Custom middleware 自定义中间件

通过在代理执行流程的特定点运行钩子来构建自定义中间件。

### 2.3.1 Hooks 钩子

中间件提供了两种拦截代理执行的钩子：

#### 2.3.1.1 Node-style hooks  Node 风格的钩子

按顺序在特定执行点运行。用于日志记录、验证和状态更新。
选择中间件所需的钩子。您可以选择节点式钩子或包装式钩子。

before_agent\before_model\after_model\after_agent

#### 2.3.1.2 Wrap-style hooks  缠绕式钩子

拦截执行过程，并在调用处理程序时进行控制。可用于重试、缓存和转换。

wrap_model_call\wrap_tool_call

### 2.3.2 State updates 状态更新

节点式和包装式钩子都可以更新代理状态。它们的机制有所不同：

#### 2.3.2.1 node-style hooks: 

直接返回一个字典。该字典会使用图的 reducer 应用于 agent 状态。

#### 2.3.2.2 wrap-style hooks: 
对于模型调用，返回一个包含 Command 的 ExtendedModelResponse ，以便在模型响应中注入状态更新。对于工具调用，直接返回一个 ` Command 。
当您需要根据模型或工具调用期间运行的逻辑（例如摘要触发点、使用情况元数据或从请求或响应计算出的自定义字段）来跟踪或更新状态时，请使用这些钩子。

+ Composition with multiple middleware 包含多个中间件的组合: 当多个中间件层返回 ExtendedModelResponse 时，它​​们的命令组成如下：
  + 命令通过 reducer 执行： 每个 Command 都会成为一次独立的状态更新。对于消息而言，这意味着它们是累加的。
  + 冲突时，外层优先： 对于非 reducer 状态字段，命令先从内层应用，再从外层应用。最外层中间件的值在键冲突时优先。
  + 可重试： 如果外部中间件实现了可能导致多次调用 handler() 的逻辑（例如，重试逻辑），则会丢弃先前调用的命令。

### 2.3.3 Create middleware 创建中间件

#### 2.3.3.1 Decorator-based middleware 基于装饰器的中间件
适用于单钩子中间件，快速简便。使用装饰器封装各个函数。

可供选择的装饰器:
+ node-style: 
  + @before_agent 在agent程序完成前运行(每次调用运行一次)
  + @before_model 在每次模型响应之前运行
  + @after_model 在每次模型响应后运行
  + @after_agent 在agent程序完成后运行(每次调用运行一次)
+ wrap-style:
  + @wrap_model_call 使用自定义逻辑包装每个模型调用
  + @wrap_tool_call 使用自定义逻辑包装每个工具调用
+ convenience:
  + @dynamic_prompt 生成动态系统prompt

#### 2.3.3.2 Class-based middleware  基于类的中间件
对于具有多个钩子或配置的复杂中间件，类的功能更强大。当需要为同一个钩子定义同步和异步实现，或者想要在单个中间件中组合多个钩子时，请使用类。

### 2.3.4 Custom state schema  自定义状态模式
如果您的中间件需要跨钩子跟踪状态，则中间件可以使用自定义属性扩展代理的状态。这使得中间件能够：
+ 跟踪执行过程中的状态 ：维护在代理的整个执行生命周期中保持不变的计数器、标志或其他值
+ 在钩子之间共享数据 ：将信息从 before_model 传递到 after_model ，或在不同的中间件实例之间传递。
+ 实现横切关注点 ：在不修改核心代理逻辑的情况下，添加诸如速率限制、使用情况跟踪、用户上下文或审计日志记录等功能。
+ 做出条件决策 ：利用累积状态来确定是继续执行、跳转到不同的节点，还是动态地修改行为。

### 2.3.5 Execution order 执行顺序
使用多个中间件时，需要了解它们的执行方式，关键规则:
+ before_* 从第一个到最后一个
+ after_* 从后到前
+ wrap_* 嵌套式(第一个中间件包裹其他中间件)

### 2.3.6 Agent jumps 代理跳转
要提前退出中间件，请返回一个包含 jump_to 字典：
+ 'end': 跳转到代理执行的末尾(或第一个after_agent钩子)
+ 'tools': 跳转到工具节点
+ 'model': 跳转到模型节点(或第一个before_model钩子)

### 2.3.7 Best practices 最佳实践

+ 保持中间件的专注——每个中间件都应该做好一件事。
+ 优雅地处理错误——不要让中间件错误导致代理崩溃。
+ 使用合适的钩子类型 ：
  + node-style 用于顺序逻辑(日志记录、验证)的节点式编程
  + wrap-style 控制流(重试、回退、缓存)的包装式
+ 清楚地记录所有自定义状态属性
+ 在集成之前，对中间件进行独立的单元测试。
+ 考虑执行顺序——将关键中间件放在列表的最前面
+ 尽可能使用内置中间件

### 2.3.8 Examples 示例

#### 2.3.8.1 Dynamic prompt 动态prompt
在运行时动态修改系统提示符，以便在每次模型调用之前注入上下文信息、用户特定指令或其他信息。这是最常见的中间件用例之一。

使用 ModelRequest 上的 system_message 字段来读取和修改系统提示。它包含一个 SystemMessage 对象（即使代理是在创建时使用字符串 system_prompt 创建的）。

#### 2.3.8.2 Dynamic model selection  动态模型选择

#### 2.3.8.3 Dynamically selecting tools 动态选择工具
在运行时选择相关工具以提高性能和准确性。

好处：
+ 更简短的提示 - 仅显示相关工具，降低复杂性
+ 更准确的选择 - 模型能从更少的选项中做出正确的选择。
+ 权限控制 - 根据用户访问权限动态筛选工具

#### 2.3.8.4 Tool call monitoring  工具调用监控

参考代码

#### 2.3.8.5 Prompt caching (Anthropic) 提示缓存（Anthropic）

在使用 Anthropic 模型时，使用带有缓存控制指令的结构化内容块来缓存大型系统提示: 


# 3 Advanced usage 高级用法

护栏通过在代理执行的关键节点验证和过滤内容，帮助您构建安全、合规的人工智能应用。他们能够检测敏感信息，执行内容政策，验证输出，并在不安全行为引发问题前预防。

常见的使用场景包括:
+ 防止PII泄露
+ 检测和阻止即时注入攻击
+ 屏蔽不当或有害内容
+ 执行业务规则和合规要求
+ 验证输出质量和准确性

## 3.1 防护机制

### 3.1.1 Built-in guardrails  内置防护机制

#### 3.1.1.1 PII detection PII检测

LangChain 内置中间件用于检测和处理对话中的个人身份信息（PII）。该中间件可以检测常见的个人身份信息类型，如电子邮件、信用卡、IP 地址等。

PII 检测中间件对于医疗和金融应用等符合合规要求的案例、需要清理日志的客户服务人员，以及任何处理敏感用户数据的应用都很有帮助。

PII 中间件支持多种处理检测到的 PII 策略:
+ redact 编辑: 请用 [REDACTED_{PII_TYPE}] 替换
+ mask 掩码: 部分冷门(例如，后四位数字)
+ hash 哈希: 用确定性哈希替换
+ block 阻断: 检测到异常时抛出异常

#### 3.1.1.2 Human-in-the-loop  人机参与

LangChain 提供了内置中间件，要求执行敏感操作前获得人工批准。这是处理高风险决策最有效的护栏之一。

人机环绕中间件适用于金融交易和转账、删除或修改生产数据、向外部发送通信以及任何对业务有重大影响的操作。


### 3.1.2 Custom guardrails  自定义防护机制

对于更复杂的防护措施，您可以创建自定义中间件，使其在代理执行之前或之后运行。这样，您就可以完全控制验证逻辑、内容过滤和安全检查。

#### 3.1.2.1 Before agent guardrails  在代理之前防护

使用“before agent”钩子在每次调用开始时验证请求一次。这对于会话级检查（例如身份验证、速率限制或在任何处理开始之前阻止不当请求）非常有用。

#### 3.1.2.2 After agent guardrails  在代理之后防护

使用“代理后”钩子在返回给用户之前对最终输出进行一次验证。这对于基于模型的安全检查、质量验证或对完整代理响应进行最终合规性扫描非常有用。

### 3.1.2.3 Combine multiple guardrails 组合多个防护

您可以通过将多个防护措施添加到中间件数组中来堆叠它们。它们按顺序执行，从而允许您构建分层保护

## 3.2 Runtime 运行时

LangChain 的 create_agent 底层运行在 LangGraph 的运行时环境中。

LangGraph 公开了一个 Runtime 对象，其中包含以下信息：
+ Context: 静态信息，例如用户 ID、数据库连接或其他代理调用依赖项。
+ Store: 用于长期记忆的 BaseStore 实例
+ Stream writer: 用于通过 "custom" 流模式传输信息的对象
+ Execution info: 当前执行的身份和重试信息（线程 ID、运行 ID、尝试次数）
+ Server info: 在 LangGraph Server 上运行时的服务器特定元数据（助手 ID、图 ID、已认证用户）

### 3.2.1 Access 使用权

使用 create_agent 创建代理时，您可以指定 context_schema 来定义存储在代理 Runtime 中的 context 结构。

调用代理时，请传递包含运行相关配置的 context 参数

#### 3.2.1.1 Inside tools  内部工具

您可以通过访问工具内部的运行时信息来执行以下操作：
+ Access the context  获取上下文
+ 读取或写入长期记忆
+ 写入自定义流 （例如，工具进度/更新）

使用 ToolRuntime 参数可以访问工具内部的 Runtime 对象。

#### 3.2.1.2 Execution info and server info inside tools 工具内部包含执行信息和服务器信息

在 LangGraph Server 上运行时，可通过 runtime.execution_info 访问执行标识（线程 ID、运行 ID），并通过 runtime.server_info 访问服务器特定元数据（助手 ID、已验证用户）

#### 3.2.1.3 Inside middleware  中间件内部

您可以访问中间件中的运行时信息，以根据用户上下文创建动态提示、修改消息或控制代理行为。

在节点式钩子中，可以使用 Runtime 参数访问 Runtime 对象。对于包装式钩子 ， Runtime 对象可通过 ModelRequest 参数访问。

#### 3.2.1.4 Execution info and server info inside middleware 中间件内部的执行信息和服务器信息

中间件钩子也可以访问 runtime.execution_info 和 runtime.server_info

## 3.3 Context engineering in agents 代理中的上下文工程

### 3.3.1 Model context  模型上下文

控制每次模型调用的内容——指令、可用工具、使用的模型以及输出格式。这些决策直接影响可靠性和成本。

所有类型的模型上下文都可以从状态 （短期记忆）、 存储 （长期记忆）或运行时上下文 （静态配置）中获取信息。

#### 3.3.1.1 System Prompt系统提示

系统提示决定了语言学习模型（LLM）的行为和功能。不同的用户、不同的情境或不同的对话阶段需要不同的指令。优秀的智能体能够利用记忆、偏好和配置信息，根据对话的当前状态提供正确的指令。

#### 3.3.1.2 Message 消息

发送给 LLM 的提示信息由消息组成。管理消息内容至关重要，以确保 LLM 拥有正确的信息以便做出有效响应。

#### 3.3.1.3 Tools 工具

工具使模型能够与数据库、API 和外部系统进行交互。工具的定义和选择方式直接影响模型能否有效地完成任务。

##### 3.3.1.3.1 Defining tools  定义工具

每个工具都需要清晰的名称、描述、参数名称和参数描述。这些不仅仅是元数据——它们指导模型判断何时以及如何使用该工具。

##### 3.3.1.3.2 Selecting tools 选择工具

并非所有工具都适用于所有情况。工具过多可能会使模型不堪重负（上下文过载）并增加错误；工具过少则会限制功能。动态工具选择会根据身份验证状态、用户权限、功能标志或对话阶段来调整可用工具集。

#### 3.3.1.4 Model 模型

不同的模型各有优势、成本和适用范围。请根据当前任务选择合适的模型，该模型在代理运行过程中可能会发生变化。

#### 3.3.1.5 Response format  回复格式

结构化输出将非结构化文本转换为经过验证的结构化数据。提取特定字段或向下游系统返回数据时，自由格式文本是不够的。

工作原理： 当您提供响应格式的模式时，模型的最终响应保证符合该模式。代理会运行模型/工具调用循环，直到模型完成所有工具调用，然后将最终响应强制转换为提供的格式。

##### 3.3.1.5.1 Defining formats  定义格式

模式定义指导模型运行。字段名称、类型和描述精确地规定了输出应遵循的格式。

##### 3.3.1.5.2 Selecting formats 选择格式

动态响应格式选择会根据用户偏好、对话阶段或角色调整方案——早期返回简单格式，随着复杂性的增加返回详细格式。

### 3.3.2 Tool context  工具上下文

工具的特殊之处在于它们既能读取也能写入上下文。

最基本的情况是，当工具执行时，它会接收 LLM 的请求参数并返回一条工具消息。工具执行其工作并产生结果。

工具还可以获取模型的重要信息，使模型能够执行和完成任务。

#### 3.3.2.1 Reads 阅读

大多数实际应用工具需要的不仅仅是 LLM 的参数。它们还需要用户 ID 用于数据库查询，API 密钥用于外部服务，或者当前会话状态来做出决策。这些工具会从状态、存储和运行时上下文中读取信息。

#### 3.3.2.2 Writes 写

工具的结果可以帮助智能体完成特定任务。工具既可以直接将结果返回给模型，也可以更新智能体的内存，以便为后续步骤提供重要的上下文信息。

### 3.3.3 Life-cycle context  生命周期背景

控制核心代理步骤之间发生的事情——拦截数据流以实现诸如摘要、防护措施和日志记录等横切关注点。

#### 3.3.3.1 Example: Summarization  示例：总结

最常见的生命周期模式之一是自动压缩过长的对话历史记录。与模型上下文中所示的临时消息修剪不同，摘要功能会持续更新状态 ——用保存下来的摘要永久替换旧消息，供所有后续对话使用。

## 3.4 模型上下文协议（MCP）

### 3.4.2 Quickstart 快速开始

langchain-mcp-adapters 使代理能够使用在单个或多个 MCP 服务器上定义的工具。

### 3.4.2 Custom servers 自定义服务器

要创建自定义的 MCP 服务器，使用 FastMCP 库

要使用 MCP 工具服务器测试您的代理，请参考代码

### 3.4.3 Transports 传输方式

MCP 支持不同的客户端-服务器通信传输机制。

#### 3.4.3.1 HTTP

http 传输（也称为 streamable-http ）使用 HTTP 请求进行客户端-服务器通信。有关更多详细信息，请参阅 MCP HTTP 传输规范。

+ 1.Passing headers  传递信息头
  + 当通过 HTTP 连接到 MCP 服务器时，您可以使用连接配置中的 headers 字段包含自定义标头（例如，用于身份验证或跟踪）。此功能支持 sse （已由 MCP 规范弃用）和 streamable_http 传输。
+ 2.Authentication 认证
  + langchain-mcp-adapters 库在底层使用官方的 MCP SDK，这允许你通过实现 httpx.Auth 接口来提供自定义的认证机制。
+ 

#### 3.4.3.2 Stdio 标准输入输出

### 3.4.4 Stateful sessions 状态会话

### 3.4.5 核心功能

#### 3.4.5.1 Tools 工具

#### 3.4.5.2 Resource 资源

#### 3.4.5.3 Prompts 提示词

### 3.4.6 Advanced features 高阶功能

#### 3.4.6.1 Tool interceptors 工具拦截器

#### 3.4.6.2 Progress notifications 进度通知

#### 3.4.6.3 日志记录

#### 3.4.6.4 Elicitation 提取














使用结构化输出时，不支持预绑定模型（已调用 bind_tools 的模型）。如果需要使用结构化输出进行动态模型选择，请确保传递给中间件的模型未预先绑定。
使用wrap_model_call实现