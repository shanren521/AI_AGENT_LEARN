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




使用结构化输出时，不支持预绑定模型（已调用 bind_tools 的模型）。如果需要使用结构化输出进行动态模型选择，请确保传递给中间件的模型未预先绑定。
使用wrap_model_call实现
















