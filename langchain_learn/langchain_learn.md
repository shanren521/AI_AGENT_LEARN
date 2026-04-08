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

+ 短期记忆: state 访问状态，Command更新代理的状态
+ Context: 上下文提供在调用时传递的不可变配置数据。可用于用户 ID、会话详细信息或在对话过程中不应更改的应用程序特定设置。runtime.context



使用结构化输出时，不支持预绑定模型（已调用 bind_tools 的模型）。如果需要使用结构化输出进行动态模型选择，请确保传递给中间件的模型未预先绑定。
使用wrap_model_call实现
















