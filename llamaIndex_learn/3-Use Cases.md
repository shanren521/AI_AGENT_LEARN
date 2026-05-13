# 1 Agents代理

一个“代理”是一个自动推理和决策引擎。它接收用户输入/查询，并可以做出内部决策以执行该查询，从而返回正确结果。代理的关键组件可以包括但不限于:
+ 将复杂问题分解成更小的问题
+ 选择使用外部工具 + 构思调用工具的参数
+ 规划一组任务
+ 将先前完成的任务存储在内存模块中

用例:
+ 代理式 RAG：在您的数据上构建一个上下文增强型研究助手，不仅能回答简单问题，还能处理复杂的研究任务。
+ 报告生成：使用多代理研究员+作家工作流程+LlamaParse 生成多模态报告。笔记本。
+ 客户支持：查看构建多代理礼宾员的工作流程的入门模板。
+ 生产力助手：构建一个可以在电子邮件、日历等常见工作流工具上操作的代理。查看我们的 GSuite 代理教程。
+ 编程助手：构建一个可以在代码上操作的代理。查看我们的代码解释器教程。

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.core.agent.workflow import AgentWorkflow


def multiply(a: int, b: int) -> int:
    """Useful for multiplying two numbers."""
    return a * b

memory = ChatMemoryBuffer(token_limit=8192)
agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-3.5-turbo-0613"),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)

# 多模态系统
msg = ChatMessage(
    role="user",
    blocks=[
        TextBlock(text="Follow what the image says."),
        ImageBlock(path='./images/xxx.png')
    ]
)

async def main():
    response = await agent.arun("What is 2 multiplied by 2?", memory=memory)
    print(response)

# 多智能体系统
multi_agent = AgentWorkflow(agents=[FunctionAgent(), FunctionAgent()])
    
if __name__ == "__main__":
    await agent.run(msg)
    asyncio.run(main())
```

# 2 Chatbots 聊天机器人

# 3 Structured Data Extraction 结构化数据提取

# 4 Fine-tuning 微调

# 5 Querying Graphs 查询图

# 6 Multi-Modal 多模态

# 7 Prompting 提示

# 8 Question-Answering RAG 问答RAG

# 9 Querying CSVS 查询CSV文件

# 10 解析表格和图表

# 11 文本转SQL




