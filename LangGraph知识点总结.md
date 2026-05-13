# LangGraph 面试知识点大全

## 1. LangGraph 是什么

LangGraph 是 LangChain 生态中的 Agent 工作流编排框架。

核心思想：

```text
使用 Graph（图）管理 Agent 的状态流转
```

适用于：

- AI Agent
- Multi-Agent
- Agentic RAG
- Tool Calling
- Human-in-the-loop
- 长任务工作流

---

# 2. LangGraph 与 LangChain 的区别

## LangChain

偏：

- Chain
- Prompt
- Tool
- Runnable
- RAG

特点：

```text
线性流程
```

---

## LangGraph

偏：

- 状态机
- Graph Workflow
- Agent Runtime

特点：

```text
非线性 + 状态驱动 + 支持循环
```

---

# 3. LangGraph 核心概念

## Graph

工作流图。

由：

- Node
- Edge
- State

组成。

---

## State

共享状态。

整个 Agent 工作流的数据中心。

示例：

```python
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: list
    user_input: str
    next_step: str
```

---

## Node

节点。

本质：

```text
一个函数
```

示例：

```python
def chatbot(state):
    return {
        "messages": state["messages"] + ["hello"]
    }
```

---

## Edge

节点之间的连接关系。

---

# 4. StateGraph

最常用 Graph。

```python
from langgraph.graph import StateGraph
```

特点：

- 支持复杂状态
- 支持条件路由
- 支持循环
- 企业级最常用

---

# 5. MessageGraph

适用于聊天场景。

默认状态：

```python
messages
```

适合：

- Chatbot
- Conversation Agent

---

# 6. START 与 END

```python
from langgraph.graph import START, END
```

---

## START

工作流入口。

---

## END

工作流结束。

---

# 7. add_node

添加节点。

```python
graph.add_node("chatbot", chatbot)
```

---

# 8. add_edge

添加普通边。

```python
graph.add_edge("a", "b")
```

含义：

```text
a 执行完 -> b
```

---

# 9. Conditional Edge（高频）

Agent 核心能力。

```python
graph.add_conditional_edges(
    "agent",
    router,
    {
        "tool": "tool_node",
        "end": END
    }
)
```

---

## router

```python
def router(state):
    if state["need_tool"]:
        return "tool"
    return "end"
```

---

# 10. compile

编译工作流。

```python
app = graph.compile()
```

本质：

```text
生成 Runnable Runtime
```

---

# 11. invoke

同步调用。

```python
result = app.invoke(input_state)
```

---

# 12. stream

流式执行。

```python
for event in app.stream(input_state):
    print(event)
```

优势：

- 实时输出
- Debug
- UI 展示

---

# 13. Reducer

状态合并机制。

常见：

```python
Annotated[list, add_messages]
```

作用：

```text
自动追加 messages
```

---

# 14. TypedDict

LangGraph 推荐使用。

原因：

- 类型检查
- IDE 提示
- 状态规范化

---

# 15. Checkpoint（高频）

用于保存执行状态。

作用：

- 中断恢复
- 长任务恢复
- Human-in-the-loop
- 持久化 Agent

---

# 16. MemorySaver

内存级 checkpoint。

```python
from langgraph.checkpoint.memory import MemorySaver
```

---

# 17. 数据库 Checkpoint

生产环境常用：

- SQLite
- Postgres

原因：

```text
服务重启后仍能恢复状态
```

---

# 18. Human-in-the-loop

人工介入。

典型场景：

- 审批
- 风控
- 人工确认

---

## interrupt

```python
interrupt(...)
```

作用：

```text
暂停工作流等待人工输入
```

---

# 19. Command

动态控制流程。

```python
Command(
    goto="tool_node"
)
```

作用：

```text
运行时动态跳转
```

---

# 20. Send

用于并行任务。

```python
Send("worker", data)
```

作用：

```text
fan-out 并发执行
```

---

# 21. Parallel Execution

LangGraph 支持：

```text
并行节点执行
```

适用于：

- 多工具调用
- 多检索源
- 多 Agent

---

# 22. ReAct Agent

经典 Agent 模式。

```text
Reason + Act
```

流程：

```text
思考 -> 调工具 -> 观察 -> 再思考
```

---

# 23. Tool Calling

Agent 调工具。

常见：

```python
from langgraph.prebuilt import ToolNode
```

---

# 24. Structured Output

结构化输出。

常见方案：

- Pydantic
- JSON Schema

作用：

```text
避免 LLM 输出不可解析
```

---

# 25. RAG + LangGraph

LangGraph 非常适合：

- Query Rewrite
- Retrieval
- Rerank
- Retry
- Hallucination Check

因为：

```text
RAG 本质是复杂状态流
```

---

# 26. Agentic RAG

Agent 自主决定：

- 是否检索
- 是否重写 Query
- 是否再次检索

---

# 27. Reflection

Agent 自我反思。

流程：

```text
生成答案
↓
自检
↓
修复答案
```

---

# 28. Retry

失败重试。

实现：

```text
失败 -> retry node
```

---

# 29. Error Handling

常见：

- try-except
- fallback node
- retry

---

# 30. Streaming

类型：

- token streaming
- event streaming
- state streaming

---

# 31. Observability

Agent 可观测性。

需要：

- trace
- log
- state history
- token usage

---

# 32. LangSmith

官方调试平台。

功能：

- Trace
- Prompt 调试
- Token 分析
- State 查看

---

# 33. 短期记忆 vs 长期记忆

## 短期记忆

当前 state。

---

## 长期记忆

向量数据库：

- 用户画像
- 历史行为

---

# 34. Async 支持

LangGraph 支持：

```python
async def
```

原因：

```text
LLM 是 IO 密集型
```

---

# 35. Runnable

LangGraph 基于：

```text
Runnable Interface
```

---

# 36. FSM（有限状态机）

LangGraph 本质：

```text
FSM + LLM
```

---

# 37. DAG vs Graph

## DAG

不能循环。

---

## LangGraph

支持循环：

```text
Agent 可以持续思考
```

---

# 38. 为什么 Graph 更适合 Agent

因为 Agent 本质：

```text
思考
↓
行动
↓
观察
↓
继续思考
```

属于循环系统。

---

# 39. Multi-Agent

LangGraph 非常适合多 Agent。

---

## Supervisor 模式

```text
Supervisor
    ↓
多个 Worker
```

---

## Router 模式

任务路由。

---

## Planner-Executor

规划与执行分离。

---

# 40. LangGraph vs AutoGen

## LangGraph

优点：

- 可控
- 状态清晰
- 企业级

---

## AutoGen

优点：

- 多 Agent 对话方便

缺点：

- 可控性差

---

# 41. LangGraph vs CrewAI

## CrewAI

偏角色协作。

---

## LangGraph

偏底层工作流编排。

---

# 42. LangGraph vs Airflow

## Airflow

传统 DAG。

---

## LangGraph

AI Agent Graph。

---

# 43. MCP（高频）

MCP：

```text
Model Context Protocol
```

作用：

```text
统一 Tool 协议
```

---

## MCP 与 LangGraph 的关系

MCP：

```text
工具标准
```

LangGraph：

```text
Agent 编排框架
```

---

# 44. 面试高频问题

## Q1：LangGraph 与 LangChain 区别？

LangChain 偏线性调用。

LangGraph 偏状态驱动工作流。

---

## Q2：为什么 Agent 适合 Graph？

因为 Agent 是循环系统。

---

## Q3：State 的作用？

保存：

- 上下文
- 工具结果
- 中间推理

---

## Q4：Conditional Edge 的作用？

实现动态路由。

---

## Q5：为什么需要 checkpoint？

实现：

- 恢复
- 持久化
- 人工介入

---

## Q6：LangGraph 如何实现循环？

Node 指回前置 Node。

---

## Q7：如何避免死循环？

- 最大迭代次数
- timeout
- 条件终止

---

## Q8：如何实现 Human-in-the-loop？

使用：

```python
interrupt()
```

---

## Q9：LangGraph 最大优势？

- 状态管理
- 条件路由
- 多 Agent
- checkpoint

---

## Q10：LangGraph 最大挑战？

- 状态复杂
- 调试困难
- Prompt 不稳定

---

# 45. 企业级最佳实践

## 状态最小化

避免：

```text
把全部历史放入 state
```

---

## 节点单一职责

一个 node 做一件事。

---

## Tool 解耦

Tool 独立封装。

---

## Structured Output

必须结构化输出。

---

## 增加 Fallback

避免 Agent 崩溃。

---

## 使用 Checkpoint

支持恢复。

---

# 46. 高频代码模板

## 基础模板

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: list

def chatbot(state):
    return {
        "messages": state["messages"] + ["hello"]
    }

graph = StateGraph(State)

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

result = app.invoke({
    "messages": []
})

print(result)
```

---

# 47. 条件路由模板

```python
def router(state):
    if state["need_tool"]:
        return "tool"

    return "end"

graph.add_conditional_edges(
    "agent",
    router,
    {
        "tool": "tool_node",
        "end": END
    }
)
```

---

# 48. ToolNode 模板

```python
from langgraph.prebuilt import ToolNode
```

---

# 49. Checkpoint 模板

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

app = graph.compile(
    checkpointer=memory
)
```

---

# 50. 一句话总结

```text
LangGraph = 面向 AI Agent 的状态机工作流框架
```

核心能力：

```text
状态管理 + 条件路由 + 循环 + 多 Agent + 持久化
```

# 51. LangGraph 最大挑战是什么？
1. 状态设计复杂
2. 调试困难
3. Prompt 不稳定
4. 多 Agent 协调复杂
5. Token 成本高

# 52. 如何优化 LangGraph Agent？
+ Prompt 优化
+ Tool 限制
+ 减少循环
+ Cache
+ Rerank
+ 小模型路由

# 53. 如何设计一个 Agent 系统？

建议回答结构：

1. 用户输入层
+ API
+ WebSocket
2. Agent 编排层
LangGraph：

+ state
+ routing
+ workflow
3. Tool 层
+ search
+ db
+ code executor
4. Memory 层
+ checkpoint
+ vector db
5. Observability
+ LangSmith
+ logging