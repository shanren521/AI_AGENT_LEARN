1. 与LangChain的区别:
+ LangChain: 偏向于Agent/workflow编排
+ LlamaIndex: 偏向于数据索引 + 检索

2. RAG核心原理:
+ 向量检索 -> 找到相关文档 -> 将文档和问题进行拼接为新的prompt -> 模型生成答案

3. LlamaIndex核心架构(五步流程):
+ Load(加载): SimpleDirectoryReader等数据加载器
+ Parse(解析): 文档分割、节点提取(SentenceSplitter, HierarchicalNodeParser)
+ Index(索引): VectorStoreIndex构建向量索引
+ Store(存储): StorageContext持久化到本地或向量数据库
+ Query(查询): QueryEngine/ChatEngine执行检索和生成

4. LlamaIndex四大基础类(面试必考):
+ Document: 原始数据容器,包含text(文本内容)、metadata(元数据)、doc_id(文档ID)
+ Node: 最小处理单元,是Document被切分后的chunk,包含text、metadata、relationships(节点关系)
+ Index: 数据结构,用于存储和检索节点,核心方法from_documents()、as_query_engine()、insert()
+ Response: 查询结果,包含response(答案文本)、source_nodes(溯源节点)、metadata(元数据)

5. 文档加载与处理:
+ SimpleDirectoryReader: 自动检测.txt/.pdf/.docx/.md等格式
  - 参数: input_dir、input_files、recursive=True、required_exts、file_metadata
+ SentenceSplitter: 基于句子边界的智能分块
  - chunk_size: 每块最大长度(默认1024 tokens)
  - chunk_overlap: 块间重叠(默认20 tokens),保持上下文连续性
  - separator、paragraph_separator、include_metadata、include_prev_next_rel
+ HierarchicalNodeParser: 层级节点解析器,创建父子节点关系
  - 支持多粒度切分(如父块1536,子块512)
  - get_leaf_nodes()获取最小粒度节点

6. 索引类型与构建:
+ VectorStoreIndex: 最常用的向量索引,将文档转为向量嵌入
  - from_documents(documents, transformations, storage_context, show_progress)
  - docStoreStrategy=DocStoreStrategy.UPSERTS避免重复
+ 其他索引: ListIndex、TreeIndex、KeywordTableIndex等
+ Settings全局配置: Settings.llm、Settings.embed_model统一管理LLM和Embedding模型

7. Embedding嵌入模型:
+ 作用: 将文本转换为数值向量表示,语义相似的文本向量距离更近
+ 常用模型: OpenAIEmbedding(text-embedding-ada-002/text-embedding-3-small)、HuggingFaceEmbedding(BAAI/bge-base-en-v1.5)
+ 配置方式: Settings.embed_model或直接传入index构造函数

8. 检索策略(面试高频):
+ similarity_top_k: 返回最相似的K个节点(默认2)
+ VectorIndexRetriever: 基础向量检索器
+ AutoMergingRetriever: 自动合并检索器,利用层级节点关系
  - 先检索叶子节点,再智能合并到父节点,平衡精确度和上下文完整性
+ QueryFusionRetriever: 融合多个检索器结果
  - 支持RRF(倒数排名融合)、relative_score、dist_based_score等融合模式
  - num_queries: LLM生成查询变体数,提升召回率
  - use_async=True异步并行检索
+ BM25Retriever: 基于关键词的BM25算法检索器
+ VectorIndexAutoRetriever: 自动根据问题决定是否使用元数据过滤

9. 混合检索(Hybrid Search):
+ 结合向量检索(语义理解)和BM25(关键词匹配)
+ QueryFusionRetriever融合两者结果
+ 优势: 同时捕捉语义相似性和精确关键词匹配

10. 后处理器(Postprocessors):
+ SimilarityPostprocessor: 基于相似度阈值过滤(similarity_cutoff 0-1)
+ LimitRetrievedNodesLength: 限制检索节点的总token长度
+ 自定义后处理器: 实现postprocess_nodes方法

11. 响应合成模式(Response Mode)(面试必考):
+ default/refine: 
  - 工作方式: 先用第一个文本块生成答案,再用后续文本块迭代优化
  - 优点: 答案精准、连贯; 缺点: 慢,调用LLM次数多
  - 适合: 需要精准详细回答的场景
+ tree_summarize:
  - 工作方式: 所有检索文本一次性输入LLM进行总结
  - 优点: 快,适合长文本总结; 缺点: 受LLM上下文长度限制
  - 适合: 汇总多个文档信息
+ compact:
  - 工作方式: 尽量塞满上下文窗口,减少LLM调用次数
  - 优点: 速度与效果平衡; 适合: 普通问答追求性价比
+ simple_summarize: 强制所有文本拼接后直接总结
+ accumulate: 对每段文本分别生成答案,最后拼接
+ compact_accumulate: 先塞满文本块再逐条生成答案
+ no_text: 不生成自然语言答案,只返回检索到的原始节点

12. 查询引擎(QueryEngine) vs 聊天引擎(ChatEngine):
+ QueryEngine: 无记忆的端到端问答流程
  - index.as_query_engine(similarity_top_k, response_mode, node_postprocessors)
+ ChatEngine: 有记忆的多轮对话流程
  - index.as_chat_engine(chat_mode)
  - chat_mode选项:
    * "context": 每轮注入检索上下文
    * "condense_plus_context"(推荐): 将"当前问题+历史"重写为独立查询再检索
    * "condense_question": 仅重写问题,不注入上下文
    * "simple": 纯对话,不检索

13. 记忆系统(Memory):
+ ChatMemoryBuffer: 管理对话历史的内存缓冲区
  - token_limit: 记忆的最大token数,超限后旧消息被丢弃
  - chat_store: 外部持久化存储(RedisChatStore、PostgresChatStore)
  - chat_store_key: 区分不同用户的存储键
+ Context: Agent工作流中的状态管理
  - 可序列化(JsonSerializer/JsonPickleSerializer),支持跨会话恢复
  - ctx.store.edit_state()读写状态变量

14. 路由机制(Routing):
+ RouterQueryEngine: 根据查询内容路由到最合适的查询引擎
+ LLMSingleSelector: 使用LLM从多个选项中选择最合适的
+ QueryEngineTool: 将查询引擎包装为工具
  - description是关键参数,LLM根据描述做路由决策,描述越具体路由越准确

15. Agent智能体(面试重点):
+ FunctionAgent: 基于函数调用的Agent
  - tools: 工具列表(可以是函数或BaseTool)
  - llm: LLM实例
  - system_prompt: 系统提示词
  - output_cls: Pydantic模型定义结构化输出
  - can_handoff_to: 允许交接给其他Agent
+ AgentWorkflow: 多Agent工作流编排
  - agents: Agent列表
  - root_agent: 根Agent名称
  - initial_state: 初始状态字典
+ AgentRunner: 管理Agent执行循环
  - chat(message)、run_task(task)

16. 工具(Tools):
+ FunctionTool: 将Python函数包装为工具
  - from_defaults(fn=your_function)
+ QueryEngineTool: 将查询引擎包装为工具
+ 内置工具集: YahooFinanceToolSpec、TavilyToolSpec等
+ 工具定义规范: 必须有清晰的docstring描述用途

17. 多Agent模式(面试高级考点):
+ 模式一: AgentWorkflow线性集群
  - 声明多个Agent,通过can_handoff_to定义交接关系
  - AgentWorkflow自动管理交接和执行顺序
  - 适合: 开箱即用的多Agent行为
+ 模式二: Orchestrator指挥代理
  - 一个协调者Agent将子Agent作为工具调用
  - 协调者决定每一步调用哪个子Agent
  - 适合: 需要集中控制决策逻辑
+ 模式三: Custom Planner自定义规划器
  - 编写LLM提示输出结构化计划(XML/JSON)
  - Python代码解析计划并强制执行
  - 适合: 极致灵活性,需要特定计划格式

18. 人工介入(Human-in-the-loop):
+ InputRequiredEvent: 发出等待人类输入的事件
+ HumanResponseEvent: 接收人类响应
+ ctx.wait_for_event(): 等待特定事件
+ 应用场景: 危险操作确认、敏感决策审核

19. 流式输出(Streaming):
+ query_engine(streaming=True): 启用流式响应
+ response.print_response_stream(): 打印流式输出
+ AgentStream: Agent工作流中的流式事件
+ handler.stream_events(): 遍历Agent执行过程中的事件

20. 结构化输出(Structured Output):
+ output_cls: Pydantic模型定义输出结构
  - agent.run()返回response.structured_response
  - response.get_pydantic_model(ModelClass)获取Pydantic对象
+ structured_output_fn: 自定义结构化输出解析函数
  - 输入ChatMessage序列,返回字典
  - 适合复杂验证逻辑

21. 存储与持久化:
+ StorageContext: 定义文档、嵌入和索引的存储后端
  - persist(persist_dir): 保存到磁盘
  - from_defaults(persist_dir): 从磁盘加载
  - vector_store: 指定向量数据库(Chroma、Pinecone、Qdrant等)
  - docstore: 文档存储(SimpleDocumentStore)
+ load_index_from_storage(storage_context): 从存储加载索引
+ index.refresh(new_documents): 用新数据刷新索引

22. 向量数据库集成:
+ Chroma: 本地轻量零成本,自动持久化到磁盘
+ Pinecone: 生产级托管向量数据库
+ Qdrant: 高性能、低内存占用,中小规模生产首选
+ Milvus: 超大规模企业级,支持万亿级向量
+ pgvector: 复用PostgreSQL能力,SQL原生支持混合检索
+ FAISS: 工业级检索引擎,本地检索性能天花板

23. 高级查询引擎:
+ SubQuestionQueryEngine: 子问题查询引擎
  - 将复杂查询分解为多个简单子问题
  - 并行执行子查询,最后综合答案
  - 适合: 对比分析、多维度查询
+ SQLAutoVectorQueryEngine: SQL+向量混合查询
  - 先查SQL找结构化数据,再查向量找非结构化细节
  - 适合: 结合表格数据和文档数据的场景
+ RetrieverQueryEngine: 基于自定义检索器的查询引擎
  - from_args(retriever, llm, node_postprocessors)

24. 文档管理与去重:
+ Document(id_=thread_id, metadata={"date": timestamp}): 设置唯一ID
+ index.ref_doc_info: 查看已索引文档信息
+ refresh(new_documents, update_kwargs): 增量更新索引
  - delete_kwargs={"delete_from_docstore": True}删除旧文档
+ 避免重复嵌入,节省token成本

25. 多模态支持:
+ ChatMessage支持TextBlock和ImageBlock
+ ImageBlock(path="image.png"): 图像输入
+ 多模态LLM: gpt-4o等支持图文混合输入

26. 异步编程:
+ nest_asyncio.apply(): 解决事件循环嵌套冲突
+ await agent.arun(): 异步运行Agent
+ await query_engine.aquery(): 异步查询
+ use_async=True: 异步并行检索

27. 性能优化技巧:
+ 调整chunk_size和chunk_overlap平衡精度和速度
+ similarity_top_k不宜过大(通常3-5)
+ 使用compact响应模式减少LLM调用
+ 启用use_async异步并行
+ 持久化索引避免重复嵌入
+ 使用局部模型(Ollama)降低成本

28. 常见面试题总结:
【基础概念】
Q: LlamaIndex和LangChain的区别?
A: LlamaIndex专注数据索引和检索,RAG场景更强;LangChain侧重Agent和工作流编排,通用性更强

Q: RAG的核心流程是什么?
A: 加载文档→切分节点→生成嵌入→构建索引→检索相似节点→拼接prompt→LLM生成答案

Q: Document和Node的区别?
A: Document是原始数据容器,Node是Document切分后的最小处理单元,Index操作的是Node

【核心技术】
Q: 向量嵌入(Embedding)的作用是什么?
A: 将文本转换为数值向量,语义相似的文本向量距离更近,支持语义检索

Q: similarity_top_k的作用?
A: 控制检索返回的最相似节点数量,默认2,过大会增加噪音,过小可能遗漏关键信息

Q: 什么是混合检索?
A: 结合向量检索(语义理解)和BM25(关键词匹配),通过QueryFusionRetriever融合结果,提升召回率

Q: 响应合成模式有哪些?区别是什么?
A: refine(迭代优化,精准但慢)、tree_summarize(一次性总结,快但受限上下文)、compact(平衡速度与效果)

【进阶应用】
Q: 如何实现多轮对话记忆?
A: 使用ChatEngine代替QueryEngine,chat_mode选择"condense_plus_context",配合ChatMemoryBuffer管理历史

Q: Agent的核心组成是什么?
A: LLM(决策大脑)+ Tools(执行能力)+ Memory(状态记忆)+ Planning(任务规划)

Q: 多Agent如何协作?
A: 三种模式:AgentWorkflow自动交接、Orchestrator集中调度、Custom Planner自定义规划

Q: 如何优化RAG性能?
A: 调整chunk大小、选择合适的top_k、使用后处理器过滤、启用异步、持久化索引、选择合适响应模式

【实战场景】
Q: 如何处理大规模文档更新?
A: 使用Document唯一ID,index.refresh()增量更新,避免全量重建索引

Q: 如何实现复杂问题的分解查询?
A: 使用SubQuestionQueryEngine自动将复杂问题拆解为子问题并行查询

Q: 如何结合结构化数据(SQL)和非结构化数据(文档)?
A: 使用SQLAutoVectorQueryEngine,先SQL查询表格数据,再向量检索文档细节

Q: 如何实现人工审核机制?
A: 使用InputRequiredEvent和HumanResponseEvent,在关键步骤暂停等待人工确认

29. 最佳实践建议:
+ 从小规模开始,逐步添加复杂性
+ 优先使用VectorStoreIndex + SentenceSplitter基础组合
+ 调试时启用verbose=True查看详细过程
+ 生产环境务必持久化索引
+ 根据场景选择合适的response_mode
+ 多轮对话使用ChatEngine而非QueryEngine
+ 复杂任务考虑多Agent协作
+ 敏感操作加入Human-in-the-loop机制
+ 监控token消耗和响应时间
+ 定期评估检索质量(相关性、覆盖率)

30. 学习资源:
+ 官方文档: https://developers.llamaindex.ai
+ GitHub示例: https://github.com/run-llama/llama_index
+ LlamaHub工具库: 丰富的预建工具和阅读器
+ create-llama: 快速生成全栈Web应用
+ SEC Insights: 财务文档查询高级案例




