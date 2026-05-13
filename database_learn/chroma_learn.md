# 1.Chroma是什么

核心概念：
+ Document：原始文本，例如一段知识库内容
+ Embedding：文本转成的向量，例如[0.12, -0.53, ...]
+ Collection：类似数据库里的表，保存文档、向量、元数据
+ Metadata：附加字段，例如来源、时间、分类、用户ID
+ Query：把问题转向量，再找最相似的文档

典型的RAG流程：
```
文档 -> 切分 chunk -> 生成 embedding -> 存入 Chroma
用户问题 -> 生成 query embedding -> Chroma 相似度检索 -> 拼接上下文 -> LLM 回答
```

示例:
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection(name="my_collection")
collection.add(
    ids=["1", "2", "3"],
    documents=[
        "Chroma 是一个向量数据库，适合构建 RAG 应用。",
        "MySQL 是关系型数据库，适合结构化数据查询。",
        "Redis 常用于缓存、分布式锁和高性能键值存储。"
    ],
    metadatas=[
        {"type": "vector_db"},
        {"type": "sql_db"},
        {"type": "cache"}
    ]
)

result = collection.query(
    query_texts=["什么数据库适合做语义搜索？"],
    n_results=2
)
print(result)
```

# 2.持久化

默认 chromadb.Client() 通常适合临时测试。生产或本地开发建议使用持久化客户端
```python
import chromadb


# ./chroma_data 会保存索引和数据，下次程序启动还能继续使用。
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="product_base")
collection.add(
    ids=["doc_001"],
    documents=["用户可以在订单页面申请退款。"],
    metadatas=[{"source": "faq", "category": "refund"}]
)
result = collection.query(
    query_texts=["如何退款?"],
    n_results=1
)
print(result["documents"])
```

# 3.Collection增删改查

```python

# 创建或获取collection
collection = client.get_or_create_collection(name="products")

# 添加数据
collection.add(
    ids=["p1", "p2"],
    documents=["iPhone 适合移动办公和拍照。", "机械键盘适合长时间打字"],
    metadatas=[
        {"category": "phone", "price": 5999},
        {"category": "keyboard", "price": 399}
    ]
)

# 查询数据
items = collection.get(
    ids=["p1"],
    include=["documents", "metadatas"]
)

# 修改数据
collection.update(
    ids=["p1"],
    documents=["iPhone 适合移动办公和拍照。"],
    metadatas=[{"category": "phone", "price": 5899}]
)

# 不存在则插入，存在则更新
collection.upsert(
    ids=["p3"],
    documents=["MacBook Pro 适合商务办公和开发。"],
    metadatas=[{"category": "laptop", "price": 18999}]
)

# 删除数据
collection.delete(ids=["p1"])

# 删除collection
client.delete_collection(name="productsd")
```

# 4.查询与过滤

```python

# 普通语义查询
result = collection.query(
    query_texts=["适合办公的设备"],
    n_results=3
)

# 按metadata过滤
result = collection.query(
    query_texts=["适合办公"],
    n_results=5,
    where={"price": {"$lt": 1000}}
)

# 文档内容过滤
result = collection.query(
    query_texts=["办公"],
    n_results=5,
    where_document={"$contains": "通勤"}
)

# 多条件过滤
where={
    "$and": [
        {"category": "phone"},
        {"price": {"$gte": 3000}}
    ]
}

# 多字段过滤
where={
    "category": {"$in": ["phone", "laptop"]}
}
```

# 5.使用自己的embedding

Chroma 可以自动处理文本向量化，但生产中通常会显式指定 embedding 模型，方便控制效果、成本和一致性。

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_data")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="products",
    embedding_function=embedding_fn
)
collection.add(
    ids=["1"],
    documents=["Chroma 可以用于语义搜索和RAG"]
)
result = collection.query(
    query_texts=["怎么做知识库检索"],
    n_results=1
)
print(result["documents"])
```

# 6.构建一个简单RAG知识库

```python
import chromadb

client = chromadb.PersistentClient(path="./faq_db")
collection = client.get_or_create_collection(name="company_faq")
docs = [
    "退款申请路径：进入订单详情页，点击申请售后，选择退款原因后提交。",
    "发票开具路径：进入个人中心，选择发票管理，填写抬头信息。",
    "会员权益包括专属折扣、优先客服、生日优惠券。",
    "订单发货后可以在物流页面查看快递单号和运输状态。"
]

collection.upsert(
    ids=[f"faq_{i}" for i in range(len(docs))],
    documents=docs,
    metadatas=[
        {"module": "refund"},
        {"module": "invoice"},
        {"module": "member"},
        {"module": "logistics"},
    ]
)

def retrieve(question: str, top_k: int = 3):
    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return result

question = "用户在哪里申请退款？"
hits = retrieve(question)
context = "\n".join(hits["documents"][0])
prompt = f"""
请基于以下资料回答用户问题。

资料：
{context}

问题：
{question}
"""
res = llm.invoke(prompt)
```

# 7.文档切分Chunk

生产中不要把一本 PDF 或一整篇文章直接塞进去。应该切分成小块。

```python
def split_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

long_doc = """
Chroma 是一个向量数据库。它可以保存 embeddings、documents 和 metadata。
在 RAG 应用中，通常先把知识库切分成多个 chunk，然后向量化并写入数据库。
查询时，根据用户问题检索最相关的 chunk，再交给大模型生成答案。
"""

chunks = split_text(long_doc)

collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    metadatas=[{"source": "intro", "chunk_index": i} for i in range(len(chunks))]
)
```

**经验值**：
+ FAQ、短文档：300-800 中文字符。
+ 技术文档：500-1200 中文字符。
+ 法律/合同：chunk 可以稍大，但 metadata 要细。
+ overlap 通常 10%-20%。
+ chunk 太小：上下文不足。
+ chunk 太大：召回不精准，LLM 成本上升。

# 8.Metadata设计

Metadata 可以用来过滤、排序、聚合、关联。
metadata 不是装饰品，它决定生产检索质量和权限控制。

```python
# 推荐字段
metadata = {
    "tenant_id": "company_a",
    "user_id": "u123",
    "source": "help_center",
    "doc_id": "refund_policy_v2",
    "chunk_index": 3,
    "category": "refund",
    "created_at": "2026-05-12",
    "version": "v2",
    "visibility": "internal"
}

# 查询时做权限过滤
result = collection.query(
    query_texts=["退款规则是什么?"],
    n_results=5,
    where={
        "$and": [
            {"tenant_id": "company_a"},
            {"visibility": "internal"}
        ]
    }
)
```

# 9.距离、相似度和排序

Chroma 使用余弦相似度来度量两个向量的相似度。
Chroma 查询结果里通常会有 distances。一般距离越小，越相似。

```python
result = collection.query(
    query_texts=["怎么开发票？"],
    n_results=3,
    include=["documents", "distances", "metadatas"]
)
for doc, distance in zip(result["documents"][0], result["distances"][0]):
    print(distance, doc)
```

**生产常见策略**：

+ 设置 top_k，比如先取 10 条。
+ 设置阈值，距离太大的结果丢弃。
+ 加 metadata 过滤。
+ 用 reranker 对召回结果重排。
+ 对最终上下文做去重、截断和引用来源展示。

# 10.生产实践重点

**生产环境最容易出问题的地方**：

+ embedding 模型变更后，新旧向量不能混用。
+ collection 里数据重复写入，导致召回重复。
+ chunk 太粗或太碎，导致答案不准。
+ 没有 metadata 权限过滤，造成数据泄露。
+ top_k 过小召回不足，top_k 过大噪声太多。
+ 没有记录文档版本，旧知识和新知识混在一起。
+ 没有删除或失效旧 chunk，知识库越用越脏。
+ 没有评估集，只靠肉眼判断 RAG 效果。
+ 查询只用向量相似度，没有关键词、过滤、rerank 辅助。
+ 没有监控检索命中率、距离分布、空召回率、用户反馈。

**推荐生产流程**

```
原始文档入库
-> 文档解析
-> 清洗
-> chunk 切分
-> embedding
-> Chroma upsert
-> 检索评估
-> 灰度上线
-> 监控召回质量
```

# 11.面试中会问到的生产问题

**Chroma 和传统数据库有什么区别？**

答：传统数据库主要做结构化精确查询，Chroma 存储向量，用于语义相似度检索。它适合“意思相近”的搜索，不适合替代 MySQL 做事务、复杂 JOIN、强一致业务数据。

**RAG 中为什么需要向量数据库？**

答：LLM 本身没有实时私有知识，向量数据库负责从外部知识库中检索相关上下文，再让模型基于上下文回答，降低幻觉并支持私有数据。

**文档应该怎么切分？**

答：按语义边界优先，比如标题、段落、章节；再控制 chunk 大小和 overlap。chunk 太大会召回不精准，太小会上下文不足。生产中要用评估集调参。

**embedding 模型换了怎么办？**

答：通常要重建 collection 或至少重算所有向量。不同模型的向量空间不同，不能直接混用。需要记录 embedding model name、version、dimension。

**如何避免重复数据？**

答：使用稳定 ID，例如 {doc_id}_{chunk_index}_{version}；导入时用 upsert；文档更新时删除旧版本 chunk 或通过 version/is_active 控制有效数据。

**如何做多租户隔离？**

答：Chroma 默认支持多租户，通过 metadata.tenant_id 来做权限控制。

**如何处理权限问题？**

答：权限必须在检索层过滤，不能只依赖 prompt。metadata 中保存用户、组织、角色、可见范围，query 时添加强制过滤条件。

**为什么检索结果不准？**

答：常见原因是 chunk 不合理、embedding 模型不适合中文/业务领域、metadata 缺失、top_k 不合适、文档噪声多、问题改写差、没有 rerank。

**top_k 怎么设置？**

答：没有固定答案。通常先取 5-20 条，再根据 token 预算、距离阈值和 rerank 结果筛选。FAQ 可以小一点，复杂文档问答可以大一点。

**怎么评估 RAG 效果？**

答：构建问题-标准答案-相关文档集，评估 recall@k、命中文档比例、最终答案准确率、引用正确率、无答案拒答能力和用户反馈。

**Chroma 适合什么规模？**

答：适合原型、本地开发、中小规模 AI 应用和轻量生产场景。大规模、高并发、复杂分布式场景需要关注部署方式、性能、备份、水平扩展和运维能力。

**Chroma 数据如何备份？**

答：如果使用 PersistentClient(path=...)，需要备份该持久化目录；生产中还要备份原始文档、chunk 结果、metadata、embedding 模型版本，因为这些决定能否重建索引。

**删除文档时怎么删干净？**

答：记录 doc_id，删除时按 metadata 删除对应所有 chunk。例如 collection.delete(where={"doc_id": "xxx"})。不要只删一个 chunk ID。

**什么时候需要 reranker？**

答：当初召回结果语义相近但排序不稳定、问题复杂、top_k 较大、答案要求高时，用 reranker 对候选 chunk 重新排序能明显提升质量。

**生产中如何降低幻觉？**

答：检索层提高召回质量，prompt 要求仅基于上下文回答，设置无答案兜底，展示引用来源，加入答案校验或二次判断，对低相似度结果拒答。