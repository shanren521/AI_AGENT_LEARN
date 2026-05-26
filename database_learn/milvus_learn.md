# 1.Milvus是什么？

Milvus 是向量数据库，用来存储、索引和检索向量。典型场景包括语义搜索、RAG 知识库、图片检索、推荐系统、异常检测、多模态检索。

## 1.1 核心概念

| 概念            | 类比          | 说明                                         |
|---------------|-------------|--------------------------------------------|
| Collection    | SQL表        | 存放一类数据，例如文档片段                              |
| Entity        | SQL 行       | 一条记录，包含主键、向量、文本、元数据                        |
| Field         | SQL 列       | 如 id、text、vector、category                  |
| Vector Field  | 向量列         | 存 embedding，如 FLOAT_VECTOR                 |
| Scalar Field  | 普通列         | 存标签、时间、作者、权限等                              |
| Index         | 数据库索引       | 加速向量近邻搜索                                   |
| Search        | 向量搜索        | 根据 query vector 找相似实体                      |
| Query         | 标量查询        | 根据条件查数据，如 category == "rag"                |
| Filter        | WHERE 条件    | 向量搜索前做元数据过滤                                |
| Load          | 加载到查询节点     | Standalone/Distributed 中搜索前需要加载 collection |

# 2. 案例

```python
from pymilvus import MilvusClient, DataType
import math

client = MilvusClient(uri="http://127.0.0.1:19530")
collection = "book_docs"

if client.has_collection(collection):
    client.drop_collection(collection)
    
def embed(text: str, dim: int = 768):
    v = [0.0] * dim
    for i, b in enumerate(text.encode("utf-8")):
        v[i % dim] += (b % 31) - 15
    norm = math.sqrt(sum(x *x for x in v)) or 1.0
    return [x / norm for x in v]

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("title", DataType.VARCHAR, max_length=256)
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("chunk", DataType.VARCHAR, max_length=2048)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

client.create_collection(
    collection_name=collection,
    schema=schema,
    index_params=index_params,
    consistency_level="Bounded"
)

docs = [
    {"id": 1, "title": "Milvus 入门", "category": "basic", "chunk": "Milvus 是面向 AI 应用的向量数据库。"},
    {"id": 2, "title": "索引调优", "category": "perf", "chunk": "HNSW 适合高召回，IVF_FLAT 适合内存受限场景。"},
    {"id": 3, "title": "RAG 检索", "category": "rag", "chunk": "RAG 通常先切分文档，再向量化并写入 Milvus。"},
    {"id": 4, "title": "部署模式", "category": "ops", "chunk": "Milvus Lite 适合原型，Standalone 适合单机生产。"},
]

for d in docs:
    d["embedding"] = embed(d["chunk"])

client.insert(collection_name=collection, data=docs)
client.load_collection(collection_name=collection)

res = client.search(
    collection_name=collection,
    data=[embed("Milvus 怎么用于 RAG 知识库？")],
    anns_field="embedding",
    limit=3,
    output_fields=["title", "category", "chunk"],
    search_params={"metric_type": "COSINE"}
)

for hit in res[0]:
    print(hit["distance"], hit["entity"])
```
	 	
连接 Milvus -> 定义 schema -> 定义 index -> 创建 collection -> 插入数据 -> load -> search/query		
		
# 3. Schema设计

Schema 是 Milvus 的数据结构契约。常用字段类型: 

| 类型                     | 用途                           |
|------------------------|------------------------------|
| INT64                  | 主键、数字 ID                     |
| VARCHAR                | 文本、类别、用户 ID、文档 ID            |
| FLOAT_VECTOR           | 稠密向量，最常见                     |
| SPARSE_FLOAT_VECTOR    | 稀疏向量，BM25/SPLADE/BGE-M3 稀疏部分 |
| JSON                   | 灵活元数据                        |
| ARRAY                  | 标签列表、多个属性                    |
| BOOL/FLOAT/DOUBLE/INT* | 普通标量字段                       |

**关键参数**：

| 参数                   | 说明             | 建议                    |
|----------------------|----------------|-----------------------|
| auto_id              | 主键是否自动生成       | 业务已有 ID 时设 False      |
| is_primary           | 是否主键字段         | 每个 collection 必须有主键   |
| max_length           | VARCHAR 最大长度   | 按实际文本长度预留             |
| dim                  | 向量维度           | 必须等于 embedding 模型输出维度 |
| enable_dynamic_field | 未定义字段是否进 $meta | 原型可开，生产建议谨慎           |

		
# 4. 向量索引选择

Milvus 支持多种索引。新手默认用 AUTOINDEX，除非你明确知道召回、延迟、内存目标。

| 索引                    | 场景        | 核心参数                     | 特点               |
|-----------------------|-----------|--------------------------|------------------|
| AUTOINDEX             | 默认首选      | metric_type              | Milvus 自动选择策略    |
| FLAT                  | 小数据、精确搜索  | 无                        | 暴力搜索，召回最高但慢      |
| HNSW                  | 高召回、内存充足  | M, efConstruction, 搜索 ef | 快、准、吃内存          |
| IVF_FLAT              | 内存较紧、数据较大 | nlist, 搜索 nprobe         | 聚类后搜索部分桶         |
| IVF_PQ                | 极度节省内存    | nlist, m, nprobe         | 压缩强，召回损失更明显      |
| SPARSE_INVERTED_INDEX | BM25/稀疏向量 | 稀疏索引参数                   | 全文/稀疏检索          |
| GPU_*                 | GPU 场景    | 视索引而定                    | 需要 GPU Milvus 部署 |

**HNSW 示例**

```python
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={
        "M": 16,
        "efConstruction": 200
    }
)

"""
M 管稠密程度
efConstruction 管建图精度
ef 只管搜索召回
"""
```
		
**IVF_FLAT 示例**

```python
index_params.add_index(
    field_name="embedding",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={
        "nlist": 1024  # 聚类簇的总数 sqrt(向量总数)
    }
)

res = client.search(
    collection_name=collection,
    data=[query_vector],
    anns_field="embedding",  # 用于近似最近邻搜索的向量字段名
    limit=3,
    search_params={
        "metric_type": "COSINE",
        "params": {"nprobe": 32}  # 查询时选取多少个最近簇搜索  只影响查询，不影响建索引
    }
)
```
		
**参数直觉：**

| 参数             | 含义           | 调大后        |
|----------------|--------------|------------|
| M              | HNSW 每个点的连接数 | 召回提升，内存增加  |
| efConstruction | HNSW 建图质量    | 索引更好，构建更慢  |
| ef             | HNSW 搜索宽度    | 召回提升，延迟增加  |
| nlist          | IVF 聚类桶数量    | 桶更细，构建成本增加 |
| nprobe         | IVF 搜索桶数量    | 召回提升，延迟增加  |

# 5.距离度量

| metric_type | 适合场景               | 分数解释  |
|-------------|--------------------|-------|
| COSINE      | 文本/图片 embedding 常用 | 越大越相似 |
| IP          | 内积、归一化向量、部分稀疏向量    | 越大越相似 |
| L2          | 欧氏距离、未归一化向量        | 越小越相似 |
| BM25        | 全文检索               | 越大越相关 |

生产里最常见错误是：模型适合 COSINE，但索引用了 L2；或者建索引是 COSINE，搜索时写成 IP。索引和搜索的 metric_type 要一致。

# 6.搜索、过滤、查询

```python
# 向量搜索
res = client.search(
    collection_name="book_docs",
    data=[embed("怎么提升检索召回?")],
    anns_field="embedding",
    limit=3,
    output_fields=["title", "category", "chunk"],
    search_params={"metric_type": "COSINE"}
)


# 带过滤的搜索
res = client.search(
    collection_name="book_docs",
    data=[embed("怎么提升检索召回?")],
    anns_field="embedding",
    filter="category=='tech'",
    limit=3,
    output_fields=["title", "category", "chunk"],
    search_params={"metric_type": "COSINE"}
)


# 标量查询，不做向量相似度
rows = client.query(
    collection_name="book_docs",
    filter="category in ['rag', 'perf']",
    output_fields=["id", "title", "category"],
    limit=10
)


# 删除
client.delete(collection_name="book_docs", filter="id in [1, 2, 3]")


# 插入
client.upsert(
    collection_name="book_docs",
    data=[{
        "id": 2,
        "title": "搜索调优新版?",
        "category": "perf",
        "chunk": "HNSW的ef越大通常召回越高，但延迟增加",
        "embedding": embed("HNSW 的 ef 越大通常召回越高，但延迟也会上升。"),
    }]
)
```

**search常用参数**

| 参数              | 说明                     |
|-----------------|------------------------|
| collection_name | 目标collection           |
| data            | query vector 列表        |
| anns_field      | 要搜索的向量字段               |
| filter          | 标量过滤表达式                |
| limit           | TopK数量                 |
| output_fields   | 返回哪些字段                 |
| search_params   | 距离度量、索引搜索参数            |
| partition_names | 限定分区                   |
| offset          | 分页偏移，和limit总和通常小于16384 |
| group_by_field  | 按字段分组，避免同一文档返回太多chunk  |


# 7.RAG 知识库流程

```python
"""
离线：原始文档 -> 清洗 -> 切 chunk -> 生成 embedding -> 写入 Milvus
在线问答：用户问题 -> 生成 query embedding -> Milvus TopK 检索 -> 拼上下文 -> 给 LLM 回答
"""
```

```python
def retrieve(question: str, top_k: int = 5):
    results = client.search(
        collection_name="book_docs",
        data=[embed(question)],
        anns_field="embedding",
        limit=top_k,
        output_fields=["title", "category", "chunk"],
        search_params={"metric_type": "COSINE"}
    )

    return [
        {
            "score": hit["distance"],
            "title": hit["entity"]["title"],
            "text": hit["entity"]["chunk"],
        }
        for hit in results[0]
    ]

def build_prompt(question: str):
    chunks = retrieve(question)
    context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
    return f"""请只基于以下资料回答问题。

资料:
{context}

问题:
{question}
"""
```

# 8.全文检索与BM25

向量搜索擅长语义相似，但对精确词、型号、错误码、函数名有时不够稳。Milvus 也支持基于 BM25 的全文检索：文本字段启用 analyzer，BM25 function 自动生成稀疏向量。

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient("http://localhost:19530", token="root:Milvus")
collection = "bm25_docs"

if client.has_collection(collection):
    client.drop_collection(collection)
    
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=1024, enable_analyzer=True)
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

bm25 = Function(
    name="text_bm25_emb",
    input_field_names=["text"],
    output_field_names=["sparse"],
    function_type=FunctionType.BM25
)

schema.add_function(bm25)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="sparse",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25"
)

client.create_collection(
    collection,
    schema,
    index_params=index_params
)

client.insert(
    collection,
    [
        {"text": "Milvus supports vector search and full text search."},
        {"text": "BM25 is useful for exact keyword retrieval."},
        {"text": "RAG often combines dense retrieval and lexical retrieval."},
    ]
)

client.load_collection(collection)

res = client.search(
    collection_name=collection,
    data=["what is BM25 useful for?"],
    anns_field="sparse",
    output_fields=["text"],
    limit=3,
)

```

**BM25关键字段**

| 参数/字段                | 说明               |
|----------------------|------------------|
| enable_analyzer=True | 让Milvus对文本分词、规范化 |
| FunctionType.BM25    | 内置BM25稀疏向量生成     |
| SPARSE_FLOAT_VECTOR  | 存BM25生成的稀疏向量     |
| metric_type="BM25"   | 全文检索指标           |
| data=["自然语言问题"]      | 查询时传原始文本，不传向量    |


# 9.生产调优清单

| 问题     | 优先检查                                       |
|--------|--------------------------------------------|
| 搜索慢    | TopK是否过大，过滤是否复杂、索引是否合适                     |
| 召回低    | metric是否匹配、chunk是否太短/太长、ef/nprobe是否太小      |
| 内存高    | HNSW是否过大、是否可用IVF/PQ、是否需要分区                 |
| 新数据查不到 | consistency、flush/load、是否搜索growing segment |
| 结果重复   | RAG chunk是否按doc_id分组，使用group_by_field      |
| 精确词查不到 | 加BM25或混合索引                                 |
| 多租户隔离  | 用tenant_id标量过滤、分区或独立collection             |
| 权限过滤   | 把ACL元数据写入标量字段，搜索时强制filter                  |










