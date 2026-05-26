from qdrant_client.http.models import SearchParamsfrom qdrant_client.http.models import Distancefrom qdrant_client.http.models import FieldConditionfrom qdrant_client.http.models import HasIdConditionfrom openai import vector_stores

# 1.什么是Qdrant

Qdrant 是一个用 Rust 编写的高性能向量数据库和向量搜索引擎。它在 AI 应用中扮演"长期记忆"的角色——将文本、图片、音频等内容转换为向量（embedding）后存储，并提供毫秒级的相似性搜索。

Qdrant 是一个向量数据库，主要用于：

+ 语义搜索：比如“手机续航好”能搜到“电池容量大、充电快”的商品。
+ RAG 检索增强生成：先从知识库找相关文档，再交给大模型回答。
+ 推荐系统：找和当前商品、文章、用户行为相似的内容。
+ 多模态搜索：图片搜图片、文字搜图片、图片搜文字。
+ 混合搜索：关键词检索 + 向量检索 + 重排。

**Qdrant的关键能力一览：**

| 能力            | 说明                                       |
| --------------- | ------------------------------------------ |
| 毫秒级 ANN 搜索 | HNSW 索引 + 过滤，百万级向量亚 10ms 延迟   |
| 丰富的过滤      | 全文搜索、数值范围、地理位置、嵌套对象过滤 |
| 量化压缩        | Scalar（4×）、Binary（32×）内存压缩        |
| GPU 加速        | 索引构建速度提升 4×、检索速度提升 40×      |
| 多租户          | 单集合 + `is_tenant` 标志，支撑数千租户    |
| 混合搜索        | RRF 融合、相关性反馈                       |
| 分布式          | Raft 共识、多副本、跨 AZ 容灾              |

# 2.核心概念

Qdrant 里最重要的几个概念：

| 概念             | 含义                                   |
|----------------|--------------------------------------|
| Collection     | 类似 MySQL 的表，一组向量数据                   |
| Point          | 一条数据，包含 id、vector、payload            |
| Vector         | 向量，通常由 embedding 模型生成                |
| Payload        | 附加字段，类似 JSON，比如 title、price、category |
| Distance       | 相似度度量，比如 Cosine、Dot、Euclid           |
| Filter         | 按 payload 过滤，比如只搜 category = book    |
| Index          | 加速搜索或过滤的数据结构                         |
| HNSW           | Qdrant 默认常用的近似最近邻索引算法                |
| Dense Vector   | 稠密向量，常见于 embedding 模型 -- 理解语义相似性     |
| Sparse Vector  | 稀疏向量，常见于关键词/BM25/SPLADE 类检索 -- 精确关键词匹配 |
| Hybrid Search  | 稠密向量 + 稀疏向量组合检索                      |


## 2.1 Collection集合

一个 Collection 中所有向量必须有相同的维度。

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(":memory:")  # 内存版数据库
client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # 768 维向量，使用 Cosine 距离度量
)
```

## 2.2 Point数据点

Point 是 Qdrant 中的基本存储单元，每个 Point 包含三个部分：

- **id**：唯一标识符（正整数或 UUID）
- **vector**：高维嵌入向量
- **payload**（可选）：附加的 JSON 结构元数据

```python
from qdrant_client.models import PointStruct

client.upsert(
    collection_name="articles",
    points=[
        PointStruct(
            id=1,
            vector=[0.12, 0.34, ...],  # 768 维向量
            payload={
                "title": "Qdrant: A Modern Vector Database",
                "author": "Qdrant Team",
                "tags": ["database", "vector", "AI"],
                "views": 1000,  # 阅读量
                "create_at": "2026-05-13"
            }
        )
    ]
)
```

## 2.3 Vector向量

向量是数据的数值化表示，由 Embedding 模型生成。常见的 Embedding 模型

| 模型                   | 维度     | 适用场景     |
| ---------------------- | -------- | ------------ |
| text-embedding-3-small | 512/1536 | 通用文本     |
| bge-large-zh-v1.5      | 1024     | 中文语义搜索 |
| all-MiniLM-L6-v2       | 384      | 轻量英文     |
| CLIP ViT-L/14          | 768      | 图文多模态   |

```python
# 使用 OpenAI 生成 embedding
from openai import OpenAI
client_openai = OpenAI()
response = client_openai.embeddings.create(
    model="text-embedding-3-small",
    input="Qdrant是一个高性能向量数据库"
)
vector = response.data[0].embedding  # 1536维浮点数列表
```

## 2.4 Payload附加字段

Payload 是附加在 Point 上的任意 JSON 数据。Payload 支持索引以加速过滤。

**Payload 类型限制：**

- 支持：string、integer、float、bool、nested object、array
- 不支持：日期类型（用字符串或时间戳代替）

| 索引类型 | 适用场景              | 创建方式                               |
| -------- | --------------------- | -------------------------------------- |
| keyword  | 精确匹配（tag、状态） | `payload_schema={"field": "keyword"}`  |
| integer  | 数值范围过滤          | `payload_schema={"field": "integer"}`  |
| float    | 浮点数范围过滤        | `payload_schema={"field": "float"}`    |
| text     | 全文搜索              | `payload_schema={"field": "text"}`     |
| geo      | 地理位置搜索          | `payload_schema={"field": "geo"}`      |
| bool     | 布尔过滤              | `payload_schema={"field": "bool"}`     |
| datetime | 时间范围过滤          | `payload_schema={"field": "datetime"}` |

## 2.5 Distance距离度量

Qdrant 在**创建 Collection 时**确定距离度量，之后不可更改。选择错误的度量是生产中最常见的错误之一。

| 度量               | 公式理解              | 何时使用                       |
| ------------------ | --------------------- | ------------------------------ |
| Cosine             | 夹角越小越相似（0~2） | 文本语义搜索（最常用）         |
| Dot（内积）        | 值越大越相似          | 推荐系统、需要保留原始向量大小 |
| Euclid（欧几里得） | 距离越小越相似        | 图像相似度、地理位置           |
| Manhattan          | 绝对差之和            | 稀疏向量、特定 Embedding 模型  |

**关键陷阱：Cosine 距离时 Qdrant 会自动归一化**

```python
# 如果使用cosine距离存储向量，Qdrant会自动归一化
# 这意味着你查询时拿到的向量可能与存入的不同
# 如果需要保留原始向量值，使用Dot距离代替

# 正确做法：用Dot距离 + 自己归一化(等价于cosine)
client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(size=768, distance=Distance.DOT)
)
```

## 2.6 Filter 过滤

Filter 是 Qdrant 中按条件筛选 Point 的机制，每个 Filter 包含三种条件组合：
+ must：必须全部满足（AND）
+ should：至少满足一个（OR）
+ must_not：必须全部不满足（NOT）

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

client.search(
    collection_name="articles",
    query_vector=[0.12, 0.34, ...],  # 768维查询向量
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="book")
            )
        ]
    ),
    limit=10
)
```

## 2.7 Index 索引

Index 是 Qdrant 中加速查询的数据结构，分为两类:

+ 向量索引：HNSW 图索引，加速近似最近邻搜索
+ Payload 索引：B-tree / Hash 索引，加速字段过滤

```python
client.create_payload_index(
    collection_name="articles",
    field_name="category",
    field_schema="keyword"
)
```

## 2.8 HNSW 图索引

HNSW 是 Qdrant 默认的向量索引算法，核心特点:

+ 多层图结构：上层稀疏快速定位，下层稠密精确搜索
+ 近似最近邻：牺牲少量精确度换取数量级速度提升
+ 参数可调：m（连接数）、ef（搜索深度）平衡召回与性能

```python
from qdrant_client.models import HnswConfigDiff, VectorParams, Distance

client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(
        m=16,  # 每个节点最大连接数
        ef_construct=200,  # 构建时的搜索深度
        ef=10  # 查询时的搜索深度
    )
)
```

## 2.9 Dense Vector 稠密向量

Dense Vector 是 Qdrant 中最常见的语义搜索向量类型，每个 Point 包含：

+ 维度固定：如 768、1536 维，所有维度均有非零值
+ 来源：BERT、OpenAI Embedding、CLIP 等神经网络模型
+ 用途：语义相似度搜索，捕获深层语义关系

```python
from qdrant_client.models import PointStruct

client.upsert(
    collection_name="articles",
    points=[
        PointStruct(
            id=1,
            vector=[0.023, -0.156, 0.892, ...],  # 768维稠密向量
            payload={
                "title": "深度学习入门",
                "model": "text-embedding-3-small",
                "type": "dense"
            }
        )
    ]
)
```

## 2.10 Sparse Vector 稀疏向量

Sparse Vector 是 Qdrant 中用于关键词检索的向量类型，特点：

+ 大部分为零：仅少数维度有非零值，存储高效
+ 维度即词汇：每个维度对应一个词或 token ID
+ 来源：TF-IDF、BM25、SPLADE 等模型

```python
from qdrant_client.models import PointStruct, SparseVector

client.upsert(
    collection_name="articles",
    points=[
        PointStruct(
            id=1,
            vector={
                "sparse": SparseVector(
                    indices=[101, 205, 3402, 8901],  # 非零维度索引
                    values=[0.85, 0.72, 0.91, 0.63]   # 对应权重
                )
            },
            payload={
                "title": "Qdrant 入门指南",
                "model": "SPLADE",
                "type": "sparse"
            }
        )
    ]
)
```

## 2.11 Hybrid Search 混合搜索

Hybrid Search 是 Qdrant 中组合多种向量检索的策略，核心机制：

+ 多路召回：同时执行稠密向量搜索和稀疏向量搜索
+ 结果融合：RRF（倒数秩融合）或加权打分合并结果
+ 优势互补：稠密捕获语义，稀疏捕获精确关键词

```python
from qdrant_client.models import Prefetch, Fusion, SparseVector

client.query_points(
    collection_name="articles",
    prefetch=[
        Prefetch(
            query=[0.12, 0.34, ...],  # 稠密向量语义检索
            using="dense",
            limit=20
        ),
        Prefetch(
            query=SparseVector(
                indices=[101, 205, 3402, 8901],  # 稀疏向量关键词检索
                values=[0.85, 0.72, 0.91, 0.63]
            ),
            using="sparse",
            limit=20
        )
    ],
    query=Fusion(Fusion.RRF)  # RRF 融合排序（倒数秩融合）
)
```

**RRF 融合排序（倒数秩融合）**

是一种无需训练参数的多路搜索结果融合算法。

核心公式：

$$\text{score(d)} = \sum_{i=1}^{n} \frac{1} {k + {rank_i}(d)}$$

其中:
+ d ：某个文档/Point
+ N ：搜索路数（如稠密 + 稀疏 = 2 路）
+ $rank_i(d)$: 文档 d  在第 i  路结果中的排名（从 1 开始）
+ k ：常数平滑因子，通常取 60

k 越大：排名差距越被拉平，结果越"民主"
k 越小：头部排名优势越明显，越"精英"

关键特性：
+ 排名越靠前，贡献越大，但差距被平滑（不是线性） 
+ 避免某一路的绝对分数主导（不同路分数不可比） 
+ 未出现的文档该路贡献为 0

# 3.三种客户端模式

```python
from qdrant_client import QdrantClient

# 模式1：内存模式（测试/实验用，数据不持久化）
client = QdrantClient(":memory:")

# 模式2：本地持久化（原型开发）
client = QdrantClient(path="./qdrant_data")

# 模式3：远程服务（生产环境）
client = QdrantClient(
    url="http://localhost:6333",
    # api_key="..."  # Qdrant Cloud 需要
)
```

# 4.基本CRUD操作

## 4.1 创建Collection

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, HnswConfig, OptimizersConfig,
    WalConfig, QuantizationConfig, ScalarQuantization
)

client = QdrantClient("http://localhost:6333")

client.create_collection(
    collection_name="database",
    vectors_config=VectorParams(
        size=768,  # 向量维度
        distance=Distance.COSINE,  # 距离度量
        on_disk=False,  # 向量存储在内容(True=磁盘)
    ),
    # HNSW 索引配置
    hnsw_config=HnswConfig(
        m=16,  # 每个节点最大连接数
        ef_construct=200,  # 构建时的搜索深度
        on_disk=False,  # 索引存储在内存(True=磁盘)
    ),
    # WAL 配置
    wal_config=WalConfig(
        wal_capacity_mb=32,  # WAL容量
    ),
    # 量化配置(可选，节省内存)
    quantization_config=QuantizationConfig(
        scalar=ScalarQuantization(
            type="int8",  # 量化类型
            always_ram=True,  # 量化向量常驻内存
        )
    )
)
```

## 4.2 插入向量(Upsert)

```python
import numpy as np
from qdrant_client.models import PointStruct

# 生成模拟数据
def generate_vectors(count, dim):
    return np.random.randn(count, dim).astype(np.float32)


vectors = generate_vectors(1000, 768)

# 批量插入(推荐每次64-256个点)
batch_size = 64

for i in range(0, len(vectors), batch_size):
    batch = vectors[i: i+batch_size]
    points = [
        PointStruct(
            id=i + j,
            vector=vec.tolist(),
            payload={
                "source": "mybase",
                "category": np.random.choice(["tech", "science", "art"]),
                "timestamp": 168000000000 + i + j,
            }
        )
        for j, vec in enumerate(batch)
    ]
    client.upsert(
        collection_name="mybase",
        points=points,
        wait=True  # 等待写入
    )
print(f"插入完成，总向量数: {client.count('mybase').count}")
```

## 4.3 更新向量

```python
# 更新单个节点的向量
client.update_vectors(
    collection_name="mybase",
    points=[
        PointStruct(
            id=42,
            vector=new_vector
        )
    ]
)

# 更新payload向量
client.set_payload(
    collection_name="mybase",
    payload={"category": "updated_category", "editor": "Alice"},
    points=[42]
)
```

## 4.4 条件更新

```python
# 仅当title字段不存在时才更新--防止覆盖

client.set_payload(
    collection_name="mybase",
    payload={"title": "New Title"},
    points=[42],
    filter=Filter(
        must_not=[HasIdCondition(has_id=[42])]
    )
)
```

## 4.5 删除

```python
# 按ID删除

client.delete(
    collection_name="mybase",
    points_selector=PointIdsList(points=[1, 2, 3])
)

# 按条件删除(危险，可能出发大量tombstone)
client.delete(
    collection_name="mybase",
    points_selector=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="tech")
            )
        ]
    )
)
```

## 4.6 检索向量

```python
# 按ID批量获取
points = client.retrieve(
    collection_name="mybase",
    ids=[1, 2, 3],
    with_vectors=True,  # 是否返回向量
    with_payload=True,  # 是否返回payload
)
for point in points:
    print(f"ID: {point.id}, Payload: {point.payload}")
```


## 4.7 批量获取向量

```python
records, next_page = client.scroll(
    collection_name="mybase",
    limit=10,
    with_payload=True,
    with_vectors=False,
)

for r in records:
    print(r.id, r.payload)
```


# 5.向量搜索

## 5.1 基础相似性搜索

```python
query_vector = embedding_model.encode("什么是向量数据库?").tolist()
results = client.query_points(
    collection_name="my_base",
    query=query_vector,
    limit=5,
    with_payload=True,
    with_vectors=False
)
for hit in results.points:
    print(f"ID: {hit.id}, Score: {hit.score:.4f}")
    print(f"Title: {hit.payload.get('title', 'N/A')}")
    print("---")
```

## 5.2 搜索参数详解

```python
results = client.query_points(
    collection_name="my_base",
    query=query_vector,
    limit=5,  # 返回数量
    offset=0,  # 分页偏移量
    with_payload=True,  # 返回payload
    with_vectors=False,  # 返回向量(通常不需要)
    score_threshold=0.5,  # 最低相似度阈值
    using="text_vector",  # 指定向量索引
    search_params=models.SearchParams(
        hnsw_ef=100,  # HNSW 搜索精度(越大越准但越慢)
        exact=False,  # True=暴力搜索
        indexed_only=False  # True=只搜索已有索引段
    )
)
```

## 5.3 批量搜索(高吞吐场景)

```python
# 一次请求搜索多个查询向量，分摊网络开销
query_vectors = [
    embedding_model.encode(q).tolist()
    for q in ["a question?", "b question?", "c question?"]
]
results = client.query_batch_points(
    collection_name="my_base",
    requests=[
        models.QueryRequest(query=qv, limit=5)
        for qv in query_vectors
    ]
)
```

## 5.4 分组搜索(按字段去重)

```python
# 搜索10篇最相关的文章，但按作者去重，每个作者最多1篇
results = client.query_points_groups(
    collection_name="my_base",
    query=query_vector,
    group_by="author",  # 按payload中的字段分组
    group_size=1,  # 每组返回的数量
    limit=10,  # 总共返回数量
)
```

## 5.5 推荐搜索(正负例混合)

```python
# 给定用户喜欢的(正例)和不喜欢的(负例)，找最匹配的
results = client.recommend(
    collection_name="my_base",
    positive=[1, 2, 3],  # 用户点赞的文章ID
    negative=[4, 5, 6],  # 用户不喜欢的文章ID
    limit=5,
    strategy=models.RecommendStrategy.AVERAGE_VECTOR,  # 对正例向量取平均
)
```

## 5.6 发现搜索(Discover API)

```python
# "文档56和文档23都不错，但我想要更好的"
# 用两个参考点的向量差来定位搜索方向

results = client.discover(
    collection_name="my_base",
    target=56,  # 从56出发
    context=[{"positive": 23, "negative": 17}],  # 上下文约束
    limit=10
)
```

# 6.过滤搜索

## 6.1 过滤表达式语法

```python
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

# 条件组合：Must(AND)、MustNot(NOT)、Should(OR)
query_filter = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="tech")),
        FieldCondition(key="views", range=Range(gte=1000))
    ],
    # 数据不能是已删除的
    must_not=[
        FieldCondition(key="is_deleted", match=MatchValue(value=True))
    ],
    # 可选项
    should=[
        FieldCondition(key="is_featured", match=MatchValue(value=True)),
        FieldCondition(key="rating", range=Range(gte=4.5))
    ]
)
results = client.query_points(
    collection_name="my_base",
    query=query_vector,
    query_filter=query_filter,
    limit=10
)
```

## 6.2 各类过滤条件

```python
from qdrant_client.models import (
    FieldCondition, MatchValue, MatchAny, MatchExcept,
    MatchText, Range, GeoRadius, DatetimeRange, Filter,
    HasIdCondition, IsEmptyCondition, IsNullCondition
)

# 精确匹配
FieldCondition(key="status", match=MatchValue(value="published"))

# 多值匹配(OR)
FieldCondition(key="category", match=MatchAny(any=[1, 2, 3]))

# 排除匹配(NOT IN)
FieldCondition(key="author", match=MatchExcept(**{"except": ["spam_bot"]}))

# 全文搜索匹配(需要text索引)
FieldCondition(key="content", match=MatchText(text="vector database"))

# 数值范围
FieldCondition(key="views", range=Range(gte=500, lt=1000))

# 地理位置范围
FieldCondition(key="location", 
               geo_radius=GeoRadius(center={"lon": 116.40, "lat": 39.90}, radius=10000.0))

# 时间范围(datetime索引)
FieldCondition(key="create_at", range=DatetimeRange(gte="2026-01-01T00:00:00Z", lt="2027-01-01T00:00:00Z"))

# 嵌套对象过滤
FieldCondition(key="metadata.source", match=MatchValue(value="official"))

# 数组包含
FieldCondition(key="tags[]", match=MatchValue(value="AI"))

# 是否有某个字段
Filter(must=[HasIdCondition(has_id=[1,2,3])])         # 指定 ID
Filter(must=[IsEmptyCondition(is_empty={"key": "deleted_at"})])  # 字段为空
Filter(must=[IsNullCondition(is_null={"key": "title"})])         # 字段为 null
```

## 6.3 Payload索引创建

```python
# 在collection创建时定义
client.create_collection(
    collection_name="my_base",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    # 定义payload索引
    payload_schema={
        "category": "keyword",
        "price": "float",
        "stock": "integer",
        "description": "text",
        "warehouse_location": "geo",
        "listed_at": "datetime",
    }
)

# 动态创建索引(已经存在的collection)
client.create_payload_index(
    collection_name="my_base",
    field_name="brand",
    field_schema="keyword",
    wait=True
)
```

## 6.4 Filterable HNSW 过滤优先于搜索

Qdrant 支持在 HNSW 图遍历过程中应用过滤器，这比先搜索后过滤快约 1.6 倍

```python
# 原理：图遍历时跳过不满足过滤条件的节点，而非遍历完再过滤
results = client.query_points(
    collection_name="my_base",
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="tech")),
        ]
    ),
    search_params=models.SearchParams(
        filter_condition_check="during_traversal",  # 图遍历时检查
        # 或 "after_traversal"（先搜后过滤，兼容旧版）
    ),
    limit=5
)
```

# 7.混合搜索与多向量

## 7.1 命名向量(Named Vectors)

一个 Point 可以存储多个不同用途的向量

```python
# 创建支持多向量的collection
client.create_collection(
    collection_name="my_base",
    vectors_config={
        "text": VectorParams(size=768, distance=Distance.COSINE),
        "image": VectorParams(size=512, distance=Distance.COSINE),
    }
)

# 插入多向量的数据
client.upsert(
    collection_name="my_base",
    points=[
        PointStruct(
            id=1,
            vector={
                "text": text_embedding,  # 768维向量
                "image": image_embedding,  # 512维向量
            },
            payload={"title": "Red Sneakers", "price": 79.99},
        )
    ]
)

# 指定用哪个向量搜索
results = client.query_points(
    collection_name="my_base",
    query=image_embedding,
    using="image",  # 用图片向量搜索
    limit=5
)
```

## 7.2 Sparse Vectors 稀疏向量 + BM25

```python
# 创建支持稀疏向量的collection
from qdrant_client.models import SparseVectorParams

client.create_collection(
    collection_name="my_base",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(size=1000000)  # 稀疏向量的最大长度
    }
)

# 插入稀疏向量的数据(BM25生成的词频向量)
from qdrant_client.models import SparseVector

client.upsert(
    collection_name="my_base",
    points=[
        PointStruct(
            id=1,
            vector={
                "dense": dense_embedding,
                "sparse": SparseVector(indices=[1, 2, 3],  # 非零元素的索引
                                       values=[0.1, 0.2, 0.3])  # 对应的值
            },
            payload={"text": "Qdrant supports hybrid search"}
        )
    ]
)

# 稀疏向量搜索(适合关键词精确匹配)
results = client.query_points(
    collection_name="my_base",
    query=SparseVector(indices=[101, 404], values=[0.1, 0.2]),
    using="sparse",
    limit=5
)

```

## 7.3 RRF(Reciprocal Rank Fusion) 混合搜索

```python
from qdrant_client.models import QueryRequest, Fusion, Sample, Prefetch, FusionQuery

# 同时发多个查询

results = client.query_points(
    collection_name="my_base",
    prefetch=[
        # 子查询1：稠密向量搜索
        Prefetch(
            query=dense_query_vector,
            using="dense",
            limit=5
        ),
        # 子查询2：稀疏向量搜索
        Prefetch(
            query=SparseVector(indices=[101, 404], values=[0.1, 0.2]),
            using="sparse",
            limit=5
        )
    ],
    # 智能合并成一组最合理的结果, 最后重新排序，输出最公平、最综合的最终结果
    query=FusionQuery(
        fusion=Fusion.RRF  # 倒数排名融合
    ),
    limit=5
)
```

## 7.4 相关性反馈搜索

```python
# 第一步：初始搜索
initial_results = client.query_points(
    collection_name="my_base",
    query=query_vector,
    limit=5
)

# 第二步：用户反馈
relevant_ids = [p.id for p in initial_results.points[:5]]   # 前5个相关
irrelevant_ids = [p.id for p in initial_results.points[15:]] # 最后5个不相关

# 第三步：用反馈优化搜索
refined_results = client.query_points(
    collection_name="my_base",
    query=query_vector,
    limit=5,
    search_params=SearchParams(
        relevance_feedback=RelevanceFeedback(
            positive=relevant_ids,  # 正反馈
            negative=irrelevant_ids  # 负反馈
        )
    )
)
```

# 8. 索引优化

## 8.1 HNSW 索引原理

分层可导航小世界图

是向量数据库（Qdrant/ES/Milvus）默认用的高性能近似最近邻 ANN 索引，专门解决：亿级向量快速相似度检索。

高层：节点少、稀疏、做「快速跳转导航」
中层：过渡
底层：包含全量所有节点，做精细搜索

## 8.2 关键参数

```python
from qdrant_client.models import HnswConfig

HnswConfig(
    m=16,  # 每个节点最大连接数
    ef_construct=100,  # 索引构建时的搜索宽度, 范围：4*m 到10*m，增大更高质量的图，但构建更慢，推荐：100-200
    full_scan_threshold=10000,  # 向量小于此值时，不建索引，全量搜索(暴力搜索)
    max_indexing_threads=0,  # 索引构建最大线程数，0为自动
    on_disk=False,  # 索引存磁盘(True=省内存但慢)
    payload_m=None,  # payload 索引的 M 值（独立于向量索引）
)
```

## 8.3 调参实践

```python
# 场景1：高召回率（问答系统、法律检索）
high_recall_config = HnswConfig(
    m=32,                    # 更高连通性
    ef_construct=200,        # 构建时更充分搜索
)
# 搜索时使用更高 ef：
search_params = models.SearchParams(hnsw_ef=256)

# 场景2：高吞吐低延迟（推荐系统、实时搜索）
high_throughput_config = HnswConfig(
    m=8,                     # 降低内存和构建成本
    ef_construct=64,
)
search_params = models.SearchParams(hnsw_ef=64)  # 保持默认

# 场景3：存储敏感（边缘设备、大容量）
memory_efficient_config = HnswConfig(
    m=8,
    on_disk=True,            # 索引放磁盘
)
```

**关键认知：hnsw_ef 增大提高召回率但降低吞吐**

```
hnsw_ef=64  → QPS: 500, 召回率: 95%
hnsw_ef=128 → QPS: 300, 召回率: 97%
hnsw_ef=256 → QPS: 120, 召回率: 98.5%
```

## 8.4 大数据批量导入最佳实践

```python
# 步骤1：创建 Collection 但不建 HNSW 索引
client.create_collection(
    collection_name="large_dataset",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    hnsw_config=HnswConfig(m=0),              # ★ 关键：m=0 禁用 HNSW
    optimizers_config=OptimizersConfig(
        indexing_threshold=1_000_000,          # 延迟索引
        default_segment_number=8,
    ),
)

# 步骤2：批量导入（并行上传，此时不建索引，速度极快）
import concurrent.futures

def upload_batch(batch_data):
    points = [PointStruct(id=d[0], vector=d[1], payload=d[2]) for d in batch_data]
    client.upsert(collection_name="large_dataset", points=points)

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    executor.map(upload_batch, data_batches)

# 步骤3：导入完成后，启用 HNSW
client.update_collection(
    collection_name="large_dataset",
    hnsw_config=HnswConfig(
        m=16,
        ef_construct=100,
    ),
    optimizers_config=OptimizersConfig(
        indexing_threshold=20000,
    ),
)

# 步骤4：等待索引构建完成
import time
while True:
    info = client.get_collection("large_dataset")
    if info.indexed_vectors_count >= info.vectors_count * 0.99:
        print("索引构建完成")
        break
    print(f"索引进度: {info.indexed_vectors_count}/{info.vectors_count}")
    time.sleep(5)
```

## 8.5 段Segment优化

```python
# Qdrant 内部将 Collection 划分为多个 Segment
# 段数过多 → 搜索要访问更多段 → 延迟增加
# 段数过少 → 索引构建慢、内存峰值高

optimizers_config = OptimizersConfig(
    default_segment_number=2,           # 目标段数
    max_segment_size=200000,            # 单个段最大向量数
    memmap_threshold=50000,             # 超过此大小用 mmap
    indexing_threshold=20000,           # 超过此向量数才建索引
    
    # 清理策略
    flush_interval_sec=5,               # WAL 刷盘间隔
    vacuum_min_vector_number=1000,      # 触发清理的最小向量数
    
    # 段合并
    max_optimization_threads=2,         # 优化线程数（控制 CPU 占用）
)
```

# 9.量化与内存优化

## 9.1 内存消耗估算

一条经验公式：对于 100 万个 768 维 float32 向量
```
原始向量: 1M × 768 × 4 bytes = 3.07 GB
HNSW 索引 (m=16): 约 1M × 16 × 2 × (4+8) bytes ≈ 384 MB
Payload 索引: 取决于字段数和基数
总计: 约 3.5-4 GB


# 量化后：
int8 量化: 1M × 768 × 1 byte = 768 MB (4×压缩)
binary 量化: 1M × 768 / 8 bytes = 96 MB (32×压缩)
```

## 9.2 Scalar量化(int8)

```python
from qdrant_client.models import QuantizationConfig, ScalarQuantization


# 在创建collection时配置
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    quantization_config=QuantizationConfig(
        scalar=ScalarQuantization(
            type="int8",  # 将float32量化为int8
            quantile=0.99,  # 忽略1%的离群值
            always_ram=True  # *** 量化向量常驻内存，全精度放磁盘
        )
    )
)

# # 对已存在的 Collection 启用量化
client.update_collection(
    collection_name="existing_collection",
    quantization_config=QuantizationConfig(
        scalar=ScalarQuantization(type="int8", always_ram=True)
    ),
)
```

**量化取舍：**

- 召回率损失通常 1-3%，大多数场景可接受
- 内存节省 4×，QPS 提升明显（更多数据可放入内存）
- 创建索引的时间也会缩短

## 9.3 Binary量化(Binary)

```python
from qdrant_client.models import BinaryQuantization

client.create_collection(
    collection_name="binary_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    quantization_config=QuantizationConfig(
        binary=BinaryQuantization(
            always_ram=True,
        )
    ),
)


# Binary 量化需要配合 over-sampling + rescore 维持召回率
results = client.query_points(
    collection_name="binary_collection",
    query=query_vector,
    limit=10,
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            ignore=False,          # 使用量化向量搜索
            rescore=True,          # ★ 用全精度向量重新打分
            oversampling=2.0,      # 先取 2× 候选再重排
        )
    ),
)
```

Binary 量化最多节省 32× 内存，但必须配合 rescore + oversampling 才能维持可接受的召回率。

## 9.4 存储层级策略

```python
# 三层存储策略：平衡性能与成本
client.create_collection(
    collection_name="tiered_storage",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
        on_disk=True,              # 第3层：全精度向量存磁盘
    ),
    hnsw_config=HnswConfig(
        m=16,
        on_disk=True,              # 第3层：HNSW 图也存磁盘
    ),
    quantization_config=QuantizationConfig(
        scalar=ScalarQuantization(
            type="int8",
            always_ram=True,       # 第1层：量化向量常驻内存（热数据）
        )
    ),
    optimizers_config=OptimizersConfig(
        memmap_threshold=20000,    # 第2层：mmap 映射（温数据）
    ),
)
```

## 9.5 诊断先于治疗

在盲目启用量化之前，先确认内存占用的真正来源

```python
# 查看collection 的遥测数据
# 查看 Collection 的遥测数据
import requests

telemetry = requests.get("http://localhost:6333/collections/knowledge_base/telemetry").json()

print(f"向量数量: {telemetry['result']['vectors_count']}")
print(f"段数: {telemetry['result']['segments_count']}")
for segment_id, seg_data in telemetry['result']['segments'].items():
    print(f"段 {segment_id}:")
    print(f"  向量大小: {seg_data['vector_size'] / 1024 / 1024:.1f} MB")
    print(f"  Payload 大小: {seg_data['payload_size'] / 1024 / 1024:.1f} MB")
    print(f"  索引大小: {seg_data['hnsw_index_size'] / 1024 / 1024:.1f} MB")
```

**决策树：**

```
内存超限？
├─ 向量本身占大头？ → 启用量化
├─ HNSW 图占大头？ → 减小 m, 或 on_disk=true
├─ Payload 占大头？ → 修剪 payload 或 on_disk_payload=true
└─ 段数过多？ → 减少 default_segment_number
```

# 10.多租户架构

## 10.1 反模式：一租户-collection

```python
# ❌ 错误做法：每个用户一个 Collection
for tenant_id in range(1000):
    client.create_collection(
        collection_name=f"tenant_{tenant_id}",  # 1000个 Collection！
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
# 问题：每个 Collection 都有独立的 WAL、Segment、HNSW 图
# 1000个 Collection 的元数据开销就可达 GB 级别
```

## 10.2 正确做法：单collection+is_tenant

```python
#  推荐：单 Collection 多租户
client.create_collection(
    collection_name="tenants",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    # 多租户配置
    shard_number=4,  # 分片数
    replication_factor=2,  # 副本数
)

# 在payload中标记租户
client.upsert(
    collection_name="tenants",
    points=[
        PointStruct(
            id=global_id,
            vector=vec,
            payload={
                "tenant_id": tenant_id,  # 租户标识
                "document_type": "invoice"
            }
        )
    ]
)

# 创建租户专用索引(is_tenant=True是关键)
client.create_payload_index(
    collection_name="tenants",
    field_name="tenant_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
    wait=True
)



```

is_tenant 标志的作用：
- 构建 per-tenant 的 HNSW 子图
- 搜索时如果指定 tenant_id 过滤，只遍历该租户的子图
- 大幅减少搜索时需要检查的节点数

## 10.3 分层多租户

```python
# 小租户共享分片，大租户独占分片
# Qdrant 自动检测租户数据量并决定是否需要提升

# 创建支持分层多租户的collection
client.create_collection(
    collection_name="tenants",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    shard_number=3,  # 初始分片数
    sharding_method=models.ShardingMethod.CUSTOM,  # 分片方式(CUSTOM大租户专用)
    # 分片键映射(大租户独占分片)
    shard_key={  # 自定义分片键配置
        "shard_1": models.ShardKey(
            shard_key="shard_1",  # 插入数据时必须带这个 shard_key 才能写入对应分片
            points_count=500000,  # 预分配点数（预估该分片存多少数据）
            state=models.ShardState.ACTIVE  # 分片状态
        )
    }
)
```

## 10.4 租户隔离搜索

```python
# 搜索时强制指定租户
results = client.query_points(
    collection_name="tenants",
    query=query_vector,
    query_filter=Filter(
        must=[
            FilterCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id)
            )
        ]
    ),
    limit=5
)
```

# 11.生产部署

## 11.1 集群配置

```
最小生产集群（3节点）：
┌─────────────────────────────────────┐
│           负载均衡器 (nginx)          │
├──────────┬──────────┬───────────────┤
│  Node 1  │  Node 2  │   Node 3      │
│ Shard 0  │ Shard 0  │  Shard 1      │
│ Shard 1  │ Shard 2  │  Shard 2      │
│ (Raft    │ (Raft    │  (Raft        │
│  Leader) │Follower) │  Follower)    │
└──────────┴──────────┴───────────────┘

关键配置：
- 最少 3 节点（Raft 共识需要奇数）
- replication_factor ≥ 2（每份数据至少存两份）
- shard_number 对齐节点数
```

## 11.2 Docker Compose 集群部署
```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant-node-1:
    image: qdrant/qdrant:v1.17.1
    container_name: qdrant-node-1
    restart: always
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/node1:/qdrant/storage
    environment:
      QDRANT__CLUSTER__ENABLED: "true"
      QDRANT__CLUSTER__P2P__PORT: 6335
      QDRANT__CLUSTER__CONSENSUS__MAX_MESSAGE_QUEUE_SIZE: 10000
    command: >
      ./qdrant
      --uri 'http://qdrant-node-1:6335'

  qdrant-node-2:
    image: qdrant/qdrant:v1.17.1
    container_name: qdrant-node-2
    restart: always
    volumes:
      - ./data/node2:/qdrant/storage
    environment:
      QDRANT__CLUSTER__ENABLED: "true"
      QDRANT__CLUSTER__P2P__PORT: 6335
    command: >
      ./qdrant
      --bootstrap 'http://qdrant-node-1:6335'
      --uri 'http://qdrant-node-2:6335'

  qdrant-node-3:
    image: qdrant/qdrant:v1.17.1
    container_name: qdrant-node-3
    restart: always
    volumes:
      - ./data/node3:/qdrant/storage
    environment:
      QDRANT__CLUSTER__ENABLED: "true"
      QDRANT__CLUSTER__P2P__PORT: 6335
    command: >
      ./qdrant
      --bootstrap 'http://qdrant-node-1:6335'
      --uri 'http://qdrant-node-3:6335'
```

## 11.3 创建分布式collection

```python
client.create_collection(
    collection_name="distributed_kb",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    shard_number=3,              # 分片数（对齐节点数）
    replication_factor=2,        # 每片 2 个副本
    write_consistency_factor=2,  # 写入一致性的最小确认数
)
```

**Consistency Factor 选择：**

- `write_consistency_factor = replication_factor`：强一致性（金融、订单系统）
- `write_consistency_factor = replication_factor // 2 + 1`：多数确认（推荐通用场景）
- `write_consistency_factor = 1`：最终一致性（推荐系统、非关键场景，写入最快）

## 11.4 快照与备份

```python
# 创建快照
client.create_snapshot(collection_name="production_kb")

# 恢复快照
client.recover_snapshot(
    collection_name="production_kb_restored",
    snapshot_url="file:///qdrant/snapshots/production_kb/snapshot-2026-05-13.snapshot",
)
```

Qdrant Cloud 上的自动备份（Web UI 或 API 配置）
- 支持每日自动备份
- 可配置保留天数

## 11.5 版本升级注意事项

```python
# ⚠️ Qdrant 不支持跨版本降级
# 升级路径必须逐版本进行：
# 1.15 → 1.16 → 1.17（正确）
# 1.15 → 1.17（错误！可能导致数据损坏）

# 升级前检查清单：
# 1. 确认当前版本
print(client.service_version())

# 2. 创建快照
client.create_snapshot(collection_name="all_collections")

# 3. 逐个节点滚动升级（集群模式）
#    先升级 Follower → 最后升级 Leader

# 4. 验证升级后的功能
info = client.get_collection("critical_collection")
assert info.status == "green"
```

# 12.性能优化

## 12.1 全链路优化

```
                       优化检查点
                     ↓             ↓
Embedding 生成 → Qdrant 写入 → 索引构建 → 搜索查询 → 结果返回
    ↓               ↓            ↓          ↓
  批量调用        并行上传     延期索引   批量查询
  模型缓存        WAL配置     m=0大导    分组API
```

## 12.2 写入优化

```python
# 1. 使用异步客户端 + 并行上传
from qdrant_client import AsyncQdrantClient
import asyncio

async def bulk_upload_async(data_batches):
    client = AsyncQdrantClient(url="http://localhost:6333")
    async def upload(batch):
        points = [PointStruct(id=i, vector=vector) for i, vector in enumerate(batch)]
        await client.upsert(
            collection_name="my_collection",
            points=points,
            wait=False  # 异步等待
        )

    # 控制并发数
    semaphore = asyncio.Semaphore(20)
    async def bounded_upload(batch):
        async with semaphore:
            return await upload(batch)
    await asyncio.gather(*[bounded_upload(batch) for batch in data_batches])

# 2. 禁用索引 + 增大 WAL
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    hnsw_config=HnswConfig(m=0),  # 禁用hnsw索引
    wal_config=WalConfig(wal_capacity_mb=1024,  # 增大WAL 减少刷盘次数
                         wal_segments_ahead=1),
    optimizers_config=OptimizersConfig(
        indexing_threshold=10000,  # 推迟索引
        max_optimization_threads=4,
    ),
)

# 3. 导入完成后启用索引
client.update_collection(
    collection_name="my_collection",
    hnsw_config=HnswConfig(m=16, ef_construct=100),
    optimizers_config=OptimizersConfig(
        indexing_threshold=10000,
    )
)
```

## 12.3 搜索优化

```python
# 1. 使用批量搜索代替逐条搜索
# 慢：N次网络往返

for query in user_queries:
    result = client.query_points(query=query, limit=5)

# 快：1次网络往返
results = client.query_batch_points(
    collection_name="kb",
    requests=[models.QueryRequest(query=query, limit=5) for query in user_queries]
)

# 2. 异步搜索
async def search_concurrently(queries):
    client = AsyncQdrantClient(url="http://localhost:6333")
    tasks = [
        client.query_points(
            collection_name="kb",
            query=q,
            limit=5,
        )
        for q in queries
    ]
    return await asyncio.gather(*tasks)


# 3. 搜索参数权衡
# 低延迟场景（推荐、实时搜索）
fast_params = models.SearchParams(hnsw_ef=64, exact=False, indexed_only=True)

# 高精度场景（问答、法律、医疗）
accurate_params = models.SearchParams(hnsw_ef=256, exact=False, indexed_only=False)

# 4. 使用 Payload 索引避免全表扫描
# 没有索引 → 扫描所有向量 → 过滤 → 返回（慢）
# 有索引 → 只访问匹配过滤条件的向量 → 返回（快 10-100×）
client.create_payload_index(
    collection_name="kb",
    field_name="status",
    field_schema="keyword",
    wait=True,
)
```

## 12.5 冷启动问题

```python
# 服务重启后首次查询可能很慢（100×+）
# 原因：OS page cache 为空，需要从磁盘加载数据

# 缓解策略1：预热脚本
def warmup(client, collection_name="kb", num_queries=100):
    import random, numpy as np
    for _ in range(num_queries):
        random_vector = np.random.randn(768).astype(np.float32).tolist()
        client.query_points(
            collection_name=collection_name,
            query=random_vector,
            limit=1,
        )

# 缓解策略2：使用 always_ram=true 的量化配置
# 让热数据始终在内存中，减少 cold cache 影响

# 缓解策略3：Kubernetes 的 preStop hook
# 在 Pod 终止前执行请求，让新 Pod 预热
```

# 13. 监控与可观测性

## 13.1 关键指标

```python
# 获取 Collection 级别的遥测数据
import requests

# Collection 遥测
response = requests.get(
    "http://localhost:6333/collections/knowledge_base/telemetry"
)

telemetry = response.json()["result"]

print(f"向量总数: {telemetry['vectors_count']}")
print(f"已索引向量: {telemetry['indexed_vectors_count']}")
print(f"段数: {telemetry['segments_count']}")

# 集群遥测
cluster_telemetry = requests.get(
    "http://localhost:6333/cluster/telemetry"
).json()

# 索引优化状态
opt_status = requests.get(
    "http://localhost:6333/collections/knowledge_base/optimizations"
).json()
```

## 13.2 Prometheus 指标

```bash
# Qdrant 1.17 支持独立的 /metrics 端口
# 启动时指定：
# docker run -p 6333:6333 -p 9090:9090 qdrant/qdrant \
#   --metrics-port 9090

# prometheus.yml
scrape_configs:
  - job_name: 'qdrant'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

关键 Prometheus 指标：

- `qdrant_collection_vectors_count`：向量总数
- `qdrant_collection_indexed_vectors_count`：已索引向量数（差距大=索引滞后）
- `qdrant_grpc_responses_duration_seconds`：请求延迟
- `qdrant_rest_responses_duration_seconds`：HTTP 请求延迟
- `qdrant_segments_count`：段数

## 13.3 健康检查

```python
# 应用层健康检查
def health_check(client):
    try:
        # 不仅检查服务活着，还要验证搜索功能正常
        test_vector = [0.0] * 768
        client.query_points(
            collection_name="health_check_collection",
            query=test_vector,
            limit=1,
        )
        return True, "healthy"
    except Exception as e:
        return False, str(e)

# Collection 状态
info = client.get_collection("kb")
# status 可能值：
# - "green": 正常
# - "yellow": 正在优化（索引构建中）
# - "red": 异常（副本不可用）
print(f"状态: {info.status}")
```

# 14. 面试高频问题

**搜索延迟从 10ms 飙升到 200ms，怎么排查？**

```
第1层：确认是基础设施问题还是 Qdrant 本身问题
├─ 检查 CPU、内存、磁盘 I/O 使用率
├─ 检查网络延迟（是否跨 AZ？）
└─ 检查是否有其他进程争抢资源

第2层：检查 Qdrant 内部状态
├─ segments_count 是否激增？→ 段过多，每个查询需要遍历更多段
├─ indexed_vectors_count vs vectors_count 差距是否大？→ 索引滞后
├─ optimizer_status 是否显示长时间 yellow？
└─ 是否有大量 tombstone 积累？（频繁删除导致）

第3层：检查索引和查询参数
├─ hnsw_ef 是否被调得过高？
├─ 是否从内存模式切到了磁盘模式？
└─ 过滤条件是否走了全表扫描（缺少 payload 索引）？

第4层：检查数据层面
├─ 向量维度是否与创建时一致？
├─ 是否有大量零向量污染？
└─ Payload 是否膨胀？
```

```python
# 快速诊断脚本
def diagnose_slow_search(client, collection_name):
    issues = []
    
    info = client.get_collection(collection_name)
    telemetry = requests.get(
        f"{client._client.rest_uri}/collections/{collection_name}/telemetry"
    ).json()["result"]
    
    # 检查1：段数过多
    if telemetry["segments_count"] > 10:
        issues.append(f"段数过多({telemetry['segments_count']})，建议减少 default_segment_number")
    
    # 检查2：索引滞后
    indexed = telemetry.get("indexed_vectors_count", 0)
    total = telemetry.get("vectors_count", 0)
    if total > 0 and indexed / total < 0.9:
        issues.append(f"索引滞后({indexed}/{total} = {indexed/total:.1%})")
    
    # 检查3：状态异常
    if info.status != "green":
        issues.append(f"Collection 状态异常: {info.status}")
    
    return issues
```

**为什么存入的向量和查出来的不一样？（Cosine 归一化陷阱）**

```python
# 结果不一样！
# 原因：Cosine 距离时，Qdrant 自动对向量做 L2 归一化
# [0.5, 0.5, 0.5, 0.5] → [0.5, 0.5, 0.5, 0.5] 恰好已经是单位向量
# 但对于其他向量，归一化会改变值

# 解决方案：
# 1. 如果必须保留原始向量，使用 Dot 距离
# 2. 在应用层自己归一化（等价于 Cosine）
import numpy as np
normalized = original / np.linalg.norm(original)
```

**大量删除后内存不释放怎么办？**

```python
# 问题重现：
# 删除了 50% 的数据，但内存使用几乎没降

# 原因：
# Qdrant 的删除是"软删除"（标记 tombstone）
# 被标记的点在段重建之前不会真正释放内存
# 段重建需要等待 optimizer 触发

# 解决方案1：手动触发优化
client.update_collection(
    collection_name="my_collection",
    optimizers_config=OptimizersConfig(
        vacuum_min_vector_number=100,  # 降低触发清理的阈值
    ),
)

# 解决方案2：对时间分界数据用 Shard Key Rotation
# 按月建分片，过期直接删分片（即时释放，无 tombstone）
client.create_collection(
    collection_name="logs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    shard_number=1,
    sharding_method=models.ShardingMethod.CUSTOM,
)

# 写入时指定分片键
client.upsert(
    collection_name="logs",
    points=[...],
    shard_key_selector="2026-05",  # 当前月
)

# 删除旧月分片（即时释放）
# Qdrant Cloud 支持，开源版需额外配置
```

**OOM 崩溃怎么解决？**

```
OOM 原因诊断：
1. 向量全部在内存 → 启用量化或 on_disk
2. HNSW 图在内存且 m 过大 → 减小 m 或 on_disk
3. Payload 膨胀 → 修剪或用 on_disk_payload
4. 段数过多 → 减少 default_segment_number
5. 批量写入时缓存峰值 → 分批写入 + 减小 WAL
6. Page cache 占满（非泄漏！）→ 这是正常的 OS 行为
```

```python
# 内存优化配置模板
client.create_collection(
    collection_name="memory_optimized",
    vectors_config=VectorParams(
        size=768, distance=Distance.COSINE, on_disk=True  # 向量存磁盘
    ),
    hnsw_config=HnswConfig(
        m=8,           # 低连接数
        on_disk=True,  # 索引存磁盘
    ),
    quantization_config=QuantizationConfig(
        scalar=ScalarQuantization(
            type="int8",
            always_ram=True,  # 仅量化向量在内存
        )
    ),
    optimizers_config=OptimizersConfig(
        default_segment_number=2,            # 少段数
        memmap_threshold=20000,
    ),
)

# 预期内存：原来 32GB → 约 6-8GB（对 1000 万 768 维向量）
```

**多租户下某租户搜索变慢，如何隔离？**

```python
# 问题：SaaS 平台中，某大租户的数据增长导致全局搜索变慢

# 诊断步骤：
# 1. 确认是否使用了 is_tenant 索引
# 2. 检查 HNSW 图是否被大租户主导
# 3. 考虑分层多租户

# 解决方案：分层多租户（Tiered Multitenancy）
# - 小租户：共享分片（m=0，全扫描即可，数据量小）
# - 中租户：共享分片 + is_tenant + payload_m 子图
# - 大租户：独占分片 + 独立 HNSW 索引

# 在 Qdrant Cloud 中，大租户可以自动提升到独立分片
```

**版本升级导致服务不可用怎么办？**

```python
# 升级前强制检查清单：
# 1. 备份数据
# 2. 阅读 Release Notes 中 Breaking Changes
# 3. 逐版本升级（不能跳版本）
# 4. 集群模式：先升级 Follower，最后升级 Leader

# 升级脚本示例：
import subprocess, time

def rolling_upgrade(nodes):
    snapshots = []
    for node in nodes:
        # 1. 创建快照
        snap = client.create_snapshot(collection_name="critical_data")
        snapshots.append(snap)
    
    # 2. 先升级 Follower
    for node in nodes[1:]:
        upgrade_node(node)
        time.sleep(30)  # 等待稳定
        assert node_healthy(node)
    
    # 3. 最后升级 Leader（触发 Leader 选举）
    upgrade_node(nodes[0])
    time.sleep(30)
    assert node_healthy(nodes[0])
    assert cluster_status() == "green"
```

**Embedding 模型换了，已有数据怎么办？**

```python
# 场景：从 text-embedding-ada-002 (1536d) 换到 bge-large-v1.5 (1024d)

# 错误做法：
# 直接改 Collection 的向量维度（不支持）

# 正确做法（零停机迁移）：
# 1. 创建新 Collection（新维度）
client.create_collection(
    collection_name="kb_v2",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# 2. 双写：同时写入旧和新 Collection
def dual_write(data, old_vec, new_vec):
    # 写旧 Collection
    client.upsert(collection_name="kb_v1", points=[...old_vec])
    # 写新 Collection
    client.upsert(collection_name="kb_v2", points=[...new_vec])

# 3. 后台回填历史数据
def backfill_historical():
    old_points = scroll_all("kb_v1")
    for batch in old_points:
        # 用新模型重新编码
        new_vectors = new_model.encode([p.payload["text"] for p in batch])
        client.upsert(collection_name="kb_v2", points=batch)

# 4. 验证通过后切换查询流量到 kb_v2
# 5. 删除旧 Collection
```

**如何保证搜索结果的业务一致性？**

```python
# 问题：用户刚上传的文档搜不到（写入延迟 vs 索引延迟）

# 方案1：写入后立即探测
def upsert_with_verification(client, collection_name, points, timeout=10):
    client.upsert(collection_name=collection_name, points=points, wait=True)
    
    # 用刚写入的向量立即搜索验证
    start = time.time()
    while time.time() - start < timeout:
        result = client.query_points(
            collection_name=collection_name,
            query=points[0].vector,
            limit=1,
            search_params=models.SearchParams(indexed_only=False),  # ★ 包含未索引段
        )
        if result.points and result.points[0].id == points[0].id:
            return True
        time.sleep(0.5)
    return False

# 方案2：Write-After-Read 模式
# 写入后立即用 indexed_only=False 查询验证
# 如果搜不到，降级到精确检索或报错

# 方案3：读写分离 + 最终一致性
# 允许短暂不可见（适合推荐、内容发现场景）
# 用 wait=False 加速写入，接受 1-5 秒的索引延迟
```
