# 1.PGVector是什么？

PGVector 是一个 Postgres 扩展，用于向量存储和相似度检索。

本质是: Postgres 加向量存储 + 相似度检索能力

关系型数据库 + 向量数据库 二合一，ACID 事务、SQL 语法、向量搜索全支持

用于:
+ 存储embedding 向量
+ 向量相似度检索
+ RAG 检索
+ AI Agent Memory
+ Semantic Search
+ 推荐系统

## 1.1 PGVector/Qdrant/Milvus的区别

### 1.1.1 核心定位对比

| 维度       | pgvector             | Qdrant          | Milvus          |
| -------- | -------------------- | --------------- | --------------- |
| **本质**   | PostgreSQL 扩展，非独立数据库 | Rust 编写的轻量专用向量库 | Go 编写的企业级分布式向量库 |
| **核心主打** | 关系 + 向量混合、零额外运维、强事务  | 低延迟、高过滤性能、单机高性能 | 水平扩展、十亿级向量、高可用  |
| **架构形态** | 单机（依赖 PostgreSQL）    | 单机为主，支持分布式      | 原生分布式集群         |

### 1.1.2 关键能力对比

| 能力维度             | pgvector                                       | Qdrant                               | Milvus                               |
| ---------------- | ---------------------------------------------- | ------------------------------------ | ------------------------------------ |
| **规模上限**         | 单机 ≤500 万（舒适区），极限约 5000 万                      | 单机 ≤1 亿（HNSW + 磁盘），分布式可达数亿           | 分布式 十亿～百亿级，生产稳定                      |
| **P99 延迟 / QPS** | ~120ms / 500 QPS                               | <20ms / 1500+ QPS                    | ~50ms / 1000+ QPS                    |
| **索引支持**         | IVFFlat、HNSW（v0.7+），仅余弦 / 内积 / L2              | HNSW（内存 + 磁盘）、支持量化（4–8x 压缩）          | HNSW、IVF、DiskANN、GPU 索引，最丰富          |
| **元数据过滤**        | 一般，依赖 Postgres 索引，混合查询需手写 SQL                  | 优秀，原生优化标签 / 时间 / 数值过滤，复杂条件快          | 优秀，支持多字段过滤、范围查询、布尔组合                 |
| **事务与一致性**       | 完整 ACID，强一致                                    | 最终一致，无事务                             | 弱一致（分布式），无跨节点事务                      |
| **部署运维**         | 极低，已有 Postgres 直接安装扩展，复用 DBA 技能                | 低，单 Docker 容器，内置 Dashboard，新手友好      | 高，依赖 etcd、MinIO、Kafka，需分布式运维经验       |
| **生态集成**         | LangChain / LlamaIndex 支持好，SQL 友好，适合关系型数据 + 向量 | Python / Rust / Go SDK 全，RAG / 推荐生态强 | 生态最完善，Attu 可视化、Zilliz Cloud 托管、企业级支持 |


### 1.1.3 选型决策树

| 如果你符合以下条件                                                                        | 推荐选择         | 核心理由                     |
| -------------------------------------------------------------------------------- | ------------ | ------------------------ |
| 已有 PostgreSQL 栈，不想引入新组件；向量规模 <500 万；需要强事务（文档频繁更新 / 删除）；做轻量 RAG / 内部工具 / POC      | **pgvector** | 零额外运维、SQL 熟悉、事务安全        |
| 追求极致低延迟（对话 Agent、实时推荐）；向量规模 100 万～1 亿，单机足够；依赖复杂元数据过滤（多标签、时间范围）；喜欢简单部署 + 高性能，预算有限 | **Qdrant**   | Rust 优化、低延迟首选、过滤性能强、运维简单 |
| 向量规模 >1 亿，需分布式水平扩展；企业级生产环境，要求高可用 / 容灾；高并发（QPS>1000），需独立集群；做大规模 RAG、召回引擎、多模态搜索    | **Milvus**   | 分布式、高可用、生态全、水平扩展能力最强     |


### 1.1.4 Agent/RAG场景推荐

| 场景                  | 推荐方案         | 选型理由                  |
| ------------------- | ------------ | --------------------- |
| 个人 / 小团队 Agent 开发   | **pgvector** | 低成本、易上手、SQL 熟悉、轻量 RAG |
| 生产级 RAG（千万级向量）      | **Qdrant**   | 低延迟、高过滤、运维简单、单机高性能    |
| 企业级多 Agent 系统（亿级向量） | **Milvus**   | 分布式、高可用、生态全、可水平扩展     |


### 1.1.5 HNSW/IVFFlat/Flat/IVF/DiskANN/GPU索引

| 索引技术        | 存储位置  | 延迟        | 规模上限     | 召回率        | 核心优势        | 核心劣势          |
| ----------- | ----- | --------- | -------- | ---------- | ----------- | ------------- |
| **HNSW**    | 内存    | 低（1-5ms）  | 单机千万级    | 极高（>95%）   | 通用、召回高、增量插入 | 内存占用大         |
| **IVF**     | 内存/磁盘 | 中（5-20ms） | 亿级（配合量化） | 中高（85-95%） | 省内存、构建快     | 边界漏搜、精度略低     |
| **DiskANN** | 磁盘为主  | 中（5-20ms） | 单机十亿级    | 高（>90%）    | 超大规模、低内存成本  | 依赖 SSD、延迟不如内存 |
| **GPU 索引**  | 显存    | 极低（批量）    | 受显存限制    | 高（>95%）    | 批量吞吐极高      | 显存贵、单条查询不划算   |


**HNSW(Hierarchical Navigable Small World)**

分层可导航小世界图

+ 原理: 把向量组织成一个多层图结构
  + 最上层：像"高速公路"，节点极少，连接稀疏，用于快速定位大致区域
  + 中间层：像"国道"，节点增多，逐步缩小范围
  + 最底层：像"城市道路"，包含全部节点，连接稠密，用于精确找到最近邻
+ 特点:
  + 召回率高：通常 >95%，能找到真正的最近邻
  + 内存占用大：需要存储多层图结构，每个节点要存邻居指针
  + 增量友好：新数据插入时不需要重建整个索引
  + 构建较慢：图结构构建需要大量距离计算
+ 适用:
  + 中等规模（百万级）、内存充足、要求高召回率的通用场景。Qdrant、Milvus、pgvector 都支持。


**IVF(Inverted File Index)**

倒排文件索引

+ 原理: 先对所有向量做聚类（K-means），分成 N 个"桶"（专业叫 Voronoi 单元）
  + 建索引时：每个向量被分配到距离最近的中心点代表的桶里
  + 查询时：计算查询向量与各个中心点的距离，只去最近的几个桶里搜索，而不是搜全量

+ 特点:
  + 内存效率高：尤其是配合 PQ/SQ 量化，可压缩 10-20 倍
  + 构建快：聚类比建图快得多
  + 边界问题：如果查询向量恰好在两个桶的边界上，最近邻可能在没搜的那个桶里，导致漏搜
  + 召回率中等：通常 85-95%，不如 HNSW

+ 适用:
  + 大规模数据（千万级以上）、内存受限、能接受轻微精度损失的场景。Milvus 原生支持，pgvector 早期版本主要用这个。

**IVF_FLAT**

倒排文件索引 + 无压缩

+ 原理: 
  + 先 K-means 聚类，把向量分到 N 个桶
  + 桶内直接存原始向量（比如 768 维的 float32，每个向量 3KB）
  + 查询时：先找最近的几个桶中心 → 进桶后暴力计算桶内所有原始向量的精确距离

+ 特点:
  + 精度: 最高，因为算的是原始向量的精确距离，无信息损失
  + 内存: 最大,桶内存原始向量，无压缩
  + 速度: 取决于桶大小，桶越大越慢
  + 参数: nlist（桶数量）、nprobe（查询时搜几个桶）

+ 适用:
  + 数据量不大（百万级）、内存充裕、要求 100% 召回率的场景。

**IVF_PQ**

倒排文件索引 + 乘积量化

+ 原理: 把高维向量分段压缩
  + 分段
  + 每段单独聚类
  + 存储
  + 查询时：
    + 查询向量也分段，与各段码中心点算距离，预计算距离表
    + 桶内向量距离=查表相加，不做原始向量运算

+ 特点:
  + M（分段数，越大精度越高）、nbits（每段码本大小，通常 8）
  + 速度: 快，距离计算变查表加法
  + 内存: 极小，压缩率50-100倍
  + 精度: 中等，信息有损失

+ 适用:
  + 大规模数据、内存极度紧张的场景，比如十亿级向量单机部署。

**IVF_SQ**

倒排文件索引 + 标量量化

+ 原理: 对向量的每个维度独立压缩
  + 找每维的最小/最大值
  + 线性映射到整数
  + 存储
  + 查询时: 查询向量也量化到同样范围，计算量化后的整数距离（可用 SIMD 加速）
+ 特点:
  + 精度: 中高，比 PQ 高，比 FLAT 低，召回率 90-95%
  + 内存: 中等，压缩率 4-8 倍，不如 PQ 极端
  + 速度: 很快，整数运算 + SIMD 优化
  + 优势: 实现简单、解码快、适合 GPU

**DiskANN(Disk-Based Approximate Nearest Neighbor)**

基于磁盘的近似最近邻搜索(解决"数据量太大，内存装不下"的问题。)

+ 原理: 三层架构
  + 图索引（Vamana 图）：在内存中存导航图结构，知道去哪里找数据
  + 压缩向量：在内存中存低精度版向量（比如 32 维压缩到 8 维），用于快速预筛选
  + 原始向量 + 邻居列表：存在 SSD 磁盘上

+ 特点:
  + 单机十亿级：内存只需原来的 1/10，大部分数据放磁盘
  + 高召回率：和纯内存 HNSW 接近（>90%）
  + 延迟比内存高：通常 5-20ms（内存 HNSW 是 1-5ms）
  + 依赖 SSD 性能：需要高 IOPS 的 NVMe 盘

+ 适用:
  + 超大规模（十亿级向量）、内存预算紧张、但能接受稍高延迟的场景。Milvus 支持，Qdrant 也有类似实现（磁盘 HNSW）。

**GPU 索引**

基于 GPU 并行计算的向量索引

+ 原理: 不是一种新的"数据结构"，而是把现有索引（通常是 IVF 或 brute-force）放到 GPU 上加速
  + 把向量数据加载到显存
  + 查询时利用 GPU 的数千个 CUDA 核心并行计算距离
  + 一个查询批次（batch）同时处理，而不是 CPU 串行处理

+ 特点:
  + 批量查询吞吐极高：一次查 1000 条比 CPU 快 10-50 倍
  + 单条查询优势不大：GPU 启动有开销，查 1 条不一定比 CPU 快
  + 显存是瓶颈：RTX 4090 24GB，A100 80GB，能存的向量有限
  + 成本高：需要独立显卡或云 GPU 实例

+ 用法:
  + GPU brute-force：不建索引，直接用 GPU 暴力算距离，适合小数据集（百万级）超高并发
  + GPU IVF：把 IVF 索引放 GPU，桶内搜索并行化，适合亿级数据

+ 适用:
  + 高并发在线服务（同时很多用户查询）、批量相似性计算、推荐系统实时召回。Milvus 原生支持 GPU 索引，Zilliz Cloud 提供托管 GPU 实例。

# 2. PGVector 核心能力

+ 向量存储：存embedding
+ 向量搜索：similarity search
+ ANN检索：HNSW/IVF
+ SQL查询：原生SQL
+ 混合检索：keyword + vector
+ Metadata Filter：JSONB
+ JOIN查询：与业务数据结合
+ RAG：文档召回
+ Agent Memory：长期记忆

# 3.使用

## 3.1 向量数据类型

```sql
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
);
```
VECTOR(384)表示：384维向量

BGE-small -> 384
OpenAI ada-002 -> 1536
bge-large -> 1024

## 3.2 插入向量

```sql
INSERT INTO documents (content, embedding)
VALUES (
        'This is a document.',
        '[0.1, 0.2, 0.3, ..., 0.384]'
       );
```

## 3.3 向量距离

pgvector 支持 L2 距离、Cosine 距离和 Dot 距离

| 运算符   | 说明              |
| ----- | --------------- |
| `<->` | L2 distance     |
| `<#>` | inner product   |
| `<=>` | cosine distance |

<-> L2 距离(欧式距离)：两点之间的直线距离，未归一化，越小越像，图像检索、人脸特征、音频特征
<#> 内积(点积)：向量方向+长度的匹配程度，必须配合归一化使用，越大越像，推荐系统排序
<=> Cosine 距离(余弦距离)：只看方向，忽略长度，越小越像，最适合自然语言处理、大模型 embedding、语义检索

## 3.4 相似度搜索

```sql
SELECT *
FROM documents
ORDER BY embedding <=> '[0.2, 0.1, 0.3]'
LIMIT 5;
```

## 3.5 Python操作PGVector

pip install psycopg2-binary

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="admin",
    password="password"
)

cur = conn.cursor()

cur.execute("""
SELECT content
FROM documents
ORDER BY embedding <=> %s
LIMIT 5
""", ("[0.1, 0.2, 0.3]"))

rows = cur.fetchall()

for row in rows:
    print(row)

```

## 3.6 Embedding 模型

| 模型                     | 维度   |
| ---------------------- | ---- |
| BGE-small              | 384  |
| BGE-base               | 768  |
| OpenAI ada-002         | 1536 |
| text-embedding-3-small | 1536 |
| text-embedding-3-large | 3072 |


## 3.7 Sentence Transformers

pip install sentence-transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-small-en"
)

text = "LangChain is AI framework"
embedding = model.encode(text)
print(len(embedding))
```

## 3.8 存储embedding到PGVector

```python
from sentence_transformers import SentenceTransformer
import psycopg2

model = SentenceTransformer(
    "BAAI/bge-small-en"
)
conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="admin",
    password="password"
)

cur = conn.cursor()

text = "LangChain is AI framework"
embedding = model.encode(text).tolist()

cur.execute("""
INSERT INTO documents (content, embedding)
VALUES (%s, %s)
""", (text, embedding))

conn.commit()
```

**完整Semantic Search**
```python
# 插入数据
docs = [
    "LangChain tutorial",
    "PGVector tutorial",
    "AI Agent development",
]

query = "vector database"
query_embedding = model.encode(query).tolist()

cur.execute("""
SELECT content, embedding <=> %s AS distance
FROM documents
ORDER BY distance
LIMIT 3
""", (query_embedding,))
```

## 3.9 索引

不使用索引：全表扫描, 数据量大后：查询极慢

### 3.9.1 IVF_FLAT 索引

```sql
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### 3.9.2 HNSW 索引

```sql
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops);
```

### 3.9.3 IVF_FLAT VS HNSW

比较维度	IVF_FLAT	                HNSW
算法原理	倒排索引 + 平坦存储	        分层可导航小世界图
查询速度	中等，高召回时nprobe增大导致变慢	极快，低延迟下易达高召回
构建速度	较快（需训练聚类）	            较慢（逐点插入构图）
召回率	通过增大nprobe提升	            efSearch可控，易达高召回
内存占用	低（≈原始向量）	            高（1.2~2倍原始数据）
增量更新	支持但可能退化，需重建	        天然支持实时插入
主要参数	nlist/nprobe	            M/efConstruction/efSearch
GPU支持	成熟	                        有限
适用场景	十亿级静态库、内存敏感	        千万级在线、动态、低延迟

## 3.10 向量操作符

相似度计算方式	        运算符	索引操作类            	说明
cosine 余弦距离	    <=>	    vector_cosine_ops	    语义检索最常用，只看方向、忽略长度
L2 欧氏距离	        <->	    vector_l2_ops	        计算向量间直线距离，通用场景
inner product 内积	<#>	    vector_ip_ops	        计算负内积，需配合 DESC 排序

## 3.11 Metadata Filter JSONB

```sql
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(384)
);

INSERT INTO documents (content, metadata, embedding)
VALUES ('LangChain', '{"category": "ai"}', '[0.1, 0.2, 0.3]');


// 过滤 + 向量搜索 ->>是PostgresSQL的JSON操作符
SELECT * FROM documents
WHERE metadata->>'category' = 'ai'
ORDER BY embedding <=> '[0.1, 0.2, 0.3]'
LIMIT 5;
```

## 3.12 LangChain + PGVector

```python
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

CONNECTION = "postgresql+psycopg://admin:123456@localhost:5432/ai"

vector_store = PGVector(
    embeddings=OpenAIEmbeddings(),
    collection_name="documents",
    connection=CONNECTION
)
```

## 3.13 添加文档

```python
vector_store.add_texts(
    ["LangGraph tutorial",
    "AI Agent memory",]
)
```

## 3.14 相似度搜索

```python
results = vector_store.similarity_search(
    "What is pgvector?",
    k=3
)
```

## 3.15 Hybrid Search 混合搜索(全文检索+向量检索)

全文检索是索引优化查询，默认不区分大小写，只会匹配完整单词(搜langchain，不会匹配langchainx，slangchain)

to_tsvector(content) 是全文检索核心函数，把文本content转换成tsvector格式(全文检索专用)

plainto_tsquery 是全文检索核心函数，把文本转换成tsquery格式(全文检索专用)，会自动处理空格，多词，只做精准匹配

@@ 是全文检索匹配运算符，不是普通的等号，判断左边的文本向量是否包含右边的检索关键词

**全文检索**

```sql
SELECT * FROM documents
WHERE to_tsvector(content)
@@ plainto_tsquery('langchain');
```

**Hybrid 混合**

```sql
SELECT * FROM documents
WHERE to_tsvector(content)
@@ plainto_tsquery('langchain')
ORDER BY embedding <=> '[0.1, 0.2, 0.3]'
LIMIT 5;
```

## 3.16 Chunking(RAG核心)

1. 拆分文本，切块

chunk size设置
| 场景    | chunk |
| ----- | ----- |
| 通用    | 500   |
| Agent | 800   |
| Code  | 300   |
| 法律    | 1000  |

chunk_overlap设置，块间重叠长度，避免语义断裂

## 3.17 处理海量数据的四种优化手段

+ 1.批量插入：execute_values() psycopg2库提供的高效批量插入方法，一次性执行多条INSERT语句
  + 普通方式：单条插入，一次网络往返+事务提交
  + 批量插入：psycopg2.extras.execute_values() 多条记录打包成一条SQL，一次网络往返+事务提交
  + ```python
    from psycopg2 import extras 
    extras.execute_values(
      cursor,
      "INSERT INTO users (name, age) VALUES %s",
      data,  # [(name1, age1), (name2, age2), ...]
      page_size=1000  # 每批 1000 条
      )
    conn.commit()  # 只提交一次
    ```
+ 2.异步：asyncpg 基于asyncio，纯异步
  + psycopg2(同步)，阻塞等待，文本协议，连接池需额外管理，并发能力低
  + asyncpg(异步)，非阻塞，二进制协议(解析更快)，内置高性能连接池，并发能力强
  + 适用场景：Web服务、高并发读写(实时处理数据)、需要同时执行多个独立查询
  + ```python
    import asyncpg
    import asyncio

    async def main():
        conn = await asyncpg.connect('postgresql://user:pass@localhost/db')
        # 并发执行多个查询
        results = await asyncio.gather(
            conn.fetch("SELECT * FROM orders WHERE status='pending'"),
            conn.fetch("SELECT * FROM users WHERE active=true"),
            conn.fetch("SELECT count(*) FROM logs")
        )
        await conn.close()
    
    asyncio.run(main())
    ```
+ 3.分区表：PARTITION BY
  + 将大表按规则拆分成多个小表（物理上独立，逻辑上是一张表）。
  + 查询裁剪：带分区条件的查询只扫描相关分区，避免全表扫描
  + 维护高效：对单个分区做 VACUUM、索引重建更快
  + 数据管理：可按时间归档旧分区（直接 detach + drop）
  + ```sql
    -- 按时间范围分区（最常用，如日志表）
    CREATE TABLE events (
        id bigint,
        created_at timestamp,
        data jsonb
    ) PARTITION BY RANGE (created_at);
    
    -- 创建具体分区
    CREATE TABLE events_2024_01 PARTITION OF events
        FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
    
    CREATE TABLE events_2024_02 PARTITION OF events
        FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
    
    -- 查询时自动裁剪：只扫描 events_2024_01
    SELECT * FROM events WHERE created_at BETWEEN '2024-01-15' AND '2024-01-20';
    ```
+ 4.多索引：metadata + vector 为不同查询场景创建多个索引，覆盖各种过滤条件。
  + | 索引类型             | 适用场景                | 示例                                 |
    | ---------------- | ------------------- | ---------------------------------- |
    | **B-tree**       | 等值/范围查询（默认）         | `WHERE id = 100`, `WHERE age > 18` |
    | **Hash**         | 等值查询（比 B-tree 小）    | `WHERE name = 'Alice'`             |
    | **GiST**         | 空间数据、范围类型           | 地理坐标、时间范围                          |
    | **GIN**          | 多值类型（数组、JSONB、全文检索） | `WHERE tags @> '{ai}'`             |
    | **BRIN**         | 大块有序数据（超大数据）        | 时序数据，极小存储开销                        |
    | **HNSW/IVFFlat** | 向量相似性搜索（需 pgvector） | 语义搜索、RAG                           |
  + 在 AI/RAG 应用中，通常需要先过滤 metadata，再向量检索
  + ```sql
    -- 场景：在"技术类"文档中找语义相似的文本
    SELECT * FROM documents
    WHERE category = "technology"
    AND embedding <-> query_vec < 0.3
    ORDER BY embedding <-> query_vec
    LIMIT 5;
    ```
  + 优化要点：
    + 先用B-tree过滤掉90%的数据，再用向量检索

实际生产建议：
+ 日志/时序数据：分区表 + BRIN 索引 + 批量插入
+ AI 向量应用：pgvector + HNSW + metadata B-tree 索引 + 异步连接池
+ 高并发 Web：asyncpg + 连接池 + 合理的复合索引

## 3.18 PGVector 性能问题

+ 查询慢
  + 无所引
  + list太小
  + HNSW ef_search太低
+ 内存高
  + HNSW图结构
+ 召回不准
  + embedding 模型差
  + chunk不合理

## 3.19 HNSW 参数

+ ef_search
  + SET hnsw.ef_search = 100; 越高越准越慢
+ m 构图复杂度

## 3.20 IVF 参数

lists

WITH (lists=100)

一般是lists=sqrt(N)

## 3.21 Agent Memory

PGVector 可以作为：长期记忆

Memory结构:
+ user_id
+ conversation
+ summary
+ embedding
+ timestamp

## 3.22 面试重点（高频）

+ pgvector 原理
+ IVF vs HNSW
+ cosine similarity
+ ANN 是什么 近似最近邻搜索。
+ chunking 为什么重要
+ metadata filtering
+ hybrid search
+ embedding pipeline
+ 向量维度为什么必须一致





