1. collection schema 固化：维度、distance、payload 字段、索引字段。
2. 写入链路可靠：批量 upsert、重试、幂等 id、失败补偿。
3. 查询链路稳定：超时、限流、top_k 控制、过滤条件强制注入。
4. 数据隔离：tenant_id、权限字段、软删除字段。
5. 监控：QPS、P95/P99 延迟、内存、磁盘、segment、索引构建状态。
6. 备份恢复：snapshot、定期演练恢复。
7. 模型版本管理：embedding_model_version 写入 payload。
8. 迁移策略：新旧 collection 双写、灰度切流、回滚。
9. 评估体系：构建 query set，评估 recall、MRR、nDCG、答案命中率。

````
第 1 阶段：理解向量、embedding、相似度、collection、point、payload。
第 2 阶段：会 Docker 启动 Qdrant，会 Python create/upsert/query/filter。
第 3 阶段：接入真实 embedding 模型，做文本语义搜索。
第 4 阶段：做 RAG：文档切分、向量化、检索、拼 prompt、生成答案。
第 5 阶段：学习 payload index、过滤、多租户、更新删除、scroll。
第 6 阶段：学习 HNSW、召回率、延迟、量化、内存、磁盘。
第 7 阶段：学习 hybrid search、reranker、评估指标。
第 8 阶段：学习生产部署、监控、备份、迁移、权限和故障处理。
````

# 1.Qdrant 和 Elasticsearch 有什么区别？

```
Elasticsearch 传统强项是倒排索引、关键词检索、聚合分析；Qdrant 强项是向量相似度搜索、payload 过滤、RAG/推荐/多模态。
生产中经常组合使用：ES 做关键词和复杂文本检索，Qdrant 做语义向量检索，也可以在 Qdrant 里做 hybrid。
```

# 2.Qdrant 和 Milvus 有什么区别？

```
Qdrant和Milvus都是非常出色的开源向量数据库，是构建AI应用的重要基石。它们主要的区别在于核心定位和设计哲学：Qdrant是一个“优等生”，
擅长在单节点上将性能、易用性和过滤功能做到极致；而Milvus则是一位“指挥官”，专为大规模、分布式、云原生的企业级环境而生。

选择 Qdrant 如果：

    你的数据量在千万至十亿级，希望获得最高的单机性价比和低延迟。

    你的应用有复杂的标量（元数据）过滤需求，且对响应速度有极致要求。

    你的团队希望快速上手，避免复杂的分布式系统运维，或需要部署在边缘设备上。

选择 Milvus 如果：

    你正为百亿甚至万亿级的海量向量数据构建生产级系统。

    你的业务需要未来能够无缝扩展，希望架构具备存算分离的灵活性，并可能用到GPU加速。

    你需要一个功能丰富、生态完善、社区支持强大的成熟数据库解决方案。
```

# 3.为什么搜出来的结果不准？

```
embedding 模型不适合业务领域；
chunk 切分太粗或太碎；
query 和 document 的 embedding 方式不一致；
top_k 太小；
没有 hybrid search；
没有 reranker；
payload 过滤条件过严；
数据本身脏或缺失；
使用了错误的 distance；
向量维度或模型版本混用。
```

# 4.Qdrant 查询慢怎么排查？

```
先看是向量搜索慢、过滤慢、网络慢还是生成慢。
检查 P95/P99、数据量、向量维度、top_k、hnsw_ef、过滤字段是否建 payload index。
看是否使用 on_disk、磁盘是否 SSD/NVMe、内存是否不足。
如果过滤很重，优化 payload index 或调整数据分片。
如果召回参数太高，降低 hnsw_ef 或加缓存。
```


# 5.过滤字段为什么要建 payload index？

```
因为没有索引时，过滤可能需要扫描更多 payload，数据量大时会明显影响查询性能。高频过滤字段，如 tenant_id、category、status、created_at，生产中一般应建索引。
```

# 6.多租户怎么做？

```
大量小租户：同一个 collection，用 tenant_id payload filter，强制每次查询带 tenant_id。
少量大租户或强隔离：每个租户独立 collection，甚至独立集群。

要强调权限过滤必须在服务端注入(gateway层实现)，不能依赖前端传参。
```

# 7.embedding 模型升级怎么办？

```
新建 collection；
用新模型重新向量化；
新旧 collection 双写；
灰度切查询流量；
评估效果；
确认后切换；
保留回滚窗口。


payload 里建议记录: model name、model version
```

# 8.Qdrant 里的 id 怎么设计？

```
建议幂等、稳定、可追踪。

document_id + chunk_index

sha256(tenant_id + source_id + chunk_index)

这样重复导入可以 upsert 覆盖，而不是产生重复数据
```

# 9.删除文档怎么处理？

```
如果一个文档被切成多个 chunk，需要能按 document_id 删除所有 chunk。可以用 payload filter 删除，或维护 document_id 到 point ids 的映射。生产中也常用软删除字段：

{"is_deleted": true}

查询时强制过滤：
is_deleted != true
```

# 10.RAG 里 chunk 怎么切？

```
按标题、段落、语义边界切；
每块不要太短，否则上下文不足；
不要太长，否则 embedding 表达被稀释；
保留 overlap；
payload 保存 source、page、section、chunk_index；
对表格、代码、FAQ 使用特殊切分策略。

pdf切分：
    PyMuPDF (fitz)：速度极快，能保留段落和文本块信息，适合快速原型。
    
    pdfplumber：对表格、文本坐标提取精准，适合需要准确获取页内布局的场景。
    
    Unstructured（推荐）：专为 RAG 设计，能自动识别标题、段落、列表、表格等“文档元素”，输出结构化的 Document 对象，极大降低后续分块难度。
    
    Marker/PaddleOCR：需要处理扫描版 PDF 时，先用 OCR 引擎转为可读文本。
    
    页眉页脚/水印：预处理时用正则或坐标过滤，避免无用字符污染 chunk。
    
    扫描件乱码：OCR 后文本可能缺少自然分段，需要结合空格和标点规则重建句子边界，再进行分块。

PDF切分落地建议流程：
    解析：用 Unstructured(自动去除页眉页脚) 或 PyMuPDF 提取出带结构标签的文本元素。
    
    清洗：去除页眉页脚、多余空行，合并被切断的句子。
    
    预分段：以标题、章节等自然边界将文档先划分为“节”。
    
    自适应分块：对每节采用递归分割，以段落、句子为断点，限制在目标长度内，并添加重叠。
    
    特殊处理：将表格单独整取，图片用多模态模型描述，作为独立 chunk 注入。
    
    测试迭代：用一组真实问题测试检索，观察是否返回冗余碎片或截断答案，调整大小和重叠。
```

**代码实现**
```python
import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Element, Text, Title, Table, Image, NarrativeText, ListItem
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 多模态描述图片（示例使用 OpenAI，需设置 OPENAI_API_KEY）
try:
    import openai
    from PIL import Image as PILImage
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False

# ------------------------------------------------------------
# 步骤1：解析 PDF - 使用 Unstructured 提取带结构标签的元素
# ------------------------------------------------------------
def parse_pdf_to_elements(pdf_path: str) -> List[Element]:
    """
    使用 hi_res 策略提取 PDF 中的文本、表格、图片等元素，
    保留类别标签（Title, NarrativeText, Table, Image 等）及页面坐标。
    """
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",          # 高精度布局分析
        infer_table_structure=True, # 自动识别表格结构
        include_page_breaks=True,   # 保留分页信号
    )
    return elements

# ------------------------------------------------------------
# 步骤2：清洗 - 去除页眉页脚，合并跨页断句
# ------------------------------------------------------------
def clean_elements(elements: List[Element]) -> List[Element]:
    """
    简单清洗：
    - 忽略页面顶部/底部常见页眉页脚模式（可按需细化）
    - 合并因换页而中断的句子（前一个文本不以句号结尾，且后一个以小写开头）
    """
    cleaned = []
    i = 0
    while i < len(elements):
        el = elements[i]
        # 跳过明显页眉页脚（可根据实际文档调整正则）
        if el.text.strip().lower().startswith("page"):
            i += 1
            continue
        # 合并断句：如果当前元素是 NarrativeText/Text，且文本不以结束标点结尾，
        # 同时下一个元素也是文本且以小写字母开头（英文）或没有明显句号（中文可自定义）
        if (isinstance(el, (Text, NarrativeText)) and 
            i + 1 < len(elements) and 
            isinstance(elements[i+1], (Text, NarrativeText))):
            cur_text = el.text.strip()
            next_el = elements[i+1]
            next_text = next_el.text.strip()
            # 简易判断：当前文本不以. ! ? 。 ！ ？结尾，下一个不以大写或数字开头
            if (cur_text and cur_text[-1] not in '.!?。！？') and \
               (next_text and next_text[0].islower()):
                # 合并到下一个元素
                next_el.text = cur_text + " " + next_text
                i += 1
                continue
        cleaned.append(el)
        i += 1
    return cleaned

# ------------------------------------------------------------
# 步骤3：预分段 - 利用标题/书签将文档划分为逻辑“节”
# ------------------------------------------------------------
def group_by_sections(elements: List[Element]) -> Dict[str, List[Element]]:
    """
    使用 Title 元素作为节边界，构建 {节标题: [该节元素列表]} 的结构。
    文档开头第一个 Title 之前的元素归入“前言”节。
    """
    sections = {}
    current_title = "前言"
    current_list: List[Element] = []
    
    for el in elements:
        if isinstance(el, Title):
            # 保存上一节
            if current_list:
                sections[current_title] = current_list
            current_title = el.text.strip()
            current_list = []
        else:
            current_list.append(el)
    # 最后一节
    if current_list:
        sections[current_title] = current_list
    return sections

# ------------------------------------------------------------
# 步骤4：自适应分块 - 对每节文本递归分割，保留语义完整
# ------------------------------------------------------------
def chunk_section_text(
    section_title: str,
    text_elements: List[Element],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    将同一节内的文本类元素拼合，用 RecursiveCharacterTextSplitter 分块。
    分隔符优先用段落、句子边界，保持 chunk 语义独立。
    """
    # 拼接文本（非表格/图片）
    full_text = "\n\n".join(
        el.text for el in text_elements 
        if isinstance(el, (Text, NarrativeText, ListItem))
    )
    if not full_text.strip():
        return []
    
    # 中文优先使用句号、换行等分隔符
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        keep_separator=False,
    )
    chunks = splitter.create_documents(
        texts=[full_text],
        metadatas=[{"section": section_title}]
    )
    # 为每个 chunk 添加章节元数据
    for chunk in chunks:
        chunk.metadata["section"] = section_title
    return chunks

# ------------------------------------------------------------
# 步骤5：特殊处理 - 表格和图片作为独立 chunk 注入
# ------------------------------------------------------------
def process_table(table_element: Table) -> Optional[Document]:
    """将表格元素转为 Markdown 表格文本，保留在独立 chunk 中"""
    try:
        # unstructured 提取的表格可能以 html 或纯文本形式存储
        table_html = table_element.metadata.text_as_html
        if table_html:
            # 简单转换为 markdown（实际可用 pandoc 或专用转换器）
            # 此处保留 HTML，LLM 也能理解，或改为纯文本表示
            table_text = table_html
        else:
            table_text = table_element.text
        return Document(
            page_content=table_text,
            metadata={
                "type": "table",
                "page_number": table_element.metadata.page_number,
                "section": None  # 后续填充
            }
        )
    except Exception:
        return None

def process_image(image_element: Image) -> Optional[Document]:
    """
    使用多模态模型（如 GPT-4V）对图片生成文字描述，
    将其作为可检索的文本 chunk。
    需设置环境变量 OPENAI_API_KEY 并安装 openai、Pillow。
    """
    if not HAS_MULTIMODAL:
        # 若无法使用多模态，返回占位文本或跳过
        return Document(
            page_content=f"[图片，位于第{image_element.metadata.page_number}页]",
            metadata={"type": "image"}
        )
    
    try:
        # 提取图片数据（unstructured 可能将图片保存为临时文件或 base64）
        image_path = image_element.metadata.image_path
        if not image_path or not os.path.exists(image_path):
            return None
        
        # 使用 OpenAI GPT-4V 生成描述
        client = openai.OpenAI()
        with open(image_path, "rb") as img_file:
            import base64
            b64_img = base64.b64encode(img_file.read()).decode("utf-8")
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请用中文详细描述这张图片的内容，包括图中的文字、图表含义等。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        description = response.choices[0].message.content
        return Document(
            page_content=description,
            metadata={
                "type": "image",
                "page_number": image_element.metadata.page_number,
                "original_image": image_path
            }
        )
    except Exception as e:
        print(f"图片处理失败: {e}")
        return None

# ------------------------------------------------------------
# 步骤6：主流程整合
# ------------------------------------------------------------
def pdf_to_chunks(
    pdf_path: str,
    text_chunk_size: int = 1000,
    overlap: int = 200,
    process_images: bool = False
) -> List[Document]:
    """
    完整流水线：PDF → 结构化解析 → 清洗 → 按章节预分段 → 文本自适应分块
    → 表格/图片独立 chunk。
    返回所有可存入向量数据库的 Document 列表。
    """
    # 1. 解析
    raw_elements = parse_pdf_to_elements(pdf_path)
    # 2. 清洗
    cleaned_elements = clean_elements(raw_elements)
    # 3. 预分段（按标题）
    sections = group_by_sections(cleaned_elements)
    
    final_chunks: List[Document] = []
    
    for section_title, sec_elements in sections.items():
        text_elements = []
        for el in sec_elements:
            # 处理表格
            if isinstance(el, Table):
                table_doc = process_table(el)
                if table_doc:
                    table_doc.metadata["section"] = section_title
                    final_chunks.append(table_doc)
                continue
            # 处理图片（需显式启用）
            if process_images and isinstance(el, Image):
                img_doc = process_image(el)
                if img_doc:
                    img_doc.metadata["section"] = section_title
                    final_chunks.append(img_doc)
                continue
            # 收集文本类元素
            if isinstance(el, (Text, NarrativeText, ListItem, Title)):
                # 标题也可作为独立小 chunk，但这里将其并入文本流
                text_elements.append(el)
        
        # 对当前节的文本部分进行自适应分块
        if text_elements:
            text_chunks = chunk_section_text(
                section_title, text_elements,
                chunk_size=text_chunk_size,
                chunk_overlap=overlap
            )
            # 补充页码等元数据（取第一个文本元素的页码，或设为节起始页）
            for chunk in text_chunks:
                first_page = text_elements[0].metadata.page_number if text_elements[0].metadata.page_number else 1
                chunk.metadata["page_number"] = first_page
                chunk.metadata["source"] = os.path.basename(pdf_path)
                final_chunks.append(chunk)
    
    return final_chunks

# ------------------------------------------------------------
# 步骤7：测试迭代 - 查看分块结果（示例）
# ------------------------------------------------------------
def quick_eval(chunks: List[Document]):
    """打印前几个 chunk 用于人工评估"""
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} ---")
        print(f"元数据: {chunk.metadata}")
        print(f"内容 (前200字符): {chunk.page_content[:200]}...\n")

# ------------------------------------------------------------
# 使用示例
# ------------------------------------------------------------
if __name__ == "__main__":
    pdf_file = "example.pdf"  # 替换为实际路径
    all_chunks = pdf_to_chunks(
        pdf_file,
        text_chunk_size=800,
        overlap=150,
        process_images=False  # 设为 True 并配置 OPENAI_API_KEY 以启用图片描述
    )
    print(f"总共生成 {len(all_chunks)} 个 chunks")
    quick_eval(all_chunks)
    
    # 后续可将 all_chunks 存入向量数据库（如 Chroma, Pinecone）
```

# 11.top_k 越大越好吗？

```
向量召回 top 20-50；
reranker 重排；
最终给 LLM top 3-8。
```

# 12.如何评估检索效果？

```
构建测试集:
    query
    标准答案
    应该命中的文档/chunk
    
Recall@K：
    前 K 个结果中，覆盖了多少比例的全量相关文档，医疗、法律等漏掉相关文档代价极高的领域，宁可错杀不可放过

MRR(Mean Reciprocal Rank，平均倒数排名)：
    多个查询下，第一个相关文档排名的倒数的平均值，
    对位置极度敏感：排第1得1.0，排第2骤降到0.5，排第10只有0.1，问答系统、导航搜索（用户通常只点第一个结果）

nDCG(Normalized Discounted Cumulative Gain，归一化折损累计增益):
    这是最全面的排序评估指标
    NDCG 的核心优势：
        支持多等级相关度（不像 Precision 只有 0/1）
        位置越靠前，高相关文档的增益越大
        归一化后不同查询之间可比
    
Hit Rate(命中率):
    与 MRR 的区别：MRR 关心第一个命中的位置，Hit Rate 只关心有没有。
    二值指标：要么 1（命中），要么 0（未命中）。
答案正确率/精确率@K：前 K 个结果中，相关文档的比例
无答案拒答率
F1@K：当 Precision 和 Recall 差距极大时（如 P=1.0, R=0.01），算术平均会虚高，调和平均能惩罚这种不平衡。

MAP（Mean Average Precision，平均精确率均值）:
    对每个查询，计算每个相关文档位置的 Precision，再取平均；最后对所有查询取平均。
    全量召回场景，如学术检索（用户会翻很多页）。不像 MRR 只看第一个。

实践建议：RAG 系统日常监控用 NDCG@5 + HitRate@5 + Faithfulness，季度深度评估加测 Context Recall + MAP。

```

# 13.Qdrant 内存不够怎么办？

```
降低向量维度；
使用 quantization；
启用 on_disk；
减少不必要 payload；
优化 payload index；
冷热数据拆分；
扩容节点；
检查副本数和 shard 配置。
```

# 14.为什么同样的问题有时搜不到关键词完全匹配的文档？

```
纯 dense embedding 偏语义，不保证关键词精确匹配。解决:
    加 sparse vector；
    加 BM25；
    做 hybrid search；
    提高关键词字段权重；
    用 reranker；
    对专有名词、编号、函数名做特殊处理。
```

# 15.如何保证 Qdrant 和业务数据库一致？

```
业务数据库是主库；
Qdrant 是检索索引；
写入业务库后通过消息队列异步更新 Qdrant；
使用 outbox pattern 保证事件不丢；
失败重试；
定期全量校验和补偿。
```

# 16.生产事故：写入成功但查询不到，可能是什么原因？

```
是否写入了正确 collection；
是否 wait=True；
是否过滤条件排除了；
tenant_id 是否正确；
vector 维度是否一致；
是否写入了 named vector 但查询 using 错了；
是否软删除字段生效；
索引构建是否完成；
查询 embedding 模型是否和入库模型一致。
```

# 17.快照和备份为什么重要？

```
因为 Qdrant 存的是可重建但成本较高的索引数据。没有备份时，重新 embedding 和导入可能耗时、耗钱，还会影响线上恢复时间。生产需要定期 snapshot，并演练恢复。
```

# 18.如何设计 payload？

```
{
  "tenant_id": "t1",
  "document_id": "doc_001",
  "chunk_id": "doc_001_0001",
  "text": "...",
  "source": "manual.pdf",
  "page": 12,
  "category": "faq",
  "created_at": "2026-05-13T10:00:00Z",
  "embedding_model": "text-embedding-3-small",
  "is_deleted": false
}

用于过滤的字段要结构化；
用于展示的字段要完整；
用于排查的字段要可追踪；
大字段谨慎存，避免 payload 过重。
```

# 19.怎么评估

```
L1 检索评估：召回率、精确率、MRR（不调用 LLM）
L2 生成评估：Faithfulness、Answer Relevance、Context Precision
    关键检查清单
    [ ] Faithfulness：答案是否胡编？（对比 context）
    [ ] Context Recall：正确答案是否在检索结果里？
    [ ] Hallucination Rate：多少比例在编造？
    [ ] Latency P99：长尾延迟是否可接受？

L3 端到端：人工评分 / A-B 测试 / 用户满意度
```

# 20.对分块的补充

```
1. 智能分块：不止是切分文本
    分块是整个RAG流程的起点，直接决定了模型能看到什么信息。目前的主流趋势是从“一刀切”转向“自适应”。

    主流基线策略：递归字符分割。在缺少特定优化目标时，这通常是最稳妥的起点。实践表明，采用400-512 tokens的块大小，配合10-20% 的重叠（chunk overlap），能较好地平衡语义完整性与检索精度，在通用文档上表现可靠。

    行业实践：向自适应进化。单一的“固定规则”切分已无法满足要求。根据文档结构（如PDF、法律条款、科研论文）采用不同规则是常见的优化路径。

    文档结构感知：对于具有清晰逻辑结构的文档（如产品手册、财报），基于章节、标题或语义边界进行切分，能保证知识的完整性。例如，针对医疗临床文档采用自适应分块，准确率可达87%，而固定大小分块仅为13%。

    自适应动态分块：近期的研究和趋势表明，更前沿的思路是根据查询的复杂性动态调整检索的粒度。在索引时，可以将知识库构建成父子文档结构；检索时，先在大块中寻找相关章节（父亲），再从中定位最相关的具体片段（孩子），以实现精准检索。

    句子级切分：一篇系统性的分析研究指出，基于句子级别的切分（sentence chunking） 效率非常高，在成本和效果上能与更复杂的语义切分方法相媲美。

小提示：有些研究中指出，过多的重叠会增加索引成本且收益有限，但在生产实践中，适度的重叠仍是保证召回率的常用手段。
```

# 21.文档更新与删除：保持数据同步

```
知识库与向量库不一致是RAG在生产中的常见痛点。文档更新后，向量库里旧版本的向量依然存在，会导致模型“看到”过时甚至矛盾的信息

解决方案：采用增量更新 + 哈希去重的策略。
    监控数据源变化（如文件系统事件）。
    当文档变化时，解析并切分新文档。
    为每个chunk计算唯一的哈希值。
    对比存储在RecordManager（记录管理器）中的旧哈希值，只更新那些哈希值发生变化的chunk。
    从向量库中删除已不存在或不再有权限访问的chunk。
```

# 22.怎么提升检索召回?

```
混合检索：
    向量检索 + 关键词检索（BM25）并行召回，再融合排序，同一查询同时走向量索引和倒排索引，用 RRF（Reciprocal Rank Fusion）或加权打分融合两路结果。
    RRF：不同检索系统对同一查询返回各自的 Top-K 列表，RRF 不依赖文档的具体打分分数（因为不同系统的分数不可比），而是只利用文档的排名位置来融合。排名越靠前，贡献的融合分越高。
        K一般设置为60
重排序：
    先用轻量模型快速召回 Top-K，再用精排模型重排

```


# 23.RAG知识库流程

离线：原始文档 -> 清洗 -> 切 chunk -> 生成 embedding -> 写入 Milvus
在线问答：用户问题 -> 生成 query embedding -> Milvus TopK 检索 -> 拼上下文 -> 给 LLM 回答

生产中 RAG 不只是“存进去搜出来”，重点还有：

+ **chunk 怎么切**；
  + 目标：把长文本切成有语义、不丢失上下文的小段
  + 三大分块策略：
    + 固定长度分块：
      + 按字符数 / Token 数切 
      + 工具：tiktoken（OpenAI）、LangChain RecursiveCharacterTextSplitter
    + 语义分块：
      + 按语义段落切，不切断完整意思
      + 工具：SemanticChunk、GLM Embedding 分块
    + 文档结构分块：
      + 按标题层级切：# 标题 → 段落
      + 工具：Unstructured、Markdown 结构化分块
    + 避坑：
      + ❌ chunk 太大：向量泛化，检索不准 
      + ❌ chunk 太小：上下文断裂，回答不完整 
      + ❌ 无重叠（overlap=0）：切断一句话，检索不到 
      + ❌ 不做长度校验：超长文本直接 Embedding 报错
+ **embedding 模型怎么选**；
  + 目标：把文本变成数学向量，用于相似度计算(关键规则：提问和知识库必须用同一个 Embedding 模型)
  + 本地开源模型：
    + BGE-base-zh、m3e-base、text2vec-large-chinese（中文最强）
    + 框架：sentence-transformers
  + 云端API Embedding：
    + OpenAI Embedding、阿里通义、百度千帆、ZhipuAI
  + 混合方案：本地小模型 + 云端高精度
  + 避坑： 
    + ❌ 向量维度不匹配：建表维度 ≠ 模型输出维度（m3e=768 维） 
    + ❌ 不做归一化：Milvus 检索精度下降 
    + ❌ 超长文本直接 Embedding：模型截断导致信息丢失 
    + ❌ 中英文混用不匹配：用中文模型处理英文
+ **文档清洗**：
  + 目标：去掉无用字符，保留干净正文
  + 方法：
    + 基础清洗：去空格、换行、制表符、特殊符号 
    + 格式清洗：去除页眉页脚、页码、水印、多余空行 
    + 正则清洗：剔除邮箱、链接、多余标点 
    + 工具清洗：Unstructured 自动清洗、LangChain TextSplitter 预处理
  + 避坑： 
    + ❌ 过度清洗：把表格、序号、关键格式删掉 
    + ❌ 保留大量空白 / 换行：导致分块不准确 
    + ❌ 中英文混排乱码
+ **metadata 怎么设计**；
  + 文档级主字段（同一份文档所有 Chunk 共用）：
    + ```
      字段名	 |      类型 	|     用途说明	 |       业务作用
      doc_id |	   string   |	文档全局唯一 ID	 批量删文档、全量更新、溯源主键
      doc_title |	string   |	文档标题	         混合检索关键词命中、前端展示
      doc_type |	string   | 	文档类型	         分类过滤：制度 / 合同 / 技术手册 / 工单 / 公告
      file_format |	 string	    原文件格式	     pdf/word/md/txt/excel
      source_path |	 string	    原始存储路径	         运维溯源、重新解析入口
      create_time |	 int/timestamp	文档入库时间	  时间范围检索、新旧内容筛选
      update_time |	 int/timestamp	最后更新时间	  增量更新判断、时效内容过滤
      doc_version |	 int	         文档版本号	  版本迭代，旧版本可保留可下线
      is_valid |	bool	       数据有效状态	   软删除：true 有效 /false 废弃
      language |	string	       语言类型	      zh/en/ 混合，适配不同 Embedding
      ```
  + 权限管控字段（企业必填，行级权限）
    + ```
      字段名	            类型	                用途	
      dept_ids	        array[string]	    可访问部门列表	按部门过滤
      role_ids	        array[string]	    可访问角色列表	管理员 / 普通员工 / 高管权限隔离
      user_whitelist	array[string]	    指定可见用户 ID	绝密文档仅指定人可见
      access_level	    int	                密级等级	1 公开 / 2 内部 / 3 机密 / 4 绝密
      ```
  + Chunk 块级独有字段（同文档不同块不同值）
    + ```
      字段名	            类型	                用途说明	
      chunk_id	       string	            块唯一标识	精准定位单条切片
      chunk_index	   int	                块序号	还原文档顺序、上下文拼接排序
      parent_section	string	            所属章节 / 标题层级	结构化检索，如「第三章 保密条款」
      chunk_length	   int	                文本字符长度	统计分析、异常块过滤
      semantic_tag	  array[string]	        语义标签	人工 / 模型打标：人事 / 财务 / 技术 / 法务
      keyword_list	  array[string]	        提取核心关键词	强化 BM25 混合检索命中
      ```
  + 工程运维 & 评估扩展字段
    + ```
      字段名	            类型	        作用	
      emb_model_name	string	   生成向量所用模型名	防止混模型入库，溯源对齐
      emb_dim	          int	   向量维度	建表校验、排查维度不匹配
      split_strategy	string	   分块策略	semantic/recursive/fixed，用于数据分析
      data_source_tag	string	   数据源标签	内部知识库 / 外部资料 / 爬虫数据
      ```
+ **top_k 取多少**；
  + topk=3~5（太多会引入噪声）
+ **是否加 reranker**；
  + Embedding 做的是粗召回（速度快，但精度一般）
  + Reranker 做精排（用 cross-encoder 重新打分，准度极高）
+ **是否做 hybrid search**；
  + 只靠向量检索 = 会丢关键词、专业术语、编号、条款、代码。
  + 向量检索（语义） + 全文检索（关键词/精确匹配） → 结果融合 → Reranker 精排
+ **权限过滤怎么保证**；
  + 必须在检索时做权限过滤，不能等 LLM 之后过滤
  + Milvus 多租户 + 行级权限过滤
  + chunk字段：
    + user_id: 可访问用户ID 
    + role_id: 角色ID 
    + dept_id: 部门ID
  + 避坑：
    + 越权访问
    + 普通员工看到高管文档
    + 合规风险
+ **文档更新和删除怎么同步**；
  + chunk字段：
    + doc_id: 文档唯一ID 
    + doc_version: 版本号 
    + status: 有效/无效
+ **检索结果如何评估**。
  + RAG 好不好，不是看 LLM 回答，是看检索召回率
  + 评估指标：
    + Hit Rate@10：前 10 条是否包含正确答案  Hit Rate > 90% 才算合格
    + MRR@10：正确答案排在第几名（越靠前越好）
    + Precision@3：前 3 条有多少是真正有用的
  + 评估工具：
    + RAGAS 
    + LlamaIndex Evaluator 
    + 自定义评估脚本


**24.RAGAS**

RAGAS = Retrieval-Augmented Generation Assessment

开源 RAG 自动化评估标杆框架（GitHub：explodinggradients/ragas）

+ 核心逻辑：用大模型当裁判 LLM-as-Judge，语义打分，告别字符串匹配（BLEU/ROUGE 完全不适合 RAG）
+ 最大优势：可无标准答案（无 Ground Truth）评估，敏感内网、私有化完美适配

| 低分指标                | 故障环节   | 立刻优化动作                            |
|---------------------|--------|-----------------------------------|
| Context Recall 低    | 检索漏信息  | 加大 TopK、优化语义分块、BGE 微调、开 BM25 混合检索 |
| Context Precision 低 | 检索噪声多  | 调高余弦阈值、Rerank 过滤、精简 Chunk、权限严格过滤  |
| Faithfulness 极低     | LLM 幻觉 | 强约束 Prompt、增加上下文、本地 Rerank、换低幻觉模型 |
| Answer Relevancy 低  | 答非所问   | Query 改写、精简上下文、限制回答长度             |


评估低分样本时，通过doc_id+chunk_index直接定位原始文档位置，快速复盘是分块问题还是文档内容问题。


# 25.Reranker

流程:
+ Milvus 向量检索（粗召回）：返回 Top10～Top15
+ Reranker 模型精排：给每一条「问题 + 文本」计算相关性分数
+ 按分数从高到低排序
+ 取 Top3～Top5 送给 LLM

效果：
+ 向量检索是双编码器（Bi-Encoder）：快，但不准 
+ Reranker 是交叉编码器（Cross-Encoder）：慢一点，但极准
		
推荐模型：
+ bge-reranker-v2 新版本
+ bge-reranker-large → 精度最高（推荐）
		

# 26.数据库选型

## 一、先分清三类库职责（绝不混库）
+ 向量数据库：存 Embedding 向量 + 轻量元数据 → 负责相似度检索、混合检索、权限过滤
+ 业务关系数据库：存完整文档信息、用户权限、版本、日志、业务字段 → 做业务增删改查、权限体系、文档管理
+ 缓存数据库：缓存高频 Query 向量、重排结果、LLM 应答 → 提速降开销

## 二、向量数据库选型（核心重点）

选型核心评判维度：
+ 部署方式：单机 / 集群 / 轻量本地 / 云服务
+ 支持能力：Hybrid 混合检索 (BM25)、标量过滤、动态增删改、分区、TTL
+ 性能：百万 / 千万级向量 QPS、召回延迟
+ 生态：Python 友好、LangChain/LlamaIndex 适配、RAG 工程适配
+ 隐私：是否外网通信、私有化部署难度
+ 成本：开源免费 / 商业授权 / 云按量计费
+ 扩展：多租户、冷热分离、数据同步

**主流向量库横向对比**

| 数据库          | 优势                                                                                             | 劣势                                                           | 适用场景                                  |
|--------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------|---------------------------------------|
| **Milvus**   | 1. 原生支持 BM25 混合检索<br>2. 标量过滤极强，权限过滤友好<br>3. 动态增删改稳定，文档更新友好<br>4. 集群成熟，国产生态完善<br>5. 支持多字段、数组元数据 | 轻量版功能少，生产需 Docker 集群                                         | 企业私有化 RAG 首选<br>知识库、内部文档、敏感数据、权限 RAG  |
| **FAISS**    | 检索速度极快、纯本地无服务、轻量化                                                                              | 1. 无原生持久化，断电丢数据<br>2. 无内置全文检索<br>3. 无完善过滤、事务、版本能力<br>4. 运维弱  | 实验测试、离线批量检索、本地小型 Demo                 |
| **Chroma**   | 极简开箱即用、代码最少、本地零部署                                                                              | 大数据量性能拉胯、无集群、无复杂过滤                                           | 个人学习、小型测试、个人知识库                       |
| **Qdrant**   | 检索性能强、分片好、过滤快、UI 友好                                                                            | 国内生态弱，混合检索不如 Milvus 顺手                                       | 海外业务、高并发检索场景                          |
| **PGVector** | 基于 Postgres，向量 + 关系一体化<br>不用维护两套库，事务强                                                          | 千万级以上向量性能弱于专业向量库                                             | 中小体量、传统业务系统内嵌 RAG<br>不想额外运维新组件        |
| **Weaviate** | 语义结构化强、实体检索强                                                                                   | 部署重、学习成本高                                                    | 知识图谱 + RAG 融合场景                       |

**向量库最终选型结论**

+ 企业生产、内网私有化、有权限管控、文档频繁更新
  + ✅ 首选 Milvus理由：混合检索 + 元数据过滤 + 软删除 + 版本更新 + 多租户完美匹配你整套 RAG 架构
+ 项目体量小、100 万向量以内、不想多部署服务
  + ✅ PGVector（PostgreSQL + 向量插件）一套库搞定业务数据 + 向量，开发最简
+ 本地调试、个人项目、快速跑通流程
  + ✅ Chroma / FAISS
+ 高并发线上公开 RAG 服务
  + ✅ Qdrant / 托管云向量库

## 三、业务关系数据库选型（存文档全量信息、权限、用户）

作用：
+ 存储：完整文档内容、上传记录、用户信息、角色权限、版本日志、RAG 评估日志、操作记录不和向量库抢存储，向量库只存 Chunk + 向量 + 轻量 metadata

选型分级：
+ 中小型项目、内网企业系统
  + ✅ MySQL 8.0
  + 稳定、运维简单、权限表、部门角色表设计成熟，适配绝大多数权限体系
+ 中大型、高并发、多租户、复杂业务
  + ✅ PostgreSQL
  + JSON 字段友好、数组字段原生支持、事务强、可无缝搭配 PGVector
+ 超大规模知识库、海量文档归档
  + ✅ TiDB / OceanBase 分布式关系库

固定搭配最优组合
+ Milvus 向量库 + MySQL 业务库 → 最通用企业组合
+ PGVector + PostgreSQL → 最简一体化组合

## 四、缓存数据库选型

**用途**：
+ 缓存高频问题 Query 向量
+ 缓存重排后上下文结果
+ 缓存 LLM 通用回答
+ 限流、会话上下文缓存

**首选：Redis**
+ 字符串存向量、List 存检索结果、Hash 存会话信息
+ 支持 TTL 自动过期，自动清理冷数据
+ 极大降低重复向量化、重复检索开销

## 五、按你的 RAG 架构精准定最终栈
你的业务特征回顾
+ 敏感内部文档，必须私有化、数据不出网
+ 要：语义分块 + BGE 向量 + Hybrid 混合检索 + Reranker 精排
+ 要：行级权限过滤、文档增删改版本管理
+ 要：RAGAS 评估、全链路本地开源

## 最终推荐技术栈（直接落地）
+ 向量数据库：Milvus 2.5+（Docker 集群部署）
  + 开启 BM25 索引做混合检索
  + 存储 chunk 文本 + 向量 + 权限 / 版本 / 分块策略等元数据
  + 检索阶段直接做部门、角色、密级过滤
+ 业务主数据库：MySQL8.0
  + 存储原始文档信息、用户、角色、部门、权限配置、上传记录、更新日志
  + 文档全量正文、附件路径存在这里，向量库只存切片
+ 缓存数据库：Redis
  + 缓存热门 query 向量、检索结果、会话上下文
  + 减轻 Embedding 服务与 Milvus 压力

# 27.Qdrant dense vector  搜索通常基于 HNSW 近似最近邻索引。你需要理解几个方向：

```
参数/概念	             影响

m	                图连接数量，越大召回越好、内存越高 管稠密程度

ef_construct	       建索引质量，越大构建越慢、召回越好 管建图精度

ef	                查询时搜索宽度，越大召回越好、延迟越高  只管搜索召回

quantization	    降低内存，提高速度，可能损失精度

on_disk	            降低内存压力，但可能增加延迟

payload index	    加速过滤

shard/replica	    扩展容量和可用性
```

+ 如果召回率不够，可以提高 hnsw_ef、调整 m/ef_construct、增加 top_k、换 embedding 模型、加 reranker。
+ 如果延迟太高，可以降低 hnsw_ef、做量化、优化 payload index、减少向量维度、使用更快存储、拆分 collection 或扩容。

# 28.IVF_FALT/HNSW/RRF/BM25/SPLADE/BGE-M3/ANN

```
层级	        定位	                     核心作用
基础方法	ANN (近似最近邻搜索)	       效率基石：牺牲少量精度，换取检索效率的巨大提升，是大规模向量检索的核心思想。
索引层	IVF_FLAT / HNSW	           加速引擎：实现ANN思想的具体算法，构建高效的数据结构以加速查询。
召回层	BM25 / SPLADE	           检索通道：生成不同维度的候选集，负责从原始数据中“召回”相关结果。
融合层	RRF (倒数排名融合)	           排序大师：将不同召回通道的结果进行智能融合，输出最终排名。(融合语义搜索/SPLADE/BGE+关键词搜索/BM25)
嵌入层	BGE-M3	                   桥梁模型：一种前沿的Embedding模型，能将原始文本转换为向量以供检索。
```

## 1. 基础方法：ANN (Approximate Nearest Neighbor) - 效率的基石
ANN（近似最近邻） 是一类算法的统称，其核心是用可接受的精度损失，来换取检索效率的极大提升。

+ 背景：当数据量达到百万、亿级时，精确的暴力搜索（Brute-Force Search） 需要遍历所有向量并逐一计算相似度，耗时会线性增长，完全不现实。

+ 思想：ANN会构建一个特殊的“目录”（即索引，如IVF_FLAT、HNSW），搜索时先快速浏览目录缩小范围，再在有限的候选集中进行精确计算，从而避免大海捞针式的搜索。

+ 权衡：ANN在召回率（找到真正相关结果的比例）和延迟（查询速度）之间寻找最佳平衡。

## 2. 索引层：IVF_FLAT 与 HNSW - 加速的核心引擎
IVF_FLAT 和 HNSW 是实现ANN思想的具体算法，是现代向量数据库（如Milvus、Faiss）的标配。

IVF_FLAT (Inverted File with Flat) - 先聚类，再搜索

IVF_FLAT 的核心思想是“分而治之”，通过聚类缩小搜索范围。

+ 工作原理:

  + 构建索引：用K-means算法将所有向量划分为 nlist 个簇，每个簇有一个中心点。向量按原始形式（Flat）存储在所属的簇中。
  + 执行查询：计算查询向量与所有簇中心的距离，选取最近的 nprobe 个候选簇，然后在候选簇内部执行原始向量的精确搜索。
+ 关键参数:

+ nlist：簇的数量。值越大，聚类越精细，但构建和粗筛的计算量也越大。

+ nprobe：搜索时探查的簇数量。值越大，召回率越高，但速度会变慢。

HNSW (Hierarchical Navigable Small World) - 走捷径，抄近道

HNSW 的核心思想是构建一个多层图结构来模拟“高速公路”导航，实现真正的“对数级”搜索速度。

+ 工作原理:

  + 构建多层图：构建一个多层的图网络。底层包含所有数据点，是“精细地图”；高层是底层节点的子集，节点间连接稀疏但“跳”得更远，是“高速路网”。
  + 执行查询：从最顶层“高速路网”的任意入口开始，贪婪地寻找局部最近邻节点，然后逐层下降到更精细的层。整个过程就像从高速出口快速定位，最后在街区中进行精确查找，极大减少了搜索路径的长度。
+ 关键参数:

  + M：每个节点在图中最多拥有的邻居数。M越大，图越稠密，召回率越高，但内存占用和索引构建时间也越大。

  + efConstruction / efSearch：控制构建和查询时动态候选列表的大小，与召回率正相关。

+ 对比总结:

  + 速度：HNSW更快，尤其是在千万级以上的大规模数据中，性能优于IVF_FLAT。

  + 内存：HNSW需存储图的连接关系，内存占用通常高于IVF_FLAT。

  + 动态性：IVF_FLAT对动态插入新数据适应性较差，可能需要重训练；HNSW支持较好的增量插入。

  + 精确度：两者都能通过参数调优达到很高的召回率（如98%以上）。

## 3.召回层：BM25 与 SPLADE - 建立多元检索通道

在混合检索架构中，常常结合关键词检索（BM25） 和语义检索（如SPLADE） 两种方式取长补短。

+ **BM25 - 经典的关键词匹配算法**
  + BM25 是基于统计的传统稀疏向量算法，核心是通过词频（TF） 和逆文档频率（IDF） 来打分。
  + 核心公式与思想：
    + IDF(qi)：衡量词的稀有度，稀有词的权重越高。 
    + f(qi,D)：词在文档D中的出现频率。 
    + |D|/avgdl：文档长度与平均长度的比值。 
    + k1, b：调节参数，控制词频饱和度和文档长度归一化。
    + BM25(qi,D) = IDF(qi) * (f(qi,D) * (k1 + 1) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl)))
    + 注意：一个重要的设计是“词频饱和度”，即一个词出现10次的相关性远不是出现1次的10倍，因为再多也带不来额外信息，有效地抑制了长文档的虚高得分。

+ **SPLADE - 语义搜索/学习型的稀疏向量模型**
  + BGE是稠密向量模型，适合长文本。
  + SPLADE 是基于BERT等预训练模型的“学习型稀疏向量” 模型
  + 核心思想:
    + 利用语义扩展：不仅能捕捉文档中已有的词，还能“联想”出语义相关的其他重要术语。例如，搜索"car"时，能赋予"vehicle"、"automobile"更高的权重。
    + 生成稀疏表示：模型输出一个高维但元素绝大部分为零的向量，非零值代表重要术语的权重。
+ 对比总结：
  + 原理：BM25基于统计信息，SPLADE基于神经网络的语义理解。 
  + 优势：BM25经典、可解释、计算快；SPLADE能捕捉同义词、一词多义等深层语义，准确率通常更高

## 4. 融合层：RRF (Reciprocal Rank Fusion) - 倒数排序融合
RRF 是一种无监督的排序融合算法，能有效合并多个相关性列表。

+ 核心思想：如果一个文档同时在多个独立的排名列表中，且在每个列表中排名都靠前，那么它大概率是用户真正想要的，因此应被赋予更高的最终得分。

+ 计算公式：RRFscore = Σ 1 / (k + rank_i)

  + rank_i：文档在第i个检索结果列表中的排名（从1开始）。

  + k：平滑常数（通常取60），用于防止排名靠后的文档得分过低。

+ 优势：

  + 无需训练：开箱即用，特别适合缺乏标注数据的场景。

  + 鲁棒性强：能有效综合不同信号的强度，比简单的分数加和或加权更可靠。

改进融合权重：RRF 是无监督的，但你可以根据验证集学习一个加权 RRF（给不同召回通道不同的 k 值或权重）。


## 5. 嵌入层：BGE-M3 - 新时代的桥梁模型
BGE-M3 是一个强大的多语言Embedding模型，因其支持多语言、多功能、多粒度的“三多”特性而著称。

+ 多语言 (Multi-Linguality)：支持100+ 种语言，可实现跨语言检索。

+ 多功能 (Multi-Functionality)：同时支持稠密向量（用于语义匹配）和学习型稀疏向量（如SPLADE，用于关键词匹配）。

+ 多粒度 (Multi-Granularity)：能处理从短句到长达8192个词元（tokens） 的长文档。

BGE-M3的一个关键优势是，它统一了从文本到不同Embedding类型的转换过程，为混合检索架构的数据准备环节提供了极大的便利。 


# 29.为什么要把工具封装进MCP适配层，再交给ReAct Agent调用？？

核心是解耦，Agent负责思考、决策，MCP适配层负责工具协议、参数校验、权限控制、超时和审计、结果缓存、错误处理等。

鉴权、限流、日志、缓存、重试、参数校验全部塞适配层，Agent 纯业务推理。

提升可维护性、可扩展性。


# 30.做增删改查场景时，如何避免直接把错误写入落库？

核心思路：先校验、预校验、事务兜底、事后拦截、异常回滚，多层拦截阻断错误入库

模型不能直接写库，模型只生成操作意图和结构化参数，真正执行必须经过后端校验、权限判断、风险确认、事务控制和操作审计。

删除、修改、更新操作，必须经过二次确认。

采用乐观锁 / 悲观锁，防止并发篡改产生错误数据，任意步骤报错、校验失败，立即触发事务回滚


# 31.Agent常见的工作模式有哪些？

+ ReAct边思考边执行的工具调用
+ Plan and Execute 适合先规划、再执行的复杂任务。
+ Reflection 反思自省 Agent在执行动作后，额外增加自我评估环节，判断结果是否符合预期，若不符合则调整执行策略。
+ Multi-Agent 多智能体协作：分工角色（决策 / 执行 / 审核），组队完成复杂业务。
+ RAG 检索增强：先查知识库再回答，知识问答、文档查询。

# 32.如何理解RAG？

本质是模型回答问题之前，先查询知识库，然后根据知识库结果生成问题，再向模型提问。

解决模型不知道最新知识、私有资源、隐私数据等问题。

# 33.RAG的基础链路怎么讲？

基础链路分六步：
+ 数据接入
+ 数据清洗
+ 文档切块
+ Embedding向量化
+ 向量召回
+ 重排和生成

真正影响效果的不只是向量数据库，而是清洗、切块、rerank、元数据

# 34.长期记忆和短期记忆怎么区分？

+ 短期记忆：服务当前对话，最近几轮对话、当前任务状态、工具调用结果
+ 长期记忆：跨服务会话，用户偏好、历史任务、长期目标

先考虑短期记忆，再考虑长期记忆

# 35.短期记忆压缩什么时候触发？

+ 上下文窗口即将超限：对话轮次、token 快塞满，自动压缩
+ 单轮对话累积过长：多轮闲聊 / 步骤堆叠，冗余信息变多
+ 任务阶段切换：一个子任务结束，清空冗余、浓缩关键结论
+ 主动阈值触发：设定轮数 / Token 阈值，到达就批量压缩
+ 调用工具前后：大量工具返回日志冗余，精简再留存

砍掉无效对话、日志细节，只保留关键结论、状态、核心参数，腾出窗口空间。


# 36.智能压缩和机械压缩有什么区别？

+ 机械压缩：
  + 按规则裁剪，比如只保留最近几轮，超过token就截断
+ 智能压缩：
  + 让模型总结历史上下文，保留目标、约束、结论和下一步。

生产中近的保留原文，远的提取摘要。工具日志，机械裁剪。


# 37.多智能体场景里模型选型怎么平衡成本和效果？

原则是分层用模型。

简单分类、格式转换、摘要，用小模型

执行、检索、工具调用：中端模型

规划、复杂推理、关键决策、最终审核，用强模型

重思考用贵模型，干活跑腿用中等，文本整理用轻量；分级调用、按需升降级，控成本不掉效果。

# 38.Agent有没有熔断机制？

必须要有，同一个工具失败3次，停止重试，还要限制最大执行步数、单次调用超时、总任务超时。

参数错误：修改参数。权限错误：直接终止。服务错误：降级/切换工具。

避免卡死宕机、无效耗损算力成本、规避业务风险、防止错误连锁扩散。

# 39.如果基础模型服务卡死怎么办？后端怎么兜底？

引入模型网关治理，六大策略：
+ 超时控制
+ 有限重试
+ 备用模型切换
+ 服务商切换
+ 限流排队
+ 降级策略

异步队列：将请求放入队列，轮询结果，避免阻塞等待。

健康检查与自动重启：定期探测服务健康，异常时重启容器或切流。

事后恢复：
+ 上报异常日志，定位阻塞原因
+ 重启异常服务实例，清理积压请求队列
+ 调整并发、上下文长度等参数规避复现

# 40.大模型评测项目说一下？

做了adapter，scenario、agent、metric主要的装饰器类，即插即用。

adapter主要是进行prompt的构建封装，scenario是数据的加载，agent是针对不同的数据集适配评测逻辑，metric用来计算评估指标，主要是通过把模型回答、标准答案、问题都传给评测模型，进行打分。


# 41.Agent结合python从架构师角度分析一下做的项目？

通过@AgentTool可以完成工具的参数校验，认证

通过业务场景动态切换工具，动态切换数据集，动态切换参数，动态切换模型。

因为http是无状态的，agent是需要有状态，所以通过redis和持久化数据库做了一套分级的记忆系统

短期通过redis list去维护窗口的大小，长期就是向量化，存入向量数据库，

考虑到大模型推理高延迟，通过fastapi+AsyncIO实现异步的响应机制，确保前端能实时的看到agent的思考过程

# 42.并发层面，如果一个agent在分析财报时，如果同时调用三个工具，三个工具的API响应时间不一样，如何通过python并发性优化这个过程？

使用asyncio实现异步IO，create_task：同时发起调用，通过asyncio.gather()函数将三个工具调用任务进行并发执行。gather：等所有返回，不互相阻塞

如果其中一个工具调用失败，可以通过try-except捕获异常，返回错误信息给用户。并且通过cancel取消剩余任务。

# 43.agent在频繁处理大模型返回的长文本时，比如几万字的报告分析，文本密集型应用导致的内存溢出，怎么处理？

Python 字符串是不可变对象，频繁拼接 = 频繁创建新对象，gc.collect()  # 强制垃圾回收

大文本分段切分（Chunking） ，禁止保存完整历史上下文（关键！）

只保留最近 2~3 轮 ，历史用智能压缩 ，超长篇丢向量库，不存内存

用 multiprocessing 处理大文本，完成后进程退出释放全部内存，使用生成器（Generator），不占列表内存

用 io.StringIO 或 bytearray 重复使用同一块内存，而不是 str1 + str2，减少临时内存分配

使用 stream=True 逐块接收文本，边收边处理，不保存完整结果

# 44.工具a返回list，工具b返回dict，agent是如何同一处理，并把他们输入给LLM的下一个思考步骤的？

定义一个同一返回的包装类， 工具 A (List)、工具 B (Dict) 原始结构保留 Agent 内置序列化器，把两种类型转JSON/Markdown/YAML通用字符串
把序列化结果追加到对话上下文 LLM 只读取文本上下文，自动解析列表、字典语义


# 45.如果有100万个文档切片，如何在后端实现毫秒级的相似度检索？

1. 向量化：使用向量数据库，如 Milvus、Pinecone、Weaviate 等，将文档切片向量化，并建立索引。HNSW/IVF_FLAT 等算法

# 46.记忆分级存储怎么设计的？

1. 短期活跃记忆：Redis list存储最近几轮原始对话，窗口大小限制，自动压缩，设置过期时间，自动清理
2. 超过10轮的历史对话，用智能压缩，只保留关键信息，把核心信息存到redis里面，下次请求带着摘要和最近几轮的对话就可以了
3. 长期记忆：向量化，用向量数据库，如 Milvus、Pinecone、Weaviate 等，将文档向量化，并建立索引。HNSW/IVF_FLAT 等算法


# 47.在python层面，如何保证并发一致性？如果用户在短时间内连续发了几条消息，如何防止记忆被写乱？

用redis做了一个redlock

同一用户 ID 同一时间只能有一个写入操作（串行化） ，读写加锁：用户级细粒度锁，不是全局锁，记忆必须原子更新：读 → 修改 → 写，一步提交

在agent思考之前，必须先获取锁，获取锁成功后，才能开始思考，获取锁失败，则等待锁释放，或者超时，则放弃

在数据库（如 PostgreSQL）或支持 CAS（Compare-And-Swap）的存储中，记录一个版本号字段，如果版本不一致，会触发状态的重载

为了防止网络重试导致的重复处理，可以要求客户端为每条消息附带 idempotency key，服务端记录已处理的 key，对重复请求直接返回缓存结果。

# 48.一个系统，需要先调用agent a，在调用agent b，如果agent a成功了，agent b失败了，陷入中间状态，如何处理这种分布式事务？

用补偿回滚 + 状态机 + 重试 + 兜底归档解决

采用 Saga 模式：定义补偿操作，并在 B 失败时异步执行补偿。

持久化事务状态 + 后台重试补偿，确保最终一致性（补偿可能最终成功或需人工处理）。

保证幂等性：Agent 调用应能通过业务键去重。

考虑边界情况：如果补偿也失败，需要告警人工介入。

# 49.workflow和agent的区别？

Workflow（工作流）是 “按提前规划好的流水线”，Agent（智能体）是 “只给目标、自己找路的决策者”。核心差别在于：流程是否写死、下一步由谁决定、有没有自主决策与闭环能力。

Workflow：步骤、分支、顺序全由人预先写死（硬编码 / 配置），控制流在代码 / 配置里，每次跑都一样，确定性高、稳定、可审计。

Agent：只给最终目标，下一步由大模型自己推理决定，可动态选工具、改路径、试错调整，控制流在模型里，不确定性高、灵活、能处理开放问题。

Workflow：DAG（有向无环图），节点是固定任务 / 分支，无记忆、无推理，按编排执行。

Agent：LLM + 记忆 + 工具集 + 规划模块，闭环循环：感知→思考→行动→观察→调整。

# 50.路由/管理/执行三层架构设计？

路由层（入口层）接收请求、参数校验、路由分发、权限拦截、接口限流只转发，不写业务逻辑

管理层（业务层）编排流程、事务控制、规则判断、调用聚合业务决策、组装调用执行层

执行层（底层层）原子操作、数据库 CRUD、第三方调用、工具计算只做单一基础动作，无业务判断

# 51.滑动窗口和动态摘要怎么选？

最主要的区别：保留原始信息 vs. 提炼信息

追求事实准确、需要回溯原文 → 选滑动窗口

追求长时记忆、提取关键主题 → 选动态摘要

**滑动窗口**：
+ 优点
  + 零信息损失（窗口内）：保留的对话是原文，没有扭曲。

  + 实现极其简单：维护一个队列，超了就 pop。

  + 快速：无需调用 LLM 做摘要，计算成本极低。

  + 可预测：上下文大小严格受控，不会超出模型限制。

+ 缺点
  + 无法跨窗口记忆：一旦滑出窗口，信息永久消失。例如用户在第 1 轮说“我是素食者”，第 100 轮问“推荐餐厅”，系统早已忘记。

  + 丢失长程依赖：任何超出窗口大小的关联都无法建立。

**动态摘要**：
+ 优点
  + 无限记忆：理论上可以记住整个对话的精髓。

  + 存储高效：摘要远小于原始日志。

  + 能提炼高层主题：例如总结出“用户是一个喜欢冒险、预算有限的旅行者”。

+ 缺点
  + 信息损失不可逆：摘要本质是有损压缩。可能丢失具体数字、精确措辞、微妙语气。

  + 误差累积：多次摘要可能产生“摘要的摘要”，导致事实扭曲或幻觉。

  + 成本高：每次生成摘要都需要 LLM 调用（token 消耗 + 延迟）。

  + 实现复杂：需要设计触发策略（每隔多少轮？基于 token 阈值？），以及摘要的更新机制（重写整个摘要 vs. 增量合并）。

**特殊情况：必须保留原文且需要长期记忆**

例如：法律咨询、医疗对话，任何精确事实都不允许丢失。

解决方案：滑动窗口 + 向量检索（RAG）。

+ 窗口保留最近几轮。
+ 更早的对话 chunk 后存入向量数据库。
+ 每次生成回复时，除了窗口内容，还检索相关历史片段。
+ 这样做到了“无限记忆 + 原文可查”，但实现复杂且需要索引更新。

```
层级	            技术	                作用

L1 近期记忆	滑动窗口（例如最近 8 轮）	保证即时对话的自然流畅

L2 中期记忆	动态摘要（每 20 轮生成一次）	捕捉会话主题演变

L3 长期记忆	向量存储 + 定期摘要	    跨会话的记忆


优先从 L1 窗口 查找事实。
如果不在窗口内，则查询 L3 向量库 获取原文片段。
L2 摘要 用于生成高层次的个性化回复（例如“根据之前的对话，你似乎更喜欢安静的地方……”），而不用于精确问答。
```

# 52.关于企业知识库智能助手的问题

## 52.1 技术选型

1. 模型：LLM
+ Qwen3
+ DeepSeek-R1
+ DeepSeek-V3
+ GPT-4o

2. Embedding

+ Sentence-transformers
+ bge-m3
+ bge-large-zh-v1.5
+ qwen3-embedding

3. 向量库

+ pgvector
+ chroma
+ milvus

4. 重排序

+ bge-reranker-v2-m3

## 52.2 核心功能

1. 文档解析

+ 支持：PDF、DOCX、PPTX、TXT、HTML、Markdown、JSON、CSV、EXCEL、SQL、YAML、JSONL、XML、ZIP、RAR、MP3、MP4、MPEG、AVI、WAV、OGG、M4A、M4V、M4P、M4R、M4B、M4E、M4O、M4A、M4V、M4P、M4R、M4B、M4E、M4O、M4A、M4V、M4P、M4R、M4B、M4E、
+ 自动提取：标题、段落、表格、图片OCR

2. 文档Chunk切分

+ Recursive Chunk： RecursiveCharacterTextSplitter
+ Markdown Chunk： 按照#、##、###层级切分
+ Parent Child Chunk： 按照父子关系切分 解决上下文丢失问题

3. 向量化

+ embedding_model.embed_documents() BGE-M3批量入库Miluvs

4. Hybrid Search

+ BM25 + 向量检索
+ 融合召回：keyword recall + semantic recall 提高召回率

5. 重排序

+ BGE-Reranker 重新排序Top50结果返回Top5 提高答案准确率

6. 多轮会话

+ 基于LangGraph实现：历史记忆、上下文压缩(摘要)、会话恢复 支持连续问答

7. Agent工具调用

+ SQL Tool 查询数据库
+ Search Tool 知识库检索
+ OCR Tool 图片识别
+ API Tool 调用内部系统

8. 权限控制

+ 基于用户 ID、部门、角色、权限    控制文档访问权限
+ 检索前过滤：    避免越权访问

## 52.3 面试问点

1. 为什么要用chunk？

+ 大模型和Embedding模型都有上下文长度限制，不能直接处理几十页甚至上百页文档，因此需要切分。
+ 从检索效果角度来说：如果整个文档作为一个向量，Embedding会把整本书压缩成一个向量。相关信息被稀释。
+ chunk之后检索时只召回对应的chunk即可

2. Chunk Size 如何确定？

+ 没有固定值，需要结合：Embedding模型、文档类型、LLM上下文长度
+ FAQ：200-300
+ 制度文档：500-800
+ 技术文档：800-1200
+ API文档：1000-1500
+ 实际会做AB测试：300、500、800 比较Recall、MRR、HitRate选择最优方案

3. Chunk Overlap 有什么作用？

+ 避免上下文被切断
+ overlap一般为chunk size 的 10% - 20%

4. BGE和OpenAI Embedding区别？

+ OpenAI：效果好、英文强、开箱即用，费用高、网络依赖
+ BGE：中文优化、可私有化部署、成本低、速度快，英文弱、需要自行维护

5. 为什么M3比v1.5效果更好？

+ M3支持：Dense Retrieval、Sparse Retrieval、Multi-Vector Retrieval
+ v1.5仅支持：只有Dense Embedding

6. 为什么Hybrid Search效果更好？

+ 同时兼顾关键词匹配、语义匹配，召回率更高

7. BM25原理？

+ 关键词相关性排序算法
+ TF词频：词出现越多，得分越高
+ IDF逆文档频率：越稀有，权重越高

8. 为什么需要Rerank？

+ 可能召回相关的信息，但是与问题匹配程度不高
+ 提升答案准确度

9. 为什么选择LangGraph？

+ 状态持久化
+ 支持循环
+ 可观测性(配合LangSmith)

10. State 怎么设计？

+ 一般：messages、query、retrieved_docs、tool_results、final_answer
+ 生产环境：user_id、session_id、tenant_id、permissions

11. Milvus和PGVector区别？

+ Milvus：分布式、高可用、海量数据、向量搜索、索引丰富，支持GPU，运维复杂
+ PGVector：部署简单、直接利用PostgreSQL

12. 知识库更新怎么办？

+ 采用增量更新，向量库更新时，只更新向量，不更新原始文本。重新chunk，重新Embedding，chunk_id没有改变的跳过不更新。

13. 如何避免幻觉？

+ 通过RAG让答案只从知识库中获取，避免幻觉。
+ prompt约束：仅根据提供的内容回答。
+ Rerank：提高召回质量。
+ 引用来源，方便人工验证。

14. 如何做答案溯源？

+ 保存document_id、chunk_id、source_file、page_no

15. 如何做权限隔离？

+ 检索前过滤：    避免越权访问

16. 如何做多租户？

+ 通过tenant_id来管理，保证租户数据隔离
+ 对于大租户单独使用一个collection，多个小租户使用一个collection

17. 如何做缓存？

+ Query Cache 相同问题直接返回
+ Embedding Cache 相同文本不重复Embedding
+ Retrieval Cache 缓存TopK结果

18. 如何评估RAG效果？

+ 离线指标：Recall@k、MRR平均倒数排名(正确答案排第几)、HitRate、NDCG(相关性、排名位置)
+ 在线指标：回答准确率、用户满意度、人工评分、幻觉率

19. ReAct原理

+ 边思考边执行：Thought、Action、Observation

20. MCP协议原理

+ 模型上下文协议：client、MCP Protocol、MCP Server、Tool

21. LangGraph Checkpoint机制

+ 状态持久化
+ 保存：State、Messages、Tool Results
+ 恢复：从服务崩溃的步骤执行

22. Agent Memory设计

+ Short-Term Memory：会话记忆，存储最近几条对话，短期记忆
+ Long-Term Memory：用户偏好，项目信息，知识库记忆，存储知识库，长期记忆
+ Semantic Memory：语义记忆，根据用户输入，生成语义记忆

23. 多Agent协作

+ Supervisor模式：RAG Agent、SQL Agent、Code Agent
+ Supervisor负责：路由、任务拆解、结果汇总

24. Deep Research原理

+ 搜索、规划、反思、总结

25. Dify工作流底层实现

+ DAG 有向无环图
+ 核心模块：Workflow Engine、Node Executor、Variable Manager、Tool Runtime、LLM Runtime

26. RAG系统最重要的指标是什么？

+ Recall@K、MRR、NDCG


27. 对复杂PDF识别，如何分块更好，方案方法

```
财资类 PDF：多表格、财报、凭证、多层图文、跨页表格、公式、水印、多栏排版、页眉页脚干扰、扫描件（图片型 PDF）、双层混合 PDF。直接固定长度切块会割裂表格、上下文，丢失财务勾稽关系。
  使用LangChain的RecursiveCharacterTextSplitter配合自定义分隔符（\n\n、。、；等）。
  1）先做 PDF 结构化解析
    1. 原生可复制 PDF：用 PyMuPDF/Pdfplumber 提取布局元数据：文字块坐标、表格区域、图片、标题层级、段落间距、栏数（单栏 / 双栏财报）；
    2. 扫描图片 PDF：先 OCR（PaddleOCR / 阿里云 OCR）+ 版面分析（如LayoutLMv3、YOLO微调后的版面分析器），识别文本框、表格框、图片框，还原逻辑层级；
    3. 过滤噪声：页眉、页脚、页码、水印、空白行、广告栏。
    
  2）逻辑分块（拒绝固定字符切块）
    1. 层级语义分块（财务优先）
      a.按文档结构：一级标题→二级标题→段落 / 表格为最小块；
      b.财报场景：资产负债表、利润表、附注、现金流各自独立块；
      c.规则：同一张完整表格不拆分，跨页表格合并为单块；
      d.对复杂布局，先调用布局分析模型（如LayoutLMv3、YOLO微调后的版面分析器）识别区域（标题、正文、表格、页眉页脚），再按区域分块。
    2. 动态自适应长度
      a.基础阈值：单块上限 2000–4000token（适配大模型上下文窗口）；
      b.若单逻辑单元超长（长篇附注）：句子滑动切块，重叠 10%–15token 保证语义连贯；
      c.若逻辑单元过短（零散短句）：向上合并至上级标题块。
    3. 多维度绑定块元数据
      每个块附加：页码、chunk_id、docs_id、类型（文本 / 表格 / 图表）、所属章节、财务科目标签，后续检索 / 问答可精准溯源。
  
  3)推荐工具链
    解析：marker、docling、Unstructured（支持复杂版式）
    表格：camelot、tabula、Table Transformer
    OCR：PaddleOCR、Tesseract（扫描版 PDF）
    向量化：切分后做 Embedding，存入向量库
  4)推荐切分
    先结构后内容：先识别文档大纲，再按章节切分
    表格特殊处理：表格不要拆开，保持行列完整性，可转为 JSON/Markdown
    重叠窗口（Overlap）：相邻 chunk 保留 10-20% 重叠，避免跨块信息丢失
    元数据标记：每个 chunk 标注来源页码、章节、文档类型、chunk_id、docs_id，便于检索时过滤
```

| 维度          | 方案                       | 说明                                                    |
|-------------|--------------------------|-------------------------------------------------------|
| **基于文档结构**  | 按段落/章节/标题切分              | 利用 PDF 解析库（如 PyMuPDF、pdfplumber）提取文本块和层级结构            |
| **基于视觉布局**  | 按页面区域切分                  | 识别表格区、正文区、图表区，分别处理                                    |
| **基于语义**    | 语义分块（Semantic Chunking）  | 用 Embedding 模型判断句子相似度，在语义转折处切分                        |
| **基于递归**    | 递归字符切分                   | 先按大粒度（段落）切，过长的再按句子、单词递归切                              |
| **基于表格**    | 表格单独提取为结构化数据             | 用 OCR + 表格识别（如 Table Transformer）转为 Markdown/HTML 表格  |


28. Function Calling 完整全过程

```
流程：
  1. 注册函数：开发者定义函数签名（名称、描述、参数及JSON Schema），和用户消息一起传给模型。
  
  2. 模型判断：模型根据用户意图和函数描述，决定是否调用函数，并输出包含函数名和参数（JSON格式）的响应。
  
  3. 解析调用：客户端收到tool_calls字段，解析出函数名和参数，执行函数（如查询数据库、调用API）。
  
  4. 返回结果：将函数执行结果以tool角色消息回传给模型。
  
  5. LLM生成最终回答：模型结合原始用户消息和函数结果，生成最终答案。

```

29. 用户意图识别，有没有把机器学习和模型结合

```
结合方式：

级联架构：

  第一层（轻量ML模型）：训练一个文本分类器（如BERT微调、FastText、SVM）快速识别常见意图（如查询余额、转账、咨询产品）。分类置信度高时直接路由到对应技能；低置信度或拒识时，降级到大模型。

  第二层（大语言模型）：使用Few-shot提示词或小模型无法处理的复杂/长尾意图，让大模型理解后直接回答或触发函数。

联合训练：

  使用大模型生成合成意图数据，用于微调小模型，降低标注成本。
  
  将意图类别嵌入提示，要求模型输出结构化标签（JSON格式），同时进行意图分类和实体抽取。

主动学习：

  大模型处理后的真实对话日志，经人工校对后回流训练小模型，持续优化分类器。

优势：小模型低延迟、低成本处理大量高频请求；大模型兜底复杂情况及推理任务。
```


30. MCP原理

```
原理：
  用于统一大模型与外部数据源、工具之间的交互。其核心目标是为每个模型提供动态、安全、可组合的上下文环境。
  让Agent（Host）与具体工具 / 数据源解耦，方便维护
  
  “权限控制”：
    支持动态权限、细粒度授权、能力（Capability）协商
    每次工具调用可鉴权、按角色 /scope 控制访问
    可做数据脱敏、最小权限、审批流
  “日志追踪”：
    所有请求 / 响应、工具调用、资源访问全链路可日志
    会话 ID、用户、时间、参数、结果都可记录，满足合规审计
  “上下文（Context）的标准化管理”：
    上下文分段、分页、流式传输，避免一次性加载超大资源打爆 token
    支持上下文裁剪、优先级控制、增量更新
  “有状态会话 + 生命周期管理”：
    连接初始化、能力协商、心跳、关闭，会话全程可控
    支持长会话、多轮工具调用、跨请求状态保持
  “安全边界与隔离（沙箱思想）”：
    Server 独立进程 / 服务，与模型隔离
    网络、文件、执行权限可严格收敛，防止提权 / 注入
  
  
主要特点：

  资源（Resources）：模型可声明的数据源（文件、数据库、API端点），客户端按需获取。
  
  工具（Tools）：模型可调用的函数或操作，与Function Calling相似但协议统一。
  
  提示（Prompts）：预定义的提示模板，方便复用。
  
  传输层：基于JSON-RPC或SSE，支持本地/远程通信。

工作流程：

  MCP客户端（如IDE插件、Agent应用）连接MCP服务器。
  
  服务器提供可用资源、工具和提示的列表。
  
  模型在生成时，客户端根据模型需求动态获取资源或调用工具，并将结果注入上下文。

价值：解决多模型、多工具、多数据源的集成碎片化问题，使Agent生态更具互操作性。
```

| 对比项       | Function Calling         | MCP             |
|-----------|--------------------------|-----------------|
| **标准**    | 各厂商自有（OpenAI、Google 不同）  | 统一开放标准          |
| **生态**    | 每个模型单独适配                 | 一次开发，多模型通用      |
| **连接范围**  | 主要连接 API                 | 连接文件、数据库、API 等  |
| **双向通信**  | 单向（模型→工具）                | 支持双向实时通信        |


31. skills怎么写？

```
1. 定义技能元信息

  skill名称（唯一）、描述（何时触发）、版本、依赖。
  
  skill输入参数（类型、是否必填、示例）。
  
  skill输出格式（文本/JSON/文件等）。

{
  "name": "pdf_table_extractor",
  "description": "从PDF文件中提取表格数据，返回CSV格式",
  "parameters": {
    "type": "object",
    "properties": {
      "file_path": {"type": "string", "description": "PDF文件路径"},
      "page_range": {"type": "string", "description": "例如 '1-3'"}
    },
    "required": ["file_path"]
  }
}
```

32. 上下文怎么压缩，流程怎么编排?

```
常用压缩方法：

  摘要压缩：使用LLM对历史对话或文档进行摘要（如每5轮对话生成1条摘要）。
  
  滑动窗口：只保留最近N轮对话，更早的内容丢弃或仅保留系统级摘要。
  
  关键信息抽取：利用BERT等模型提取对话中的实体、用户目标、未完成任务，仅存储这些结构化信息。
  
  Token截断：按token数硬截断，但可能丢失语义。配合RecursiveCharacterTextSplitter按语义边界截断。
  
  RAG压缩：将长上下文转为向量库，检索时只取最相关片段，替代全量上下文。
  
常见编排模式：

  顺序链（Sequential Chain）：A→B→C，如“提取PDF表格 → 清洗数据 → 分析趋势”。
  
  路由链（Router Chain）：根据意图判断走哪个子链（如查询走RAG，计算走函数）。
  
  并行链（Parallel Chain）：同时执行多个独立任务，最后合并结果（如同时查天气和新闻）。
  
  循环/重试：当结果不符合要求时，重新调用模型或工具。
  
  人机协同：关键节点挂起等待人工确认或补充信息。

```

33. 微调怎么去做?

```
1. 准备数据集

  指令微调：收集(指令, 输入, 输出)三元组，例如{"instruction": "从下列财报中计算流动比率", "input": "流动资产=500, 流动负债=250", "output": "2.0"}。
  
  对话微调：整理多轮对话，保留角色（user/assistant/system）。
  
  数据量：高质量几千条即可见效果，一般需1万~10万条。

2. 选择基座模型

  开源：Llama 3、Qwen2、Baichuan2、DeepSeek-V2（中文财资能力较强）。
  
  商业API：OpenAI GPT-4o微调、智谱GLM微调。
  
3. 微调方式

  全量微调（Full Fine-tuning）：更新所有参数，效果好但昂贵，需多卡。
  
  LoRA / QLoRA：仅更新低秩适配器，显存占用低（QLoRA可4bit量化），主流选择。
  
  P-tuning v2：适合NLU任务。
  
4. 评估与部署

  使用领域内测试集（如财务问答、报表生成）计算BLEU、ROUGE、人工评分。
  
  合并LoRA权重到基座模型后部署，或动态加载LoRA。
  
5. 持续微调：定期用新收集的对话日志进行增量微调，防止灾难性遗忘可采用EWC或Replay机制。
```

================================================================

逻辑隔离要防 prompt injection 绕过 filter，在应用层强制注入过滤条件，不要信任 LLM 生成的 filter。

铁律：永远保留原始文本，不要只存向量。




| 问题    | 核心解                               |
|-------|-----------------------------------|
| 搜不准   | **多路召回 + 重排序 + 查询改写**             |
| 慢     | **并行 + 缓存 + 流式 + 限制输出长度**         |
| 隔离    | **metadata filter（轻量）或物理隔离（合规）**  |
| 迁移    | **保留原文，可重新编码，双写验证**               |
| 评估    | **RAGAS 自动化 + 人工兜底 + A-B 测试**     |


0603 面试问题：
    1. 对有1000万文档的信息如何设计rag，rag用的框架是什么？
    2. langgraph的持久化如何实现？
    3. asyncio 执行过程，主要的方法
    4. langchain有load文件的方法，其中最主要的两个方法是什么？
    5. langchain中断和恢复
    6. langchain核心组件