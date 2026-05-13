import os
import nest_asyncio

nest_asyncio.apply()

from markdown_docs_reader import MarkdownDocsReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import Document, MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def load_markdown_docs(filepath):
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath,
        required_exts=[".md"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive=True  # 递归读取子文件夹
    )

    documents = loader.load_data()
    documents = [
        Document(text="\n\n".join(
            document.get_content(metadata_mode=MetadataMode.ALL)
            for document in documents
        ))
    ]

    large_chunk_size = 1536
    # 将文档切分为3个层级，父块大小1536，子块大小512，返回所有节点和叶子节点(最小块)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[
            large_chunk_size,
            large_chunk_size // 3
        ]
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes, get_leaf_nodes(nodes)


def get_query_engine_tool(directory, description, postprocessors=None):
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./data_{os.path.basename(directory)}"
        )
        index = load_index_from_storage(storage_context)

        retriever = AutoMergingRetriever(
            index.as_retriever(similarity_top_k=12),
            storage_context=storage_context
        )
    except:
        nodes, leaf_nodes = load_markdown_docs(directory)

        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=docstore)

        index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

        index.storage_context.persist(persist_dir=f"./data_{os.path.basename(directory)}")

        retriever = AutoMergingRetriever(
            index.as_retriever(similarity_top_k=12),
            storage_context=storage_context
        )

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=postprocessors or [],
    )

    return QueryEngineTool(query_engine=query_engine, metadata=ToolMetadata(name=directory, description=description))


def create_query_engine(
        data_dir: str = "./data",  # 你的文档放在这个文件夹
        persist_dir: str = "./storage_index",  # 索引持久化目录
        force_reindex: bool = False
):
    """
    创建并返回一个 QueryEngine（完全匹配你调用的函数）
    自动读取文档 → 构建向量索引 → 返回查询引擎
    """
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # ======================
    # 2. 加载已有索引 或 重建索引
    # ======================
    if os.path.exists(persist_dir) and not force_reindex:
        # 从磁盘加载索引
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        # 读取文档
        documents = SimpleDirectoryReader(data_dir).load_data()

        # 创建索引
        index = VectorStoreIndex.from_documents(documents)

        # 保存到磁盘
        index.storage_context.persist(persist_dir=persist_dir)

    # ======================
    # 3. 创建并返回查询引擎
    # ======================
    query_engine = index.as_query_engine(
        response_mode="compact",  # 精简回答
        verbose=True
    )

    return query_engine
