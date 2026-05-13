import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import MetadataMode

from llamaIndex_learn.llama_docs_robot.markdown_docs_reader import MarkdownDocsReader


def load_markdown_docs(filepath):
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath,
        required_exts=["*.rst", "*.ipynb", "*.py", "*.bat", "*.txt", "*.png", "*.jpg", "*.jpeg", "*.csv", "*.html", "*.js", "*.css", "*.pdf", "*.json"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive=True  # 递归读取子文件夹
    )

    return loader.load_data()

getting_started_docs = load_markdown_docs("../docs/getting_started")
community_docs = load_markdown_docs("../docs/community")
data_docs = load_markdown_docs("../docs/core_modules/data_modules")
agent_docs = load_markdown_docs("../docs/core_modules/agent_modules")
model_docs = load_markdown_docs("../docs/core_modules/model_modules")
query_docs = load_markdown_docs("../docs/core_modules/query_modules")
supporting_docs = load_markdown_docs("../docs/core_modules/supporting_modules")
tutorials_docs = load_markdown_docs("../docs/end_to_end_tutorials")
contributing_docs = load_markdown_docs("../docs/development")

text_template = "Content Metadata:\n{metadata_str}\n\nContent:\n{content}"

metadata_template = "{key}: {value},"
metadata_seperator= " "

for doc in agent_docs:
    doc.text_template = text_template
    doc.metadata_template = metadata_template
    doc.metadata_seperator = metadata_seperator

agent_docs[0].excluded_embed_metadata_keys = ["File Name"]
agent_docs[0].excluded_llm_metadata_keys = ["File Name"]








