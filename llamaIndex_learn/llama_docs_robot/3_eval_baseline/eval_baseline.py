import os
import time
import sys
import nest_asyncio
nest_asyncio.apply()
import  random
random.seed(42)
import asyncio
import numpy as np

sys.path.append(os.path.join(os.getcwd(), ".."))

from llamaIndex_learn.llama_docs_robot.markdown_docs_reader import MarkdownDocsReader
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import Document
from llama_index.core.evaluation import DatasetGenerator, ResponseEvaluator
from llama_index.core.prompts import Prompt
from llama_index.core import Response

def load_markdown_docs(filepath):
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath,
        required_exts=[".md"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive=True
    )

    documents = loader.load_data()

    # exclude some metadata from the LLM
    for doc in documents:
        doc.excluded_llm_metadata_keys = ["File Name", "Content Type", "Header Path"]

    return documents

getting_started_docs = load_markdown_docs("../docs/getting_started")
community_docs = load_markdown_docs("../docs/community")
data_docs = load_markdown_docs("../docs/core_modules/data_modules")
agent_docs = load_markdown_docs("../docs/core_modules/agent_modules")
model_docs = load_markdown_docs("../docs/core_modules/model_modules")
query_docs = load_markdown_docs("../docs/core_modules/query_modules")
supporting_docs = load_markdown_docs("../docs/core_modules/supporting_modules")
tutorials_docs = load_markdown_docs("../docs/end_to_end_tutorials")
contributing_docs = load_markdown_docs("../docs/development")

try:
    getting_started_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./getting_started_index"))
    community_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./community_index"))
    data_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./data_index"))
    agent_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./agent_index"))
    model_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./model_index"))
    query_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./query_index"))
    supporting_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./supporting_index"))
    tutorials_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./tutorials_index"))
    contributing_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./contributing_index"))
except:
    getting_started_index = VectorStoreIndex.from_documents(getting_started_docs)
    getting_started_index.storage_context.persist(persist_dir="./getting_started_index")

    community_index = VectorStoreIndex.from_documents(community_docs)
    community_index.storage_context.persist(persist_dir="./community_index")

    data_index = VectorStoreIndex.from_documents(data_docs)
    data_index.storage_context.persist(persist_dir="./data_index")

    agent_index = VectorStoreIndex.from_documents(agent_docs)
    agent_index.storage_context.persist(persist_dir="./agent_index")

    model_index = VectorStoreIndex.from_documents(model_docs)
    model_index.storage_context.persist(persist_dir="./model_index")

    query_index = VectorStoreIndex.from_documents(query_docs)
    query_index.storage_context.persist(persist_dir="./query_index")

    supporting_index = VectorStoreIndex.from_documents(supporting_docs)
    supporting_index.storage_context.persist(persist_dir="./supporting_index")

    tutorials_index = VectorStoreIndex.from_documents(tutorials_docs)
    tutorials_index.storage_context.persist(persist_dir="./tutorials_index")

    contributing_index = VectorStoreIndex.from_documents(contributing_docs)
    contributing_index.storage_context.persist(persist_dir="./contributing_index")


getting_started_tool = QueryEngineTool.from_defaults(
    query_engine=getting_started_index.as_query_engine(),
    name="Getting Started",
    description="Useful for answering questions about installing and running llama index, as well as basic explanations of how llama index works."
)

community_tool = QueryEngineTool.from_defaults(
    query_engine=community_index.as_query_engine(),
    name="Community",
    description="Useful for answering questions about integrations and other apps built by the community."
)

data_tool = QueryEngineTool.from_defaults(
    query_engine=data_index.as_query_engine(),
    name="Data Modules",
    description="Useful for answering questions about data loaders, documents, nodes, and index structures."
)

agent_tool = QueryEngineTool.from_defaults(
    query_engine=agent_index.as_query_engine(),
    name="Agent Modules",
    description="Useful for answering questions about data agents, agent configurations, and tools."
)

model_tool = QueryEngineTool.from_defaults(
    query_engine=model_index.as_query_engine(),
    name="Model Modules",
    description="Useful for answering questions about using and configuring LLMs, embedding modles, and prompts."
)

query_tool = QueryEngineTool.from_defaults(
    query_engine=query_index.as_query_engine(),
    name="Query Modules",
    description="Useful for answering questions about query engines, query configurations, and using various parts of the query engine pipeline."
)

supporting_tool = QueryEngineTool.from_defaults(
    query_engine=supporting_index.as_query_engine(),
    name="Supporting Modules",
    description="Useful for answering questions about supporting modules, such as callbacks, service context, and avaluation."
)

tutorials_tool = QueryEngineTool.from_defaults(
    query_engine=tutorials_index.as_query_engine(),
    name="Tutorials",
    description="Useful for answering questions about end-to-end tutorials and giving examples of specific use-cases."
)

contributing_tool = QueryEngineTool.from_defaults(
    query_engine=contributing_index.as_query_engine(),
    name="Contributing",
    description="Useful for answering questions about contributing to llama index, including how to contribute to the codebase and how to build documentation."
)

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        getting_started_tool,
        community_tool,
        data_tool,
        agent_tool,
        model_tool,
        query_tool,
        supporting_tool,
        tutorials_tool,
        contributing_tool
    ],
    # enable this for streaming
    # response_synthesizer=get_response_synthesizer(streaming=True),
    verbose=False
)

response = query_engine.query("How do I install llama index?")
print(str(response))

documents = SimpleDirectoryReader("../docs", recursive=True, required_exts=[".md"]).load_data()

all_text = ""

for doc in documents:
    all_text += doc.text

giant_document = Document(text=all_text)


gpt4_service_context = ServiceContext.from_defaults(llm=OpenAI(llm="gpt-4", temperature=0))

question_dataset = []
if os.path.exists("question_dataset.txt"):
    with open("question_dataset.txt", "r") as f:
        for line in f:
            question_dataset.append(line.strip())
else:
    # generate questions
    data_generator = DatasetGenerator.from_documents(
        [giant_document],
        text_question_template=Prompt(
            "A sample from the LlamaIndex documentation is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Using the documentation sample, carefully follow the instructions below:\n"
            "{query_str}"
        ),
        question_gen_query=(
            "You are an evaluator for a search pipeline. Your task is to write a single question "
            "using the provided documentation sample above to test the search pipeline. The question should "
            "reference specific names, functions, and terms. Restrict the question to the "
            "context information provided.\n"
            "Question: "
        ),
        # set this to be low, so we can generate more questions
        service_context=gpt4_service_context
    )
    generated_questions = data_generator.generate_questions_from_nodes()

    # randomly pick 40 questions from each dataset
    generated_questions = random.sample(generated_questions, 40)
    question_dataset.extend(generated_questions)

    print(f"Generated {len(question_dataset)} questions.")

    # save the questions!
    with open("question_dataset.txt", "w") as f:
        for question in question_dataset:
            f.write(f"{question.strip()}\n")


def evaluate_query_engine(evaluator, query_engine, questions):
    async def run_query(query_engine, q):
        try:
            return await query_engine.aquery(q)
        except:
            return Response(response="Error, query failed.")

    total_correct = 0
    all_results = []
    for batch_size in range(0, len(questions), 5):
        batch_qs = questions[batch_size:batch_size + 5]

        tasks = [run_query(query_engine, q) for q in batch_qs]
        responses = asyncio.run(asyncio.gather(*tasks))
        print(f"finished batch {(batch_size // 5) + 1} out of {len(questions) // 5}")

        for response in responses:
            eval_result = 1 if "YES" in evaluator.evaluate(response) else 0
            total_correct += eval_result
            all_results.append(eval_result)

        # helps avoid rate limits
        time.sleep(1)

    return total_correct, all_results

evaluator = ResponseEvaluator(service_context=gpt4_service_context)

total_correct, all_results = evaluate_query_engine(evaluator, query_engine, question_dataset)

print(f"Hallucination? Scored {total_correct} out of {len(question_dataset)} questions correctly.")

hallucinated_question = np.array(question_dataset)[np.array(all_results) == 0]
print(f"Hallucinated questions: {hallucinated_question}")











