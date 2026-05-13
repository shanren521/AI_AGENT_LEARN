import os
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import Prompt
from llama_index.core.llms import ChatMessage

with open("../docs/starter_example.md", "r") as f:
    text = f.read()

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

text_qa_template = Prompt(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)

refine_template = Prompt(
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question: {query_str}. "
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

question = "How can I install llama-index?"
prompt = text_qa_template.format(context_str=text, query_str=question)

response = llm.complete(prompt)
result = response.text

stream_response = llm.stream_complete(prompt)
for chunk in stream_response:
    print(chunk.delta, end="")

chat_history = [
    ChatMessage(role="system", content="You are a helpful QA chatbot that can answer questions about llama-index."),
    ChatMessage(role="user", content="How do I create an index?"),
]

response = llm.chat(chat_history)
print(response.message)

