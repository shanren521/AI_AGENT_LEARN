import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "text-embedding-3-small"

def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in resp.data]
















