import os
import redis
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

embedding_model = DashScopeEmbeddings(model="text-embedding-v3")

redis_url = "redis://localhost:6379"

redis_client = redis.from_url(redis_url)

print(redis_client.ping())

config = RedisConfig(
    index_name="couplet",
    redis_url=redis_url
)

vector_store = RedisVectorStore(embedding_model, config=config)

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR.parent / "resource" / "couplet_test.csv"

lines = []
with open(csv_path, "r", encoding="utf-8") as f:
    for line in f:
        print(line)
        lines.append(line.strip())

vector_store.add_texts(lines)