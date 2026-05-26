from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from embedder_demo import embed_texts

def ingest_docs(client: QdrantClient, collection: str, docs: list[dict]):
    vectors = embed_texts([doc["text"] for doc in docs])

    points = [
        PointStruct(
            id=doc["id"],
            vector=vector,
            payload={
                "text": doc["text"],
                "source": doc["source"],
                "tenant_id": doc["tenant_id"],
                "embedding_model": "text-embedding-3-small"
            }
        )
        for doc, vector in zip(docs, vectors)
    ]
    client.upsert(collection, points, wait=True)






