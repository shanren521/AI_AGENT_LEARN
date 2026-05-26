from qdrant_client import QdrantClient


def get_qdrant_client():
    return QdrantClient(
        url="http://localhost:6333",
        timeout=30,
    )





