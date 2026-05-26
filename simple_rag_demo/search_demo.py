from qdrant_client.models import Filter, FieldCondition, MatchValue
from embedder_demo import embed_texts

def search_docs(client, collection: str, question: str, tenant_id: str, limit: int = 5):
    query_vector = embed_texts([question])[0]

    query_filter = Filter(
        must=[
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id)
            )
        ]
    )

    result = client.query_points(
        collection_name=collection,
        vector=query_vector,
        filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return [
        {
            "id": point.id,
            "score": point.socre,
            "text": point.text,
            "source": point.payload["source"],
        }
        for point in result.points
    ]









