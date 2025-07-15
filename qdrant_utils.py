import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding

COLLECTION_NAME = "voice-rag-agent"

def setup_qdrant(qdrant_url, api_key):
    client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=60.0)
    embedder = TextEmbedding()
    dim = len(list(embedder.embed(["test"]))[0])

    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    return client, embedder

def store_embeddings(client, embedder, documents):
    batch = []
    for i, doc in enumerate(documents):
        try:
            vector = list(embedder.embed([doc.page_content]))[0]
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"content": doc.page_content, **doc.metadata}
            )
            batch.append(point)
        except Exception:
            continue

        if len(batch) >= 64 or i == len(documents) - 1:
            client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
            batch = []