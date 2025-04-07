from testcontainers.redis import RedisContainer
from testcontainers.core.container import DockerContainer

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from redisvl.schema import IndexSchema
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.redis import RedisVectorStore

# Instead of creating a new container, just use the existing Redis connection
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# Since we're using HuggingFace instead of Bedrock, we'll keep that
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Custom index schema for Redis
custom_schema = IndexSchema.from_dict({
    "index": {
        "name": "docs",
        "prefix": "chunk",
        "key_separator": ":"
    },
    "fields": [
        {"type": "tag", "name": "id"},
        {"type": "tag", "name": "doc_id"},
        {"type": "text", "name": "text"},
        {
            "type": "vector",
            "name": "vector",
            "attrs": {
                # Adjusted dims for all-MiniLM-L6-v2 which uses 384 dimensions
                "dims": 384,
                "algorithm": "flat",
                "distance_metric": "cosine",
            },
        },
    ],
})

# Load documents
documents = SimpleDirectoryReader('data').load_data()

# Create ingestion pipeline
doc_ingestion_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        REDIS_HOST, REDIS_PORT, namespace="doc-store"
    ),
    vector_store=RedisVectorStore(
        schema=custom_schema,
        redis_url=f"redis://{REDIS_HOST}:{REDIS_PORT}",
    ),
    cache=IngestionCache(
        cache=RedisCache.from_host_and_port(REDIS_HOST, REDIS_PORT),
        collection="doc-cache",
    ),
    docstore_strategy=DocstoreStrategy.UPSERTS,
)


def verify_ingestion_cli():
    import redis
    import json
    
    # Connect to Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    
    print("\nVerification Results:")
    print("--------------------")
    
    # 1. Check vector store chunks
    chunks = r.keys("chunk:*")
    print(f"Number of chunks: {len(chunks)}")
    
    # 2. Check document store
    doc_store_keys = r.keys("doc-store:*")
    print(f"Number of doc store entries: {len(doc_store_keys)}")
    
    # 3. Show index info
    try:
        index_info = r.execute_command("FT.INFO docs")
        print("\nIndex Info:")
        print(f"Number of docs in index: {index_info[index_info.index('num_docs') + 1]}")
    except Exception as e:
        print(f"Error getting index info: {e}")

# Add this to ingestion.py after the ingestion pipeline runs
if __name__ == "__main__":
    # Run ingestion
    doc_ingestion_pipeline.run(documents=documents, show_progress=True)
    
    # Verify using CLI commands
    verify_ingestion_cli()