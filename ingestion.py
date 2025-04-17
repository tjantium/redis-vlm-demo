import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # This model uses 384 dimensions
    device="cpu"  # Explicitly set device
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
                "dims": 384,  # all-MiniLM-L6-v2 uses 384 dimensions
                "algorithm": "flat",
                "distance_metric": "cosine",
                "datatype": "FLOAT32"  # Explicitly set datatype
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

def clear_and_reingest():
    """Clear existing data and re-ingest documents with new embedding model"""
    import redis
    from pathlib import Path
    
    # Connect to Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    
    try:
        # Clear existing data
        logger.info("Clearing existing data...")
        r.flushdb()  # This will clear all data in the current Redis database
        
        # Drop existing index if it exists
        try:
            r.execute_command("FT.DROPINDEX docs")
            logger.info("Dropped existing index")
        except Exception as e:
            logger.info("No existing index to drop")
        
        # Verify data directory exists
        data_dir = Path('data')
        if not data_dir.exists():
            logger.error("Data directory not found!")
            return False
            
        # Load and process documents
        logger.info("Loading documents...")
        documents = SimpleDirectoryReader('data').load_data()
        
        # Create new ingestion pipeline with updated schema
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
        
        # Run ingestion pipeline
        logger.info("Starting document ingestion...")
        doc_ingestion_pipeline.run(documents=documents, show_progress=True)
        
        # Verify ingestion
        verify_ingestion_cli()
        return True
        
    except Exception as e:
        logger.error(f"Error during re-ingestion: {str(e)}")
        return False

if __name__ == "__main__":
    # Run re-ingestion
    success = clear_and_reingest()
    if success:
        logger.info("Re-ingestion completed successfully!")
    else:
        logger.error("Re-ingestion failed!")