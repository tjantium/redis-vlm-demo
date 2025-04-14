from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.llms import ChatMessage, MessageRole
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

# Initialize embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Initialize LLM with proper chat template
llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    max_new_tokens=256,
    generate_kwargs={
        "temperature": 0.7,
        "top_p": 0.95
    },
    system_prompt="You are a helpful AI assistant that answers questions about the Chevy Colorado 2022."
)

# # Option 2: Using TinyLlama (better performance, still accessible)
# llm = HuggingFaceLLM(
#     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     device_map="auto",
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "top_p": 0.95},
#     system_prompt="You are a helpful AI assistant that answers questions about the Chevy Colorado 2022."
# )

# # Option 3: Using FLAN-T5-small (good for Q&A tasks)
# llm = HuggingFaceLLM(
#     model_name="google/flan-t5-small",
#     tokenizer_name="google/flan-t5-small",
#     device_map="auto",
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7},
# )

# Set global configurations
Settings.llm = llm
Settings.embed_model = embed_model

# Instead, define Redis connection details directly
REDIS_HOST = "localhost"
REDIS_PORT = 6379

def build_rag_agent(vector_store):
    # Create retriever with similarity top-k
    retriever = VectorIndexRetriever(
        index=VectorStoreIndex.from_vector_store(vector_store),
        similarity_top_k=3
    )

    # Create response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        use_async=True
    )

    # Create query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[]
    )

    # Create chat memory
    chat_storage = RedisChatStore(
        redis_url=f"redis://{REDIS_HOST}:{REDIS_PORT}", 
        ttl=300
    )
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_storage,
        chat_store_key="chat_history"
    )

    # Create agent tools
    tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="car_manual",
                description=(
                    "Provides detailed information about the Chevy Colorado 2022 car. "
                    "Use specific questions to get accurate information from the manual."
                )
            ),
        )
    ]

    # Update the context using chat message format
    system_message = ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are a knowledgeable customer support agent for the Chevy Colorado 2022. "
            "Use the car manual tool to provide accurate and detailed information. "
            "If you're not sure about something, say so rather than making assumptions."
        )
    )

    # Create ReAct agent with system message
    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        memory=chat_memory,
        system_message=system_message
    )

    return agent

def chat_with_agent(agent, query: str) -> str:
    try:
        response = agent.chat(query)
        return str(response)
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Initialize semantic cache with BGE embeddings
cache = SemanticCache(
    name="chevy_cache",
    prefix="cache",
    distance_threshold=0.2,
    ttl=60,
    vectorizer=HFTextVectorizer(model="BAAI/bge-small-en-v1.5")
)

def invoke_agent(prompt: str) -> str:
    if cached_result := cache.check(prompt=prompt):
        response = cached_result[0]['response']
        return response
    response = agent.chat(prompt)
    cache.store(prompt=prompt, response=response.response)
    return response.response

if __name__ == "__main__":
    # Import the vector store from ingestion
    from ingestion import doc_ingestion_pipeline
    
    # Build RAG agent
    agent = build_rag_agent(doc_ingestion_pipeline.vector_store)
    
    # Example queries to test the agent
    test_queries = [
        "What is the seating capacity of the Chevy Colorado 2022?",
        "What are the safety features available?",
        "Tell me about the engine specifications.",
    ]
    
    print("\nTesting RAG Agent:")
    print("=================")
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = chat_with_agent(agent, query)
        print(f"Response: {response}")

    # Interactive mode
    while True:
        user_query = input("\nEnter your question (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        response = chat_with_agent(agent, user_query)
        print(f"Response: {response}")
