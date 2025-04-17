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
from pydantic import BaseModel
import logging

# Initialize embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Initialize LLM with TinyLlama
llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    max_new_tokens=256,
    context_window=2048,
    generate_kwargs={
        "temperature": 0.1,
        "do_sample": True,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    },
    system_prompt=(
        "You are a specialized assistant that ONLY provides factual information about the Chevy Colorado 2022. "
        "NEVER engage in general conversation or roleplay. "
        "ONLY use the provided car manual tool to get accurate information. "
        "ALWAYS give direct, specific answers about the 2022 Chevy Colorado. "
        "If information is not available in the manual, say so directly. "
        "Do not make assumptions or provide information about other vehicles."
    )
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
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Instead, define Redis connection details directly
REDIS_HOST = "localhost"
REDIS_PORT = 6379

logger = logging.getLogger(__name__)

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

    # Create agent tools
    tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="car_manual",
                description=(
                    "This tool provides ONLY factual information about the 2022 Chevy Colorado. "
                    "Use it to look up specific features, specifications, and details from the manual. "
                    "Do not engage in conversation or discuss other vehicles. "
                    "Return direct, factual answers about the 2022 Chevy Colorado only."
                )
            ),
        )
    ]

    # Create ReAct agent with system message
    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        system_message=ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a specialized tool that ONLY provides information about the 2022 Chevy Colorado. "
                "Your responses must ONLY contain factual details from the car manual. "
                "Do not engage in conversation, roleplay, or discuss other vehicles. "
                "If asked about anything not related to the 2022 Chevy Colorado, "
                "respond with: 'I can only provide information about the 2022 Chevy Colorado.'"
            )
        )
    )

    return agent

def chat_with_agent(agent, query: str, context: str | None = None) -> str:
    try:
        # If context is provided, prepend it to the query
        if context:
            enhanced_query = f"Context: {context}\nQuestion: {query}"
            logger.info(f"Enhanced query with context: {enhanced_query}")
            response = agent.chat(enhanced_query)
        else:
            logger.info(f"Processing query without context: {query}")
            response = agent.chat(query)
        
        # Extract the final answer from the ReAct agent's response
        if hasattr(response, 'response'):
            response_text = response.response
            
            # If response contains "Answer: ", extract everything after it
            if "Answer: " in response_text:
                final_answer = response_text.split("Answer: ")[-1].strip()
                return final_answer
            
            # If response contains "Final Answer: ", extract everything after it
            if "Final Answer: " in response_text:
                final_answer = response_text.split("Final Answer: ")[-1].strip()
                return final_answer
            
            # Remove any conversation or roleplay patterns
            response_text = response_text.strip()
            if "User:" in response_text or "Assistant:" in response_text:
                return "Error: Invalid response format. Please try again."
            
            return response_text.strip()
            
        elif hasattr(response, 'content'):
            return response.content.strip()
        else:
            logger.error(f"Unexpected response type: {type(response)}")
            return "Error: Unexpected response format"
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Error processing query: {str(e)}"

# Initialize semantic cache with BGE embeddings
cache = SemanticCache(
    name="chevy_cache",
    prefix="cache",
    distance_threshold=0.2,
    ttl=300,  # Increased TTL to 5 minutes
    vectorizer=HFTextVectorizer(model="BAAI/bge-small-en-v1.5")
)

def refresh_cache():
    """Clear the semantic cache to force fresh responses"""
    try:
        cache.flush()
        logger.info("Successfully cleared semantic cache")
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False

def invoke_agent(prompt: str, force_refresh: bool = False) -> str:
    """
    Invoke the agent with optional cache refresh
    Args:
        prompt: The query prompt
        force_refresh: If True, bypass cache and store new response
    """
    if force_refresh:
        refresh_cache()
    
    if not force_refresh and (cached_result := cache.check(prompt=prompt)):
        response = cached_result[0]['response']
        # Validate cached response
        if 'car car car' in response.lower() or len(set(response.split())) < 5:
            logger.warning("Found corrupted cached response, forcing refresh")
            refresh_cache()
            return invoke_agent(prompt, force_refresh=True)
        return response
        
    response = agent.chat(prompt)
    response_content = response.response if hasattr(response, 'response') else str(response)
    
    # Validate response before caching
    if 'car car car' not in response_content.lower() and len(set(response_content.split())) >= 5:
        cache.store(prompt=prompt, response=response_content)
    
    return response_content

class QueryRequest(BaseModel):
    query: str
    context: str | None = None
    force_refresh: bool = False  # Add option to force cache refresh

# if __name__ == "__main__":
#     # Import the vector store from ingestion
#     from ingestion import doc_ingestion_pipeline
    
#     # Build RAG agent
#     agent = build_rag_agent(doc_ingestion_pipeline.vector_store)
    
#     # Example queries to test the agent
#     test_queries = [
#         "What is the seating capacity of the Chevy Colorado 2022?",
#         "What are the safety features available?",
#         "Tell me about the engine specifications.",
#     ]
    
#     print("\nTesting RAG Agent:")
#     print("=================")
#     for query in test_queries:
#         print(f"\nQuery: {query}")
#         response = chat_with_agent(agent, query)
#         print(f"Response: {response}")

#     # Interactive mode
#     while True:
#         user_query = input("\nEnter your question (or 'quit' to exit): ")
#         if user_query.lower() == 'quit':
#             break
#         response = chat_with_agent(agent, user_query)
#         print(f"Response: {response}")
