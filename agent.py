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
from typing import Optional
import re

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
    """Build a RAG agent with improved configuration."""
    # Define the system prompt
    system_prompt = """You are a specialized assistant for the 2022 Chevy Colorado. 
    Your task is to provide direct, factual information about the vehicle's features, specifications, and options.
    DO NOT engage in general conversation or roleplay.
    DO NOT include conversation history or examples in your responses.
    DO NOT use phrases like "Here's how to use the tool" or "Let me show you an example".
    ALWAYS provide direct answers about the Chevy Colorado.
    If you don't know the answer, say "I don't have information about that specific feature"."""

    # Define the tools
    tools = [
        Tool(
            name="Car Manual",
            func=lambda query: vector_store.similarity_search(query, k=3),
            description="""Use this tool to look up specific features, specifications, and details about the 2022 Chevy Colorado.
            Input should be a direct question about the vehicle.
            The tool will return relevant information from the manual.
            DO NOT include conversation examples or tool usage instructions in the response.""",
        )
    ]

    # Initialize the LLM with stricter parameters
    llm = HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        context_window=2048,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
        device_map="auto",
        tokenizer_kwargs={"max_length": 2048},
    )

    # Create the agent with improved configuration
    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        system_prompt=system_prompt,
        max_iterations=3,
        handle_parsing_errors=True,
    )

    return agent

def chat_with_agent(agent, query: str, context: Optional[str] = None) -> str:
    """Chat with the agent and validate the response."""
    try:
        # Prepare the query
        if context:
            full_query = f"Context: {context}\n\nQuestion: {query}"
        else:
            full_query = query

        # Get response from agent
        response = agent.chat(full_query)

        # Extract the actual answer
        if isinstance(response, str):
            answer = response
        else:
            answer = str(response)

        # Clean up the response
        answer = answer.strip()
        
        # Remove any conversation history or examples
        if "User:" in answer or "Assistant:" in answer:
            # Extract only the last answer
            parts = answer.split("Answer:")
            if len(parts) > 1:
                answer = parts[-1].strip()
            else:
                parts = answer.split("Final Answer:")
                if len(parts) > 1:
                    answer = parts[-1].strip()
                else:
                    # If no clear answer marker, take the last line
                    lines = answer.split("\n")
                    answer = lines[-1].strip()

        # Validate the response
        if len(answer) < 20:  # Minimum length check
            raise ValueError("Response too short")
            
        if "Here's how" in answer or "Let me show" in answer:
            raise ValueError("Response contains example text")
            
        if "User:" in answer or "Assistant:" in answer:
            raise ValueError("Response contains conversation history")

        return answer

    except Exception as e:
        logger.error(f"Error in chat_with_agent: {str(e)}")
        raise

# Initialize semantic cache with BGE embeddings
cache = SemanticCache(
    name="rag_cache",
    prefix="cache",
    distance_threshold=0.15,
    ttl=120,  # 2 minutes TTL
    vectorizer=HFTextVectorizer(model="BAAI/bge-small-en-v1.5")
)

def debug_cache_operation(operation: str, query: str, response: Optional[str] = None, error: Optional[Exception] = None):
    """Log detailed information about cache operations."""
    log_data = {
        "operation": operation,
        "query": query,
        "response_length": len(response) if response else 0,
        "response_preview": response[:100] + "..." if response else None,
        "error": str(error) if error else None,
        "cache_size": cache.size() if hasattr(cache, 'size') else "unknown",
        "cache_stats": cache.stats() if hasattr(cache, 'stats') else "unknown"
    }
    logger.info(f"Cache Debug: {log_data}")

def validate_cached_response(response: str) -> tuple[bool, str]:
    """Validate a cached response and return (is_valid, reason)."""
    if not response or not isinstance(response, str):
        return False, "Empty or invalid response type"
        
    # Basic validation
    if len(response) < 20 or len(response) > 1000:
        return False, f"Invalid length: {len(response)}"
        
    words = response.split()
    if len(words) < 5 or len(words) > 200:
        return False, f"Invalid word count: {len(words)}"
        
    # Check for corruption patterns
    if response.count("or") > 5:
        return False, f"Excessive 'or' repetition: {response.count('or')}"
    if response.count("ing") > 10:
        return False, f"Excessive 'ing' repetition: {response.count('ing')}"
    if response.count("tion") > 10:
        return False, f"Excessive 'tion' repetition: {response.count('tion')}"
        
    # Check word variety
    unique_words = len(set(word.lower() for word in words))
    if unique_words < len(words) * 0.4:
        return False, f"Low word variety: {unique_words}/{len(words)} unique words"
        
    # Check for specific corruption patterns
    if re.search(r'(\w+)\1{2,}', response):
        return False, "Repeated word pattern detected"
    if re.search(r'[^\w\s.,!?-]', response):
        return False, "Unusual characters detected"
    if re.search(r'\b\w{15,}\b', response):
        return False, "Very long words detected"
        
    return True, "Valid response"

def get_cached_response(query: str) -> Optional[str]:
    """Get a validated cached response with detailed logging."""
    try:
        response = cache.get(query)
        if response:
            is_valid, reason = validate_cached_response(response)
            if is_valid:
                debug_cache_operation("cache_hit", query, response)
                return response
            else:
                debug_cache_operation("invalid_cache_hit", query, response, Exception(reason))
                clear_cache()  # Clear cache on invalid response
        return None
    except Exception as e:
        debug_cache_operation("cache_error", query, error=e)
        return None

def cache_response(query: str, response: str) -> bool:
    """Cache a validated response with detailed logging."""
    try:
        is_valid, reason = validate_cached_response(response)
        if is_valid:
            cache.set(query, response)
            debug_cache_operation("cache_store", query, response)
            return True
        else:
            debug_cache_operation("invalid_cache_store", query, response, Exception(reason))
            return False
    except Exception as e:
        debug_cache_operation("cache_error", query, response, e)
        return False

def clear_cache() -> bool:
    """Clear the cache with detailed logging."""
    try:
        cache.flush()
        debug_cache_operation("cache_clear", "all")
        return True
    except Exception as e:
        debug_cache_operation("cache_error", "clear", error=e)
        return False

def invoke_agent(prompt: str, force_refresh: bool = False) -> str:
    """
    Invoke the agent with improved cache handling and debugging.
    """
    try:
        if force_refresh:
            clear_cache()
            debug_cache_operation("force_refresh", prompt)
        
        # Try to get cached response
        if not force_refresh:
            cached_response = get_cached_response(prompt)
            if cached_response:
                return cached_response
        
        # Generate new response
        response = agent.chat(prompt)
        response_content = response.response if hasattr(response, 'response') else str(response)
        
        # Cache the response if valid
        if cache_response(prompt, response_content):
            debug_cache_operation("response_cached", prompt, response_content)
        
        return response_content
        
    except Exception as e:
        debug_cache_operation("agent_error", prompt, error=e)
        return f"Error: {str(e)}"

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
