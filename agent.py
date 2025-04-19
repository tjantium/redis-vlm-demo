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

# Initialize LLM with stricter parameters
llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    max_new_tokens=256,  # Reduced from 512 to prevent token overflow
    context_window=2048,
    generate_kwargs={
        "temperature": 0.1,
        "do_sample": False,  # Disable sampling for more deterministic output
        "top_k": 1,  # Only use the most likely token
        "repetition_penalty": 1.2,  # Penalize repetition
        "no_repeat_ngram_size": 3  # Prevent repeating phrases
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
    # Create retriever with similarity top-k
    retriever = VectorIndexRetriever(
        index=VectorStoreIndex.from_vector_store(vector_store),
        similarity_top_k=3
    )

    # Create response synthesizer with compact mode
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
                    "Use this tool to get information about the 2022 Chevy Colorado. "
                    "The tool will search the car manual and return relevant information. "
                    "No special formatting is required - just ask your question directly."
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
                "You are a helpful assistant that provides information about the 2022 Chevy Colorado. "
                "When answering questions:\n"
                "1. Use the car_manual tool to find relevant information\n"
                "2. Return direct, factual answers without any special formatting\n"
                "3. Do not include any thought process or action formatting\n"
                "4. If you can't find the information, say so directly\n"
                "5. Keep your answers concise and focused on the question"
            )
        )
    )

    return agent

def validate_llm_response(response: str) -> tuple[bool, str]:
    """Validate LLM responses with minimal restrictions."""
    if not response or not isinstance(response, str):
        return False, "Empty or invalid response type"
        
    # Basic validation for LLM responses
    if len(response.strip()) < 5:  # Very minimal length requirement
        return False, f"Response too short: {len(response)} characters"
        
    # Only check for obvious corruption
    if re.search(r'(\w+)\1{4,}', response):  # Only flag extreme repetition
        return False, "Response contains extreme repetition"
        
    # Check for conversation history
    if "User:" in response or "Assistant:" in response:
        return False, "Response contains conversation history"
        
    return True, "Valid response"

def validate_cached_response(response: str) -> tuple[bool, str]:
    """Validate cached responses with stricter rules."""
    if not response or not isinstance(response, str):
        return False, "Empty or invalid response type"
        
    # Stricter validation for cached responses
    if len(response) < 10:
        return False, f"Cached response too short: {len(response)} characters"
    if len(response) > 2000:
        return False, f"Cached response too long: {len(response)} characters"
        
    words = response.split()
    if len(words) < 3:
        return False, f"Too few words in cached response: {len(words)}"
        
    # Check for corruption patterns in cache
    if response.count("or") > 10:
        return False, f"Excessive 'or' repetition in cache: {response.count('or')}"
    if response.count("ing") > 15:
        return False, f"Excessive 'ing' repetition in cache: {response.count('ing')}"
    if response.count("tion") > 15:
        return False, f"Excessive 'tion' repetition in cache: {response.count('tion')}"
        
    # Check for specific corruption patterns
    if re.search(r'(\w+)\1{3,}', response):
        return False, "Repeated word pattern in cache"
    if re.search(r'[^\w\s.,!?-]', response):
        return False, "Unusual characters in cache"
        
    return True, "Valid cached response"

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
        logger.info(f"Raw response: {response}")

        # Extract the actual answer
        if isinstance(response, str):
            answer = response
        else:
            # Try different ways to extract the response
            if hasattr(response, 'response'):
                answer = response.response
            elif hasattr(response, 'content'):
                answer = response.content
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

        # Remove any special tokens or formatting
        answer = re.sub(r'<.*?>', '', answer)  # Remove HTML tags
        answer = re.sub(r'\[.*?\]', '', answer)  # Remove markdown links
        answer = re.sub(r'\{.*?\}', '', answer)  # Remove JSON objects
        answer = re.sub(r'\(.*?\)', '', answer)  # Remove parentheses
        answer = re.sub(r'\*\*.*?\*\*', '', answer)  # Remove bold text
        answer = re.sub(r'\*.*?\*', '', answer)  # Remove italic text
        answer = re.sub(r'`.*?`', '', answer)  # Remove code blocks
        
        # Normalize whitespace
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.strip()

        # Validate the LLM response with minimal restrictions
        is_valid, reason = validate_llm_response(answer)
        if not is_valid:
            logger.warning(f"Invalid LLM response: {reason}")
            return f"Error: Could not generate a valid response. {reason}"

        logger.info(f"Final answer: {answer}")
        return answer

    except Exception as e:
        logger.error(f"Error in chat_with_agent: {str(e)}")
        return f"Error: Could not generate a valid response. Please try again with a different question."

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

def get_cached_response(query: str) -> Optional[str]:
    """Get a validated cached response."""
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
    """Cache a validated response."""
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
