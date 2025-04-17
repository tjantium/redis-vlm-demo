from temporalio import activity
from typing import Optional, Dict, Any
from dataclasses import dataclass
from agent import build_rag_agent, chat_with_agent, get_cached_response, cache_response, clear_cache
from ingestion import doc_ingestion_pipeline
import asyncio
import logging
import time
import re
import hashlib
import json
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "QUERY_TIMEOUT": 90,  # 90 seconds timeout
    "MAX_RETRIES": 3,
    "INITIAL_RETRY_DELAY": 1,
    "MAX_RETRY_DELAY": 10,
    "PERFORMANCE_THRESHOLD": 60,  # Warning threshold in seconds
    "MIN_RESPONSE_LENGTH": 50,    # Minimum length for a valid response
    "MAX_REPEATED_WORDS": 3,      # Maximum number of times a word can be repeated
    "MAX_REPEATED_PHRASES": 2,    # Maximum number of times a phrase can be repeated
    "CACHE_TTL": 300,             # Cache TTL in seconds
    "MAX_CACHE_RETRIES": 2,       # Maximum number of cache retries
    "DEBUG_MODE": True,           # Enable debug mode
    "DEBUG_LOG_FILE": "cache_debug.log"  # Debug log file
}

class CacheDebugger:
    def __init__(self):
        self.operations: list[Dict[str, Any]] = []
        self.enabled = CONFIG["DEBUG_MODE"]
        self.log_file = CONFIG["DEBUG_LOG_FILE"]
    
    def log_operation(self, operation: str, query: str, cache_key: str, result: Any, error: Optional[str] = None):
        """Log a cache operation with details."""
        if not self.enabled:
            return
            
        operation_data = {
            "timestamp": time.time(),
            "operation": operation,
            "query": query,
            "cache_key": cache_key,
            "result": result,
            "error": error
        }
        
        self.operations.append(operation_data)
        
        # Write to log file
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(operation_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to debug log: {str(e)}")
    
    def get_operations(self) -> list[Dict[str, Any]]:
        """Get all logged operations."""
        return self.operations
    
    def clear_operations(self):
        """Clear all logged operations."""
        self.operations = []
        try:
            with open(self.log_file, "w") as f:
                f.write("")
        except Exception as e:
            logger.error(f"Failed to clear debug log: {str(e)}")

# Initialize debugger
cache_debugger = CacheDebugger()

def debug_cache_operation(operation: str, query: str, cache_key: str, result: Any, error: Optional[str] = None):
    """Debug a cache operation."""
    cache_debugger.log_operation(operation, query, cache_key, result, error)
    logger.debug(f"Cache {operation}: query='{query}', key='{cache_key}', result={result}, error={error}")

def generate_cache_key(query: str) -> str:
    """Generate a consistent cache key for the query."""
    # Normalize the query
    normalized_query = query.lower().strip()
    # Create a hash of the normalized query
    return hashlib.md5(normalized_query.encode()).hexdigest()

def validate_response(response: str) -> str:
    """Validate and clean the response."""
    if not response or not isinstance(response, str):
        return "Error: Invalid response received"
    
    # Remove any conversation transcripts
    response = re.sub(r'\*\*User:.*?\*\*', '', response, flags=re.DOTALL)
    response = re.sub(r'\*\*Assistant:.*?\*\*', '', response, flags=re.DOTALL)
    
    # Extract actual answer if response contains thought process
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    elif "Final Answer:" in response:
        response = response.split("Final Answer:")[-1].strip()
    
    # Basic cleanup
    response = response.strip()
    
    # Remove any special tokens
    response = re.sub(r'<.*?>', '', response)
    response = re.sub(r'\[.*?\]', '', response)
    response = re.sub(r'\{.*?\}', '', response)
    
    # Normalize whitespace
    response = re.sub(r'\s+', ' ', response)
    response = response.strip()
    
    # Check for corrupted responses
    if len(response) < 10:
        return "Error: Response too short"
    
    if response.count(' ') < 3:
        return "Error: Response too simple"
    
    # Check for repetitive content
    words = response.lower().split()
    if len(words) > 0:
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Only check words longer than 2 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Check for excessive repetition
        total_words = len(words)
        unique_words = len(set(words))
        
        # If more than 50% of the words are the same word
        for word, freq in word_freq.items():
            if freq > total_words * 0.2 and word not in {'the', 'and', 'for', 'with', 'this', 'that'}:
                return "Error: Response contains excessive repetition"
        
        # If there's too little variety in words
        if unique_words < total_words * 0.2:  # Less than 20% unique words
            return "Error: Response lacks variety"
        
        # Check for sequential repetition
        prev_word = None
        repeat_count = 1
        for word in words:
            if word == prev_word:
                repeat_count += 1
                if repeat_count > 3:  # More than 3 same words in sequence
                    return "Error: Response contains sequential repetition"
            else:
                repeat_count = 1
            prev_word = word
    
    # If we get here, the response is valid
    return response

def clean_response(response: str) -> str:
    """Clean the response by removing corrupted patterns and normalizing text."""
    # Remove HTML tags
    response = re.sub(r'<.*?>', '', response)
    
    # Remove markdown links
    response = re.sub(r'\[.*?\]', '', response)
    
    # Remove JSON objects
    response = re.sub(r'\{.*?\}', '', response)
    
    # Remove parentheses
    response = re.sub(r'\(.*?\)', '', response)
    
    # Remove corrupted patterns
    corrupted_patterns = [
        r'[a-z]{2,}ns\b',    # Words ending with 'ns'
        r'[a-z]{2,}rs\b',    # Words ending with 'rs'
        r'[a-z]{2,}sns\b',   # Words ending with 'sns'
        r'[a-z]{2,}ories\b', # Words ending with 'ories'
        r'[a-z]{2,}ments\b', # Words ending with 'ments'
        r'[a-z]{2,}nsns\b',  # Words ending with 'nsns'
        r'[a-z]{2,}rsns\b',  # Words ending with 'rsns'
        r'[a-z]{2,}srs\b',   # Words ending with 'srs'
        r'[a-z]{2,}nsrs\b',  # Words ending with 'nsrs'
        r'[a-z]{2,}rsrs\b',  # Words ending with 'rsrs'
    ]
    for pattern in corrupted_patterns:
        response = re.sub(pattern, '', response)
    
    # Normalize whitespace
    response = re.sub(r'\s+', ' ', response)
    response = response.strip()
    
    return response

@retry(
    stop=stop_after_attempt(CONFIG["MAX_RETRIES"]),
    wait=wait_exponential(
        multiplier=CONFIG["INITIAL_RETRY_DELAY"],
        min=CONFIG["INITIAL_RETRY_DELAY"],
        max=CONFIG["MAX_RETRY_DELAY"]
    ),
    retry_error_cls=(asyncio.TimeoutError,)
)
async def execute_rag_query(agent, query: str, context: Optional[str] = None) -> str:
    """Execute RAG query with retry logic and performance monitoring."""
    start_time = time.time()
    try:
        logger.info(f"Executing RAG query: {query}")
        if context:
            logger.info(f"With context: {context}")
            
        response = await asyncio.wait_for(
            asyncio.to_thread(
                chat_with_agent,
                agent,
                query,
                context
            ),
            timeout=CONFIG["QUERY_TIMEOUT"]
        )
        
        # Clean the response first
        response = clean_response(response)
        
        # Validate response
        validation_message = validate_response(response)
        if validation_message != "Response is valid":
            logger.error(f"Invalid response: {validation_message}")
            raise ValueError(f"Invalid response: {validation_message}")
        
        duration = time.time() - start_time
        logger.info(f"Query executed successfully. Duration: {duration:.2f}s, Response length: {len(response)}")
        
        # Log performance warning if query takes too long
        if duration > CONFIG["PERFORMANCE_THRESHOLD"]:
            logger.warning(f"Query execution took {duration:.2f}s, exceeding performance threshold")
            
        return response
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        logger.error(f"Query execution timed out after {duration:.2f}s")
        raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error executing query after {duration:.2f}s: {str(e)}")
        raise

@activity.defn
async def debug_cache(query: str) -> Dict[str, Any]:
    """Debug endpoint to inspect cache operations."""
    cache_key = generate_cache_key(query)
    debug_info = {
        "query": query,
        "cache_key": cache_key,
        "operations": cache_debugger.get_operations(),
        "cached_response": None,
        "is_valid": False,
        "validation_message": None
    }
    
    try:
        # Try to get cached response
        cached_response = get_cached_response(cache_key)
        if cached_response:
            debug_info["cached_response"] = cached_response
            # Validate cached response
            validation_message = validate_response(cached_response)
            debug_info["is_valid"] = validation_message == "Response is valid"
            debug_info["validation_message"] = validation_message
    except Exception as e:
        debug_info["error"] = str(e)
    
    return debug_info

@activity.defn
async def clear_cache_debug() -> Dict[str, Any]:
    """Clear cache and debug logs."""
    try:
        clear_cache()
        cache_debugger.clear_operations()
        return {"status": "success", "message": "Cache and debug logs cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@activity.defn
async def process_rag_query(query: str, force_refresh: bool = False) -> str:
    """Process a RAG query with improved caching, validation, and performance monitoring."""
    start_time = time.time()
    cache_key = generate_cache_key(query)
    cache_retries = 0
    
    try:
        logger.info(f"Processing RAG query: {query}")
        logger.info(f"Cache key: {cache_key}")
        logger.info(f"Force refresh: {force_refresh}")
        
        # Initialize agent
        agent = build_rag_agent(doc_ingestion_pipeline.vector_store)
        
        # Clear cache if force refresh is requested
        if force_refresh:
            try:
                clear_cache()
                debug_cache_operation("clear", query, cache_key, "success")
                logger.info("Cache cleared successfully")
            except Exception as e:
                debug_cache_operation("clear", query, cache_key, "failed", str(e))
                logger.warning(f"Failed to clear cache: {str(e)}")
                # Continue processing even if cache clearing fails
        
        # Try to get cached response
        if not force_refresh:
            while cache_retries < CONFIG["MAX_CACHE_RETRIES"]:
                try:
                    cached_response = get_cached_response(cache_key)
                    debug_cache_operation("get", query, cache_key, cached_response is not None)
                    
                    if cached_response:
                        # Validate cached response
                        validated_response = validate_response(cached_response)
                        if not validated_response.startswith("Error:"):
                            duration = time.time() - start_time
                            logger.info(f"Using valid cached response. Total duration: {duration:.2f}s")
                            return validated_response
                        else:
                            logger.warning(f"Invalid cached response: {validated_response}")
                            # Clear cache for this key
                            try:
                                clear_cache()
                                debug_cache_operation("clear_invalid", query, cache_key, "success")
                                logger.info("Cache cleared due to invalid response")
                            except Exception as e:
                                debug_cache_operation("clear_invalid", query, cache_key, "failed", str(e))
                                logger.warning(f"Failed to clear cache: {str(e)}")
                            cache_retries += 1
                            continue
                    break
                except Exception as e:
                    debug_cache_operation("get", query, cache_key, "failed", str(e))
                    logger.warning(f"Cache retrieval error: {str(e)}")
                    cache_retries += 1
                    if cache_retries >= CONFIG["MAX_CACHE_RETRIES"]:
                        logger.error("Max cache retries reached")
                        break
        
        # Process query
        response = await execute_rag_query(agent, query)
        
        # Validate and clean the response
        validated_response = validate_response(response)
        if validated_response.startswith("Error:"):
            logger.error(f"Invalid response: {validated_response}")
            return validated_response
        
        # Cache the valid response
        try:
            cache_response(cache_key, validated_response, ttl=CONFIG["CACHE_TTL"])
            debug_cache_operation("store", query, cache_key, "success")
            logger.info("Response cached successfully")
        except Exception as e:
            debug_cache_operation("store", query, cache_key, "failed", str(e))
            logger.warning(f"Failed to cache response: {str(e)}")
        
        duration = time.time() - start_time
        logger.info(f"Query processing completed. Total duration: {duration:.2f}s")
        
        return validated_response

    except Exception as e:
        duration = time.time() - start_time
        debug_cache_operation("process", query, cache_key, "failed", str(e))
        logger.error(f"Error processing RAG query after {duration:.2f}s: {str(e)}")
        return f"Error: {str(e)}" 