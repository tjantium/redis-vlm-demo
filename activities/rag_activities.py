from temporalio import activity
from typing import Optional
from dataclasses import dataclass
from agent import build_rag_agent, chat_with_agent
from ingestion import doc_ingestion_pipeline
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
import asyncio
import logging
import re
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize semantic cache
cache = SemanticCache(
    name="chevy_cache",
    prefix="cache",
    distance_threshold=0.2,
    ttl=300,  # 5 minutes
    vectorizer=HFTextVectorizer(model="BAAI/bge-small-en-v1.5")
)

@dataclass
class RAGActivityParams:
    query: str
    context: Optional[str] = None
    force_refresh: bool = False

def refresh_cache():
    """Clear the semantic cache to force fresh responses"""
    try:
        cache.flush()
        logger.info("Successfully cleared semantic cache")
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False

def validate_response(response: str) -> str:
    """Validate and clean the response."""
    if not response or not isinstance(response, str):
        return "Error: Invalid response received"
    
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
    
    # Basic quality checks
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
    
    return response

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_cls=(asyncio.TimeoutError,)
)
async def execute_rag_query(agent, query: str, context: Optional[str] = None) -> str:
    """Execute RAG query with retry logic"""
    return await asyncio.wait_for(
        asyncio.to_thread(
            chat_with_agent,
            agent,
            query,
            context
        ),
        timeout=60.0  # Increased timeout to 60 seconds
    )

class RAGActivities:
    def __init__(self):
        self.agent = None
        self.vector_store = None
        logger.info("RAGActivities initialized")

    async def _get_cached_response(self, query: str, force_refresh: bool = False) -> Optional[str]:
        """Try to get a response from cache"""
        if force_refresh:
            refresh_cache()
            return None
        
        if cached_result := cache.check(prompt=query):
            response = cached_result[0]['response']
            # Validate cached response
            if 'car car car' in response.lower() or len(set(response.split())) < 5:
                logger.warning("Found corrupted cached response, forcing refresh")
                refresh_cache()
                return None
            return response
        return None

    async def _store_in_cache(self, query: str, response: str):
        """Store a valid response in cache"""
        if 'car car car' not in response.lower() and len(set(response.split())) >= 5:
            try:
                cache.store(prompt=query, response=response)
                logger.info("Response stored in cache")
            except Exception as e:
                logger.error(f"Failed to store in cache: {str(e)}")

    @activity.defn
    async def process_rag_query(self, params: RAGActivityParams) -> str:
        try:
            logger.info(f"Processing RAG query: {params.query}")
            if params.context:
                logger.info(f"With context: {params.context}")
            
            # Initialize agent if needed
            if not self.agent:
                logger.info("Initializing agent...")
                self.agent = build_rag_agent(doc_ingestion_pipeline.vector_store)
                logger.info("Agent initialized successfully")
            
            try:
                # Try cache first
                if cached_response := await self._get_cached_response(params.query, params.force_refresh):
                    logger.info("Using cached response")
                    return cached_response
                
                # If no cache hit or force refresh, query the agent
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        chat_with_agent,
                        self.agent,
                        params.query,
                        params.context
                    ),
                    timeout=45.0
                )
                
                # Store valid response in cache
                await self._store_in_cache(params.query, response)
                
                logger.info(f"Raw response: {response}")
                return response
                
            except asyncio.TimeoutError:
                logger.error("Query processing timed out")
                return "Error: Query processing timed out. Please try again."
            except Exception as e:
                logger.error(f"Error during query processing: {str(e)}")
                return f"Error: {str(e)}"
                
        except Exception as e:
            logger.error(f"Unexpected error in process_rag_query: {str(e)}")
            return f"Error: {str(e)}"
