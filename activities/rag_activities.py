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
import time

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
        self.similarity_threshold = 0.7  # Set similarity threshold
        logger.info("RAGActivities initialized")

    async def _check_similarity(self, query: str, context: Optional[str]) -> bool:
        """Check if the retrieved context meets similarity threshold"""
        if not context:
            return False
            
        try:
            # Get similarity score from vector store
            similarity_score = await self.vector_store.similarity_search_with_score(
                query,
                k=1,
                score_threshold=self.similarity_threshold
            )
            
            if not similarity_score or similarity_score[0][1] < self.similarity_threshold:
                logger.info(f"Similarity score {similarity_score[0][1] if similarity_score else 'None'} below threshold {self.similarity_threshold}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking similarity: {str(e)}")
            return False

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
    async def validate_context(self, query: str, context: Optional[str], force_refresh: bool) -> bool:
        """Activity to validate context relevance"""
        start_time = time.time()
        try:
            logger.info(f"Starting context validation for query: {query}")
            
            # Check if context is empty or too short
            if not context or not isinstance(context, str):
                logger.info("Context validation failed: No context provided")
                return False
                
            if len(context.strip()) < 10:
                logger.info(f"Context validation failed: Too short ({len(context.strip())} characters)")
                return False
                
            # Check for meaningful content
            words = context.lower().split()
            if len(words) < 5:
                logger.info(f"Context validation failed: Insufficient words ({len(words)} words)")
                return False
                
            duration = time.time() - start_time
            logger.info(f"Context validation passed in {duration:.2f}s with {len(words)} words")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Context validation error after {duration:.2f}s: {str(e)}")
            return False

    @activity.defn
    async def check_similarity(self, query: str, context: Optional[str], force_refresh: bool) -> bool:
        """Activity to check context similarity"""
        start_time = time.time()
        try:
            logger.info(f"Starting similarity check for query: {query}")
            
            if not self.vector_store:
                logger.info("Initializing vector store for similarity check...")
                self.vector_store = doc_ingestion_pipeline.vector_store
                
            # Get similarity score from vector store
            similarity_score = await self.vector_store.similarity_search_with_score(
                query,
                k=1,
                score_threshold=self.similarity_threshold
            )
            
            duration = time.time() - start_time
            
            if not similarity_score or similarity_score[0][1] < self.similarity_threshold:
                score = similarity_score[0][1] if similarity_score else None
                logger.info(f"Similarity check failed in {duration:.2f}s: Score {score} below threshold {self.similarity_threshold}")
                return False
                
            logger.info(f"Similarity check passed in {duration:.2f}s with score: {similarity_score[0][1]}")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Similarity check error after {duration:.2f}s: {str(e)}")
            return False

    @activity.defn
    async def process_rag_query(self, query: str, context: Optional[str], force_refresh: bool) -> str:
        start_time = time.time()
        try:
            logger.info(f"Starting query processing for: {query}")
            if context:
                logger.info(f"With context length: {len(context)} characters")
            
            # Initialize agent if needed
            if not self.agent:
                logger.info("Initializing agent for query processing...")
                self.agent = build_rag_agent(doc_ingestion_pipeline.vector_store)
                self.vector_store = doc_ingestion_pipeline.vector_store
                logger.info("Agent initialized successfully")
            
            try:
                # Try cache first
                if cached_response := await self._get_cached_response(query, force_refresh):
                    duration = time.time() - start_time
                    logger.info(f"Query processed from cache in {duration:.2f}s")
                    return cached_response
                
                # If no cache hit or force refresh, query the agent
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        chat_with_agent,
                        self.agent,
                        query,
                        context
                    ),
                    timeout=45.0
                )
                
                # Store valid response in cache
                await self._store_in_cache(query, response)
                
                duration = time.time() - start_time
                logger.info(f"Query processed successfully in {duration:.2f}s, response length: {len(response)}")
                return response
                
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                logger.error(f"Query processing timed out after {duration:.2f}s")
                return "Error: Query processing timed out. Please try again."
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Error during query processing after {duration:.2f}s: {str(e)}")
                return f"Error: {str(e)}"
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Unexpected error in process_rag_query after {duration:.2f}s: {str(e)}")
            return f"Error: {str(e)}"
