from temporalio import activity
from typing import Optional
from dataclasses import dataclass
from agent import build_rag_agent, chat_with_agent, get_cached_response, cache_response, clear_cache
from ingestion import doc_ingestion_pipeline
import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_cls=(asyncio.TimeoutError,)
)
async def execute_rag_query(agent, query: str, context: Optional[str] = None) -> str:
    """Execute RAG query with retry logic and detailed logging."""
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
            timeout=60.0
        )
        
        logger.info(f"Query executed successfully. Response length: {len(response)}")
        return response
    except asyncio.TimeoutError:
        logger.error("Query execution timed out")
        raise
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise

@activity.defn
async def process_rag_query(query: str, force_refresh: bool = False) -> str:
    """Process a RAG query with improved caching, validation, and debugging."""
    try:
        logger.info(f"Processing RAG query: {query}")
        logger.info(f"Force refresh: {force_refresh}")
        
        # Initialize agent
        agent = build_rag_agent(doc_ingestion_pipeline.vector_store)
        
        # Clear cache if force refresh is requested
        if force_refresh:
            clear_cache()
            logger.info("Cache cleared due to force refresh")
        
        # Try to get cached response
        if not force_refresh:
            cached_response = get_cached_response(query)
            if cached_response:
                logger.info("Using cached response")
                return cached_response
            else:
                logger.info("No valid cached response found")
        
        # Process query
        response = await execute_rag_query(agent, query)
        
        # Cache the response if valid
        if cache_response(query, response):
            logger.info("Response cached successfully")
        else:
            logger.warning("Response validation failed, not caching")
        
        return response

    except Exception as e:
        logger.error(f"Error processing RAG query: {str(e)}")
        return f"Error: {str(e)}" 