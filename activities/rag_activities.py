from temporalio import activity
from typing import Optional
from dataclasses import dataclass
from agent import build_rag_agent, chat_with_agent
from ingestion import doc_ingestion_pipeline
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGActivityParams:
    query: str
    context: Optional[str] = None

class RAGActivities:
    def __init__(self):
        self.agent = None
        self.vector_store = None
        logger.info("RAGActivities initialized")

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
            
            # Add timeout for the agent response with proper error handling
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        chat_with_agent, 
                        self.agent, 
                        params.query,
                        params.context
                    ),
                    timeout=30.0
                )
                logger.info("Query processed successfully")
                return response
            except asyncio.TimeoutError:
                logger.error("Query processing timed out")
                return "Error: Query processing timed out after 30 seconds"
            except Exception as e:
                logger.error(f"Error during query processing: {str(e)}")
                return f"Error: {str(e)}"
                
        except Exception as e:
            logger.error(f"Unexpected error in process_rag_query: {str(e)}")
            return f"Error: {str(e)}"
