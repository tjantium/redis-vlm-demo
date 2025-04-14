from temporalio import workflow
from temporalio.common import RetryPolicy
from dataclasses import dataclass
from typing import Optional, List
from datetime import timedelta, datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGEvent:
    timestamp: datetime
    event_type: str
    details: dict

@dataclass
class RAGWorkflowParams:
    query: str
    context: Optional[str] = None

@workflow.defn
class RAGWorkflow:
    def __init__(self):
        self._query = ""
        self._response = ""
        self._conversation_history = []
        self._status = "initialized"
        self._events: List[RAGEvent] = []
        self._start_time = None
        self._end_time = None
        logger.info("RAGWorkflow initialized")

    def _log_event(self, event_type: str, details: dict = None):
        event = RAGEvent(
            timestamp=workflow.now(),
            event_type=event_type,
            details=details or {}
        )
        self._events.append(event)
        logger.info(f"RAG Event: {event_type} - {details}")

    @workflow.run
    async def run(self, params: RAGWorkflowParams) -> str:
        self._start_time = workflow.now()
        self._status = "processing"
        self._query = params.query
        self._log_event("workflow_started", {"query": params.query})
        
        try:
            # Log query processing start
            self._log_event("query_processing_started")
            
            # Execute the RAG pipeline with proper timeout configuration
            self._response = await workflow.execute_activity(
                "process_rag_query",
                params,
                start_to_close_timeout=timedelta(seconds=60),
                task_queue="rag-task-queue",
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=10)
                )
            )
            
            # Log successful response
            self._log_event("response_generated", {"response_length": len(self._response)})
            self._status = "completed"
            
        except Exception as e:
            self._status = "failed"
            self._response = f"Error: {str(e)}"
            self._log_event("error_occurred", {"error": str(e)})
        
        self._end_time = workflow.now()
        duration = (self._end_time - self._start_time).total_seconds()
        self._log_event("workflow_completed", {
            "status": self._status,
            "duration_seconds": duration,
            "response_length": len(self._response)
        })
        
        return self._response

    @workflow.query
    def get_status(self) -> str:
        return self._status

    @workflow.query
    def get_events(self) -> List[RAGEvent]:
        return self._events

    @workflow.query
    def get_metrics(self) -> dict:
        return {
            "status": self._status,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "duration_seconds": (self._end_time - self._start_time).total_seconds() if self._end_time else None,
            "response_length": len(self._response),
            "event_count": len(self._events)
        }

    @workflow.query
    def get_response(self) -> str:
        return self._response

    @workflow.query
    def get_conversation_history(self) -> list:
        return self._conversation_history
