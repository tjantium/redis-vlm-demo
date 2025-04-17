from temporalio import workflow
from temporalio.common import RetryPolicy
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "details": self.details
        }

@dataclass
class RAGWorkflowParams:
    query: str
    context: Optional[str] = None
    force_refresh: bool = False

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
        self._attempt = 0
        self._max_attempts = 3
        logger.info("RAGWorkflow initialized")

    def _log_event(self, event_type: str, details: dict = None):
        event = RAGEvent(
            timestamp=workflow.now(),
            event_type=event_type,
            details=details or {}
        )
        self._events.append(event)
        logger.info(f"RAG Event: {event_type} - {details}")

    def _validate_response(self, response: str) -> bool:
        """Validate the response quality"""
        if not response or not isinstance(response, str):
            return False
        
        # Check for repetitive content
        words = response.lower().split()
        if len(words) < 5:
            return False
            
        # Check for excessive repetition
        word_freq = {}
        for word in words:
            if len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        for word, freq in word_freq.items():
            if freq > len(words) * 0.3 and word not in {'the', 'and', 'for', 'with'}:
                return False
        
        return True

    @workflow.run
    async def run(self, params: RAGWorkflowParams) -> str:
        self._start_time = workflow.now()
        self._status = "processing"
        self._query = params.query
        self._log_event("workflow_started", {
            "query": params.query,
            "force_refresh": params.force_refresh
        })
        
        while self._attempt < self._max_attempts:
            try:
                self._attempt += 1
                self._log_event("query_processing_started", {"attempt": self._attempt})
                
                # Execute the RAG activity
                response = await workflow.execute_activity(
                    "process_rag_query",
                    params,
                    start_to_close_timeout=timedelta(seconds=60),
                    task_queue="rag-task-queue",
                    retry_policy=RetryPolicy(
                        maximum_attempts=1,  # Don't retry at activity level
                        initial_interval=timedelta(seconds=1)
                    )
                )
                
                # Validate response
                if self._validate_response(response):
                    self._response = response
                    self._log_event("response_generated", {
                        "response_length": len(response),
                        "attempt": self._attempt
                    })
                    self._status = "completed"
                    break
                else:
                    self._log_event("invalid_response", {
                        "attempt": self._attempt,
                        "response_length": len(response) if response else 0
                    })
                    # Force cache refresh on next attempt
                    params.force_refresh = True
                    
            except Exception as e:
                self._log_event("attempt_failed", {
                    "attempt": self._attempt,
                    "error": str(e)
                })
                if self._attempt >= self._max_attempts:
                    self._status = "failed"
                    self._response = f"Error: Failed to generate valid response after {self._max_attempts} attempts"
                    self._log_event("error_occurred", {"error": str(e)})
                    break
        
        self._end_time = workflow.now()
        duration = (self._end_time - self._start_time).total_seconds()
        self._log_event("workflow_completed", {
            "status": self._status,
            "duration_seconds": duration,
            "response_length": len(self._response),
            "attempts": self._attempt
        })
        
        return self._response

    @workflow.query
    def get_status(self) -> str:
        return self._status

    @workflow.query
    def get_events(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self._events]

    @workflow.query
    def get_metrics(self) -> Dict[str, Any]:
        metrics = {
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
            "duration": str(self._end_time - self._start_time) if self._start_time and self._end_time else None,
            "event_count": len(self._events),
            "status": self._status
        }
        return metrics

    @workflow.query
    def get_response(self) -> str:
        return self._response

    @workflow.query
    def get_conversation_history(self) -> list:
        return self._conversation_history
