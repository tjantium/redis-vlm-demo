from temporalio import workflow
from temporalio.common import RetryPolicy
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import timedelta, datetime
import logging
import asyncio
from shared.config import TEMPORAL_TASK_QUEUE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGEvent:
    timestamp: datetime
    event_type: str
    details: dict
    stage: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "details": self.details,
            "stage": self.stage,
            "status": self.status
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
        self._stage = "initialized"
        self._events: List[RAGEvent] = []
        self._start_time = None
        self._end_time = None
        self._attempt = 0
        self._max_attempts = 3
        self._is_processing = False
        self._min_response_length = 100
        self._stage_metrics = {}
        logger.info("RAGWorkflow initialized")

    def _log_event(self, event_type: str, details: dict = None, stage: str = None, status: str = None):
        event = RAGEvent(
            timestamp=workflow.now(),
            event_type=event_type,
            details=details or {},
            stage=stage or self._stage,
            status=status or self._status
        )
        self._events.append(event)
        logger.info(f"RAG Event: {event_type} - Stage: {stage or self._stage} - Status: {status or self._status} - Details: {details}")

    def _update_stage_metrics(self, stage: str, start_time: datetime, end_time: datetime):
        """Update metrics for a specific stage"""
        duration = (end_time - start_time).total_seconds()
        self._stage_metrics[stage] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration
        }

    def _validate_response(self, response: str) -> bool:
        """Validate the response quality"""
        if not response or not isinstance(response, str):
            return False
        
        # Check minimum length
        if len(response.strip()) < self._min_response_length:
            logger.warning(f"Response too short: {len(response.strip())} characters")
            return False
            
        # Check for repetitive content
        words = response.lower().split()
        if len(words) < 10:  # Minimum word count
            logger.warning(f"Response has too few words: {len(words)}")
            return False
            
        # Check for excessive repetition
        word_freq = {}
        for word in words:
            if len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        for word, freq in word_freq.items():
            if freq > len(words) * 0.3 and word not in {'the', 'and', 'for', 'with'}:
                logger.warning(f"Excessive word repetition: {word} appears {freq} times")
                return False
        
        return True

    def _check_context_relevance(self, context: Optional[str]) -> bool:
        """Check if context is empty or irrelevant"""
        if not context or not isinstance(context, str):
            self._log_event("empty_context", {"message": "No context provided"})
            return False
            
        # Check if context is too short
        if len(context.strip()) < 10:
            self._log_event("short_context", {"length": len(context.strip())})
            return False
            
        # Check if context has meaningful content
        words = context.lower().split()
        if len(words) < 5:  # Minimum word count for meaningful context
            self._log_event("insufficient_context", {"word_count": len(words)})
            return False
            
        return True

    @workflow.run
    async def run(self, params: RAGWorkflowParams) -> str:
        if self._is_processing:
            logger.warning("Workflow already processing, skipping duplicate request")
            return self._response
            
        try:
            self._is_processing = True
            self._start_time = workflow.now()
            self._status = "processing"
            self._query = params.query
            self._log_event("workflow_started", {
                "query": params.query,
                "force_refresh": params.force_refresh
            })
            
            # Step 1: Context Validation
            self._stage = "context_validation"
            self._status = "validating"
            context_start_time = workflow.now()
            self._log_event("context_validation_started", {"query": params.query})
            
            context_valid = await workflow.execute_activity(
                "validate_context",
                args=[params.query, params.context, params.force_refresh],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue=TEMPORAL_TASK_QUEUE
            )
            
            context_end_time = workflow.now()
            self._update_stage_metrics("context_validation", context_start_time, context_end_time)
            
            if not context_valid:
                self._status = "failed"
                self._response = "Sorry, there are no documents in the database relevant to your question."
                self._log_event("context_validation_failed", {
                    "message": "No relevant context found",
                    "duration": self._stage_metrics["context_validation"]["duration_seconds"]
                })
                return self._response
                
            self._status = "completed"
            self._log_event("context_validation_passed", {
                "message": "Context validation successful",
                "duration": self._stage_metrics["context_validation"]["duration_seconds"]
            })
            
            # Step 2: Similarity Check
            self._stage = "similarity_check"
            self._status = "checking"
            similarity_start_time = workflow.now()
            self._log_event("similarity_check_started", {"query": params.query})
            
            similarity_valid = await workflow.execute_activity(
                "check_similarity",
                args=[params.query, params.context, params.force_refresh],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue=TEMPORAL_TASK_QUEUE
            )
            
            similarity_end_time = workflow.now()
            self._update_stage_metrics("similarity_check", similarity_start_time, similarity_end_time)
            
            if not similarity_valid:
                self._status = "failed"
                self._response = "Sorry, there are no documents in the database relevant to your question."
                self._log_event("similarity_check_failed", {
                    "message": "Similarity below threshold",
                    "duration": self._stage_metrics["similarity_check"]["duration_seconds"]
                })
                return self._response
                
            self._status = "completed"
            self._log_event("similarity_check_passed", {
                "message": "Similarity check successful",
                "duration": self._stage_metrics["similarity_check"]["duration_seconds"]
            })
            
            # Step 3: Process Query
            self._stage = "query_processing"
            self._status = "processing"
            query_start_time = workflow.now()
            self._log_event("query_processing_started", {"query": params.query})
            
            while self._attempt < self._max_attempts:
                try:
                    self._attempt += 1
                    self._log_event("query_processing_attempt", {
                        "attempt": self._attempt,
                        "max_attempts": self._max_attempts
                    })
                    
                    response = await workflow.execute_activity(
                        "process_rag_query",
                        params,
                        start_to_close_timeout=timedelta(seconds=90),
                        task_queue=TEMPORAL_TASK_QUEUE,
                        retry_policy=RetryPolicy(
                            maximum_attempts=1,
                            initial_interval=timedelta(seconds=1)
                        )
                    )
                    
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
                        params.force_refresh = True
                        await asyncio.sleep(1)
                        
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
            
            query_end_time = workflow.now()
            self._update_stage_metrics("query_processing", query_start_time, query_end_time)
            
            self._end_time = workflow.now()
            duration = (self._end_time - self._start_time).total_seconds()
            self._log_event("workflow_completed", {
                "status": self._status,
                "duration_seconds": duration,
                "response_length": len(self._response),
                "attempts": self._attempt,
                "stage_metrics": self._stage_metrics
            })
            
            return self._response
            
        finally:
            self._is_processing = False

    @workflow.query
    def get_status(self) -> str:
        return self._status

    @workflow.query
    def get_stage(self) -> str:
        return self._stage

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
            "status": self._status,
            "stage": self._stage,
            "stage_metrics": self._stage_metrics,
            "attempts": self._attempt
        }
        return metrics

    @workflow.query
    def get_response(self) -> str:
        return self._response

    @workflow.query
    def get_conversation_history(self) -> list:
        return self._conversation_history
