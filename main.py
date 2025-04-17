from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from temporalio.client import Client
from workflows.rag_workflow import RAGWorkflow, RAGWorkflowParams
from shared.config import get_temporal_client, TEMPORAL_TASK_QUEUE
from pydantic import BaseModel
import asyncio
import uuid
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

temporal_client: Client = None

# Update request model to include optional context
class QueryRequest(BaseModel):
    query: str
    context: str | None = None  # Make context optional with None as default
    force_refresh: bool = False  # Add force_refresh field with default value

@app.on_event("startup")
async def startup_event():
    global temporal_client
    temporal_client = await get_temporal_client()

@app.get("/query")
async def get_query_info():
    """Handle GET requests to /query with helpful information"""
    return JSONResponse(
        status_code=405,
        content={
            "error": "Method not allowed",
            "message": "This endpoint only accepts POST requests. To query:",
            "usage": {
                "method": "POST",
                "endpoint": "/query",
                "body": {
                    "query": "Your question here",
                    "context": "Optional context"
                }
            },
            "example": {
                "curl": 'curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d \'{"query": "What features does the Chevy Colorado 2022 have?"}\'',
            }
        }
    )

@app.post("/query")
async def process_query(request: QueryRequest):
    # Generate a unique workflow ID using timestamp and UUID
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    workflow_id = f"rag-{timestamp}-{unique_id}"
    
    try:
        # Start the workflow with context and force_refresh option
        handle = await temporal_client.start_workflow(
            RAGWorkflow.run,
            RAGWorkflowParams(
                query=request.query,
                context=request.context,
                force_refresh=request.force_refresh  # Pass force_refresh to workflow
            ),
            id=workflow_id,
            task_queue=TEMPORAL_TASK_QUEUE
        )
        logger.info(f"Started workflow: {workflow_id}")
        return {
            "workflow_id": workflow_id, 
            "status": "started",
            "context_provided": request.context is not None,
            "cache_refresh": request.force_refresh
        }
    except Exception as e:
        logger.error(f"Error starting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/{workflow_id}")
async def get_query_result(workflow_id: str):
    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        
        try:
            # First check if workflow exists and is running
            status = await handle.query(RAGWorkflow.get_status)
            
            # If status is not completed, return current status
            if status != "completed":
                return {
                    "workflow_id": workflow_id,
                    "status": status,
                    "message": f"Workflow is {status}"
                }
            
            # If completed, get full details
            result = await handle.query(RAGWorkflow.get_response)
            events = await handle.query(RAGWorkflow.get_events)
            metrics = await handle.query(RAGWorkflow.get_metrics)
            
            logger.info(f"Retrieved workflow {workflow_id} with status: {status}")
            return {
                "workflow_id": workflow_id,
                "status": status,
                "result": result,
                "metrics": metrics,
                "events": events
            }
        except Exception as workflow_error:
            error_msg = str(workflow_error).lower()
            if "workflow not found" in error_msg:
                logger.info(f"Workflow {workflow_id} not found yet")
                return {
                    "workflow_id": workflow_id,
                    "status": "initializing",
                    "message": "Workflow is being initialized"
                }
            elif "workflow execution not found" in error_msg:
                logger.warning(f"Workflow {workflow_id} execution not found")
                return {
                    "workflow_id": workflow_id,
                    "status": "not_found",
                    "message": "Workflow execution not found"
                }
            else:
                logger.warning(f"Error querying workflow {workflow_id}: {error_msg}")
                return {
                    "workflow_id": workflow_id,
                    "status": "error",
                    "message": f"Error querying workflow: {error_msg}"
                }
            
    except Exception as e:
        logger.error(f"Error accessing workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/history/{workflow_id}")
async def get_history(workflow_id: str):
    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        history = await handle.query(RAGWorkflow.get_conversation_history)
        return {"workflow_id": workflow_id, "history": history}
    except Exception as e:
        error_msg = str(e).lower()
        if "workflow not found" in error_msg:
            return {
                "workflow_id": workflow_id,
                "status": "not_found",
                "message": "Workflow history not found"
            }
        raise HTTPException(status_code=404, detail=str(e))
