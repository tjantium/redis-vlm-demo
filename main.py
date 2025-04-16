from fastapi import FastAPI, BackgroundTasks, HTTPException
from temporalio.client import Client
from workflows.rag_workflow import RAGWorkflow, RAGWorkflowParams
from shared.config import get_temporal_client, TEMPORAL_TASK_QUEUE
from pydantic import BaseModel
import asyncio
import uuid
import time

app = FastAPI()
temporal_client: Client = None

# Update request model to include optional context
class QueryRequest(BaseModel):
    query: str
    context: str | None = None  # Make context optional with None as default

@app.on_event("startup")
async def startup_event():
    global temporal_client
    temporal_client = await get_temporal_client()

@app.post("/query")
async def process_query(request: QueryRequest):
    # Generate a unique workflow ID using timestamp and UUID
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    workflow_id = f"rag-{timestamp}-{unique_id}"
    
    try:
        # Start the workflow with context
        handle = await temporal_client.start_workflow(
            RAGWorkflow.run,
            RAGWorkflowParams(
                query=request.query,
                context=request.context  # Pass the context to the workflow
            ),
            id=workflow_id,
            task_queue=TEMPORAL_TASK_QUEUE
        )
        return {
            "workflow_id": workflow_id, 
            "status": "started",
            "context_provided": request.context is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/{workflow_id}")
async def get_query_result(workflow_id: str):
    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        result = await handle.query(RAGWorkflow.get_response)
        status = await handle.query(RAGWorkflow.get_status)
        events = await handle.query(RAGWorkflow.get_events)
        metrics = await handle.query(RAGWorkflow.get_metrics)
        return {
            "workflow_id": workflow_id,
            "status": status,
            "result": result,
            "metrics": metrics,
            "events": events
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/history/{workflow_id}")
async def get_history(workflow_id: str):
    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        history = await handle.query(RAGWorkflow.get_conversation_history)
        return {"workflow_id": workflow_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
