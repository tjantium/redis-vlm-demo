from temporalio.client import Client
import os

TEMPORAL_TASK_QUEUE = "rag-task-queue"

async def get_temporal_client() -> Client:
    return await Client.connect("localhost:7233")
