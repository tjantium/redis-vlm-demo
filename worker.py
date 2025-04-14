import asyncio
from temporalio.worker import Worker
from temporalio.client import Client
from workflows.rag_workflow import RAGWorkflow
from activities.rag_activities import RAGActivities
from shared.config import get_temporal_client, TEMPORAL_TASK_QUEUE

async def main():
    # Create the client
    client = await get_temporal_client()

    # Run the worker
    worker = Worker(
        client,
        task_queue=TEMPORAL_TASK_QUEUE,
        workflows=[RAGWorkflow],
        activities=[RAGActivities().process_rag_query],
    )

    print(f"Starting worker, connecting to task queue: {TEMPORAL_TASK_QUEUE}")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
