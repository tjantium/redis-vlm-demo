import asyncio
from temporalio.worker import Worker
from temporalio.client import Client
from workflows.rag_workflow import RAGWorkflow
from activities.rag_activities import RAGActivities
from shared.config import get_temporal_client, TEMPORAL_TASK_QUEUE
from agent import build_rag_agent, chat_with_agent

async def main():
    # Create the client
    client = await get_temporal_client()

    # Create RAGActivities instance
    rag_activities = RAGActivities()

    # Run the worker with explicitly registered activities
    worker = Worker(
        client,
        task_queue=TEMPORAL_TASK_QUEUE,
        workflows=[RAGWorkflow],
        activities=[
            rag_activities.validate_context,
            rag_activities.check_similarity,
            rag_activities.process_rag_query
        ],
    )

    print(f"Starting worker, connecting to task queue: {TEMPORAL_TASK_QUEUE}")
    print("Registered activities:")
    print("- validate_context")
    print("- check_similarity")
    print("- process_rag_query")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
