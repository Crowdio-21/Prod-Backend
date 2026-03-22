from common.protocol import create_job_results_message
from .aggregation_handler import AggregationHandler


class JobCompletionHandler:
    """Handles job completion logic and result delivery"""

    def __init__(self, connection_manager, job_manager):
        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.aggregation_handler = AggregationHandler()

    async def handle_job_completion(self, job_id: str):
        """Handle completion of a job and send results to client"""
        print(f"🏁 [COMPLETION DEBUG] Starting job completion for {job_id}")

        # Get job info
        job = await self.job_manager.get_job_info(job_id)
        if not job:
            print(f"⚠️ [COMPLETION DEBUG] Job {job_id} not found in database")
            return

        # Get all tasks
        tasks = await self.job_manager.get_job_tasks_info(job_id)

        print(f"🏁 [COMPLETION DEBUG] Job {job_id} has {len(tasks)} tasks")

        # Log task statuses for debugging
        task_statuses = {}
        for task in tasks:
            status = getattr(task, "status", "unknown")
            task_statuses[status] = task_statuses.get(status, 0) + 1
        print(f"🏁 [COMPLETION DEBUG] Task statuses for job {job_id}: {task_statuses}")

        # Get ordered results
        results = await self.job_manager.get_job_results(job_id)

        if results is None:
            print(f"⚠️ [COMPLETION DEBUG] Could not retrieve results for job {job_id}")
            return

        # Count how many results are None vs actual values
        none_count = sum(1 for r in results if r is None)
        print(
            f"🏁 [COMPLETION DEBUG] Results for job {job_id}: {len(results)} total, {none_count} are None"
        )

        # Aggregate DNN outputs when strategy is configured.
        payload_results = results
        if getattr(job, "is_dnn_inference", False):
            strategy = getattr(job, "aggregation_strategy", None) or "average"
            aggregated = self.aggregation_handler.aggregate(results, strategy=strategy)
            payload_results = {
                "raw_results": results,
                "aggregated": aggregated,
                "aggregation_strategy": strategy,
            }

        # Send results to client
        client_websocket = self.connection_manager.get_client_websocket(job_id)

        if not client_websocket:
            print(f"⚠️ [COMPLETION DEBUG] No client websocket found for job {job_id}")
            return

        # Create and send results message
        message = create_job_results_message(payload_results, job_id)
        await client_websocket.send(message.to_json())

        # Finalize job (completed_tasks already tracked via increment_job_completed_tasks)
        await self.job_manager.finalize_job(job_id)

        print(
            f"✅ [COMPLETION DEBUG] Job {job_id} completed and results sent to client"
        )
