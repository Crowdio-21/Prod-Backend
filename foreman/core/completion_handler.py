from common.protocol import Message, MessageType, create_job_results_message

from .run_log import log_job_run


class JobCompletionHandler:
    """Handles job completion logic and result delivery"""

    def __init__(self, connection_manager, job_manager, task_dispatcher=None):
        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.task_dispatcher = task_dispatcher

    async def handle_job_failure(self, job_id: str, error: str):
        """Give up on a job and tell its client, instead of retrying it forever"""
        reason = f"Job failed after {self.job_manager.max_task_attempts} attempts: {error}"
        print(f"[JOB FAILED] {job_id}: {reason}")
        # Log before the job's cached stage names are cleared
        await log_job_run(
            self.job_manager, self.connection_manager, self.task_dispatcher, job_id, status="failed",
        )
        await self.job_manager.fail_job(job_id, reason)
        client_websocket = self.connection_manager.get_client_websocket(job_id)
        if client_websocket:
            try:
                await client_websocket.send(Message(MessageType.JOB_ERROR, {"error": reason}, job_id).to_json())
            except Exception as exc:  # noqa: BLE001
                print(f"[JOB FAILED] could not notify the client of {job_id}: {exc}")

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

        # Write the run log before finalize_job clears the pipeline's stage names.
        await log_job_run(
            self.job_manager, self.connection_manager, self.task_dispatcher, job_id,
            results_returned=len(results) - none_count, results_missing=none_count, status="completed",
        )

        # Send results to client
        client_websocket = self.connection_manager.get_client_websocket(job_id)

        if not client_websocket:
            print(f"⚠️ [COMPLETION DEBUG] No client websocket found for job {job_id}")
            return

        # Create and send results message
        message = create_job_results_message(results, job_id)
        await client_websocket.send(message.to_json())

        # Finalize job (completed_tasks already tracked via increment_job_completed_tasks)
        await self.job_manager.finalize_job(job_id)

        print(
            f"✅ [COMPLETION DEBUG] Job {job_id} completed and results sent to client"
        )
