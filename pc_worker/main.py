"""
Main entry point for the PC worker application.

This module provides a convenient way to run a worker instance.
"""

import asyncio
import sys
from pc_worker import FastAPIWorker, WorkerConfig


def main():
    """Run the worker with command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="CrowdCompute PC Worker")
    parser.add_argument(
        "--worker-id", type=str, help="Unique worker identifier", default=None
    )
    parser.add_argument(
        "--foreman-url",
        type=str,
        default="ws://localhost:9000",
        help="Foreman WebSocket URL",
    )
    parser.add_argument(
        "--api-host", type=str, default="0.0.0.0", help="API server host"
    )
    parser.add_argument(
        "--api-port", "-p", type=int, default=8001, help="API server port"
    )
    parser.add_argument(
        "--max-concurrent-tasks", type=int, default=1, help="Maximum concurrent tasks"
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Heartbeat interval in seconds",
    )
    parser.add_argument(
        "--no-auto-restart",
        action="store_true",
        help="Disable automatic restart on connection failure",
    )

    args = parser.parse_args()

    # Generate worker ID if not provided
    worker_id = args.worker_id
    if not worker_id:
        import uuid

        worker_id = f"worker-{uuid.uuid4().hex[:8]}"

    # Create worker configuration
    config = WorkerConfig(
        worker_id=worker_id,
        foreman_url=args.foreman_url,
        api_host=args.api_host,
        api_port=args.api_port,
        max_concurrent_tasks=args.max_concurrent_tasks,
        heartbeat_interval=args.heartbeat_interval,
        auto_restart=not args.no_auto_restart,
    )

    # Create and run worker
    worker = FastAPIWorker(config)

    # Run the status API and the Foreman connection together. ``worker.run()``
    # only serves the API and never connects to the Foreman.
    try:
        asyncio.run(worker.run_with_worker())
    except KeyboardInterrupt:
        print("\nWorker stopped by user")


if __name__ == "__main__":
    main()
