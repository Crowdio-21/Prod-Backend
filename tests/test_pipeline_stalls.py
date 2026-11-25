"""Regression tests for the ways a job used to hang partway."""

import asyncio
import json
import uuid

from foreman.core.completion_handler import JobCompletionHandler
from foreman.core.connection_manager import ConnectionManager
from foreman.core.job_manager import JobManager
from foreman.core.scheduling import create_scheduler
from foreman.core.scheduling.bandwidth_scheduler import BandwidthAwareScheduler
from foreman.core.scheduling.scheduler_interface import Task
from foreman.core.task_dispatcher import TaskDispatcher
from foreman.core.watchdog import TaskWatchdog
from foreman.db.base import db_session, engine, init_db
from foreman.db.models import JobModel, TaskModel

CODE = "def add(x):\n    return x\n"


def run(coro_fn):
    async def wrapper():
        await init_db()
        try:
            await coro_fn()
        finally:
            await engine.dispose()  # pooled connections belong to this event loop

    asyncio.run(wrapper())


class _Socket:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(json.loads(text))


class _Clock:
    now = 0.0

    def __call__(self):
        return self.now


class _Foreman:
    def __init__(self, workers=("w1",), **watchdog):
        self.connections = ConnectionManager()
        self.jobs = JobManager()
        self.dispatcher = TaskDispatcher(create_scheduler("fifo"), self.connections, self.jobs)
        self.completion = JobCompletionHandler(self.connections, self.jobs, self.dispatcher)
        self.clock = _Clock()
        settings = dict(disconnect_grace_s=30, resume_grace_s=60, task_timeout_s=600)
        settings.update(watchdog)
        self.watchdog = TaskWatchdog(self.connections, self.jobs, self.dispatcher, self.completion,
                                     clock=self.clock, **settings)
        for worker_id in workers:
            self.connections.add_worker(worker_id, _Socket())

    async def job(self, n_tasks=1):
        job_id = uuid.uuid4().hex
        await self.jobs.create_job(job_id, CODE, list(range(n_tasks)), n_tasks)
        await self.dispatcher.assign_tasks_for_job(job_id, CODE, [])
        return job_id

    async def tick(self, after=0.0):
        self.clock.now += after
        await self.watchdog.tick()


async def _task(task_id):
    async with db_session() as session:
        task = await session.get(TaskModel, task_id)
        return task.status, task.worker_id


async def _finish_all(foreman, job_id):
    """Take every task of this test off the shared database so later tests do not see it."""
    await foreman.jobs.fail_job(job_id, "test over")


# ---------------------------------------------------------------- fan-in counter
def test_simultaneous_upstream_results_unblock_the_next_stage():
    async def scenario():
        jobs, n, job_id = JobManager(), 12, uuid.uuid4().hex
        await jobs.create_pipeline_job(job_id, [
            {"func_code": CODE, "args_list": list(range(n))},
            {"func_code": CODE, "args_list": [0], "pass_upstream_results": True},
        ])
        async with db_session() as session:
            for i in range(n):
                task = await session.get(TaskModel, f"{job_id}_task_0_{i}")
                task.status, task.worker_id = "assigned", f"w{i}"
            await session.commit()

        async def finish(i):
            task_id = f"{job_id}_task_0_{i}"
            await jobs.mark_task_completed(task_id, job_id, f"w{i}", "1")
            return await jobs.dependency_manager.on_task_completed(task_id, job_id, i)

        unblocked = await asyncio.gather(*[finish(i) for i in range(n)])
        assert sum(len(u) for u in unblocked) == 1
        async with db_session() as session:
            downstream = await session.get(TaskModel, f"{job_id}_task_1_0")
            assert (downstream.status, downstream.dependency_count) == ("pending", 0)
            assert len(json.loads(downstream.args)["upstream_results"]) == n  # none overwritten
            assert (await session.get(JobModel, job_id)).completed_tasks == n
        await jobs.fail_job(job_id, "test over")

    run(scenario)


# ---------------------------------------------------------------- scheduler
def _scheduler():
    clock = _Clock()
    scheduler = BandwidthAwareScheduler(clock=clock, model_size=lambda _: 0)
    scheduler.observe_link_report("fast", {"down_bps": 5e7, "up_bps": 5e7, "rtt_s": 0.01})
    scheduler.observe_link_report("slow", {"down_bps": 2e5, "up_bps": 2e5, "rtt_s": 0.5})
    task = Task(id="t", job_id="j", args=json.dumps(["x" * 500_000]))
    return scheduler, clock, task


def test_scheduler_does_not_wait_for_a_disconnected_worker():
    async def scenario():
        scheduler, clock, task = _scheduler()
        assert await scheduler.batch_select_workers([task], {"fast"}, {"fast": object()})
        clock.now = 5.0
        other = Task(id="u", job_id="j", args=task.args)
        got = await scheduler.batch_select_workers([other], {"slow"}, {"slow": object()})
        assert [w for _, w in got] == ["slow"]

    asyncio.run(scenario())


def test_scheduler_forgets_a_lost_task():
    async def scenario():
        scheduler, clock, task = _scheduler()
        both = {"fast": object(), "slow": object()}
        assert await scheduler.batch_select_workers([task], {"fast"}, both)
        scheduler.observe_task_lost("t", "someone-else")
        assert "t" in scheduler._inflight
        scheduler.observe_task_lost("t", "fast")
        other = Task(id="u", job_id="j", args=task.args)
        got = await scheduler.batch_select_workers([other], {"slow"}, both)
        assert [w for _, w in got] == ["slow"]

    asyncio.run(scenario())


# ---------------------------------------------------------------- watchdog
def test_task_of_a_disconnected_worker_moves_to_another():
    async def scenario():
        foreman = _Foreman(workers=("w1",))
        job_id = await foreman.job()
        task_id = f"{job_id}_task_0"
        foreman.connections.add_worker("w2", _Socket())
        foreman.connections.remove_worker("w1")
        await foreman.tick()
        assert await _task(task_id) == ("assigned", "w1")  # still within the grace period
        await foreman.tick(after=31)
        assert await _task(task_id) == ("assigned", "w2")
        await _finish_all(foreman, job_id)

    run(scenario)


def test_restarted_worker_gets_its_task_again_unless_it_reports():
    async def scenario():
        foreman = _Foreman()
        job_id = await foreman.job()
        task_id = f"{job_id}_task_0"
        socket = foreman.connections.get_worker_websocket("w1")
        foreman.watchdog.expect_resume(task_id, "w1")
        await foreman.tick(after=59)
        assert len(socket.sent) == 1
        await foreman.tick(after=2)
        assert len(socket.sent) == 2  # sent again
        foreman.watchdog.expect_resume(task_id, "w1")
        foreman.watchdog.touch(task_id)  # progress arrived: it is still running it
        await foreman.tick(after=120)
        assert len(socket.sent) == 2
        await _finish_all(foreman, job_id)

    run(scenario)


def test_task_that_keeps_timing_out_fails_the_job():
    async def scenario():
        foreman = _Foreman(task_timeout_s=10)
        foreman.jobs.max_task_attempts = 2
        client = _Socket()
        job_id = await foreman.job()
        foreman.connections.add_client(job_id, client)
        await foreman.tick()
        await foreman.tick(after=11)  # first timeout: re-sent
        await foreman.tick()
        await foreman.tick(after=11)  # second: give up
        assert (await _task(f"{job_id}_task_0"))[0] == "failed"
        assert client.sent[-1]["type"] == "job_error"
        assert foreman.connections.is_worker_available("w1")

    run(scenario)


def test_task_that_keeps_failing_is_not_retried_forever():
    async def scenario():
        foreman = _Foreman()
        foreman.jobs.max_task_attempts = 3
        job_id = await foreman.job()
        task_id = f"{job_id}_task_0"
        outcomes = []
        for _ in range(3):
            assert await _task(task_id) == ("assigned", "w1")
            outcomes.append(await foreman.jobs.mark_task_failed(task_id, job_id, "w1", "boom"))
            foreman.connections.mark_worker_available("w1")
            if not outcomes[-1]:
                await foreman.dispatcher.assign_tasks_for_job(job_id, CODE, [])
        assert outcomes == [False, False, True]
        # an error from a worker that no longer holds the task changes nothing
        assert not await foreman.jobs.mark_task_failed(task_id, job_id, "other", "late")
        await _finish_all(foreman, job_id)

    run(scenario)


def test_concurrent_dispatch_gives_a_worker_one_task():
    async def scenario():
        foreman = _Foreman(workers=())
        job_id = uuid.uuid4().hex
        await foreman.jobs.create_job(job_id, CODE, [0, 1], 2)
        socket = _Socket()
        foreman.connections.add_worker("w1", socket)
        await asyncio.gather(*[foreman.dispatcher.assign_tasks_for_job(job_id, CODE, []) for _ in range(3)])
        assert len(socket.sent) == 1
        await _finish_all(foreman, job_id)

    run(scenario)


# ---------------------------------------------------------------- planned path
def _lost_run_scenario(during_switch):
    from tests.planning.fake_workers import submit
    from tests.planning.test_integration_fake_workers import _check_results, _heartbeats, _network

    async def scenario():
        net, clock = _network()
        net.coordinator.request_timeout_s = 10.0
        await _heartbeats(net, clock)
        send, dropped = net.send_worker, []

        async def lossy(worker_id, message):
            if worker_id == "w2" and message["type"] == "run" and not dropped:
                dropped.append(message)
                return True  # looks sent, never arrives
            return await send(worker_id, message)

        net.send_worker = lossy
        client = await submit(net, 1)
        await net.settle()
        assert dropped
        coordinator = net.coordinator
        if during_switch:
            from foreman.core.plan_coordinator import PendingSwitch

            coordinator.pending = PendingSwitch(coordinator.plan.assignment(), (), "drain", "test",
                                                coordinator.epoch + 1, clock())
        for _ in range(40):
            await net.settle()
            clock.now += 1.0
            await coordinator.tick()
        await net.settle()
        _check_results(client, 1)
        assert coordinator.pending is None

    asyncio.run(scenario())


def test_planned_request_recovers_from_a_lost_run():
    _lost_run_scenario(during_switch=False)


def test_lost_run_does_not_hold_a_drain_forever():
    _lost_run_scenario(during_switch=True)
