import asyncio

import httpx
import pytest

from foreman.core.scheduling import (
    FIFOScheduler,
    LeastLoadedScheduler,
    PerformanceBasedScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    create_scheduler,
)


@pytest.mark.parametrize("name,cls", [
    ("fifo", FIFOScheduler), ("round_robin", RoundRobinScheduler), ("performance", PerformanceBasedScheduler),
    ("least_loaded", LeastLoadedScheduler), ("priority", PriorityScheduler), ("FIFO", FIFOScheduler),
])
def test_simple_schedulers_are_created_by_name(name, cls):
    assert type(create_scheduler(name)) is cls


def test_default_scheduler_is_fifo():
    assert type(create_scheduler()) is FIFOScheduler


@pytest.mark.parametrize("name", ["aras", "edas", "mabac", "wrr", "nope"])
def test_removed_and_unknown_schedulers_are_rejected(name):
    with pytest.raises(ValueError, match="Available: fifo, round_robin"):
        create_scheduler(name)


def test_no_mcdm_code_or_api_remains():
    import importlib

    import foreman.db.models as models
    from foreman.main import app

    assert not hasattr(models, "SchedulerConfigModel")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("foreman.core.scheduling.mcdm")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("foreman.api.scheduler_routes")
    assert not [r.path for r in app.routes if getattr(r, "path", "").startswith("/api/scheduler")]


def test_startup_and_scheduling_info_work_without_mcdm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def scenario():
        from foreman.db.base import engine
        from foreman.db.seed import initialize_database
        from foreman.main import app

        try:
            assert await initialize_database() is True
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                return await client.get("/api/scheduling-info")
        finally:
            await engine.dispose()

    response = asyncio.run(scenario())
    assert response.status_code == 200
    scheduler = response.json()["scheduler"]
    assert set(scheduler) == {"algorithm", "description"}
    assert scheduler["algorithm"] == "fifo"
