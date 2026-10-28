from types import SimpleNamespace

import pytest

from common.code_instrumenter import WorkerType
from foreman.core.dnn_placement import dnn_worker_scope
from foreman.core.task_dispatcher import TaskDispatcher


class _ConnectionManager:
    def __init__(self):
        self.workers = {
            "android": SimpleNamespace(worker_type=WorkerType.ANDROID_CHAQUOPY),
            "android-kotlin": SimpleNamespace(worker_type=WorkerType.ANDROID_KOTLIN),
            "desktop": SimpleNamespace(worker_type=WorkerType.PC_PYTHON),
        }

    def get_worker_info(self, worker_id):
        return self.workers.get(worker_id)


ALL = {"android", "android-kotlin", "desktop", "ghost"}


def _dispatcher(monkeypatch, scope=None):
    if scope is None:
        monkeypatch.delenv("CROWDIO_DNN_WORKERS", raising=False)
    else:
        monkeypatch.setenv("CROWDIO_DNN_WORKERS", scope)
    return TaskDispatcher(None, _ConnectionManager(), None)


def test_dnn_tasks_are_detected():
    assert TaskDispatcher._is_dnn_task("def run(stage_input):\n    import onnxruntime as ort\n")
    assert TaskDispatcher._is_dnn_task("def partition_model(config):\n    pass\n")
    assert TaskDispatcher._is_dnn_task("def run_onnx_partition(stage_input):\n    pass\n")
    assert not TaskDispatcher._is_dnn_task("def add(value):\n    return value + 1\n")


def test_dnn_tasks_run_on_mobile_workers_by_default(monkeypatch):
    dispatcher = _dispatcher(monkeypatch)
    assert dispatcher.dnn_scope == "mobile"
    assert dispatcher._dnn_workers(ALL) == {"android", "android-kotlin"}


def test_pc_scope_is_for_validating_partitions_on_pcs(monkeypatch):
    dispatcher = _dispatcher(monkeypatch, "pc")
    assert dispatcher._dnn_workers(ALL) == {"desktop"}


def test_any_scope_allows_known_workers_only(monkeypatch):
    dispatcher = _dispatcher(monkeypatch, "any")
    assert dispatcher._dnn_workers(ALL) == {"android", "android-kotlin", "desktop"}


def test_invalid_scope_is_rejected():
    with pytest.raises(ValueError):
        dnn_worker_scope({"CROWDIO_DNN_WORKERS": "desktop"})
