from types import SimpleNamespace

from common.code_instrumenter import WorkerType
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


def test_onnx_tasks_only_use_desktop_workers():
    dispatcher = TaskDispatcher(None, _ConnectionManager(), None)

    assert dispatcher._requires_desktop_worker(
        "def run(stage_input):\n    import onnxruntime as ort\n"
    )
    assert dispatcher._requires_desktop_worker("def partition_model(config):\n    pass\n")
    assert dispatcher._onnx_workers({"android", "android-kotlin", "desktop"}) == {
        "android-kotlin",
        "desktop",
    }


def test_regular_tasks_keep_android_workers_eligible():
    dispatcher = TaskDispatcher(None, _ConnectionManager(), None)

    assert not dispatcher._requires_desktop_worker("def add(value):\n    return value + 1\n")
    assert dispatcher._onnx_workers({"android", "desktop"}) == {"desktop"}
