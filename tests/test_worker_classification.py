import pytest

from common.code_instrumenter import WorkerType
from foreman.core.connection_manager import WorkerInfo, resolve_worker_type


def info(**data):
    return WorkerInfo.from_worker_ready_message("w", None, {"worker_id": "w", **data})


def test_the_real_android_app_registration_is_a_mobile_worker():
    """What the Kotlin app actually sends (WorkerWebSocketClient.kt): runtime is "jvm", not "kotlin"."""
    phone = info(worker_type="android_kotlin", platform="android", runtime="jvm",
                 capabilities={"supports_settrace": False, "supports_frame_introspection": False})
    assert phone.worker_type is WorkerType.ANDROID_KOTLIN
    assert phone.needs_instrumentation  # unchanged: no sys.settrace on the phone


@pytest.mark.parametrize("data,expected", [
    ({"platform": "android", "runtime": "jvm"}, WorkerType.ANDROID_KOTLIN),  # no declared type: still a phone
    ({"platform": "android", "runtime": "chaquopy"}, WorkerType.ANDROID_CHAQUOPY),
    ({"worker_type": "android_chaquopy", "platform": "android", "runtime": "jvm"}, WorkerType.ANDROID_CHAQUOPY),
    ({"device_specs": {"os_type": "Windows"}}, WorkerType.PC_PYTHON),  # the PC worker sends no type
    ({"worker_type": "pc_python", "platform": "windows"}, WorkerType.PC_PYTHON),
    ({"worker_type": "toaster", "platform": "windows"}, WorkerType.PC_PYTHON),  # unknown declaration ignored
])
def test_worker_type_resolution(data, expected):
    assert info(**data).worker_type is expected


def test_a_pc_is_not_promoted_to_mobile():
    assert resolve_worker_type(WorkerType.PC_PYTHON, None, "windows", "cpython") is WorkerType.PC_PYTHON
    assert resolve_worker_type(WorkerType.PC_PYTHON, "unknown", "linux", None) is WorkerType.PC_PYTHON
