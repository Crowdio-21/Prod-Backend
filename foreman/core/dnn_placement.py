"""Which workers may run DNN tasks.

DNN work runs on mobile devices. PC workers are only for testing that the
partitioning is correct. ``CROWDIO_DNN_WORKERS`` selects the scope:

  mobile  (default) Android workers only (android_kotlin, android_chaquopy)
  pc      PC Python workers only, for validating partitions on a desktop
  any     every connected worker

Non-DNN tasks are not affected.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

from common.code_instrumenter import WorkerType

from foreman.planning.scope import SCOPES, check_scope, kind_allowed  # noqa: F401  (re-exported)

MOBILE_TYPES = frozenset({WorkerType.ANDROID_KOTLIN, WorkerType.ANDROID_CHAQUOPY})
DNN_MARKERS = (
    "onnxruntime",
    "def partition_model",
    "def run_onnx_partition",
    "def run_tflite_partition",
)


def dnn_worker_scope(environ: Optional[Mapping[str, str]] = None) -> str:
    environ = os.environ if environ is None else environ
    return check_scope(environ.get("CROWDIO_DNN_WORKERS") or "mobile")


def is_dnn_task(func_code: Optional[str]) -> bool:
    if not func_code:
        return False
    normalized = func_code.lower()
    return any(marker in normalized for marker in DNN_MARKERS)


def worker_kind(worker_type: Optional[WorkerType]) -> str:
    """"mobile", "pc" or "unknown"."""
    if worker_type in MOBILE_TYPES:
        return "mobile"
    if worker_type == WorkerType.PC_PYTHON:
        return "pc"
    return "unknown"

