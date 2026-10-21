"""Small in-process DNN routing feed for the dashboard and terminal logs."""

from collections import deque
from datetime import datetime, timezone
from typing import Any

_events = deque(maxlen=200)
_summary = {
    "dnn_jobs_seen": 0,
    "desktop_tasks_assigned": 0,
    "non_desktop_tasks_rejected": 0,
    "waiting_for_desktop_worker": 0,
}


def record_event(event: str, **data: Any) -> None:
    _events.appendleft({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **data,
    })


def increment(counter: str) -> None:
    if counter in _summary:
        _summary[counter] += 1


def snapshot() -> dict[str, Any]:
    return {"summary": dict(_summary), "events": list(_events)}
