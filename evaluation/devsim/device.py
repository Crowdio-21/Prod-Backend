"""A simulated device: speaks the real worker protocol to a real Foreman over a WebSocket.

What the Foreman sees is what it would see from a phone (or PC): the same registration, the same
heartbeats and PING replies, task results and errors. What is simulated:

* **the link.** Every message to or from the device waits for its transfer time on the device's
  current bandwidth (see ``link.py``), plus half the round trip each way. A model file the device
  has not seen is "downloaded" the same way before its first task with that model.
* **the CPU.** In ``model`` mode a task takes ``work_s x cpu_factor`` virtual seconds and returns a
  padded result of the requested size, so 15 devices never run a real model on one machine. In
  ``real`` mode the task function really runs (so results are real) and is then slowed by
  ``cpu_factor`` (this needs ``speed`` 1).
* **the metrics.** CPU, RAM and battery are plausible numbers for the device's tier. The link
  measurement it reports is the true current rate with 8% noise, in bytes per *real* second (see
  ``clock.py``); it does not cost any bandwidth.

A task hint travels in the arguments: ``{"sim": {"work_s": 0.5, "out_bytes": 20000}, ...}``, found at
the top level or under ``original_args`` (where pipeline stages put it). Without a hint the device
uses ``default_work_s`` and a result about the size of the input.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import websockets

from common.protocol import Message, MessageType

from .clock import VirtualClock
from .fleet import DeviceSpec
from .link import LinkEngine

_MODEL_RE = re.compile(r'"model_path"\s*:\s*"([^"]+)"')
LINK_NOISE = 0.08


@dataclass
class DeviceStats:
    tasks: int = 0
    errors: int = 0
    busy_s: float = 0.0  # virtual seconds from receiving a task to having sent its result
    bytes_down: int = 0
    bytes_up: int = 0
    model_downloads: int = 0
    connects: int = 0
    task_seconds: List[float] = field(default_factory=list)


def find_hint(task_args: Any) -> Dict[str, Any]:
    """The ``sim`` hint inside a task's arguments (top level or under ``original_args``)."""
    items = task_args if isinstance(task_args, list) else [task_args]
    for item in items:
        if not isinstance(item, dict):
            continue
        for holder in (item, item.get("original_args") if isinstance(item.get("original_args"), dict) else {}):
            hint = holder.get("sim")
            if isinstance(hint, dict):
                return hint
    return {}


def find_model(task_args: Any) -> Optional[str]:
    match = _MODEL_RE.search(json.dumps(task_args)[:4096])
    return match.group(1).replace("\\", "/").rsplit("/", 1)[-1] if match else None


class VirtualDevice:
    def __init__(
        self,
        spec: DeviceSpec,
        url: str,
        clock: VirtualClock,
        engine: LinkEngine,
        *,
        mode: str = "model",
        model_size: Callable[[str], Optional[int]] = lambda name: 0,
        seed: int = 0,
        heartbeat_s: float = 10.0,
        probe_s: float = 10.0,
        default_work_s: float = 0.5,
        out_ratio: float = 0.1,
        events: Optional[Callable[..., None]] = None,
    ) -> None:
        if mode not in ("model", "real"):
            raise ValueError("mode must be 'model' or 'real'")
        if mode == "real" and clock.speed != 1.0:
            raise ValueError("real mode runs the actual task function, so it needs speed 1")
        self.spec = spec
        self.url = url
        self.clock = clock
        self.engine = engine
        self.mode = mode
        self.model_size = model_size
        self.heartbeat_s = heartbeat_s
        self.probe_s = probe_s
        self.default_work_s = default_work_s
        self.out_ratio = out_ratio
        self.stats = DeviceStats()
        self._rng = random.Random(f"device:{seed}:{spec.device_id}")
        self._emit = events or (lambda *a, **k: None)
        self._cached: set = set()
        self._ws = None
        self._busy = False
        self._current_task: Optional[str] = None
        self._online = asyncio.Event()
        self._online.set()
        self._closed = False
        self._session_task: Optional[asyncio.Task] = None
        self._measured: Optional[Dict[str, float]] = None
        self._measured_at = -math.inf

    # ---------------------------------------------------------------- lifecycle
    @property
    def online(self) -> bool:
        return self._online.is_set() and self._ws is not None

    async def run(self) -> None:
        """Stay connected until ``close``; reconnect after drops and when told to come back online."""
        while not self._closed:
            await self._online.wait()
            self._session_task = asyncio.create_task(self._session())
            try:
                await self._session_task
            except asyncio.CancelledError:
                if self._closed:
                    raise
            except Exception as exc:  # connection refused, dropped, aborted transfer...
                self._emit("error", device=self.spec.device_id, error=repr(exc))
                await asyncio.sleep(0.3)

    def go_offline(self) -> None:
        """Drop off the network (the Foreman sees the socket close)."""
        self._online.clear()
        self.engine.abort(self.spec.device_id)
        if self._session_task is not None:
            self._session_task.cancel()
        self._emit("leave", device=self.spec.device_id)

    def go_online(self) -> None:
        self._online.set()
        self._emit("join", device=self.spec.device_id)

    async def close(self) -> None:
        self._closed = True
        self._online.set()
        if self._session_task is not None:
            self._session_task.cancel()

    # ---------------------------------------------------------------- one connection
    async def _session(self) -> None:
        async with websockets.connect(self.url, max_size=None, ping_interval=None, close_timeout=1) as ws:
            self._ws = ws
            self.stats.connects += 1
            heartbeat = None
            try:
                await self._send(Message(MessageType.WORKER_READY, self._registration()))
                self._emit("connect", device=self.spec.device_id)
                heartbeat = asyncio.create_task(self._heartbeats())
                async for raw in ws:
                    await self._on_message(raw)
            finally:
                if heartbeat is not None:
                    heartbeat.cancel()
                self._ws = None
                self._busy = False

    def _registration(self) -> Dict[str, Any]:
        s = self.spec
        specs = {
            "device_type": "PC" if s.kind == "pc" else "Android", "os_type": "Windows" if s.kind == "pc" else "Android",
            "os_version": "11" if s.kind == "pc" else "14", "cpu_model": f"sim-{s.tier}", "cpu_cores": s.cores,
            "cpu_threads": s.cores, "cpu_frequency_mhz": round(2800 / s.cpu_factor), "ram_total_mb": s.ram_mb,
            "ram_available_mb": round(s.ram_mb * 0.5), "gpu_model": None, "battery_level": s.battery,
            "is_charging": s.kind == "pc", "network_type": "Ethernet" if s.kind == "pc" else "WiFi",
            "python_version": "3.12" if s.kind == "pc" else None,
        }
        if s.kind == "pc":  # the PC worker sends only its id and specs
            return {"worker_id": s.device_id, "device_specs": specs}
        return {  # what WorkerWebSocketClient.kt sends
            "worker_id": s.device_id, "worker_type": "android_kotlin", "platform": "android", "runtime": "jvm",
            "capabilities": {"supports_settrace": False, "supports_frame_introspection": False},
            "device_specs": specs,
        }

    # ---------------------------------------------------------------- shaped I/O
    async def _download(self, n_bytes: int) -> None:
        await self.engine.transfer(self.spec.device_id, "down", n_bytes)
        await self.clock.sleep(self.spec.rtt_ms / 2000.0)
        self.stats.bytes_down += n_bytes

    async def _upload(self, n_bytes: int) -> None:
        await self.engine.transfer(self.spec.device_id, "up", n_bytes)
        await self.clock.sleep(self.spec.rtt_ms / 2000.0)
        self.stats.bytes_up += n_bytes

    async def _send(self, message: Message) -> None:
        text = message.to_json()
        ws = self._ws
        await self._upload(len(text.encode("utf-8")))
        if ws is None:
            raise ConnectionError("not connected")
        await ws.send(text)

    # ---------------------------------------------------------------- messages
    async def _on_message(self, raw) -> None:
        await self._download(len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw))
        message = Message.from_dict(json.loads(raw))
        if message.type == MessageType.ASSIGN_TASK:
            await self._on_task(message)
        elif message.type == MessageType.PING:
            await self._send(self._pong())
        # resume_task and checkpoint acks are not used by the simulated workloads

    def _link_report(self) -> Dict[str, float]:
        """True current link rates with measurement noise, in bytes per real second."""
        now = self.clock.now()
        if self._measured is None or now - self._measured_at >= self.probe_s:
            noise = lambda: math.exp(self._rng.gauss(0.0, LINK_NOISE))  # noqa: E731
            speed = self.clock.speed
            self._measured = {
                "down_bps": self.engine.rate_bps(self.spec.device_id, "down", now) * noise() * speed,
                "up_bps": self.engine.rate_bps(self.spec.device_id, "up", now) * noise() * speed,
                "rtt_s": self.spec.rtt_ms / 1000.0 * noise() / speed,
            }
            self._measured_at = now
        return self._measured

    def _metrics(self) -> Dict[str, Any]:
        s, busy = self.spec, self._busy
        link = self._link_report()
        return {
            "cpu_usage_percent": round(self._rng.uniform(55, 90) if busy else self._rng.uniform(3, 15), 1),
            "ram_available_mb": round(s.ram_mb * (0.3 if busy else 0.5) * self._rng.uniform(0.9, 1.1), 1),
            "battery_level": None if s.kind == "pc" else round(max(1.0, s.battery - self.clock.now() / 600.0), 1),
            "is_charging": s.kind == "pc",
            "network_speed_mbps": round(link["down_bps"] * 8 / 1e6 / self.clock.speed, 2),
            "storage_available_gb": 40.0,
        }

    def _pong(self) -> Message:
        link = self._link_report()
        data = {"worker_id": self.spec.device_id, "status": "busy" if self._busy else "online"}
        if self.spec.kind == "pc":
            data.update({"performance_metrics": self._metrics(), "link": link})
        else:  # the Android app answers a PING with flat metrics
            data.update(self._metrics())
            data["net"] = {"bw_down": link["down_bps"] * 8, "bw_up": link["up_bps"] * 8}
        return Message(MessageType.PONG, data)

    async def _heartbeats(self) -> None:
        while True:
            link = self._link_report()
            if self.spec.kind == "pc":
                message = Message(MessageType.PONG, {
                    "worker_id": self.spec.device_id, "status": "online", "current_task": self._current_task,
                    "link": link, **self._metrics()})
            else:  # the shape of WorkerWebSocketClient.sendHeartbeatMessage
                metrics = self._metrics()
                message = Message(MessageType.WORKER_HEARTBEAT, {
                    "worker_id": self.spec.device_id, "status": "busy" if self._busy else "online",
                    "current_task": self._current_task,
                    "net": {"bw_down": link["down_bps"] * 8, "bw_up": link["up_bps"] * 8},
                    "res": {"free_ram_mb": metrics["ram_available_mb"], "free_disk_mb": 40_000,
                            "battery_pct": metrics["battery_level"], "cpu_load": metrics["cpu_usage_percent"] / 100.0,
                            "thermal": 0}})
            await self._send(message)
            await self.clock.sleep(self.heartbeat_s)

    # ---------------------------------------------------------------- tasks
    async def _on_task(self, message: Message) -> None:
        task_id = message.data["task_id"]
        job_id = message.job_id
        received = self.clock.now()
        self._busy, self._current_task = True, task_id
        try:
            result = await self._execute(message.data)
            await self._send(Message(MessageType.TASK_RESULT, {"result": json.dumps(result), "task_id": task_id}, job_id))
            took = self.clock.now() - received
            self.stats.tasks += 1
            self.stats.busy_s += took
            self.stats.task_seconds.append(took)
            self._emit("task", device=self.spec.device_id, task=task_id, seconds=round(took, 3))
        except (ConnectionError, asyncio.CancelledError):
            raise
        except Exception as exc:
            self.stats.errors += 1
            self._emit("task_error", device=self.spec.device_id, task=task_id, error=repr(exc))
            await self._send(Message(MessageType.TASK_ERROR, {"error": str(exc), "task_id": task_id}, job_id))
        finally:
            self._busy, self._current_task = False, None

    async def _execute(self, data: Dict[str, Any]) -> Any:
        task_args = data["task_args"]
        model = find_model(task_args)
        if model and model not in self._cached:  # first use of a model file: download it over this link
            size = self.model_size(model) or 0
            await self._download(size)
            self._cached.add(model)
            self.stats.model_downloads += 1
            self._emit("model", device=self.spec.device_id, model=model, bytes=size)
        if self.mode == "real":
            return await self._execute_real(data)
        hint = find_hint(task_args)
        work = float(hint.get("work_s", self.default_work_s)) * self.spec.cpu_factor * self._rng.uniform(0.95, 1.05)
        await self.clock.sleep(work)
        out_bytes = hint.get("out_bytes")
        if out_bytes is None:
            out_bytes = int(self.out_ratio * len(json.dumps(task_args)))
        return {"sim": True, "device": self.spec.device_id, "pad": "x" * int(out_bytes)}

    async def _execute_real(self, data: Dict[str, Any]) -> Any:
        from pc_worker.core.worker import FastAPIWorker

        func = FastAPIWorker._deserialize_task_function(data["func_code"], data.get("task_metadata"))
        args = data["task_args"]

        def call():
            if isinstance(args, list) and len(args) == 1:
                return func(args[0])
            if isinstance(args, list) and len(args) == 2 and isinstance(args[1], dict):
                return func(*args[0], **args[1])
            return func(*args) if isinstance(args, list) else func(args)

        started = time.perf_counter()
        result = await asyncio.to_thread(call)
        await self.clock.sleep((time.perf_counter() - started) * (self.spec.cpu_factor - 1.0))
        return result
