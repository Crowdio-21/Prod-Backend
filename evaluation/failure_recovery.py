"""
Failure Recovery Tracker Module

Tracks and measures failure recovery times and checkpoint efficiency
in the CROWDio distributed computing system.

Metrics:
- Failure detection time
- Recovery time (time to reassign and complete failed tasks)
- Checkpoint overhead
- Recovery success rate
- Mean time between failures (MTBF)
- Mean time to recovery (MTTR)
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json


class FailureType(Enum):
    """Types of failures that can occur"""
    WORKER_DISCONNECT = "worker_disconnect"
    WORKER_CRASH = "worker_crash"
    TASK_TIMEOUT = "task_timeout"
    TASK_ERROR = "task_error"
    NETWORK_FAILURE = "network_failure"
    CHECKPOINT_FAILURE = "checkpoint_failure"
    MEMORY_ERROR = "memory_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Strategies for recovering from failures"""
    RESTART_TASK = "restart_task"  # Start task from beginning
    CHECKPOINT_RECOVERY = "checkpoint_recovery"  # Resume from checkpoint
    REASSIGN_TO_NEW_WORKER = "reassign_to_new_worker"
    SKIP_TASK = "skip_task"  # Mark as failed and continue
    JOB_FAILURE = "job_failure"  # Fail entire job


@dataclass
class FailureEvent:
    """Record of a failure event"""
    failure_id: str
    failure_type: FailureType
    worker_id: Optional[str]
    task_id: Optional[str]
    job_id: Optional[str]
    
    # Timing
    detected_at: float
    recovery_started_at: Optional[float] = None
    recovery_completed_at: Optional[float] = None
    
    # Recovery information
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_success: bool = False
    new_worker_id: Optional[str] = None
    
    # Checkpoint information
    checkpoint_available: bool = False
    checkpoint_age_seconds: Optional[float] = None  # Time since last checkpoint
    checkpoint_size_bytes: int = 0
    progress_recovered: float = 0.0  # Percentage of work saved (0-100)
    
    # Error details
    error_message: Optional[str] = None
    
    # Derived metrics
    detection_time: Optional[float] = None  # Time from failure to detection
    recovery_time: Optional[float] = None  # Time from detection to recovery
    total_downtime: Optional[float] = None  # Total time task was not progressing
    
    def calculate_metrics(self) -> None:
        """Calculate derived timing metrics"""
        if self.recovery_started_at and self.detected_at:
            self.detection_time = self.recovery_started_at - self.detected_at
        
        if self.recovery_completed_at and self.recovery_started_at:
            self.recovery_time = self.recovery_completed_at - self.recovery_started_at
        
        if self.recovery_completed_at and self.detected_at:
            self.total_downtime = self.recovery_completed_at - self.detected_at
    
    def to_dict(self) -> Dict[str, Any]:
        self.calculate_metrics()
        return {
            "failure_id": self.failure_id,
            "failure_type": self.failure_type.value,
            "worker_id": self.worker_id,
            "task_id": self.task_id,
            "job_id": self.job_id,
            "detected_at": self.detected_at,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "recovery_success": self.recovery_success,
            "checkpoint_available": self.checkpoint_available,
            "progress_recovered": self.progress_recovered,
            "detection_time": self.detection_time,
            "recovery_time": self.recovery_time,
            "total_downtime": self.total_downtime,
            "error_message": self.error_message,
        }


@dataclass
class WorkerReliability:
    """Reliability metrics for a worker"""
    worker_id: str
    
    # Failure counts
    total_failures: int = 0
    failures_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Recovery counts
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    
    # Timing metrics
    total_uptime: float = 0.0
    total_downtime: float = 0.0
    
    # Derived metrics
    mtbf: Optional[float] = None  # Mean Time Between Failures
    mttr: Optional[float] = None  # Mean Time To Recovery
    availability: float = 100.0  # Percentage uptime
    failure_rate: float = 0.0  # Failures per hour
    
    # Connection tracking
    connect_times: List[float] = field(default_factory=list)
    disconnect_times: List[float] = field(default_factory=list)
    
    def calculate_metrics(self) -> None:
        """Calculate derived reliability metrics"""
        total_time = self.total_uptime + self.total_downtime
        
        if total_time > 0:
            self.availability = (self.total_uptime / total_time) * 100.0
        
        if self.total_failures > 0:
            # MTBF: average uptime between failures
            if len(self.connect_times) > 1:
                uptime_periods = []
                for i in range(len(self.disconnect_times)):
                    if i < len(self.connect_times):
                        uptime_periods.append(
                            self.disconnect_times[i] - self.connect_times[i]
                        )
                if uptime_periods:
                    self.mtbf = statistics.mean(uptime_periods)
            
            # MTTR: average recovery time
            if self.successful_recoveries > 0:
                self.mttr = self.total_downtime / self.successful_recoveries
            
            # Failure rate (per hour)
            if total_time > 0:
                self.failure_rate = (self.total_failures / total_time) * 3600
    
    def to_dict(self) -> Dict[str, Any]:
        self.calculate_metrics()
        return {
            "worker_id": self.worker_id,
            "total_failures": self.total_failures,
            "failures_by_type": self.failures_by_type,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "availability_percent": self.availability,
            "mtbf_seconds": self.mtbf,
            "mttr_seconds": self.mttr,
            "failure_rate_per_hour": self.failure_rate,
        }


@dataclass
class RecoveryMetrics:
    """Aggregate recovery metrics"""
    # Failure statistics
    total_failures: int = 0
    failures_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Recovery statistics
    total_recoveries: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_success_rate: float = 0.0
    
    # Checkpoint statistics
    recoveries_with_checkpoint: int = 0
    recoveries_without_checkpoint: int = 0
    avg_progress_saved: float = 0.0  # Average % progress recovered via checkpoints
    
    # Timing statistics
    avg_detection_time: Optional[float] = None
    avg_recovery_time: Optional[float] = None
    min_recovery_time: Optional[float] = None
    max_recovery_time: Optional[float] = None
    p95_recovery_time: Optional[float] = None
    
    # System reliability
    system_mtbf: Optional[float] = None
    system_mttr: Optional[float] = None
    system_availability: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureRecoveryTracker:
    """
    Tracks failures and recovery performance across the distributed system.
    
    Provides:
    - Failure detection and logging
    - Recovery time measurement
    - Checkpoint effectiveness analysis
    - Worker reliability tracking
    - System-wide MTBF/MTTR calculation
    """
    
    def __init__(self):
        """Initialize failure recovery tracker"""
        # Failure events: failure_id -> FailureEvent
        self._failure_events: Dict[str, FailureEvent] = {}
        
        # Worker reliability: worker_id -> WorkerReliability
        self._worker_reliability: Dict[str, WorkerReliability] = {}
        
        # Active failures awaiting recovery: task_id -> failure_id
        self._pending_recoveries: Dict[str, str] = {}
        
        # Tracking state
        self._tracking_start_time: float = time.time()
        self._failure_counter: int = 0
        
        # Lock for thread-safe access
        self._lock = asyncio.Lock()
    
    # ==================== Failure Recording ====================
    
    async def record_failure(
        self,
        failure_type: FailureType,
        worker_id: Optional[str] = None,
        task_id: Optional[str] = None,
        job_id: Optional[str] = None,
        error_message: Optional[str] = None,
        checkpoint_available: bool = False,
        checkpoint_age: Optional[float] = None,
        checkpoint_size: int = 0,
        progress_at_failure: float = 0.0
    ) -> str:
        """
        Record a failure event.
        
        Args:
            failure_type: Type of failure
            worker_id: Worker where failure occurred
            task_id: Task that failed (if applicable)
            job_id: Job associated with failure
            error_message: Error details
            checkpoint_available: Whether checkpoint exists for recovery
            checkpoint_age: Seconds since last checkpoint
            checkpoint_size: Size of checkpoint in bytes
            progress_at_failure: Task progress percentage at failure time
        
        Returns:
            Failure ID for tracking
        """
        async with self._lock:
            self._failure_counter += 1
            failure_id = f"failure_{self._failure_counter}_{int(time.time())}"
            
            event = FailureEvent(
                failure_id=failure_id,
                failure_type=failure_type,
                worker_id=worker_id,
                task_id=task_id,
                job_id=job_id,
                detected_at=time.time(),
                error_message=error_message,
                checkpoint_available=checkpoint_available,
                checkpoint_age_seconds=checkpoint_age,
                checkpoint_size_bytes=checkpoint_size,
                progress_recovered=progress_at_failure if checkpoint_available else 0.0,
            )
            
            self._failure_events[failure_id] = event
            
            # Track pending recovery
            if task_id:
                self._pending_recoveries[task_id] = failure_id
            
            # Update worker reliability
            if worker_id:
                await self._update_worker_failure(worker_id, failure_type)
            
            print(f"FailureRecoveryTracker: Recorded {failure_type.value} failure {failure_id}")
            
            return failure_id
    
    async def _update_worker_failure(
        self,
        worker_id: str,
        failure_type: FailureType
    ) -> None:
        """Update worker reliability on failure"""
        if worker_id not in self._worker_reliability:
            self._worker_reliability[worker_id] = WorkerReliability(worker_id=worker_id)
        
        reliability = self._worker_reliability[worker_id]
        reliability.total_failures += 1
        
        type_str = failure_type.value
        reliability.failures_by_type[type_str] = (
            reliability.failures_by_type.get(type_str, 0) + 1
        )
        
        # Record disconnect time
        reliability.disconnect_times.append(time.time())
    
    # ==================== Recovery Recording ====================
    
    async def start_recovery(
        self,
        failure_id: str,
        recovery_strategy: RecoveryStrategy,
        new_worker_id: Optional[str] = None
    ) -> None:
        """
        Record that recovery has started for a failure.
        
        Args:
            failure_id: ID of the failure being recovered
            recovery_strategy: Strategy being used
            new_worker_id: Worker handling recovery (if different)
        """
        async with self._lock:
            if failure_id not in self._failure_events:
                return
            
            event = self._failure_events[failure_id]
            event.recovery_started_at = time.time()
            event.recovery_strategy = recovery_strategy
            event.new_worker_id = new_worker_id
    
    async def complete_recovery(
        self,
        failure_id: str,
        success: bool,
        work_redone: float = 0.0
    ) -> None:
        """
        Record recovery completion.
        
        Args:
            failure_id: ID of the failure that was recovered
            success: Whether recovery was successful
            work_redone: Percentage of work that had to be redone (0-100)
        """
        async with self._lock:
            if failure_id not in self._failure_events:
                return
            
            event = self._failure_events[failure_id]
            event.recovery_completed_at = time.time()
            event.recovery_success = success
            event.calculate_metrics()
            
            # Remove from pending
            if event.task_id and event.task_id in self._pending_recoveries:
                del self._pending_recoveries[event.task_id]
            
            # Update worker reliability
            worker_id = event.new_worker_id or event.worker_id
            if worker_id and worker_id in self._worker_reliability:
                reliability = self._worker_reliability[worker_id]
                if success:
                    reliability.successful_recoveries += 1
                else:
                    reliability.failed_recoveries += 1
                
                if event.total_downtime:
                    reliability.total_downtime += event.total_downtime
            
            print(f"FailureRecoveryTracker: Recovery {'succeeded' if success else 'failed'} for {failure_id}")
    
    async def record_task_recovery_from_checkpoint(
        self,
        task_id: str,
        checkpoint_progress: float,
        recovery_time: float
    ) -> None:
        """
        Record successful task recovery from checkpoint.
        
        Args:
            task_id: Task that was recovered
            checkpoint_progress: Progress percentage from checkpoint (0-100)
            recovery_time: Time taken to recover
        """
        async with self._lock:
            if task_id in self._pending_recoveries:
                failure_id = self._pending_recoveries[task_id]
                if failure_id in self._failure_events:
                    event = self._failure_events[failure_id]
                    event.progress_recovered = checkpoint_progress
                    event.recovery_completed_at = time.time()
                    event.recovery_success = True
                    event.recovery_strategy = RecoveryStrategy.CHECKPOINT_RECOVERY
                    event.calculate_metrics()
                    del self._pending_recoveries[task_id]
    
    # ==================== Worker Connection Tracking ====================
    
    async def record_worker_connect(self, worker_id: str) -> None:
        """Record worker connection"""
        async with self._lock:
            if worker_id not in self._worker_reliability:
                self._worker_reliability[worker_id] = WorkerReliability(worker_id=worker_id)
            
            reliability = self._worker_reliability[worker_id]
            reliability.connect_times.append(time.time())
    
    async def record_worker_disconnect(
        self,
        worker_id: str,
        graceful: bool = True
    ) -> None:
        """Record worker disconnection"""
        async with self._lock:
            if worker_id not in self._worker_reliability:
                return
            
            reliability = self._worker_reliability[worker_id]
            now = time.time()
            
            # Calculate uptime for this session
            if reliability.connect_times:
                last_connect = reliability.connect_times[-1]
                session_uptime = now - last_connect
                reliability.total_uptime += session_uptime
            
            reliability.disconnect_times.append(now)
            
            if not graceful:
                # Record as failure
                await self.record_failure(
                    failure_type=FailureType.WORKER_DISCONNECT,
                    worker_id=worker_id,
                    error_message="Unexpected disconnection"
                )
    
    # ==================== Metrics and Analysis ====================
    
    async def get_recovery_metrics(self) -> RecoveryMetrics:
        """Get aggregate recovery metrics"""
        async with self._lock:
            metrics = RecoveryMetrics()
            
            if not self._failure_events:
                return metrics
            
            recovery_times = []
            detection_times = []
            progress_saved = []
            
            for event in self._failure_events.values():
                event.calculate_metrics()
                
                metrics.total_failures += 1
                type_str = event.failure_type.value
                metrics.failures_by_type[type_str] = (
                    metrics.failures_by_type.get(type_str, 0) + 1
                )
                
                if event.recovery_completed_at:
                    metrics.total_recoveries += 1
                    if event.recovery_success:
                        metrics.successful_recoveries += 1
                    else:
                        metrics.failed_recoveries += 1
                    
                    if event.recovery_time:
                        recovery_times.append(event.recovery_time)
                
                if event.detection_time:
                    detection_times.append(event.detection_time)
                
                if event.checkpoint_available:
                    metrics.recoveries_with_checkpoint += 1
                    progress_saved.append(event.progress_recovered)
                else:
                    metrics.recoveries_without_checkpoint += 1
            
            # Calculate success rate
            if metrics.total_recoveries > 0:
                metrics.recovery_success_rate = (
                    metrics.successful_recoveries / metrics.total_recoveries * 100
                )
            
            # Calculate timing statistics
            if recovery_times:
                metrics.avg_recovery_time = statistics.mean(recovery_times)
                metrics.min_recovery_time = min(recovery_times)
                metrics.max_recovery_time = max(recovery_times)
                if len(recovery_times) > 1:
                    sorted_times = sorted(recovery_times)
                    metrics.p95_recovery_time = sorted_times[int(len(sorted_times) * 0.95)]
            
            if detection_times:
                metrics.avg_detection_time = statistics.mean(detection_times)
            
            if progress_saved:
                metrics.avg_progress_saved = statistics.mean(progress_saved)
            
            # Calculate system MTBF/MTTR
            total_uptime = sum(r.total_uptime for r in self._worker_reliability.values())
            total_downtime = sum(r.total_downtime for r in self._worker_reliability.values())
            total_time = total_uptime + total_downtime
            
            if metrics.total_failures > 0 and total_uptime > 0:
                metrics.system_mtbf = total_uptime / metrics.total_failures
            
            if metrics.successful_recoveries > 0 and total_downtime > 0:
                metrics.system_mttr = total_downtime / metrics.successful_recoveries
            
            if total_time > 0:
                metrics.system_availability = (total_uptime / total_time) * 100
            
            return metrics
    
    async def get_worker_reliability(self, worker_id: str) -> Optional[WorkerReliability]:
        """Get reliability metrics for a specific worker"""
        async with self._lock:
            reliability = self._worker_reliability.get(worker_id)
            if reliability:
                reliability.calculate_metrics()
            return reliability
    
    async def get_all_worker_reliability(self) -> Dict[str, WorkerReliability]:
        """Get reliability metrics for all workers"""
        async with self._lock:
            for reliability in self._worker_reliability.values():
                reliability.calculate_metrics()
            return dict(self._worker_reliability)
    
    async def get_failure_events(
        self,
        worker_id: Optional[str] = None,
        failure_type: Optional[FailureType] = None,
        since: Optional[float] = None
    ) -> List[FailureEvent]:
        """Get failure events with optional filtering"""
        async with self._lock:
            events = list(self._failure_events.values())
            
            if worker_id:
                events = [e for e in events if e.worker_id == worker_id]
            
            if failure_type:
                events = [e for e in events if e.failure_type == failure_type]
            
            if since:
                events = [e for e in events if e.detected_at >= since]
            
            return sorted(events, key=lambda e: e.detected_at)
    
    async def get_checkpoint_efficiency(self) -> Dict[str, Any]:
        """Analyze checkpoint effectiveness for recovery"""
        async with self._lock:
            with_checkpoint = []
            without_checkpoint = []
            
            for event in self._failure_events.values():
                if event.recovery_completed_at:
                    if event.checkpoint_available:
                        with_checkpoint.append(event)
                    else:
                        without_checkpoint.append(event)
            
            analysis = {
                "total_recoveries": len(with_checkpoint) + len(without_checkpoint),
                "with_checkpoint": {
                    "count": len(with_checkpoint),
                    "avg_recovery_time": None,
                    "avg_progress_saved": None,
                    "success_rate": None,
                },
                "without_checkpoint": {
                    "count": len(without_checkpoint),
                    "avg_recovery_time": None,
                    "success_rate": None,
                },
                "checkpoint_benefit": {
                    "time_saved": None,
                    "work_saved_percent": None,
                }
            }
            
            if with_checkpoint:
                recovery_times = [e.recovery_time for e in with_checkpoint if e.recovery_time]
                if recovery_times:
                    analysis["with_checkpoint"]["avg_recovery_time"] = statistics.mean(recovery_times)
                
                progress = [e.progress_recovered for e in with_checkpoint]
                analysis["with_checkpoint"]["avg_progress_saved"] = statistics.mean(progress)
                
                successful = sum(1 for e in with_checkpoint if e.recovery_success)
                analysis["with_checkpoint"]["success_rate"] = (successful / len(with_checkpoint)) * 100
            
            if without_checkpoint:
                recovery_times = [e.recovery_time for e in without_checkpoint if e.recovery_time]
                if recovery_times:
                    analysis["without_checkpoint"]["avg_recovery_time"] = statistics.mean(recovery_times)
                
                successful = sum(1 for e in without_checkpoint if e.recovery_success)
                analysis["without_checkpoint"]["success_rate"] = (successful / len(without_checkpoint)) * 100
            
            # Calculate checkpoint benefit
            if (analysis["with_checkpoint"]["avg_recovery_time"] and 
                analysis["without_checkpoint"]["avg_recovery_time"]):
                analysis["checkpoint_benefit"]["time_saved"] = (
                    analysis["without_checkpoint"]["avg_recovery_time"] -
                    analysis["with_checkpoint"]["avg_recovery_time"]
                )
            
            if analysis["with_checkpoint"]["avg_progress_saved"]:
                analysis["checkpoint_benefit"]["work_saved_percent"] = (
                    analysis["with_checkpoint"]["avg_progress_saved"]
                )
            
            return analysis
    
    # ==================== Export ====================
    
    async def export_report(self, filepath: str) -> None:
        """Export failure recovery report to file"""
        metrics = await self.get_recovery_metrics()
        checkpoint_analysis = await self.get_checkpoint_efficiency()
        
        async with self._lock:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "tracking_duration_seconds": time.time() - self._tracking_start_time,
                "recovery_metrics": metrics.to_dict(),
                "checkpoint_efficiency": checkpoint_analysis,
                "worker_reliability": {
                    worker_id: reliability.to_dict()
                    for worker_id, reliability in self._worker_reliability.items()
                },
                "failure_events": [
                    event.to_dict() for event in self._failure_events.values()
                ],
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"FailureRecoveryTracker: Exported report to {filepath}")
    
    async def reset(self) -> None:
        """Reset all tracking data"""
        async with self._lock:
            self._failure_events.clear()
            self._worker_reliability.clear()
            self._pending_recoveries.clear()
            self._failure_counter = 0
            self._tracking_start_time = time.time()
        print("FailureRecoveryTracker: Reset all tracking data")
