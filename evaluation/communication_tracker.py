"""
Communication Tracker Module

Tracks and measures communication overhead in the CROWDio distributed
computing system, including message sizes, latencies, and bandwidth usage.

Metrics:
- Message sizes (bytes sent/received)
- Message latency (round-trip time)
- Bandwidth utilization
- Protocol overhead
- Network efficiency
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


class MessageCategory(Enum):
    """Categories of messages in the system"""
    CONTROL = "control"  # Ping, pong, status updates
    TASK_ASSIGNMENT = "task_assignment"  # Task dispatch messages
    TASK_RESULT = "task_result"  # Task result messages
    CHECKPOINT = "checkpoint"  # Checkpoint data
    HEARTBEAT = "heartbeat"  # Heartbeat/keep-alive
    ERROR = "error"  # Error messages
    JOB_SUBMISSION = "job_submission"  # Job submission from client
    JOB_RESULT = "job_result"  # Job result to client


class EndpointType(Enum):
    """Types of communication endpoints"""
    CLIENT = "client"
    FOREMAN = "foreman"
    WORKER = "worker"


@dataclass
class MessageMetrics:
    """Metrics for a single message"""
    message_id: str
    category: MessageCategory
    
    # Endpoints
    source: EndpointType
    destination: EndpointType
    source_id: Optional[str] = None
    destination_id: Optional[str] = None
    
    # Size metrics (bytes)
    payload_size: int = 0
    total_size: int = 0  # Including headers/protocol overhead
    compressed_size: Optional[int] = None
    
    # Timing
    sent_at: Optional[float] = None
    received_at: Optional[float] = None
    latency: Optional[float] = None  # One-way latency
    
    # Related entities
    task_id: Optional[str] = None
    job_id: Optional[str] = None
    
    def calculate_latency(self) -> None:
        """Calculate latency if both timestamps available"""
        if self.sent_at and self.received_at:
            self.latency = self.received_at - self.sent_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "category": self.category.value,
            "source": self.source.value,
            "destination": self.destination.value,
            "payload_size": self.payload_size,
            "total_size": self.total_size,
            "latency": self.latency,
            "task_id": self.task_id,
            "job_id": self.job_id,
        }


@dataclass
class ConnectionMetrics:
    """Metrics for a single connection"""
    connection_id: str
    endpoint_type: EndpointType
    endpoint_id: str
    
    # Connection timing
    connected_at: float = field(default_factory=time.time)
    disconnected_at: Optional[float] = None
    
    # Message counts
    messages_sent: int = 0
    messages_received: int = 0
    
    # Data transfer (bytes)
    bytes_sent: int = 0
    bytes_received: int = 0
    
    # Latency tracking
    latency_samples: List[float] = field(default_factory=list)
    avg_latency: Optional[float] = None
    min_latency: Optional[float] = None
    max_latency: Optional[float] = None
    
    # Error tracking
    errors: int = 0
    reconnects: int = 0
    
    def add_latency_sample(self, latency: float) -> None:
        """Add a latency sample and update statistics"""
        self.latency_samples.append(latency)
        if self.latency_samples:
            self.avg_latency = statistics.mean(self.latency_samples)
            self.min_latency = min(self.latency_samples)
            self.max_latency = max(self.latency_samples)
    
    def get_connection_duration(self) -> float:
        """Get connection duration in seconds"""
        end_time = self.disconnected_at or time.time()
        return end_time - self.connected_at
    
    def get_bandwidth(self) -> Tuple[float, float]:
        """Get send/receive bandwidth in bytes/second"""
        duration = self.get_connection_duration()
        if duration <= 0:
            return 0.0, 0.0
        return self.bytes_sent / duration, self.bytes_received / duration
    
    def to_dict(self) -> Dict[str, Any]:
        send_bw, recv_bw = self.get_bandwidth()
        return {
            "connection_id": self.connection_id,
            "endpoint_type": self.endpoint_type.value,
            "endpoint_id": self.endpoint_id,
            "duration_seconds": self.get_connection_duration(),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "send_bandwidth_bps": send_bw,
            "recv_bandwidth_bps": recv_bw,
            "avg_latency": self.avg_latency,
            "errors": self.errors,
        }


@dataclass
class CommunicationMetrics:
    """Aggregate communication metrics"""
    # Message statistics
    total_messages: int = 0
    messages_by_category: Dict[str, int] = field(default_factory=dict)
    
    # Data transfer
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_payload_bytes: int = 0
    
    # Protocol overhead
    protocol_overhead_bytes: int = 0
    protocol_overhead_percent: float = 0.0
    
    # Latency statistics
    avg_latency: Optional[float] = None
    min_latency: Optional[float] = None
    max_latency: Optional[float] = None
    p95_latency: Optional[float] = None
    p99_latency: Optional[float] = None
    
    # Bandwidth
    avg_send_bandwidth: float = 0.0  # bytes/second
    avg_recv_bandwidth: float = 0.0
    peak_bandwidth: float = 0.0
    
    # Connection statistics
    total_connections: int = 0
    active_connections: int = 0
    total_reconnects: int = 0
    total_errors: int = 0
    
    # Efficiency metrics
    useful_data_ratio: float = 0.0  # payload / total bytes
    message_rate: float = 0.0  # messages per second
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CommunicationTracker:
    """
    Tracks communication overhead and network metrics.
    
    Provides:
    - Message size tracking
    - Latency measurement
    - Bandwidth monitoring
    - Protocol overhead analysis
    - Per-connection statistics
    """
    
    def __init__(self, latency_window_size: int = 1000):
        """
        Initialize communication tracker.
        
        Args:
            latency_window_size: Number of latency samples to keep for statistics
        """
        self.latency_window_size = latency_window_size
        
        # Message tracking: message_id -> MessageMetrics
        self._messages: Dict[str, MessageMetrics] = {}
        
        # Connection tracking: connection_id -> ConnectionMetrics
        self._connections: Dict[str, ConnectionMetrics] = {}
        
        # Latency samples (rolling window)
        self._latency_samples: List[float] = []
        
        # Bandwidth tracking (time series)
        self._bandwidth_samples: List[Tuple[float, int, int]] = []  # (timestamp, sent, received)
        
        # Message counters
        self._message_counter: int = 0
        self._connection_counter: int = 0
        
        # Tracking state
        self._tracking_start_time: float = time.time()
        
        # Lock for thread-safe access
        self._lock = asyncio.Lock()
        
        # Pending messages awaiting response (for round-trip tracking)
        self._pending_messages: Dict[str, float] = {}  # message_id -> sent_time
    
    # ==================== Connection Tracking ====================
    
    async def register_connection(
        self,
        endpoint_type: EndpointType,
        endpoint_id: str
    ) -> str:
        """
        Register a new connection.
        
        Returns:
            Connection ID
        """
        async with self._lock:
            self._connection_counter += 1
            connection_id = f"conn_{endpoint_id}_{self._connection_counter}"
            
            self._connections[connection_id] = ConnectionMetrics(
                connection_id=connection_id,
                endpoint_type=endpoint_type,
                endpoint_id=endpoint_id,
            )
            
            return connection_id
    
    async def close_connection(self, connection_id: str) -> Optional[ConnectionMetrics]:
        """Mark a connection as closed"""
        async with self._lock:
            if connection_id in self._connections:
                conn = self._connections[connection_id]
                conn.disconnected_at = time.time()
                return conn
            return None
    
    async def record_reconnect(self, connection_id: str) -> None:
        """Record a reconnection event"""
        async with self._lock:
            if connection_id in self._connections:
                self._connections[connection_id].reconnects += 1
    
    # ==================== Message Tracking ====================
    
    async def record_message_sent(
        self,
        category: MessageCategory,
        source: EndpointType,
        destination: EndpointType,
        payload_size: int,
        total_size: Optional[int] = None,
        source_id: Optional[str] = None,
        destination_id: Optional[str] = None,
        task_id: Optional[str] = None,
        job_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> str:
        """
        Record an outgoing message.
        
        Returns:
            Message ID for tracking response
        """
        async with self._lock:
            self._message_counter += 1
            message_id = f"msg_{self._message_counter}_{int(time.time() * 1000)}"
            
            actual_total = total_size or int(payload_size * 1.1)  # Estimate 10% overhead
            
            metrics = MessageMetrics(
                message_id=message_id,
                category=category,
                source=source,
                destination=destination,
                source_id=source_id,
                destination_id=destination_id,
                payload_size=payload_size,
                total_size=actual_total,
                sent_at=time.time(),
                task_id=task_id,
                job_id=job_id,
            )
            
            self._messages[message_id] = metrics
            self._pending_messages[message_id] = time.time()
            
            # Update connection metrics
            if connection_id and connection_id in self._connections:
                conn = self._connections[connection_id]
                conn.messages_sent += 1
                conn.bytes_sent += actual_total
            
            # Record bandwidth sample
            self._bandwidth_samples.append((time.time(), actual_total, 0))
            
            return message_id
    
    async def record_message_received(
        self,
        category: MessageCategory,
        source: EndpointType,
        destination: EndpointType,
        payload_size: int,
        total_size: Optional[int] = None,
        source_id: Optional[str] = None,
        destination_id: Optional[str] = None,
        task_id: Optional[str] = None,
        job_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        response_to: Optional[str] = None
    ) -> str:
        """
        Record an incoming message.
        
        Args:
            response_to: Message ID this is responding to (for latency calc)
        
        Returns:
            Message ID
        """
        async with self._lock:
            self._message_counter += 1
            message_id = f"msg_{self._message_counter}_{int(time.time() * 1000)}"
            
            actual_total = total_size or int(payload_size * 1.1)
            now = time.time()
            
            metrics = MessageMetrics(
                message_id=message_id,
                category=category,
                source=source,
                destination=destination,
                source_id=source_id,
                destination_id=destination_id,
                payload_size=payload_size,
                total_size=actual_total,
                received_at=now,
                task_id=task_id,
                job_id=job_id,
            )
            
            self._messages[message_id] = metrics
            
            # Calculate round-trip latency if this is a response
            if response_to and response_to in self._pending_messages:
                sent_time = self._pending_messages.pop(response_to)
                latency = (now - sent_time) / 2  # Half round-trip
                metrics.latency = latency
                self._add_latency_sample(latency)
                
                if connection_id and connection_id in self._connections:
                    self._connections[connection_id].add_latency_sample(latency)
            
            # Update connection metrics
            if connection_id and connection_id in self._connections:
                conn = self._connections[connection_id]
                conn.messages_received += 1
                conn.bytes_received += actual_total
            
            # Record bandwidth sample
            self._bandwidth_samples.append((now, 0, actual_total))
            
            return message_id
    
    def _add_latency_sample(self, latency: float) -> None:
        """Add latency sample to rolling window"""
        self._latency_samples.append(latency)
        if len(self._latency_samples) > self.latency_window_size:
            self._latency_samples.pop(0)
    
    async def record_error(self, connection_id: str) -> None:
        """Record a communication error"""
        async with self._lock:
            if connection_id in self._connections:
                self._connections[connection_id].errors += 1
    
    # ==================== Metrics Retrieval ====================
    
    async def get_communication_metrics(self) -> CommunicationMetrics:
        """Get aggregate communication metrics"""
        async with self._lock:
            metrics = CommunicationMetrics()
            
            # Message statistics
            for msg in self._messages.values():
                metrics.total_messages += 1
                cat = msg.category.value
                metrics.messages_by_category[cat] = (
                    metrics.messages_by_category.get(cat, 0) + 1
                )
                
                if msg.sent_at:
                    metrics.total_bytes_sent += msg.total_size
                if msg.received_at:
                    metrics.total_bytes_received += msg.total_size
                
                metrics.total_payload_bytes += msg.payload_size
            
            # Protocol overhead
            total_bytes = metrics.total_bytes_sent + metrics.total_bytes_received
            if total_bytes > 0:
                metrics.protocol_overhead_bytes = total_bytes - metrics.total_payload_bytes
                metrics.protocol_overhead_percent = (
                    metrics.protocol_overhead_bytes / total_bytes * 100
                )
                metrics.useful_data_ratio = metrics.total_payload_bytes / total_bytes
            
            # Latency statistics
            if self._latency_samples:
                metrics.avg_latency = statistics.mean(self._latency_samples)
                metrics.min_latency = min(self._latency_samples)
                metrics.max_latency = max(self._latency_samples)
                
                sorted_latency = sorted(self._latency_samples)
                n = len(sorted_latency)
                if n > 0:
                    metrics.p95_latency = sorted_latency[int(n * 0.95)]
                    metrics.p99_latency = sorted_latency[int(n * 0.99)]
            
            # Connection statistics
            metrics.total_connections = len(self._connections)
            metrics.active_connections = sum(
                1 for c in self._connections.values()
                if c.disconnected_at is None
            )
            metrics.total_reconnects = sum(
                c.reconnects for c in self._connections.values()
            )
            metrics.total_errors = sum(
                c.errors for c in self._connections.values()
            )
            
            # Bandwidth
            duration = time.time() - self._tracking_start_time
            if duration > 0:
                metrics.avg_send_bandwidth = metrics.total_bytes_sent / duration
                metrics.avg_recv_bandwidth = metrics.total_bytes_received / duration
                metrics.message_rate = metrics.total_messages / duration
            
            return metrics
    
    async def get_connection_metrics(
        self,
        connection_id: str
    ) -> Optional[ConnectionMetrics]:
        """Get metrics for a specific connection"""
        async with self._lock:
            return self._connections.get(connection_id)
    
    async def get_all_connection_metrics(self) -> Dict[str, ConnectionMetrics]:
        """Get metrics for all connections"""
        async with self._lock:
            return dict(self._connections)
    
    async def get_messages_by_category(
        self,
        category: MessageCategory
    ) -> List[MessageMetrics]:
        """Get all messages of a specific category"""
        async with self._lock:
            return [
                msg for msg in self._messages.values()
                if msg.category == category
            ]
    
    async def get_bandwidth_history(
        self,
        window_seconds: float = 60.0
    ) -> List[Dict[str, Any]]:
        """Get bandwidth history for the last N seconds"""
        async with self._lock:
            cutoff = time.time() - window_seconds
            history = []
            
            # Aggregate by second
            by_second: Dict[int, Dict[str, int]] = {}
            for ts, sent, recv in self._bandwidth_samples:
                if ts >= cutoff:
                    second = int(ts)
                    if second not in by_second:
                        by_second[second] = {"sent": 0, "received": 0}
                    by_second[second]["sent"] += sent
                    by_second[second]["received"] += recv
            
            for second, data in sorted(by_second.items()):
                history.append({
                    "timestamp": second,
                    "bytes_sent": data["sent"],
                    "bytes_received": data["received"],
                    "total_bandwidth": data["sent"] + data["received"],
                })
            
            return history
    
    async def get_overhead_analysis(self) -> Dict[str, Any]:
        """Analyze protocol overhead by message category"""
        async with self._lock:
            by_category: Dict[str, Dict[str, int]] = {}
            
            for msg in self._messages.values():
                cat = msg.category.value
                if cat not in by_category:
                    by_category[cat] = {
                        "count": 0,
                        "payload_bytes": 0,
                        "total_bytes": 0,
                    }
                
                by_category[cat]["count"] += 1
                by_category[cat]["payload_bytes"] += msg.payload_size
                by_category[cat]["total_bytes"] += msg.total_size
            
            analysis = {}
            for cat, data in by_category.items():
                overhead = data["total_bytes"] - data["payload_bytes"]
                overhead_pct = (overhead / data["total_bytes"] * 100) if data["total_bytes"] > 0 else 0
                
                analysis[cat] = {
                    "message_count": data["count"],
                    "payload_bytes": data["payload_bytes"],
                    "total_bytes": data["total_bytes"],
                    "overhead_bytes": overhead,
                    "overhead_percent": overhead_pct,
                    "avg_message_size": data["total_bytes"] / data["count"] if data["count"] > 0 else 0,
                }
            
            return analysis
    
    # ==================== Export ====================
    
    async def export_report(self, filepath: str) -> None:
        """Export communication analysis report to file"""
        metrics = await self.get_communication_metrics()
        overhead = await self.get_overhead_analysis()
        bandwidth = await self.get_bandwidth_history(window_seconds=300)
        
        async with self._lock:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "tracking_duration_seconds": time.time() - self._tracking_start_time,
                "aggregate_metrics": metrics.to_dict(),
                "overhead_by_category": overhead,
                "bandwidth_history_5min": bandwidth,
                "connections": {
                    conn_id: conn.to_dict()
                    for conn_id, conn in self._connections.items()
                },
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"CommunicationTracker: Exported report to {filepath}")
    
    async def reset(self) -> None:
        """Reset all tracking data"""
        async with self._lock:
            self._messages.clear()
            self._connections.clear()
            self._latency_samples.clear()
            self._bandwidth_samples.clear()
            self._pending_messages.clear()
            self._message_counter = 0
            self._connection_counter = 0
            self._tracking_start_time = time.time()
        print("CommunicationTracker: Reset all tracking data")
