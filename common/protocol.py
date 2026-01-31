"""
Network protocol definitions for CrowdCompute
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MessageType(Enum):
    """Message types for WebSocket communication"""
    # Client -> Foreman
    SUBMIT_JOB = "submit_job"
    GET_RESULTS = "get_results"
    DISCONNECT = "disconnect"
    
    # Foreman -> Worker
    ASSIGN_TASK = "assign_task"
    PING = "ping"
    RESUME_TASK = "resume_task"
    
    # Worker -> Foreman
    TASK_RESULT = "task_result"
    TASK_ERROR = "task_error"
    WORKER_READY = "worker_ready"
    WORKER_HEARTBEAT = "worker_heartbeat"
    PONG = "pong"
    TASK_CHECKPOINT = "task_checkpoint"
    
    # Foreman -> Client
    JOB_RESULTS = "job_results"
    JOB_ERROR = "job_error"
    JOB_ACCEPTED = "job_accepted"
    
    # Checkpoint Acks
    CHECKPOINT_ACK = "checkpoint_ack"


class Message:
    """Base message class"""
    
    def __init__(self, msg_type: MessageType, data: Dict[str, Any], job_id: Optional[str] = None):
        self.type = msg_type
        self.data = data
        self.job_id = job_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
            "job_id": self.job_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            msg_type=MessageType(data["type"]),
            data=data["data"],
            job_id=data.get("job_id")
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        return cls.from_dict(json.loads(json_str))


# Message factory functions
def create_submit_job_message(func_code: str, args_list: List[Any], job_id: str) -> Message:
    """Create a job submission message"""
    return Message(
        msg_type=MessageType.SUBMIT_JOB,
        data={
            "func_code": func_code,
            "args_list": args_list,
            "total_tasks": len(args_list)
        },
        job_id=job_id
    )


def create_job_accepted_message(job_id: str) -> Message:
    """Create a job accepted message"""
    return Message(
        MessageType.JOB_ACCEPTED,
        {"job_id": job_id},
        job_id
    )

def create_assign_task_message(func_code: str, task_args: List[Any], task_id: str, job_id: str) -> Message:
    """Create a task assignment message"""
    return Message(
        msg_type=MessageType.ASSIGN_TASK,
        data={
            "func_code": func_code,
            "task_args": task_args,
            "task_id": task_id
        },
        job_id=job_id
    )


def create_task_result_message(result: Any, task_id: str, job_id: str) -> Message:
    """Create a task result message"""
    return Message(
        msg_type=MessageType.TASK_RESULT,
        data={
            "result": result,
            "task_id": task_id
        },
        job_id=job_id
    )


def create_task_error_message(error: str, task_id: str, job_id: str) -> Message:
    """Create a task error message"""
    return Message(
        msg_type=MessageType.TASK_ERROR,
        data={
            "error": error,
            "task_id": task_id
        },
        job_id=job_id
    )


def create_job_results_message(results: List[Any], job_id: str) -> Message:
    """Create a job results message"""
    return Message(
        msg_type=MessageType.JOB_RESULTS,
        data={"results": results},
        job_id=job_id
    )


def create_worker_ready_message(worker_id: str) -> Message:
    """Create a worker ready message"""
    return Message(
        msg_type=MessageType.WORKER_READY,
        data={"worker_id": worker_id}
    )


def create_ping_message() -> Message:
    """Create a ping message"""
    return Message(
        msg_type=MessageType.PING,
        data={}
    )


def create_pong_message() -> Message:
    """Create a pong message"""
    return Message(
        msg_type=MessageType.PONG,
        data={}
    )


def create_task_checkpoint_message(
    task_id: str, 
    job_id: str, 
    is_base: bool, 
    delta_data_hex: str, 
    progress_percent: float, 
    checkpoint_id: int,
    compression_type: str = "gzip"
) -> Message:
    """Create a task checkpoint message from worker to foreman
    
    Args:
        task_id: Task identifier
        job_id: Job identifier
        is_base: True if this is the base checkpoint, False if delta
        delta_data_hex: Hex-encoded checkpoint data (compressed)
        progress_percent: Task progress as percentage (0-100)
        checkpoint_id: Sequential checkpoint number
        compression_type: Type of compression applied (gzip, zstd, etc)
    """
    return Message(
        msg_type=MessageType.TASK_CHECKPOINT,
        data={
            "task_id": task_id,
            "is_base": is_base,
            "delta_data_hex": delta_data_hex,
            "progress_percent": progress_percent,
            "checkpoint_id": checkpoint_id,
            "compression_type": compression_type
        },
        job_id=job_id
    )


def create_checkpoint_ack_message(task_id: str, job_id: str, checkpoint_id: int) -> Message:
    """Create a checkpoint acknowledgment message from foreman to worker"""
    return Message(
        msg_type=MessageType.CHECKPOINT_ACK,
        data={
            "task_id": task_id,
            "checkpoint_id": checkpoint_id
        },
        job_id=job_id
    )


def create_resume_task_message(
    task_id: str, 
    job_id: str, 
    func_code: str,
    checkpoint_state: dict,
    task_args: List[Any] = None,
    task_kwargs: Dict[str, Any] = None,
    progress_percent: float = 0.0,
    checkpoint_count: int = 0
) -> Message:
    """Create a task resumption message from foreman to worker
    
    Sends checkpoint state to worker so it can continue from where 
    the previous worker left off.
    
    Args:
        task_id: Task identifier
        job_id: Job identifier
        func_code: Function code to execute
        checkpoint_state: Dictionary containing the saved state from last checkpoint
        task_args: Original task arguments (required for function execution)
        task_kwargs: Original task keyword arguments
        progress_percent: Progress percentage at last checkpoint
        checkpoint_count: Total checkpoints available for this task
    """
    return Message(
        msg_type=MessageType.RESUME_TASK,
        data={
            "task_id": task_id,
            "func_code": func_code,
            "checkpoint_state": checkpoint_state,
            "task_args": task_args or [],
            "task_kwargs": task_kwargs or {},
            "progress_percent": progress_percent,
            "checkpoint_count": checkpoint_count
        },
        job_id=job_id
    )

