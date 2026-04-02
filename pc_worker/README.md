# PC Worker Package

A modular FastAPI-based worker service for the CrowdCompute distributed computing system.

## Overview

This package provides FastAPI-based worker servers that connect to the foreman, receive task assignments, execute Python functions, and return results. Workers can run on any device with Python support.

## Project Structure

```
pc_worker/
├── __init__.py              # Package initialization
├── config.py                # Configuration models
├── main.py                  # Main entry point
├── README.md                # This file
├── core/                    # Core worker functionality
│   ├── __init__.py
│   ├── app.py              # FastAPI application factory
│   └── worker.py           # Worker implementation
├── api/                     # API routes and endpoints
│   ├── __init__.py
│   ├── routes.py           # REST API routes
│   └── dashboard.py        # Web dashboard
└── schema/                  # Data models and schemas
    ├── __init__.py
    └── models.py           # Pydantic models
```

## Modules

### Core (`core/`)

#### `worker.py`
Main worker implementation with task execution engine.

**Key Classes:**
- `FastAPIWorker` - Main worker class with FastAPI integration

**Key Features:**
- WebSocket client for foreman communication
- Task execution engine with error handling
- Statistics tracking and monitoring
- Automatic reconnection and recovery
- Reports cached model partitions on connect (WORKER_READY) for model-affinity scheduling
- Supports from_cache model loading — skips download when model already exists locally

**Key Methods:**
- `connect()` - Connect to foreman
- `disconnect()` - Disconnect from foreman
- `handle_message()` - Process incoming messages
- `_handle_task_assignment()` - Handle task assignments
- `_execute_task()` - Execute assigned tasks
- `listen_for_tasks()` - Listen for task assignments
- `heartbeat()` - Send periodic heartbeats

#### `app.py`
FastAPI application factory for creating and configuring the web server.

### API (`api/`)

#### `routes.py`
REST API endpoints for worker monitoring and control.

**Endpoints:**
- `GET /` - Worker status
- `GET /stats` - Worker statistics
- `POST /restart` - Restart worker connection
- `WS /ws` - WebSocket for real-time updates

#### `dashboard.py`
HTML dashboard for worker monitoring and control.

**Features:**
- Real-time worker status display
- Task execution statistics
- Connection status monitoring
- Performance metrics
- WebSocket-based live updates

### Schema (`schema/`)

#### `models.py`
Pydantic models for data validation and serialization.

**Models:**
- `TaskResult` - Task execution result

### Configuration (`config.py`)

**Models:**
- `WorkerConfig` - Worker runtime configuration

## Usage

### Quick Start

Run a worker using the main entry point:

```bash
# With auto-generated worker ID
python -m pc_worker.main --foreman-url ws://localhost:9000 --api-port 8001

# With custom worker ID
python -m pc_worker.main --worker-id my-worker-01 --api-port 8001

# View all options
python -m pc_worker.main --help
```

### Using the Test Script

```bash
# Run a simple worker
python tests/run_worker_simple.py

# Run on a specific port
python tests/run_worker_simple.py -p 8002
```

### Programmatic Usage

```python
from pc_worker import FastAPIWorker, WorkerConfig

config = WorkerConfig(
    worker_id="worker-001",
    foreman_url="ws://localhost:9000",
    max_concurrent_tasks=1,
    auto_restart=True,
    heartbeat_interval=30,
    api_host="0.0.0.0",
    api_port=8001
)

worker = FastAPIWorker(config)
worker.run()  # Starts FastAPI server with worker
```

### Worker Configuration Options

```python
from pc_worker import WorkerConfig

config = WorkerConfig(
    worker_id="unique-worker-id",              # Unique identifier
    foreman_url="ws://192.168.1.100:9000",    # Foreman WebSocket URL
    max_concurrent_tasks=2,                    # Max concurrent tasks
    auto_restart=True,                         # Auto-restart on failure
    heartbeat_interval=30,                     # Heartbeat frequency (seconds)
    api_host="0.0.0.0",                       # API server host
    api_port=8001                              # API server port
)
```

### API Usage

```python
import requests

# Get worker status
response = requests.get("http://localhost:8001/")
status = response.json()
print(f"Worker ID: {status['worker_id']}")
print(f"Status: {status['status']}")
print(f"Current task: {status['current_task']}")

# Get worker statistics
response = requests.get("http://localhost:8001/stats")
stats = response.json()
print(f"Tasks completed: {stats['stats']['tasks_completed']}")
print(f"Tasks failed: {stats['stats']['tasks_failed']}")
print(f"Total execution time: {stats['stats']['total_execution_time']}")

# Restart worker
response = requests.post("http://localhost:8001/restart")
print("Worker restarted")
```

### Dashboard Access

Open your browser to view the real-time dashboard:
```
http://localhost:8001/dashboard
```

## Features

### Task Execution
- **Function Deserialization** - Deserialize functions from cloudpickle
- **Safe Execution** - Execute functions in controlled environment
- **Argument Handling** - Support for various argument formats
- **Error Handling** - Capture and report execution errors
- **Result Serialization** - Serialize results for transmission

### Connection Management
- **WebSocket Client** - Connect to foreman via WebSocket
- **Automatic Reconnection** - Reconnect on connection loss
- **Heartbeat Monitoring** - Send periodic heartbeats
- **Connection Recovery** - Handle network interruptions

### Monitoring and Control
- **FastAPI Server** - HTTP API for monitoring
- **Real-time Dashboard** - Web-based monitoring interface
- **Statistics Tracking** - Track performance metrics
- **Status Reporting** - Report current status and health

### Error Handling
- **Task Failures** - Handle task execution errors
- **Network Issues** - Handle connection problems
- **Resource Limits** - Respect system resource limits
- **Graceful Degradation** - Continue operation despite errors

## Architecture

```
Foreman → WebSocket → Worker → Task Execution
                ↓
            FastAPI Server → Dashboard
```

1. **Connection** - Worker connects to foreman via WebSocket
2. **Registration** - Worker announces availability
3. **Task Assignment** - Foreman assigns tasks to worker
4. **Task Execution** - Worker executes assigned tasks
5. **Result Return** - Results sent back to foreman
6. **Monitoring** - Status available via HTTP API

## Task Execution Flow

```python
# 1. Receive task assignment
task_message = {
    "type": "assign_task",
    "data": {
        "func_code": "serialized_function_hex",
        "task_args": [1, 2, 3],
        "task_id": "task-123"
    },
    "job_id": "job-456"
}

# 2. Deserialize function
func = deserialize_function_for_PC(func_code)

# 3. Execute function
result = func(*task_args)

# 4. Send result
result_message = {
    "type": "task_result",
    "data": {
        "task_id": "task-123",
        "result": result
    },
    "job_id": "job-456"
}
```

## Configuration Options

### Worker Settings
- **Worker ID** - Unique identifier for the worker
- **Foreman URL** - WebSocket URL of the foreman
- **Max Concurrent Tasks** - Maximum tasks to execute simultaneously
- **Auto Restart** - Automatically restart on failure
- **Heartbeat Interval** - Frequency of heartbeat messages (seconds)

### Network Settings
- **API Host** - Host for the FastAPI server (default: 0.0.0.0)
- **API Port** - Port for monitoring API (default: 8001)
- **Foreman URL** - WebSocket URL of foreman (default: ws://localhost:9000)

## Monitoring

### Dashboard Features
- **Real-time Status** - Current worker status
- **Task Statistics** - Completed and failed tasks
- **Performance Metrics** - Execution time and throughput
- **Connection Status** - Foreman connection status
- **Control Actions** - Restart and disconnect controls

### API Endpoints
- `GET /` - Worker status and configuration
- `GET /stats` - Detailed statistics
- `POST /restart` - Restart worker connection
- `GET /dashboard` - Web dashboard
- `WS /ws` - Real-time status updates via WebSocket

### API Documentation
FastAPI automatically generates interactive documentation:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Development

### Running Tests

```bash
# Run foreman
python tests/run_foreman_simple.py

# Run worker
python tests/run_worker_simple.py -p 8001

# Run example client
python tests/example_client.py localhost
```

### Project Guidelines

- **Modularity** - Keep components separated by responsibility
- **Type Safety** - Use Pydantic models for data validation
- **Error Handling** - Handle all exceptions gracefully
- **Logging** - Use clear, informative log messages
- **Documentation** - Document all public APIs

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `websockets` - WebSocket client
- `pydantic` - Data validation
- `cloudpickle` - Function serialization (via common module)

## License

Part of the CrowdCompute distributed computing system.
