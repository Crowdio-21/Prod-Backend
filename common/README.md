# common package

Shared runtime primitives used by SDK, foreman, and workers.

This folder is the protocol and execution glue for CrowdIO:
- typed websocket message envelope + factories
- function source serialization for remote execution
- device capability and hardware introspection
- AST-based instrumentation for mobile runtimes that do not support sys.settrace

## Files

### protocol.py
Defines network message types and factories.

Main symbols:
- MessageType: enum of all protocol message names
- CheckpointType: checkpoint kind enum (base, delta, compacted)
- RecoveryStatus: task recovery status enum
- Message: envelope with to_dict/from_dict and to_json/from_json helpers

Representative factory helpers:
- create_submit_job_message
- create_submit_job_message_with_metadata
- create_submit_pipeline_job_message
- create_assign_task_message
- create_assign_task_message_with_metadata
- create_resume_task_message
- create_resume_task_message_with_metadata
- create_task_checkpoint_message
- create_task_checkpoint_message_extended
- create_task_progress_message
- create_job_progress_message
- DNN orchestration helpers:
    - create_load_model_message
    - create_device_topology_message
    - create_topology_update_message
    - create_intermediate_feature_message
    - create_aggregation_config_message
    - create_fallback_decision_message

### serializer.py
Function and payload serialization helpers used for shipping executable logic and tensor data to workers.

Main functions:
- **Function serialization:**
  - serialize_function: gets source and strips decorators before transport
  - deserialize_function_for_PC: exec-based restoration of callable function code
- **Tensor serialization:**
  - serialize_tensor: numpy tensor to compressed transport payload
  - deserialize_tensor: transport payload back to numpy tensor
- **Tensor-aware feature payload encoding (for DNN pipelines):**
  - encode_feature_payload: recursively encodes payloads with numpy arrays as zlib-compressed transport dicts
  - decode_feature_payload: recursively decodes transport dicts and materializes numpy arrays
- **Utilities:**
  - get_runtime_info: lightweight runtime diagnostic string
  - hex_to_bytes / bytes_to_hex: utility conversion helpers

Note:
- Tensor encoding supports nested structures (dicts, lists) with automatic recursive handling
- Tensor helpers are defined in common.serializer; use this as the canonical import location
- serialize_data and deserialize_data are placeholders in the current implementation.

### device_info.py
Collects runtime device specs and lightweight performance metadata.

Main functions:
- get_device_specs
- get_lightweight_device_specs
- format_device_specs_summary
- get_performance_metrics

### code_instrumenter.py
AST transforms for mobile worker checkpointing and pause/kill control.

Main symbols:
- WorkerType
- detect_worker_capabilities
- CheckpointInstrumenter
- TaskControlInstrumenter
- ResumeInstrumenter
- instrument_for_mobile
- prepare_code_for_mobile_resume
- generate_mobile_checkpoint_wrapper

### __init__.py
Exports selected protocol, serialization, and instrumentation APIs for package-level imports.

## Quick usage

```python
from common.protocol import create_submit_job_message
from common.serializer import serialize_function, deserialize_function_for_PC


def double_value(x):
        return x * 2


func_code = serialize_function(double_value)
msg = create_submit_job_message(func_code, [1, 2, 3], "job-123")
wire_payload = msg.to_json()

restored = deserialize_function_for_PC(func_code)
assert restored(4) == 8
```

## Design notes

- Messages are JSON-friendly for websocket transport.
- Serialization is source-based to keep worker execution flexible.
- Mobile workers use AST instrumentation where runtime tracing is unavailable.
