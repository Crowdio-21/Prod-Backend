# CROWDio DNN Partitioning Guide

This document explains how the existing DNN partitioning workflow works in
this repository and how the sentiment ONNX example uses it.

## 1. Why partition a model?

A complete neural network can be too large or too expensive for one device.
DNN partitioning splits the model into several executable stages. Different
workers can run the stages, allowing the foreman to distribute work across
multiple PCs.

For the sentiment example, DistilBERT is split into three ONNX files:

```text
input_ids
   |
   v
cell_a: embeddings + first transformer layers
   |
   v
cell_b: remaining transformer layers
   |
   v
cell_c: classification head
   |
   v
sentiment logits
```

## 2. Sentiment partition contract

The exporter creates the following interfaces:

```text
cell_a
  input:  input_ids   int64   [batch, sequence_length]
  output: features_a  float32 [batch, sequence_length, hidden_size + 1]

cell_b
  input:  features_a  float32 [batch, sequence_length, hidden_size + 1]
  output: features_b  float32 [batch, sequence_length, hidden_size]

cell_c
  input:  features_b  float32 [batch, sequence_length, hidden_size]
  output: logits      float32 [batch, number_of_labels]
```

The final channel of `features_a` stores the attention mask. This lets
`cell_b` reconstruct padding information without sending a second tensor
through the pipeline.

## 3. Exporting the model

The exporter downloads the pretrained SST-2 DistilBERT model, builds the three
PyTorch partition modules, checks the PyTorch chain, and writes ONNX files.

```powershell
.\.venv\Scripts\Activate.ps1

python tests\dnn_partitioning\sentiment_onnx_pipeline\export_partitions.py
```

Output directory:

```text
tools\onnx_partitions_sentiment\
  cell_a.onnx
  cell_b.onnx
  cell_c.onnx
  tokenizer\
  partition_metadata.json
```

`partition_metadata.json` records the split index, tensor shapes, labels, and
verification results.

## 4. Dataset preparation

The example uses the public SST-2 validation dataset. It creates exactly 700
labeled sentences in `data\sentences.csv`:

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\prepare_sst2_dataset.py `
  --limit 700
```

The generated CSV has this structure:

```csv
text,label
The product worked perfectly.,1
The delivery arrived late.,0
```

The label is useful for later accuracy calculations. The current pipeline
client uses the `text` column to create model inputs.

## 5. How one input travels

For each sentence:

1. The client tokenizes the sentence into an `int64` token tensor.
2. The tensor is serialized with `tensor_transport`.
3. Stage A runs `cell_a.onnx` on a worker.
4. The resulting tensor is passed to stage B.
5. Stage B runs `cell_b.onnx` on a worker.
6. The resulting tensor is passed to stage C.
7. Stage C runs `cell_c.onnx` and returns sentiment logits.
8. The client applies softmax and prints the prediction.

The intermediate tensors are serialized so they can move between workers over
the foreman connection.

## 6. Running with eight workers

Start the foreman first:

```powershell
python tests\run_foreman_simple.py
```

Start one worker process on each PC. Every worker must have:

- Python and the worker dependencies installed
- access to the ONNX files at the configured model paths
- ONNX Runtime installed
- network access to the foreman WebSocket port `9000`

Example worker command:

```powershell
python -m pc_worker.main `
  --worker-id pc-worker-01 `
  --foreman-url ws://192.168.1.100:9000 `
  --api-port 8001 `
  --max-concurrent-tasks 1
```

Use unique IDs from `pc-worker-01` through `pc-worker-08`.

## 7. Why only eight tasks appear at once

With eight workers and `--max-concurrent-tasks 1`, the scheduler can execute
at most eight tasks simultaneously:

```text
700 input tasks submitted
        |
        +-- 8 tasks running on 8 workers
        +-- remaining tasks waiting in the foreman queue
        |
        +-- completed worker becomes available
        +-- next queued task is dispatched
```

Therefore, seeing task indexes such as `0_0`, `0_1`, `0_10`, and `0_100` does
not mean that only eight sentences were submitted. It means eight tasks are
currently active or are the first tasks visible in the worker logs. The rest
are queued and should be dispatched as workers complete.

Increase worker concurrency only if each PC has enough CPU and memory:

```powershell
python -m pc_worker.main `
  --worker-id pc-worker-01 `
  --foreman-url ws://192.168.1.100:9000 `
  --api-port 8001 `
  --max-concurrent-tasks 2
```

Eight workers with concurrency two can run up to sixteen tasks at once, but
this also increases memory use and ONNX Runtime CPU contention.

## 8. Run 700 sentences

After the foreman and workers are connected:

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\multi_input_test.py `
  --dataset data\sentences.csv `
  --text-column text `
  --limit 700
```

The client submits three pipeline stages. In the currently installed SDK,
the pipeline API uses serialized Python functions that invoke ONNX Runtime.
The foreman creates one task per input per stage, so 700 sentences create:

```text
Stage A: 700 tasks
Stage B: 700 tasks
Stage C: 700 tasks
Total:   2,100 tasks
```

The current SDK uses stage barriers: stage B starts after stage A completes,
and stage C starts after stage B completes. The worker pool still parallelizes
tasks within each stage.

## 9. Important deployment detail

The model path sent to a worker must exist on that worker. For eight physical
PCs, copy the generated directory to the same path on every PC, or configure
the client to send paths that are valid on each worker.

A shared network path can also be used if every worker has access:

```text
\\FOREMAN-PC\crowdio-models\onnx_partitions_sentiment\cell_a.onnx
```

Using a local copy on each PC is generally faster and avoids repeated network
file access during inference.

## 10. Common log messages

### `name 'deserialize_tensor' is not defined`

The serialized worker function did not include its serializer import. The
current example imports worker dependencies inside the function so each worker
can execute it independently.

### `KeyError: 'model_path'`

The SDK wraps downstream pipeline arguments inside `original_args`. The
current example unwraps that structure before reading the model path.

### Only a few tasks are visible

This is normal when the worker pool is busy. The number of active tasks is
approximately:

```text
worker_count × max_concurrent_tasks
```

Check the final client output and completed-task messages rather than counting
only the first worker log lines.

### Connection or stalled job

Make sure:

- the foreman is running on port `9000`
- all workers point to the correct foreman IP
- Windows Firewall allows TCP port `9000`
- all workers have the ONNX files
- all workers have `onnxruntime` installed
- old failed jobs are not being mistaken for a new submission
