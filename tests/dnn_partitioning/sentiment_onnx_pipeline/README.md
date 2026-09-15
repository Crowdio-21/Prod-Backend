# Sentiment ONNX DNN Pipeline Tests

This folder contains an end-to-end example for partitioning the pretrained
DistilBERT SST-2 sentiment model into three ONNX stages and running those
stages through the CROWDio DNN pipeline.

## Files

- `export_partitions.py` downloads the Hugging Face model, exports:
  `cell_a.onnx`, `cell_b.onnx`, and `cell_c.onnx`, and writes tokenizer files
  and `partition_metadata.json`.
- `single_input_test.py` sends one tokenized text through all three stages.
- `multi_input_test.py` sends independent inputs through the pipeline in
  streaming mode and prints one classification per input.

The generated artifacts are written to `tools/onnx_partitions_sentiment/` by
default. The client scripts use that same location relative to the repository
root, so they can be launched from any working directory.

## Requirements

Activate the repository virtual environment and install the model/runtime
packages:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install torch transformers onnx onnxruntime
```

The repository must also have the CROWDio SDK and its `common.serializer`
package available. These are normally provided by the existing project setup.

## Export the partitions

Run this once from the repository root:

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\export_partitions.py
```

Useful options include:

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\export_partitions.py `
  --output-dir tools\onnx_partitions_sentiment `
  --split-idx 3 `
  --max-length 64
```

The exporter performs a PyTorch chain check and, when `onnxruntime` is
installed, compares the ONNX pipeline output with the original model output.
Use `--skip-verify` only when runtime verification is unavailable or not
needed.

## Run the pipeline tests

Start the foreman and at least one worker using the project’s normal startup
commands. Then run the single-input smoke test:

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\single_input_test.py
```

Run the streaming multi-input test with two or more workers to observe work
being distributed across inputs and stages. The runner reads up to 700
sentences from a local dataset:

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\multi_input_test.py `
  --dataset data\sentences.csv `
  --text-column text `
  --limit 700
```

Supported dataset formats are `.csv`, `.json`, `.jsonl`, `.ndjson`, and `.txt`.
CSV/JSON records use the `text` field by default. A text file uses one
sentence per line. To process a different number of rows, change `--limit`.

Example CSV:

```text
text
The product worked perfectly.
The delivery arrived late.
```

The command submits 700 independent stage-A inputs. In streaming mode, each
input flows independently through `cell_a`, `cell_b`, and `cell_c`; with eight
connected workers, the foreman can distribute those tasks across the workers.

Both scripts default to `localhost:9000`. Edit the `__main__` call or import
`run(...)` from another test harness to use a different foreman host, port,
model directory, sequence length, or text list.

## Pipeline contract

```text
cell_a: input_ids [batch, seq] int64 -> features_a [batch, seq, hidden + 1] float32
cell_b: features_a                   -> features_b [batch, seq, hidden] float32
cell_c: features_b                   -> logits    [batch, num_labels] float32
```

The final channel emitted by `cell_a` carries the attention mask so `cell_b`
can reconstruct masking without a second pipeline input.
