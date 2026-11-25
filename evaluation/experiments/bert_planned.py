"""Submit BERT-base sentiment inference to a Foreman in planning mode, collect and score the results.

Unlike ``evaluation.reconfig.run live`` this does not run scenario hooks, and it decodes the
returned logits, so the predictions can be checked against the dataset labels.

  python -m evaluation.experiments.bert_planned --sentences 30
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = ROOT / "tools" / "onnx_shards_bert_base" / "tokenizer"
DATASET = ROOT / "data" / "sentences.csv"
MODEL_ID = "textattack/bert-base-uncased-SST-2"


async def run(args) -> int:
    import websockets
    from common.serializer import deserialize_tensor, serialize_tensor
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER))
    rows = list(csv.DictReader(open(DATASET, encoding="utf-8")))[: args.sentences]
    inputs = []
    for row in rows:
        ids = tokenizer(row["text"], return_tensors="np", padding="max_length", truncation=True,
                        max_length=args.max_length)["input_ids"].astype(np.int64)
        inputs.append(serialize_tensor(ids, compression="zlib"))

    async with websockets.connect(f"ws://{args.host}:{args.port}", max_size=None) as ws:
        started = time.monotonic()
        await ws.send(json.dumps({"protocol": "crowdio.reconfiguration", "version": 2, "type": "submit_inference",
                                  "data": {"model_id": args.model_id, "inputs": inputs, "deadline_s": None}}))
        deadline = started + args.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(f"timed out after {args.timeout:.0f}s with no result")
                return 1
            message = json.loads(await asyncio.wait_for(ws.recv(), remaining))
            if message.get("type") == "inference_result":
                break
        elapsed = time.monotonic() - started

    results = message["data"]["results"]
    logits = [np.asarray(deserialize_tensor(r["tensor"]), dtype=np.float64).reshape(-1) for r in results]
    predictions = [int(np.argmax(v)) for v in logits]
    hits = sum(p == int(row["label"]) for p, row in zip(predictions, rows))
    out = ROOT / "logs" / "experiments" / f"bert_planned_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "predicted", "label", "logit_neg", "logit_pos", "text"])
        for i, (p, row, v) in enumerate(zip(predictions, rows, logits)):
            writer.writerow([i, p, row["label"], round(float(v[0]), 4), round(float(v[1]), 4), row["text"]])
    print(f"{len(results)} results for {len(inputs)} inputs in {elapsed:.1f}s ({len(inputs) / elapsed:.3f} sentences/s)")
    print(f"accuracy {hits}/{len(rows)} = {100.0 * hits / len(rows):.1f}%   predictions saved to {out}")
    return 0 if len(results) == len(inputs) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sentences", type=int, default=30)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--timeout", type=float, default=900.0)
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
