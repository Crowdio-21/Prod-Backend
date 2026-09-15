#!/usr/bin/env python3
"""Download the public SST-2 validation set for the sentiment pipeline."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
import urllib.request
from pathlib import Path

DATASET_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "data" / "sentences.csv"


def download_sst2(output_path: Path, limit: int) -> int:
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "CROWDio sentiment pipeline test"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = response.read()

    with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
        content = zip_file.read("SST-2/dev.tsv").decode("utf-8")

    rows = []
    reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    for row in reader:
        sentence = (row.get("sentence") or "").strip()
        label = (row.get("label") or "").strip()
        if sentence and label in {"0", "1"}:
            rows.append((sentence, label))
        if len(rows) == limit:
            break

    if len(rows) < limit:
        raise RuntimeError(f"SST-2 provided only {len(rows)} usable rows; requested {limit}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=700)
    args = parser.parse_args()
    count = download_sst2(args.output, args.limit)
    print(f"Wrote {count} labeled sentences to {args.output.resolve()}")


if __name__ == "__main__":
    main()
