"""Retrieve public embed text as a CSV for a separate annotation review.

Run after installing the project: python scripts/fetch_annotation_context.py
    --input prepared.csv --output context.csv
This script does not read sentiment predictions, linked pages, or credentials.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ph_sentiment.annotation_context import CONTEXT_FIELDS, fetch_context
from ph_sentiment.collectors.base import HTTPClient
from ph_sentiment.csvio import read_records, write_csv
from ph_sentiment.errors import PipelineError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new context snapshot path.")
    client = HTTPClient(max_retries=1, max_wait=30)
    try:
        rows = fetch_context(read_records(args.input), client)
        write_csv(args.output, rows, CONTEXT_FIELDS)
    except (PipelineError, OSError) as exc:
        print(f"Context collection failed: {exc}")
        return 2
    finally:
        client.close()
    print(f"Saved {len(rows)} annotation context rows to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
