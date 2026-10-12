"""Reproduce the offline labeled snapshot from completed reference annotation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ph_sentiment.errors import PipelineError
from ph_sentiment.labeled_handoff import export_labeled_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, default=Path("deliverables/public_2026"))
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("docs/annotation_protocol.md"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = export_labeled_handoff(
            args.source_snapshot, args.decisions, args.context, args.protocol, args.output_dir
        )
    except (PipelineError, OSError) as exc:
        print(f"Labeled handoff failed: {exc}")
        return 2
    print(f"Completed labeled handoff: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
