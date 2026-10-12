"""Apply completed review decisions and report genuine pre-adjudication agreement."""

from __future__ import annotations

import argparse
from pathlib import Path

from ph_sentiment.annotation_review import QUALITY_FIELDS, adjudicate_annotations
from ph_sentiment.csvio import read_records, write_csv
from ph_sentiment.curation import DECISION_FIELDS
from ph_sentiment.errors import PipelineError
from ph_sentiment.handoff import _table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--sampling-seed", type=int, default=20261012)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = [
        args.output_dir / name for name in ("annotation_decisions.csv", "annotation_quality.csv")
    ]
    if any(path.exists() for path in outputs):
        parser.error("Outputs already exist; choose a new adjudication directory.")
    try:
        final, quality = adjudicate_annotations(
            read_records(args.candidates),
            _table(args.primary)[1],
            _table(args.review)[1],
            _table(args.adjudication)[1],
            sampling_seed=args.sampling_seed,
        )
        write_csv(outputs[0], final, DECISION_FIELDS)
        write_csv(outputs[1], quality, QUALITY_FIELDS)
    except (PipelineError, OSError) as exc:
        print(f"Annotation adjudication failed: {exc}")
        return 2
    print(f"Saved complete decisions and actual annotation consistency to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
