"""Build an immutable, reproducible handoff from independent reference decisions."""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

from .analysis import analyze
from .annotation_context import CONTEXT_FIELDS
from .config import load_config
from .csvio import read_records, write_csv, write_records
from .curation import DECISION_FIELDS, curate_records
from .errors import DataValidationError
from .evaluation import evaluate
from .handoff import _table, verify_integrity_manifest, write_integrity_manifest
from .live import PUBLIC_PLATFORMS
from .preprocessing import AUDIT_FIELDS, prepare_records
from .schema import parse_timestamp


def export_labeled_handoff(
    source_snapshot: Path,
    decisions_path: Path,
    context_path: Path,
    protocol_path: Path,
    output_dir: Path,
) -> Path:
    """Replay preparation and audited curation, then compute genuine outputs.

    The supplied decisions are research inputs, never classifier predictions.
    Existing snapshots are preserved. Class-count and split validation must pass
    before the staged output is published as a completed handoff.
    """
    source_snapshot, output_dir = source_snapshot.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise DataValidationError("Labeled snapshot already exists; choose a new output directory.")
    if output_dir.is_relative_to(source_snapshot) or source_snapshot.is_relative_to(output_dir):
        raise DataValidationError("Keep labeled and original snapshots in separate directories.")
    verify_integrity_manifest(source_snapshot)
    config = load_config(source_snapshot / "config.toml")
    raw = [
        record
        for platform in PUBLIC_PLATFORMS
        for record in read_records(source_snapshot / "raw" / f"{platform}.csv")
    ]
    candidates, preparation_audit = prepare_records(
        raw, config["research"]["start_date"], config["research"]["end_date"]
    )
    _, decisions = _table(decisions_path, DECISION_FIELDS, allowed=DECISION_FIELDS)
    if {row["label_source"] for row in decisions} != {"ai_assisted"}:
        raise DataValidationError(
            "This handoff requires explicitly AI-assisted reference decisions."
        )
    curated = curate_records(candidates, decisions)
    if len(curated) < config["analysis"]["minimum_records"]:
        raise DataValidationError(
            "Too few relevant labeled records; collect and annotate more data."
        )
    if {record.platform for record in curated} != set(PUBLIC_PLATFORMS):
        raise DataValidationError("Labeled handoff requires records from both public sources.")
    _, context = _table(context_path, CONTEXT_FIELDS, allowed=CONTEXT_FIELDS)
    if len(context) != len(candidates) or {
        (row["platform"], row["record_id"]) for row in context
    } != {(record.platform, record.record_id) for record in candidates}:
        raise DataValidationError("Context IDs must cover the prepared candidates exactly once.")
    protocol = protocol_path.read_text(encoding="utf-8")
    if not protocol.strip():
        raise DataValidationError("Supply the annotation protocol used for this research snapshot.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".labeled-", dir=output_dir.parent) as temporary:
        staging = Path(temporary).resolve()
        if staging.parent != output_dir.parent or not staging.name.startswith(".labeled-"):
            raise DataValidationError("Unsafe labeled handoff staging path.")
        (staging / "raw").mkdir()
        for platform in PUBLIC_PLATFORMS:
            for suffix in (".csv", "_collection_manifest.csv", "_query_diagnostics.csv"):
                filename = f"{platform}{suffix}"
                source = source_snapshot / "raw" / filename
                if source.is_file():
                    shutil.copyfile(source, staging / "raw" / filename)
        for name in ("config.toml", "collection_report.csv"):
            shutil.copyfile(source_snapshot / name, staging / name)
        write_records(staging / "processed/candidates.csv", candidates)
        write_records(staging / "processed/posts.csv", curated)
        write_records(staging / "labels/annotation.csv", curated)
        write_csv(staging / "processed/preprocessing_audit.csv", preparation_audit, AUDIT_FIELDS)
        shutil.copyfile(decisions_path, staging / "labels/annotation_decisions.csv")
        shutil.copyfile(context_path, staging / "labels/post_context.csv")
        review_files = (
            "primary_annotation_decisions.csv",
            "independent_review.csv",
            "adjudication.csv",
            "annotation_quality.csv",
        )
        available = [name for name in review_files if (decisions_path.parent / name).is_file()]
        if available and len(available) != len(review_files):
            raise DataValidationError(
                "Annotation quality audit is incomplete; supply all four review CSVs."
            )
        for filename in available:
            _table(decisions_path.parent / filename)
            shutil.copyfile(decisions_path.parent / filename, staging / "labels" / filename)
        (staging / "annotation_protocol.md").write_text(protocol, encoding="utf-8")
        counts = Counter(row["relevance"] for row in decisions)
        write_csv(
            staging / "curation_summary.csv",
            [
                {
                    "relevance": status,
                    "records": counts[status],
                    "label_source": "ai_assisted",
                    "notes": "Included posts have resolved reference labels; excluded/unresolved posts have no sentiment label.",
                }
                for status in ("include", "exclude", "unresolved")
            ],
            ("relevance", "records", "label_source", "notes"),
        )
        labels = Counter(record.gold_label for record in curated)
        write_csv(
            staging / "annotation_summary.csv",
            [
                {
                    "label": label,
                    "records": labels[label],
                    "label_source": "ai_assisted",
                    "notes": "Independent target-polarity reference annotations; not human-validated ground truth.",
                }
                for label in ("negative", "neutral", "positive")
            ],
            ("label", "records", "label_source", "notes"),
        )
        source_rows = []
        for platform in PUBLIC_PLATFORMS:
            records = [record for record in curated if record.platform == platform]
            dates = [
                parse_timestamp(record.created_at)
                .astimezone(ZoneInfo("Asia/Manila"))
                .date()
                .isoformat()
                for record in records
            ]
            source_rows.append(
                {
                    "platform": platform,
                    "raw_records": sum(r.platform == platform for r in raw),
                    "prepared_candidates": sum(r.platform == platform for r in candidates),
                    "curated_records": len(records),
                    **{
                        label: sum(r.gold_label == label for r in records)
                        for label in ("negative", "neutral", "positive")
                    },
                    "first_post_date_manila": min(dates),
                    "last_post_date_manila": max(dates),
                    "label_source": "ai_assisted",
                }
            )
        write_csv(staging / "source_summary.csv", source_rows, tuple(source_rows[0]))
        options = {
            key: value for key, value in config["analysis"].items() if key != "minimum_records"
        }
        analysis_paths = analyze(curated, staging / "analysis", **options)
        _, analyzed = _table(analysis_paths["documents"])
        decision_by_id = {row["record_id"]: row for row in decisions}
        grouped = Counter(
            (
                row["topic_id"],
                row["platform"],
                decision_by_id[row["record_id"]]["content_kind"],
                row["gold_label"],
            )
            for row in analyzed
        )
        group_totals = Counter()
        for (topic, platform, kind, _), count in grouped.items():
            group_totals[(topic, platform, kind)] += count
        reference_summary = []
        for (topic, platform, kind, label), count in sorted(grouped.items()):
            total = group_totals[(topic, platform, kind)]
            reference_summary.append(
                {
                    "topic_id": int(topic),
                    "platform": platform,
                    "content_kind": kind,
                    "label": label,
                    "count": count,
                    "total": total,
                    "share": count / total,
                    "label_source": "ai_assisted",
                    "notes": "Supplied target-polarity reference labels, not rule predictions or population approval.",
                }
            )
        write_csv(
            staging / "labels/reference_sentiment_summary.csv",
            reference_summary,
            (
                "topic_id",
                "platform",
                "content_kind",
                "label",
                "count",
                "total",
                "share",
                "label_source",
                "notes",
            ),
        )
        dictionary_fields, dictionary = _table(source_snapshot / "data_dictionary.csv")
        for row in dictionary:
            if row["field"] == "gold_label":
                row["description"] = (
                    "AI-assisted target-polarity reference annotation: negative, neutral or positive; not human-validated gold truth."
                )
        write_csv(staging / "data_dictionary.csv", dictionary, dictionary_fields)
        evaluate(
            curated,
            staging / "evaluation",
            seed=config["analysis"]["seed"],
            label_source="ai_assisted",
        )
        write_csv(
            staging / "run_summary.csv",
            [
                {
                    "raw_records": len(raw),
                    "prepared_candidates": len(candidates),
                    "curated_records": len(curated),
                    "sources_with_records": len(source_rows),
                    "analysis_status": "success",
                    "evaluation_status": "success",
                    "posts_path": "processed/posts.csv",
                    "label_source": "ai_assisted",
                }
            ],
            (
                "raw_records",
                "prepared_candidates",
                "curated_records",
                "sources_with_records",
                "analysis_status",
                "evaluation_status",
                "posts_path",
                "label_source",
            ),
        )
        _write_guide(staging, len(candidates), len(curated), counts, labels)
        write_csv(
            staging / "script_output_remarks.csv",
            [
                {
                    "script": "Public Bluesky/Mastodon collectors",
                    "output": "raw/*.csv",
                    "remarks": "Original frozen public API sample; manifests record actual query/date coverage.",
                },
                {
                    "script": "scripts/fetch_annotation_context.py",
                    "output": "labels/post_context.csv",
                    "remarks": "Public Bluesky embed text only; no linked websites, media files or author profiles.",
                },
                {
                    "script": "Independent AI annotation + curation.curate_records",
                    "output": "labels/annotation_decisions.csv; processed/posts.csv",
                    "remarks": "Full-text relevance/target-polarity decisions, reasons and text hashes; unresolved rows excluded.",
                },
                {
                    "script": "analysis.analyze",
                    "output": "analysis/*.csv",
                    "remarks": "Three LDA topics; fixed rule sentiment; cosine/Jaccard/Euclidean pairs; topic diagnostics.",
                },
                {
                    "script": "evaluation.evaluate(label_source='ai_assisted')",
                    "output": "evaluation/*.csv",
                    "remarks": "90/10 holdout and ten development folds; agreement against AI references, not human-validated accuracy.",
                },
                {
                    "script": "notebooks/philippine_government_2026.ipynb",
                    "output": "data/labeled-notebook-run/*.csv (subdirectories)",
                    "remarks": "Offline Run All; supplied labels; group completes documentation and interpretation.",
                },
            ],
            ("script", "output", "remarks"),
        )
        write_integrity_manifest(staging)
        staging.rename(output_dir)
    return output_dir


def _write_guide(
    output: Path, candidates: int, curated: int, counts: Counter, labels: Counter
) -> None:
    (output / "README.md").write_text(
        f"""# Labeled research handoff — 2026

This snapshot contains **{curated} curated, labeled records** from Bluesky and Mastodon. Independent AI annotation screened {candidates} prepared candidates: {counts["exclude"]} excluded and {counts["unresolved"]} unresolved. Sentiment references: {labels["negative"]} negative, {labels["neutral"]} neutral, {labels["positive"]} positive. The original public API acquisition and text remain traceable in raw CSVs, source manifests and the per-record decision audit.

## Run All

Download this entire repository branch, install `python -m pip install -e ".[notebook]"`, select that Python environment, and run `notebooks/philippine_government_2026.ipynb`. No credentials or live API requests are needed. Analysis and evaluation run by default, writing CSV files under `data/labeled-notebook-run/`. Existing working labels are preserved. See the root README for setup.

## What is complete

Relevance screening, positive/negative/neutral reference annotations, three-topic LDA, rule sentiment, cosine/Jaccard/Euclidean proximity, and stratified 90/10 plus ten-fold development evaluation are supplied. `file_manifest.csv` records every other snapshot file's row count, byte size and SHA-256 checksum. `scripts/build_labeled_handoff.py` reproduces the snapshot from the decision/context inputs without creating new labels or using an annotation API.

`processed/posts.csv` and `labels/annotation.csv` contain identical labeled canonical records. The historical schema calls the reference column `gold_label`; here it is **AI-assisted**, not human gold truth. `labels/annotation_decisions.csv` covers every candidate with inclusion/exclusion rationale, annotator provenance, rubric version and text hash. `labels/post_context.csv` preserves the public embed context available to annotators. Read `annotation_protocol.md` for exact definitions and scope.

## Interpretation and remaining group work

The labels describe expressed target polarity, including allegations conveyed in news reports; they do not establish whether those allegations are true or measure each author's personal approval. Some target/polarity judgments use public link-preview context while the fixed lexical classifier sees post text only. Evaluation reports **agreement with AI-assisted references**, not human-validated predictive accuracy. The annotation audit reports any independently reviewed AI sample and its actual agreement separately; this does not establish human validation. Optional independent human validation would strengthen research conclusions.

Finish the topic names, examples, discussion, limitations, instructor's report and peer evaluation. Replace the notebook's group name and save a submission copy as `GROUP NAME.ipynb`. Keep submission files uncompressed as instructed. No compulsory manual labeling step remains in this handoff.

This is a bounded, source-imbalanced sample during the elapsed portion of 2026, dominated by publicly indexed/hashtagged posts and news text. It is not a representative survey, an exhaustive archive, or a measure of objective government performance. Public post text, URLs and IDs can identify authors. Use aggregate findings and necessary traceable examples in documentation.
""",
        encoding="utf-8",
    )
