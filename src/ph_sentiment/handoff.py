"""Validate and freeze a portable, credential-free public-source handoff."""

from __future__ import annotations

import csv
import hashlib
import shutil
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import load_config
from .csvio import read_records, safe_cell, write_csv, write_records
from .errors import DataValidationError
from .live import (
    MANIFEST_FIELDS,
    PUBLIC_PLATFORMS,
    QUERY_FIELDS,
    REPORT_FIELDS,
    RUN_SUMMARY_FIELDS,
    collector_config,
)
from .preprocessing import prepare_records
from .schema import RECORD_FIELDS, parse_timestamp

ANALYSIS_FIELDS = {
    "analyzed_documents.csv": (
        *RECORD_FIELDS,
        "clean_text",
        "topic_id",
        "topic_probability",
        "topic_status",
        "sentiment_label",
        "sentiment_score",
        "sentiment_evidence",
        "sentiment_model_version",
    ),
    "topic_terms.csv": ("topic_id", "rank", "term", "probability"),
    "topic_metrics.csv": ("metric", "topic_id", "value", "evaluation_scope", "notes"),
    "sentiment_summary.csv": (
        "topic_id",
        "platform",
        "record_type",
        "month",
        "label",
        "count",
        "total",
        "share",
        "mean_score",
    ),
    "proximity_pairs.csv": (
        "record_id_a",
        "platform_a",
        "record_id_b",
        "platform_b",
        "cosine_similarity",
        "jaccard_similarity",
        "euclidean_distance",
        "pair_status",
    ),
}
ANALYSIS_FILES = tuple(ANALYSIS_FIELDS)


def _table(
    path: Path, required: tuple[str, ...] = (), *, allowed: tuple[str, ...] | None = None
) -> tuple[list[str], list[dict[str, str]]]:
    """Read strict metadata CSVs, reversing only our formula-protection marker."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)) or not set(required).issubset(fields):
                raise DataValidationError(f"Missing or duplicate CSV columns: {path.name}.")
            if allowed is not None and not set(fields).issubset(allowed):
                raise DataValidationError(f"Unexpected handoff CSV columns: {path.name}.")
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise DataValidationError(f"Malformed CSV row: {path.name}.")
                rows.append(
                    {
                        key: value[1:]
                        if value.startswith("'") and safe_cell(value[1:]) == value
                        else value
                        for key, value in row.items()
                    }
                )
            return fields, rows
    except (csv.Error, UnicodeError):
        raise DataValidationError(f"Invalid UTF-8 CSV: {path.name}.") from None


def _source_file(run_dir: Path, relative: str) -> Path:
    path = run_dir / relative
    if not path.is_file() or not path.resolve().is_relative_to(run_dir):
        raise DataValidationError(f"Missing or unsafe handoff input: {relative}.")
    return path


def export_handoff(run_dir: Path, output_dir: Path, config_path: Path) -> Path:
    """Export a complete snapshot only after verifying its records and provenance.

    Existing snapshots are immutable. Build in a checked sibling staging folder
    and rename only when every artifact and checksum has been written.
    """
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if not run_dir.is_dir():
        raise DataValidationError("--run-dir must be a completed public collection directory.")
    if output_dir.exists():
        raise DataValidationError("Handoff output already exists; choose a new snapshot directory.")
    if output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir):
        raise DataValidationError("Keep the handoff snapshot separate from its source run.")
    config_path = Path(config_path)
    config = load_config(config_path)
    if any(name in config for name in ("x", "threads", "reddit", "facebook")):
        raise DataValidationError("Use the public-source configuration for this handoff.")
    research = config["research"]
    raw_by_source = {}
    copied_files: list[tuple[str, Path]] = []
    for platform in PUBLIC_PLATFORMS:
        raw_path = _source_file(run_dir, f"raw/{platform}.csv")
        _table(raw_path, RECORD_FIELDS, allowed=RECORD_FIELDS)
        records = read_records(raw_path)
        if not records or any(
            record.platform != platform or record.gold_label for record in records
        ):
            raise DataValidationError(
                "Raw handoff sources must contain unlabelled posts from both public platforms."
            )
        for record in records:
            record.validate()
        if len({record.record_id for record in records}) != len(records):
            raise DataValidationError(f"Duplicate public post identifiers in {platform} raw CSV.")
        raw_by_source[platform] = records
        manifest_relative = f"raw/{platform}_collection_manifest.csv"
        manifest = _source_file(run_dir, manifest_relative)
        _, rows = _table(
            manifest,
            MANIFEST_FIELDS,
            allowed=MANIFEST_FIELDS,
        )
        options = collector_config(platform, config)
        if len(rows) != 1 or any(
            (
                rows[0]["platform"] != platform,
                rows[0]["collected_records"] != str(len(records)),
                rows[0]["start_date"] != research["start_date"],
                rows[0]["end_date_exclusive"] != research["end_date"],
                rows[0]["status"] not in {"success", "partial"},
                rows[0]["queries"] != " | ".join(options["queries"]),
                rows[0]["requested_max_records"] != str(options["max_records"]),
                rows[0]["max_pages"] != str(options["max_pages"]),
            )
        ):
            raise DataValidationError(
                f"Collection manifest disagrees with {platform} data/configuration."
            )
        copied_files.extend([(f"raw/{platform}.csv", raw_path), (manifest_relative, manifest)])
        diagnostic_relative = f"raw/{platform}_query_diagnostics.csv"
        if (run_dir / diagnostic_relative).is_file():
            diagnostic_path = _source_file(run_dir, diagnostic_relative)
            _table(diagnostic_path, QUERY_FIELDS, allowed=QUERY_FIELDS)
            copied_files.append((diagnostic_relative, diagnostic_path))
    raw = [record for platform in PUBLIC_PLATFORMS for record in raw_by_source[platform]]
    prepared, audit = prepare_records(raw, research["start_date"], research["end_date"])
    prepared_path = _source_file(run_dir, "processed/posts.csv")
    _table(prepared_path, RECORD_FIELDS, allowed=RECORD_FIELDS)
    if read_records(prepared_path) != prepared:
        raise DataValidationError(
            "Prepared CSV does not match a fresh preparation of the raw source files."
        )
    if len(prepared) < max(200, config["analysis"]["minimum_records"]):
        raise DataValidationError(
            "The group handoff needs at least 200 valid unique records after preparation."
        )
    if {record.platform for record in prepared} != set(PUBLIC_PLATFORMS):
        raise DataValidationError("Both public sources must have retained records in the handoff.")
    audit_path = _source_file(run_dir, "processed/preprocessing_audit.csv")
    _, saved_audit = _table(
        audit_path, ("record_id", "platform", "disposition", "reason", "row_number")
    )
    if saved_audit != [{key: str(value) for key, value in row.items()} for row in audit]:
        raise DataValidationError("Preprocessing audit disagrees with the supplied raw records.")
    copied_files.extend(
        [("processed/posts.csv", prepared_path), ("processed/preprocessing_audit.csv", audit_path)]
    )
    report_fields, report = _table(
        _source_file(run_dir, "collection_report.csv"),
        REPORT_FIELDS,
        allowed=REPORT_FIELDS,
    )
    if len(report) != 2 or {row["platform"] for row in report} != set(PUBLIC_PLATFORMS):
        raise DataValidationError("Collection report must describe only Bluesky and Mastodon.")
    for row in report:
        platform = row["platform"]
        if row["status"] not in {"success", "partial"} or row["record_count"] != str(
            len(raw_by_source[platform])
        ):
            raise DataValidationError("Collection report disagrees with raw source counts.")
        row["output_path"] = f"raw/{platform}.csv"
        if "diagnostics_path" in row:
            relative = f"raw/{platform}_query_diagnostics.csv"
            row["diagnostics_path"] = (
                relative if any(name == relative for name, _ in copied_files) else ""
            )
    summary_fields, summary = _table(
        _source_file(run_dir, "run_summary.csv"),
        RUN_SUMMARY_FIELDS,
        allowed=RUN_SUMMARY_FIELDS,
    )
    if len(summary) != 1 or any(
        (
            summary[0]["raw_records"] != str(len(raw)),
            summary[0]["prepared_records"] != str(len(prepared)),
            summary[0]["sources_with_records"] != "2",
        )
    ):
        raise DataValidationError("Run summary disagrees with the handoff dataset.")
    summary[0]["posts_path"] = "processed/posts.csv"
    if summary[0]["analysis_status"] == "success":
        for name in ANALYSIS_FILES:
            relative = f"analysis/{name}"
            analysis_path = _source_file(run_dir, relative)
            _table(analysis_path, ANALYSIS_FIELDS[name], allowed=ANALYSIS_FIELDS[name])
            copied_files.append((relative, analysis_path))
        if read_records(run_dir / "analysis/analyzed_documents.csv") != prepared:
            raise DataValidationError(
                "Baseline analysis refers to different records; rerun analysis before handoff."
            )
    else:
        # Students will reproduce analysis in the notebook; avoid copying stale files.
        summary[0]["analysis_status"] = "not_included"
    source_summary = []
    zone = ZoneInfo("Asia/Manila")
    for platform in PUBLIC_PLATFORMS:
        records = [record for record in prepared if record.platform == platform]
        dates = [
            parse_timestamp(record.created_at).astimezone(zone).date().isoformat()
            for record in records
        ]
        source_summary.append(
            {
                "platform": platform,
                "raw_records": len(raw_by_source[platform]),
                "prepared_records": len(records),
                "dropped_records": len(raw_by_source[platform]) - len(records),
                "first_post_date_manila": min(dates),
                "last_post_date_manila": max(dates),
                "manual_relevance_review": "pending",
                "human_sentiment_labels": "pending",
            }
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".handoff-", dir=output_dir.parent) as temporary:
        staging = Path(temporary).resolve()
        # Verify the absolute cleanup/move target remains under the requested parent.
        if staging.parent != output_dir.parent or not staging.name.startswith(".handoff-"):
            raise DataValidationError("Unsafe handoff staging directory.")
        for relative, source in copied_files:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(config_path, staging / "config.toml")
        write_csv(staging / "collection_report.csv", report, report_fields)
        write_csv(staging / "run_summary.csv", summary, summary_fields)
        write_csv(staging / "source_summary.csv", source_summary, tuple(source_summary[0]))
        write_records(staging / "labels/annotation.csv", prepared)
        _write_guide(staging, len(raw), len(prepared), source_summary)
        _write_dictionary(staging)
        _write_script_table(staging)
        _write_integrity_manifest(staging)
        staging.rename(output_dir)
    return output_dir


def _write_guide(output: Path, raw_count: int, count: int, sources: list[dict]) -> None:
    counts = ", ".join(f"{row['prepared_records']} {row['platform']}" for row in sources)
    text = f"""# Public-source research handoff (2026)

This immutable snapshot contains {raw_count} raw public posts and **{count} unique prepared records** ({counts}). Only Bluesky and Mastodon contributed observations. It is a bounded API sample of posts published during the elapsed portion of 2026, collected on the timestamps recorded in the source manifests. Government relevance and human sentiment labels still require review. Public post IDs, source URLs and text can identify authors; this dataset is not anonymous.

## Run the notebook

Download the entire repository branch, keeping `src/`, `configs/`, `notebooks/`, and this folder together. Follow the root `README.md` installation steps (Python 3.11+) and open `notebooks/philippine_government_2026.ipynb` in that environment. Choose Run All. The notebook runs offline against this snapshot and writes student outputs under `data/notebook-run/`. It does not contact X, Threads or any collection API.

## Files

- `raw/`: original public post CSVs and collection manifests.
- `processed/posts.csv`: cleaned starting corpus; `preprocessing_audit.csv` explains exclusions.
- `labels/annotation.csv`: canonical records with blank `gold_label`, with no model predictions.
- `collection_report.csv`, `run_summary.csv`, `source_summary.csv`: actual collection coverage/counts; paths are relative to this folder.
- `analysis/` (if present): reproducible exploratory LDA topics, rule sentiment, summaries and cosine/Jaccard/Euclidean pairs. These are not human labels or final accuracy results.
- `data_dictionary.csv`, `script_output_remarks.csv`: field definitions and script/output/remarks reference.
- `file_manifest.csv`: relative filenames, row counts, sizes and SHA-256 checksums for every other snapshot file.
- `config.toml`: collection and analysis settings, containing no credentials.

## Group responsibilities

1. Review relevance to the 2026 Philippine national government and document exclusions. Keep raw files and this snapshot unchanged. Maintain at least 200 final valid records from different sources; collect more if review reduces the corpus below that minimum.
2. The notebook creates a working annotation CSV under `data/notebook-run/labels/`. Independently fill `gold_label` with `positive`, `negative` or `neutral`, resolving disagreements. Avoid model-output files while annotating. Do not manufacture labels or class counts.
3. Follow the notebook guidance when selecting the reviewed corpus. Rerun preparation and analysis after changing it; name the three fitted topics from their words and representative posts.
4. Enable `RUN_EVALUATION` only after independently labeling the final corpus. The fixed rule classifier reserves a 10% holdout and applies ten-fold evaluation on the remaining 90%; required class counts are checked. No final evaluation metrics are supplied in this unlabelled handoff.
5. Interpret actual outputs, discuss sampling/language/source imbalance, complete the instructor's report and peer evaluation, and save the final notebook as your actual `GROUP NAME.ipynb`. Submit uncompressed files according to the assignment.

Post sentiment does not measure person-level approval or objective government performance. Do not describe API query coverage as an exhaustive archive, uniform full-year sample or representative survey.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _write_dictionary(output: Path) -> None:
    descriptions = {
        "record_id": "Stable public post ID; Bluesky AT URIs include public author DIDs. Not an anonymous identifier.",
        "platform": "Source platform: bluesky or mastodon in this snapshot.",
        "created_at": "Timezone-aware ISO-8601 publication time; study calendar dates use Asia/Manila.",
        "text": "Raw post text in raw files; normalized cleaned text in prepared/annotation files.",
        "source_url": "HTTPS provenance URL for public post verification; may identify its author.",
        "query": "Search phrase or hashtag that returned the post; not a relevance judgment.",
        "record_type": "Post type; current public collectors retrieve posts, without conversation expansion.",
        "collected_at": "Timezone-aware collection timestamp; establishes the collection snapshot.",
        "gold_label": "Blank until independently annotated as positive, negative or neutral. Never copy rule predictions here.",
    }
    write_csv(
        output / "data_dictionary.csv",
        [
            {"field": field, "type": "string", "description": descriptions[field]}
            for field in RECORD_FIELDS
        ],
        ("field", "type", "description"),
    )


def _write_script_table(output: Path) -> None:
    rows = [
        {
            "script": "collect-public / collectors.bluesky",
            "output": "raw/bluesky.csv and source manifest",
            "remarks": "Live public indexed post search; bounded pages and queries, no credentials.",
        },
        {
            "script": "collect-public / collectors.mastodon",
            "output": "raw/mastodon.csv and source manifest",
            "remarks": "Public hashtag timelines visible to mastodon.social; government/country terms need human relevance review.",
        },
        {
            "script": "prepare / preprocessing.prepare_records",
            "output": "processed/posts.csv; processed/preprocessing_audit.csv",
            "remarks": "Date filtering, normalization, empty-text rejection and ID/text deduplication.",
        },
        {
            "script": "analyze / analysis.analyze",
            "output": "analysis/*.csv",
            "remarks": "Three LDA topics by default, rule sentiment, cosine/Jaccard/Euclidean proximity and summaries. Exploratory results.",
        },
        {
            "script": "label / csvio.write_records",
            "output": "labels/annotation.csv",
            "remarks": "Blank human annotation template; no automated predictions. Review independently before evaluation.",
        },
        {
            "script": "evaluate / evaluation.evaluate",
            "output": "data/notebook-run/evaluation/*.csv after human labeling",
            "remarks": "Pending group work: unseen 90/10 holdout and ten-fold development evaluation; no fabricated scores supplied.",
        },
        {
            "script": "notebooks/philippine_government_2026.ipynb",
            "output": "data/notebook-run/ derived CSVs; GROUP NAME.ipynb submission copy",
            "remarks": "Run offline against this snapshot; complete relevance review, annotations, interpretation and final documentation.",
        },
    ]
    write_csv(output / "script_output_remarks.csv", rows, ("script", "output", "remarks"))


def _write_integrity_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            row_count = len(_table(path)[1]) if path.suffix == ".csv" else ""
            payload = path.read_bytes()
            rows.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "rows": row_count,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    write_csv(output / "file_manifest.csv", rows, ("path", "rows", "bytes", "sha256"))
