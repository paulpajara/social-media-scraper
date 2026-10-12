"""Run real API collection, retain partial progress, and produce CSV research artifacts."""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .csvio import write_csv, write_records
from .errors import DataValidationError, PipelineError
from .preprocessing import AUDIT_FIELDS, prepare_records
from .schema import Record

PLATFORM_CREDENTIALS = {
    "threads": ("THREADS_ACCESS_TOKEN",),
    "x": ("X_BEARER_TOKEN",),
}
PUBLIC_PLATFORMS = ("bluesky", "mastodon")
SUPPORTED_PLATFORMS = (*PLATFORM_CREDENTIALS, *PUBLIC_PLATFORMS)
REPORT_FIELDS = (
    "platform",
    "status",
    "record_count",
    "output_path",
    "message",
    "warnings",
    "http_status",
    "error_category",
    "diagnostics_path",
)
QUERY_FIELDS = (
    "platform",
    "query",
    "api_records",
    "retained_records",
    "status",
    "message",
    "http_status",
    "error_category",
)
MANIFEST_FIELDS = (
    "platform",
    "collected_at",
    "start_date",
    "end_date_exclusive",
    "date_timezone",
    "requested_max_records",
    "collected_records",
    "queries",
    "max_pages",
    "coverage",
    "warnings",
    "status",
    "error",
)
RUN_SUMMARY_FIELDS = (
    "completed_at",
    "raw_records",
    "prepared_records",
    "sources_with_records",
    "minimum_records",
    "activity_record_minimum_met",
    "analysis_status",
    "analysis_error",
    "posts_path",
)


@dataclass(frozen=True, slots=True)
class CollectionResult:
    records: list[Record]
    warnings: str
    status: str
    error: str = ""
    http_status: int | str = ""
    error_category: str = ""
    empty_reason: str = ""
    diagnostics_path: Path | None = None


@dataclass(frozen=True, slots=True)
class LiveResult:
    prepared_path: Path | None
    report_path: Path
    raw_paths: tuple[Path, ...]
    record_count: int
    analysis_paths: dict[str, Path]
    statuses: tuple[dict, ...]


def collector_config(platform: str, config: dict) -> dict:
    research = config["research"]
    settings = config.get(platform, {})
    options = {**research, "settings": settings}
    # All active collectors read search/timeline posts; conversation expansion
    # must be explicitly implemented before requesting it from a source.
    options["include_comments"] = settings.get("include_comments", False)
    if "queries" in settings:
        options["queries"] = settings["queries"]
    if "max_pages" in settings:
        options["max_pages"] = settings["max_pages"]
    if "max_records" in settings:
        options["max_records"] = settings["max_records"]
    return options


def collect_to_csv(platform: str, config: dict, output: Path) -> CollectionResult:
    """Collect using the real API; checkpoint valid records even if a later page fails.

    Callers protect existing output paths. No result is manufactured when the
    source fails before returning its first public record.
    """
    from .collectors import make_collector

    options = collector_config(platform, config)
    collector = make_collector(platform, options)
    error = ""
    failure: PipelineError | None = None
    diagnostics_path: Path | None = None
    try:
        try:
            records = collector.collect()
        except PipelineError as exc:
            failure = exc
            if not collector.records:
                raise
            records = list(collector.records)
            error = str(exc)
        warnings = " | ".join(collector.warnings)
    finally:
        try:
            if hasattr(collector, "query_diagnostics"):
                diagnostics_path = output.with_name(f"{output.stem}_query_diagnostics.csv")
                write_csv(
                    diagnostics_path,
                    collector.query_diagnostics,
                    QUERY_FIELDS,
                )
                if failure is not None:
                    failure.diagnostics_path = diagnostics_path
        finally:
            collector.close()
    for record in records:
        record.validate()
    status = "partial" if error else "success" if records else "empty"
    write_records(output, records)
    manifest = output.with_name(f"{output.stem}_collection_manifest.csv")
    write_csv(
        manifest,
        [
            {
                "platform": platform,
                "collected_at": datetime.now(UTC).isoformat(),
                "start_date": options["start_date"],
                "end_date_exclusive": options["end_date"],
                "date_timezone": "Asia/Manila",
                "requested_max_records": options.get("max_records", 1000),
                "collected_records": len(records),
                "queries": " | ".join(options["queries"]),
                "max_pages": options.get("max_pages", 20),
                "coverage": "Bounded API search sample; no exhaustive coverage or population representation claim.",
                "warnings": warnings,
                "status": status,
                "error": error,
            }
        ],
        MANIFEST_FIELDS,
    )
    empty_reason = ""
    if not records:
        messages = [
            row["message"]
            for row in getattr(collector, "query_diagnostics", [])
            if row.get("status") in {"empty_api", "filtered_empty"}
        ]
        empty_reason = " | ".join(dict.fromkeys(messages))
    return CollectionResult(
        records,
        warnings,
        status,
        error,
        getattr(failure, "status_code", ""),
        getattr(failure, "category", ""),
        empty_reason,
        diagnostics_path,
    )


def live_collect(
    config: dict,
    output_dir: Path,
    platforms: Sequence[str] | None = None,
    overwrite: bool = False,
    skip_analysis: bool = False,
    progress: Callable[[str], None] = print,
) -> LiveResult:
    """Attempt selected sources independently; readiness is not proof of API permission."""
    selected = list(dict.fromkeys(platforms if platforms is not None else PUBLIC_PLATFORMS))
    if not selected or any(platform not in SUPPORTED_PLATFORMS for platform in selected):
        raise DataValidationError(f"Choose a supported source: {', '.join(SUPPORTED_PLATFORMS)}.")
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise DataValidationError("Live output path must be a directory.")
    if output_dir.exists() and any(output_dir.rglob("*.csv")) and not overwrite:
        raise DataValidationError(
            "Live output directory contains CSV files; use a new directory or --overwrite."
        )
    report_path = output_dir / "collection_report.csv"
    statuses: list[dict] = []
    all_records: list[Record] = []
    raw_paths: list[Path] = []
    for platform in selected:
        row = {
            "platform": platform,
            "status": "",
            "record_count": 0,
            "output_path": "",
            "message": "",
            "warnings": "",
            "http_status": "",
            "error_category": "",
            "diagnostics_path": "",
        }
        missing = [
            name
            for name in PLATFORM_CREDENTIALS.get(platform, ())
            if not os.environ.get(name, "").strip()
        ]
        if missing:
            row.update(status="missing_credentials", message="Set locally: " + ", ".join(missing))
            progress(f"{platform}: missing credentials ({', '.join(missing)}).")
        else:
            progress(f"{platform}: collecting live public records...")
            raw_path = output_dir / "raw" / f"{platform}.csv"
            diagnostics_path = None
            try:
                result = collect_to_csv(platform, config, raw_path)
            except PipelineError as exc:
                diagnostics_path = exc.diagnostics_path
                row.update(
                    status="failed",
                    message=str(exc),
                    http_status=getattr(exc, "status_code", ""),
                    error_category=getattr(exc, "category", ""),
                )
                progress(f"{platform}: {exc}")
            else:
                diagnostics_path = result.diagnostics_path
                all_records.extend(result.records)
                raw_paths.append(raw_path)
                row.update(
                    status=result.status,
                    record_count=len(result.records),
                    output_path=str(raw_path),
                    message=result.error or result.empty_reason,
                    warnings=result.warnings,
                    http_status=result.http_status,
                    error_category=result.error_category,
                )
                progress(
                    f"{platform}: {result.status}, {len(result.records)} records -> {raw_path}"
                )
                if result.empty_reason:
                    progress(f"{platform}: {result.empty_reason}")
            if diagnostics_path is not None:
                row["diagnostics_path"] = str(diagnostics_path)
        statuses.append(row)
        # Each source checkpoints immediately; failures do not erase successful sources.
        write_csv(report_path, statuses, REPORT_FIELDS)

    prepared_path: Path | None = None
    analysis_paths: dict[str, Path] = {}
    research = config["research"]
    prepared, audit = prepare_records(all_records, research["start_date"], research["end_date"])
    if all_records:
        prepared_path = output_dir / "processed" / "posts.csv"
        write_records(prepared_path, prepared)
        write_csv(output_dir / "processed" / "preprocessing_audit.csv", audit, AUDIT_FIELDS)
    minimum = config["analysis"]["minimum_records"]
    analysis_status = "not_requested" if skip_analysis else "insufficient_records"
    analysis_error = ""
    if not skip_analysis and len(prepared) >= minimum:
        from .analysis import analyze

        progress(f"Analyzing {len(prepared)} valid unique records...")
        try:
            options = {
                key: value for key, value in config["analysis"].items() if key != "minimum_records"
            }
            analysis_paths = analyze(prepared, output_dir / "analysis", **options)
            analysis_status = "success"
        except PipelineError as exc:
            analysis_status, analysis_error = "failed", str(exc)
            progress(f"Analysis failed; collected CSVs are preserved: {exc}")
    write_csv(
        output_dir / "run_summary.csv",
        [
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "raw_records": len(all_records),
                "prepared_records": len(prepared),
                "sources_with_records": len({r.platform for r in prepared}),
                "minimum_records": minimum,
                "activity_record_minimum_met": len(prepared) >= 200,
                "analysis_status": analysis_status,
                "analysis_error": analysis_error,
                "posts_path": str(prepared_path) if prepared_path else "",
            }
        ],
        RUN_SUMMARY_FIELDS,
    )
    progress(f"Retained {len(prepared)} unique records. Source report -> {report_path}")
    return LiveResult(
        prepared_path, report_path, tuple(raw_paths), len(prepared), analysis_paths, tuple(statuses)
    )
