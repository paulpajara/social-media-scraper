"""Explicit research stages, with CSV outputs and no network calls during analysis."""

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .config import DEFAULT_CONFIG, PUBLIC_CONFIG, load_config
from .csvio import read_records, write_csv, write_records
from .errors import DataValidationError, PipelineError
from .live import PUBLIC_PLATFORMS, SUPPORTED_PLATFORMS
from .preprocessing import AUDIT_FIELDS, prepare_records
from .schema import Record


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Collect and analyze public government discussions as CSV."
    )
    subparsers = root.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("collect", "Collect one Threads, X, Bluesky, or Mastodon source through its API"),
        ("collect-live", "Collect available live sources, prepare CSVs, and analyze when ready"),
        ("collect-public", "Run credential-free Bluesky/Mastodon collection and CSV analysis"),
        ("check-access", "Check Threads token permissions and X credits without searching posts"),
        ("prepare", "Merge canonical CSV exports, clean, filter, and deduplicate"),
        ("analyze", "Compute topics, rule-based sentiment, and proximity measures"),
        ("label", "Create a blank human annotation CSV without model predictions"),
        ("evaluate", "Evaluate human labels with 90/10 split and 10-fold validation"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument(
            "--config",
            type=Path,
            default=PUBLIC_CONFIG if name == "collect-public" else DEFAULT_CONFIG,
        )
        command.add_argument(
            "--overwrite", action="store_true", help="Explicitly replace existing outputs"
        )
        if name == "check-access":
            command.add_argument("--platform", choices=("threads", "x"), action="append")
            command.add_argument("--output", type=Path, required=True)
        elif name in {"collect-live", "collect-public"}:
            command.add_argument(
                "--platform",
                choices=PUBLIC_PLATFORMS if name == "collect-public" else SUPPORTED_PLATFORMS,
                action="append",
            )
            command.add_argument(
                "--output-dir", type=Path, help="Defaults to a new timestamped data/live directory"
            )
            command.add_argument(
                "--skip-analysis", action="store_true", help="Collect and prepare only"
            )
        elif name == "collect":
            command.add_argument(
                "--platform",
                choices=SUPPORTED_PLATFORMS,
                required=True,
            )
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument(
                "--input",
                type=Path,
                required=True,
                action="append" if name == "prepare" else "store",
            )
            if name in {"analyze", "evaluate"}:
                command.add_argument("--output-dir", type=Path, required=True)
                command.add_argument(
                    "--minimum-records",
                    type=int,
                    help="Override 200 only for exploratory/smoke runs",
                )
            else:
                command.add_argument("--output", type=Path, required=True)
            if name == "prepare":
                command.add_argument("--audit-output", type=Path, required=True)
    return root


def _check_outputs(paths: list[Path], inputs: list[Path], overwrite: bool) -> None:
    if len({path.resolve() for path in paths}) != len(paths):
        raise DataValidationError("Each output must have a distinct path.")
    for path in paths:
        if path.suffix.lower() != ".csv":
            raise DataValidationError("Dataset and analysis output filenames must end in .csv.")
        if any(path.resolve() == source.resolve() for source in inputs):
            raise DataValidationError("Keep outputs separate from input datasets.")
        if path.exists() and not overwrite:
            raise DataValidationError(
                f"Output already exists: {path}. Use --overwrite explicitly to replace it."
            )


def _check_analysis_output(path: Path, source: Path, overwrite: bool) -> None:
    if source.resolve().parent == path.resolve():
        raise DataValidationError("Use a separate output directory to preserve the input dataset.")
    if path.exists() and not path.is_dir():
        raise DataValidationError("--output-dir must be a directory.")
    if path.exists() and list(path.glob("*.csv")) and not overwrite:
        raise DataValidationError(
            "Output directory already contains CSV files. Use --overwrite explicitly."
        )


def _prepared_input(path: Path, config: dict, minimum: int) -> list[Record]:
    if minimum < 3:
        raise DataValidationError("minimum_records must be at least 3.")
    records = read_records(path)
    research = config["research"]
    prepared, audit = prepare_records(records, research["start_date"], research["end_date"])
    if len(prepared) != len(records):
        reasons = Counter(row["reason"] for row in audit if row["disposition"] == "dropped")
        raise DataValidationError(
            f"Input needs preprocessing ({dict(reasons)}). Run prepare first."
        )
    if len(records) < minimum:
        raise DataValidationError(
            f"Found {len(records)} valid unique records; require at least {minimum}."
        )
    if minimum < 200:
        print(
            "Exploratory run: the activity's 200-record requirement is not enforced.",
            file=sys.stderr,
        )
    if len({record.platform for record in records}) < 2:
        print(
            "Dataset uses one platform. Add another source for the activity's different-source requirement.",
            file=sys.stderr,
        )
    return prepared


def run(args: argparse.Namespace) -> None:
    if args.command == "check-access":
        from .access import ACCESS_PLATFORMS, check_access

        _check_outputs([args.output], [], args.overwrite)
        rows = check_access(args.output, args.platform or ACCESS_PLATFORMS)
        print(f"Access report -> {args.output}")
        if any(row["status"] in {"blocked", "unverified", "missing_credentials"} for row in rows):
            raise DataValidationError(
                "Access is blocked or unverified; follow the actions in the access CSV."
            )
        return
    config = load_config(args.config)
    research = config["research"]
    if args.command in {"collect-live", "collect-public"}:
        from .live import live_collect

        output_dir = args.output_dir or Path(
            "data/public" if args.command == "collect-public" else "data/live"
        ) / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        result = live_collect(
            config,
            output_dir,
            args.platform or (PUBLIC_PLATFORMS if args.command == "collect-public" else None),
            args.overwrite,
            args.skip_analysis,
            progress=lambda message: print(message, flush=True),
        )
        if not result.record_count:
            raise DataValidationError(
                f"No valid public records collected. Check credentials, access, and queries in {result.report_path}."
            )
        if any(row["status"] in {"partial", "failed"} for row in result.statuses):
            print(
                "Some sources were incomplete; inspect collection_report.csv before interpreting coverage.",
                file=sys.stderr,
            )
        if result.record_count < config["analysis"]["minimum_records"]:
            print(
                "More records are needed before final analysis; current CSVs are preserved.",
                file=sys.stderr,
            )
        elif not args.skip_analysis and not result.analysis_paths:
            raise DataValidationError(
                "Analysis did not complete; collected data are preserved. Inspect run_summary.csv."
            )
    elif args.command == "collect":
        from .live import collect_to_csv

        manifest = args.output.with_name(f"{args.output.stem}_collection_manifest.csv")
        diagnostics = args.output.with_name(f"{args.output.stem}_query_diagnostics.csv")
        _check_outputs([args.output, manifest, diagnostics], [], args.overwrite)
        result = collect_to_csv(args.platform, config, args.output)
        records, collection_warnings = result.records, result.warnings
        if collection_warnings:
            print(f"Collection notes: {collection_warnings}", file=sys.stderr)
        print(f"Collected {len(records)} records -> {args.output}")
        if result.status == "partial":
            raise DataValidationError(
                f"Partial records preserved; collection was incomplete: {result.error}"
            )
        if not records:
            raise DataValidationError(
                "No records collected. "
                + (result.empty_reason or "Check queries, date range, and API access.")
            )
    elif args.command == "prepare":
        _check_outputs([args.output, args.audit_output], args.input, args.overwrite)
        records = [record for source in args.input for record in read_records(source)]
        prepared, audit = prepare_records(records, research["start_date"], research["end_date"])
        write_records(args.output, prepared)
        write_csv(args.audit_output, audit, AUDIT_FIELDS)
        print(
            f"Retained {len(prepared)}/{len(records)} records -> {args.output}; audit -> {args.audit_output}"
        )
        if len(prepared) < 200:
            print(
                "Collect more data: the final activity dataset needs at least 200 valid unique records.",
                file=sys.stderr,
            )
    elif args.command == "label":
        _check_outputs([args.output], [args.input], args.overwrite)
        records = read_records(args.input)
        for record in records:
            record.validate()
        if not records:
            raise DataValidationError("Cannot annotate an empty dataset.")
        if any(record.gold_label for record in records):
            raise DataValidationError(
                "Input already contains human labels. Preserve it instead of creating a blank template."
            )
        write_records(args.output, records)
        print(
            f"Annotation template -> {args.output}. Fill gold_label with positive, negative, or neutral."
        )
    else:
        _check_analysis_output(args.output_dir, args.input, args.overwrite)
        minimum = (
            args.minimum_records
            if args.minimum_records is not None
            else config["analysis"]["minimum_records"]
        )
        records = _prepared_input(args.input, config, minimum)
        if args.command == "analyze":
            from .analysis import analyze

            options = {
                key: value for key, value in config["analysis"].items() if key != "minimum_records"
            }
            outputs = analyze(records, args.output_dir, **options)
        else:
            from .evaluation import evaluate

            outputs = evaluate(records, args.output_dir, seed=config["analysis"]["seed"])
        for name, path in outputs.items():
            print(f"{name} -> {path}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # Load only the explicit project-local file; existing process environment always wins.
    load_dotenv(Path.cwd() / ".env", override=False)
    try:
        run(args)
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(
            f"File operation failed: {exc.strerror or 'check paths and permissions'}.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed CSV files remain available.", file=sys.stderr)
        return 130
    return 0
