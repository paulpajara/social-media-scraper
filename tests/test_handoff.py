"""Offline tests for portable snapshots without credentials or fabricated research labels."""

import csv
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from ph_sentiment import collectors
from ph_sentiment.cli import main, parser
from ph_sentiment.config import PUBLIC_CONFIG, load_config
from ph_sentiment.csvio import read_records, write_csv, write_records
from ph_sentiment.errors import DataValidationError
from ph_sentiment.handoff import ANALYSIS_FILES, export_handoff
from ph_sentiment.live import PUBLIC_PLATFORMS, live_collect
from ph_sentiment.schema import Record


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def rewrite(path, values):
    fields = list(rows(path)[0])
    return write_csv(path, values, fields)


class PublicFixture:
    def __init__(self, platform, count):
        self.records = [
            Record(
                f"{platform}:synthetic-{index}",
                platform,
                "2026-06-01T12:00:00Z",
                f"+ Government hospital services {platform} fixture{index}",
                f"https://example.test/{platform}/{index}",
                "government",
                collected_at="2026-10-12T05:00:00Z",
            )
            for index in range(count)
        ]
        self.warnings = ["Synthetic software-test fixture; not public research data."]

    def collect(self):
        return self.records

    def close(self):
        pass


@pytest.fixture
def public_run(tmp_path, monkeypatch):
    config = load_config(PUBLIC_CONFIG)
    config["analysis"]["max_pairs"] = 5
    fixtures = {platform: PublicFixture(platform, 100) for platform in PUBLIC_PLATFORMS}
    fixtures["bluesky"].records.append(
        replace(fixtures["bluesky"].records[0], record_id="bluesky:duplicate")
    )
    monkeypatch.setattr(collectors, "make_collector", lambda platform, _: fixtures[platform])
    run = tmp_path / "run"
    live_collect(config, run, skip_analysis=True, progress=lambda _: None)
    return run


def test_snapshot_preserves_records_and_produces_portable_integrity_checked_csvs(
    public_run, tmp_path
):
    # Only approved outputs are copied, even when the run directory has extra files.
    (public_run / ".env").write_text("TEST_SECRET=private-synthetic-secret", encoding="utf-8")
    snapshot = export_handoff(public_run, tmp_path / "handoff", PUBLIC_CONFIG)
    prepared = read_records(snapshot / "processed/posts.csv")
    assert len(prepared) == 200
    assert read_records(snapshot / "labels/annotation.csv") == prepared
    assert all(not record.gold_label for record in prepared)
    assert {record.platform for record in prepared} == set(PUBLIC_PLATFORMS)
    assert all(
        not Path(row["output_path"]).is_absolute()
        for row in rows(snapshot / "collection_report.csv")
    )
    assert rows(snapshot / "run_summary.csv")[0]["posts_path"] == "processed/posts.csv"
    assert rows(snapshot / "run_summary.csv")[0]["analysis_status"] == "not_included"
    assert not (snapshot / ".env").exists()
    assert not (snapshot / "analysis").exists()
    summary = {row["platform"]: row for row in rows(snapshot / "source_summary.csv")}
    assert summary["bluesky"]["dropped_records"] == "1"
    assert summary["mastodon"]["prepared_records"] == "100"
    assert len(rows(snapshot / "data_dictionary.csv")) == 9
    assert "predictions" not in rows(snapshot / "labels/annotation.csv")[0]
    manifest = rows(snapshot / "file_manifest.csv")
    assert {row["path"] for row in manifest} == {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and path.name != "file_manifest.csv"
    }
    for row in manifest:
        path = snapshot / row["path"]
        payload = path.read_bytes()
        assert int(row["bytes"]) == len(payload)
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()
        assert path.suffix in {".csv", ".md", ".toml"}
        if path.suffix == ".csv":
            assert int(row["rows"]) == len(rows(path))
    assert "private-synthetic-secret" not in "".join(
        path.read_text(encoding="utf-8-sig") for path in snapshot.rglob("*") if path.is_file()
    )


def test_export_includes_matching_baseline_analysis(public_run, tmp_path):
    from ph_sentiment.analysis import analyze

    prepared = read_records(public_run / "processed/posts.csv")
    analyze(prepared, public_run / "analysis", max_pairs=5)
    summary_path = public_run / "run_summary.csv"
    summary = rows(summary_path)
    summary[0]["analysis_status"] = "success"
    rewrite(summary_path, summary)
    snapshot = export_handoff(public_run, tmp_path / "handoff", PUBLIC_CONFIG)
    assert {path.name for path in (snapshot / "analysis").iterdir()} == set(ANALYSIS_FILES)
    assert read_records(snapshot / "analysis/analyzed_documents.csv") == prepared


@pytest.mark.parametrize(
    "damage", ["prepared", "audit", "manifest", "report", "summary", "labels", "missing-source"]
)
def test_invalid_snapshot_inputs_fail_before_creating_output(public_run, tmp_path, damage):
    if damage == "prepared":
        path = public_run / "processed/posts.csv"
        write_records(path, read_records(path)[1:])
    elif damage == "audit":
        path = public_run / "processed/preprocessing_audit.csv"
        rewrite(path, rows(path)[1:])
    elif damage == "manifest":
        path = public_run / "raw/bluesky_collection_manifest.csv"
        values = rows(path)
        values[0]["collected_records"] = "999"
        rewrite(path, values)
    elif damage == "report":
        path = public_run / "collection_report.csv"
        values = rows(path)
        values[0]["platform"] = "x"
        rewrite(path, values)
    elif damage == "summary":
        path = public_run / "run_summary.csv"
        values = rows(path)
        values[0]["prepared_records"] = "1"
        rewrite(path, values)
    elif damage == "labels":
        path = public_run / "raw/bluesky.csv"
        values = read_records(path)
        values[0] = replace(values[0], gold_label="positive")
        write_records(path, values)
    else:
        (public_run / "raw/mastodon.csv").unlink()
    output = tmp_path / "handoff"
    with pytest.raises(DataValidationError):
        export_handoff(public_run, output, PUBLIC_CONFIG)
    assert not output.exists()


def test_handoff_protects_existing_snapshot_and_rejects_nested_output(public_run, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    protected = write_csv(output / "annotations.csv", [{"gold_label": "positive"}], ["gold_label"])
    before = protected.read_bytes()
    with pytest.raises(DataValidationError, match="already exists"):
        export_handoff(public_run, output, PUBLIC_CONFIG)
    assert protected.read_bytes() == before
    with pytest.raises(DataValidationError, match="separate"):
        export_handoff(public_run, public_run / "handoff", PUBLIC_CONFIG)


def test_handoff_cli_uses_public_configuration_and_preserves_snapshot(
    public_run, tmp_path, monkeypatch
):
    from ph_sentiment import cli

    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    output = tmp_path / "handoff"
    args = ["handoff", "--run-dir", str(public_run), "--output-dir", str(output)]
    assert parser().parse_args(args).config == PUBLIC_CONFIG
    assert main(args) == 0
    before = (output / "file_manifest.csv").read_bytes()
    assert main(args) == 2
    assert (output / "file_manifest.csv").read_bytes() == before


def test_default_live_cli_uses_public_configuration():
    assert parser().parse_args(["collect-live"]).config == PUBLIC_CONFIG


@pytest.mark.parametrize(
    "original,replacement",
    [
        ("max_records = 400", "max_records = 401"),
        ("max_pages = 4", "max_pages = 3"),
        ("Philippines Marcos", "different government query"),
    ],
)
def test_export_rejects_configuration_with_different_collection_queries_or_budgets(
    public_run, tmp_path, original, replacement
):
    config = tmp_path / "different.toml"
    config.write_text(
        PUBLIC_CONFIG.read_text(encoding="utf-8").replace(original, replacement), encoding="utf-8"
    )
    with pytest.raises(DataValidationError, match="manifest disagrees"):
        export_handoff(public_run, tmp_path / "handoff", config)


@pytest.mark.parametrize("relative", ["raw/bluesky.csv", "processed/posts.csv"])
def test_export_rejects_unexpected_columns_in_canonical_sources(public_run, tmp_path, relative):
    path = public_run / relative
    values = rows(path)
    for row in values:
        row["unexpected_profile_attribute"] = "private-synthetic-value"
    write_csv(path, values, tuple(values[0]))
    with pytest.raises(DataValidationError, match="Unexpected handoff CSV columns"):
        export_handoff(public_run, tmp_path / "handoff", PUBLIC_CONFIG)
    assert not (tmp_path / "handoff").exists()
