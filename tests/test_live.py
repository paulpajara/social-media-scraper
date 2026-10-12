"""Credential-safe live workflow tests with synthetic API fixtures and no network."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ph_sentiment import collectors, live
from ph_sentiment.config import load_config
from ph_sentiment.csvio import read_records, write_csv
from ph_sentiment.errors import CollectionError, DataValidationError
from ph_sentiment.schema import Record

_CREDENTIALS = {
    "x": ("X_BEARER_TOKEN",),
    "threads": ("THREADS_ACCESS_TOKEN",),
}


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for names in _CREDENTIALS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_request(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Live workflow tests must use injected collectors without real HTTP.")

    monkeypatch.setattr("requests.sessions.Session.request", unexpected_request)


@pytest.fixture
def config() -> dict[str, Any]:
    return load_config()


def enable(monkeypatch: pytest.MonkeyPatch, *platforms: str) -> None:
    for platform in platforms:
        for name in _CREDENTIALS[platform]:
            monkeypatch.setenv(name, "synthetic-test-credential")


def records(platform: str, count: int = 1) -> list[Record]:
    # Unique synthetic text keeps preprocessing from conflating unrelated fixtures.
    return [
        Record(
            f"{platform}:fixture-{index}",
            platform,
            "2026-06-01T12:00:00Z",
            f"Government hospital services {platform} fixture{index}",
        )
        for index in range(count)
    ]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class FakeCollector:
    def __init__(self, rows: list[Record], failure: Exception | None = None) -> None:
        self.records = rows
        self.failure = failure
        self.warnings = ["Synthetic API fixture; bounded test coverage."]
        self.closed = False
        self.called = False

    def collect(self) -> list[Record]:
        self.called = True
        if self.failure:
            raise self.failure
        return self.records

    def close(self) -> None:
        self.closed = True


def install_collectors(
    monkeypatch: pytest.MonkeyPatch, configured: dict[str, FakeCollector]
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def factory(platform: str, settings: dict[str, Any]) -> FakeCollector:
        calls.append((platform, settings))
        return configured[platform]

    monkeypatch.setattr(collectors, "make_collector", factory)
    if hasattr(live, "make_collector"):
        monkeypatch.setattr(live, "make_collector", factory)
    return calls


def test_missing_credentials_produce_report_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    calls = install_collectors(monkeypatch, {})
    result = live.live_collect(
        config, tmp_path / "run", platforms=list(_CREDENTIALS), progress=lambda _: None
    )
    assert calls == []
    assert result.record_count == 0
    assert result.prepared_path is None
    assert result.analysis_paths == {}
    assert result.raw_paths == ()
    statuses = csv_rows(result.report_path)
    assert {row["platform"] for row in statuses} == set(_CREDENTIALS)
    assert all(row["status"] == "missing_credentials" for row in statuses)
    assert "synthetic-test-credential" not in result.report_path.read_text(encoding="utf-8-sig")


def test_source_failure_preserves_successful_source_and_closes_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    enable(monkeypatch, "threads", "x")
    successful = FakeCollector(records("threads", 2))
    failed = FakeCollector([], CollectionError("API returned HTTP 403; check read permissions."))
    calls = install_collectors(monkeypatch, {"threads": successful, "x": failed})
    result = live.live_collect(
        config, tmp_path / "run", platforms=["threads", "x"], progress=lambda _: None
    )
    assert [platform for platform, _ in calls] == ["threads", "x"]
    assert successful.closed and failed.closed
    assert result.record_count == 2
    assert result.prepared_path is not None
    assert {row.platform for row in read_records(result.prepared_path)} == {"threads"}
    statuses = {row["platform"]: row["status"] for row in csv_rows(result.report_path)}
    assert statuses["x"] == "failed"
    assert statuses["threads"] == "success"


def test_below_activity_minimum_writes_deduplication_audit_and_blank_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    enable(monkeypatch, "threads")
    unique = records("threads", 2)
    duplicate = replace(unique[0], record_id="threads:repost", text=unique[0].text.upper())
    collector = FakeCollector([*unique, duplicate])
    install_collectors(monkeypatch, {"threads": collector})
    output = tmp_path / "run"
    result = live.live_collect(config, output, platforms=["threads"], progress=lambda _: None)
    assert result.record_count == 2
    assert result.analysis_paths == {}
    assert result.prepared_path is not None
    prepared = read_records(result.prepared_path)
    assert all(row.gold_label == "" for row in prepared)
    audits = [
        rows
        for path in output.rglob("*.csv")
        if (rows := csv_rows(path)) and "disposition" in rows[0]
    ]
    assert len(audits) == 1
    assert len(audits[0]) == 3
    assert sum(row["disposition"] == "dropped" for row in audits[0]) == 1
    assert any(row["reason"] == "duplicate_text" for row in audits[0])
    assert all(path.suffix == ".csv" for path in output.rglob("*") if path.is_file())


def test_activity_scale_forwards_analysis_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    from ph_sentiment import analysis

    enable(monkeypatch, "threads", "x")
    install_collectors(
        monkeypatch,
        {"threads": FakeCollector(records("threads", 100)), "x": FakeCollector(records("x", 100))},
    )
    analysis_calls: list[tuple[list[Record], Path, dict[str, Any]]] = []

    def analyze(rows: list[Record], output_dir: Path, **options: Any) -> dict[str, Path]:
        analysis_calls.append((rows, output_dir, options))
        path = write_csv(output_dir / "analysis_fixture.csv", [{"count": len(rows)}], ["count"])
        return {"fixture": path}

    monkeypatch.setattr(analysis, "analyze", analyze)
    if hasattr(live, "analyze"):
        monkeypatch.setattr(live, "analyze", analyze)
    result = live.live_collect(
        config, tmp_path / "run", platforms=["threads", "x"], progress=lambda _: None
    )
    assert result.record_count == 200
    assert len(analysis_calls) == 1
    rows, _, options = analysis_calls[0]
    assert len(rows) == 200
    assert all(row.gold_label == "" for row in rows)
    assert options == {
        key: value for key, value in config["analysis"].items() if key != "minimum_records"
    }
    assert result.analysis_paths["fixture"].is_file()


def test_existing_run_is_protected_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    enable(monkeypatch, "threads")
    collector = FakeCollector(records("threads"))
    calls = install_collectors(monkeypatch, {"threads": collector})
    output = tmp_path / "run"
    output.mkdir()
    protected = write_csv(output / "existing.csv", [{"preserve": "human-label"}], ["preserve"])
    before = protected.read_bytes()
    with pytest.raises(DataValidationError, match="overwrite"):
        live.live_collect(config, output, platforms=["threads"], progress=lambda _: None)
    assert calls == []
    assert protected.read_bytes() == before


def test_skip_analysis_keeps_large_dataset_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    enable(monkeypatch, "threads")
    install_collectors(monkeypatch, {"threads": FakeCollector(records("threads", 200))})
    result = live.live_collect(
        config,
        tmp_path / "run",
        platforms=["threads"],
        skip_analysis=True,
        progress=lambda _: None,
    )
    assert result.record_count == 200
    assert result.analysis_paths == {}
    assert result.prepared_path is not None
    assert len(read_records(result.prepared_path)) == 200


def test_partial_collection_checkpoint_preserves_records_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    enable(monkeypatch, "threads")
    collector = FakeCollector(records("threads", 2), CollectionError("API quota exceeded."))
    install_collectors(monkeypatch, {"threads": collector})
    result = live.live_collect(
        config, tmp_path / "run", platforms=["threads"], progress=lambda _: None
    )
    assert collector.closed
    assert result.record_count == 2
    assert len(result.raw_paths) == 1
    assert len(read_records(result.raw_paths[0])) == 2
    assert csv_rows(result.report_path)[0]["status"] == "partial"
    manifests = list((tmp_path / "run").rglob("*_collection_manifest.csv"))
    assert len(manifests) == 1
    assert csv_rows(manifests[0])[0]["collected_records"] == "2"


def test_successful_empty_collection_is_saved_without_fabricated_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    enable(monkeypatch, "threads")
    collector = FakeCollector([])
    install_collectors(monkeypatch, {"threads": collector})
    result = live.live_collect(
        config, tmp_path / "run", platforms=["threads"], progress=lambda _: None
    )
    assert collector.called and collector.closed
    assert result.record_count == 0
    assert result.analysis_paths == {}
    assert len(result.raw_paths) == 1
    assert read_records(result.raw_paths[0]) == []


def test_default_live_run_uses_only_the_four_main_platforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    enable(monkeypatch, *_CREDENTIALS)
    config["research"]["include_comments"] = True
    platform_queries = ["government policy lang:en", "gobyerno lang:tl"]
    config["x"]["queries"] = platform_queries
    platforms = ("threads", "x", "bluesky", "mastodon")
    configured = {platform: FakeCollector(records(platform)) for platform in platforms}
    calls = install_collectors(monkeypatch, configured)
    result = live.live_collect(config, tmp_path / "run", progress=lambda _: None)

    supplied = dict(calls)
    assert tuple(supplied) == platforms
    assert all(options["include_comments"] is False for _, options in calls)
    assert supplied["x"]["queries"] == platform_queries
    assert supplied["threads"]["queries"] == config["research"]["queries"]
    assert result.record_count == 4
    assert all(collector.closed for collector in configured.values())
    for raw_path in result.raw_paths:
        platform = raw_path.stem
        manifest = raw_path.with_name(f"{platform}_collection_manifest.csv")
        row = csv_rows(manifest)[0]
        expected_queries = config.get(platform, {}).get("queries", config["research"]["queries"])
        assert row["platform"] == platform
        assert row["queries"] == " | ".join(expected_queries)


@pytest.mark.parametrize("platforms", [[], ["reddit"], ["facebook"], ["x", "reddit"]])
def test_removed_sources_are_rejected_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], platforms: list[str]
) -> None:
    calls = install_collectors(monkeypatch, {})
    with pytest.raises(DataValidationError, match="supported source"):
        live.live_collect(config, tmp_path / "run", platforms=platforms, progress=lambda _: None)
    assert calls == []
    assert not (tmp_path / "run").exists()


def test_default_live_run_preserves_public_results_without_authenticated_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    configured = {
        platform: FakeCollector(records(platform)) for platform in ("bluesky", "mastodon")
    }
    calls = install_collectors(monkeypatch, configured)
    result = live.live_collect(config, tmp_path / "run", progress=lambda _: None)
    assert [platform for platform, _ in calls] == ["bluesky", "mastodon"]
    assert result.record_count == 2
    assert {row["platform"]: row["status"] for row in result.statuses} == {
        "threads": "missing_credentials",
        "x": "missing_credentials",
        "bluesky": "success",
        "mastodon": "success",
    }
    assert result.prepared_path is not None
    assert {record.platform for record in read_records(result.prepared_path)} == set(configured)
