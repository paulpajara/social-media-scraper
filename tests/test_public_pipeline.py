"""Credential-free CLI integration with synthetic API records and real CSV analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from ph_sentiment import collectors
from ph_sentiment.cli import main, parser
from ph_sentiment.config import PUBLIC_CONFIG, load_config
from ph_sentiment.csvio import read_records
from ph_sentiment.errors import CollectionError
from ph_sentiment.live import PLATFORM_CREDENTIALS, PUBLIC_PLATFORMS
from ph_sentiment.schema import RECORD_FIELDS, Record


@pytest.fixture(autouse=True)
def isolated_public_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI loads only cwd/.env. A fresh directory proves no local setup is needed.
    monkeypatch.chdir(tmp_path)
    for names in PLATFORM_CREDENTIALS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    for name in (
        "BLUESKY_ACCESS_TOKEN",
        "BLUESKY_HANDLE",
        "BLUESKY_PASSWORD",
        "MASTODON_ACCESS_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    def unexpected_request(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Public pipeline tests must use injected collectors without real HTTP.")

    monkeypatch.setattr("requests.sessions.Session.request", unexpected_request)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def public_records(platform: str, count: int) -> list[Record]:
    # Synthetic fixtures exercise three topics; they are never reported as live opinion.
    texts = (
        "Good government health services hospital medicine",
        "Bad government economy inflation transport wages",
        "Government education budget schools teachers classrooms announced",
    )
    return [
        Record(
            f"{platform}:synthetic-{index}",
            platform,
            "2026-06-01T12:00:00Z",
            f"{texts[index % len(texts)]} {platform} example{index}",
            query="Philippine government",
        )
        for index in range(count)
    ]


class FakePublicCollector:
    def __init__(self, platform: str, count: int, fail: bool = False) -> None:
        self.records = public_records(platform, count)
        self.warnings = ["Synthetic public API fixture; bounded test coverage."]
        self.closed = False
        self.fail = fail

    def collect(self) -> list[Record]:
        if self.fail:
            raise CollectionError("Public API returned HTTP 503 after bounded retries.")
        return self.records

    def close(self) -> None:
        self.closed = True


def install_public_collectors(
    monkeypatch: pytest.MonkeyPatch, sources: dict[str, FakePublicCollector]
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def factory(platform: str, config: dict[str, Any]) -> FakePublicCollector:
        calls.append((platform, config))
        return sources[platform]

    monkeypatch.setattr(collectors, "make_collector", factory)
    return calls


def test_public_cli_runs_both_sources_without_credentials_and_merges_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {platform: FakePublicCollector(platform, 2) for platform in PUBLIC_PLATFORMS}
    calls = install_public_collectors(monkeypatch, sources)
    assert parser().parse_args(["collect-public"]).config == PUBLIC_CONFIG
    assert not (tmp_path / ".env").exists()
    assert main(["collect-public", "--skip-analysis"]) == 0
    runs = list((tmp_path / "data" / "public").iterdir())
    assert len(runs) == 1
    output = runs[0]
    assert output.is_dir() and output.name.endswith("Z")

    assert [platform for platform, _ in calls] == list(PUBLIC_PLATFORMS)
    assert all(source.closed for source in sources.values())
    config = load_config(PUBLIC_CONFIG)
    assert all(options["queries"] == config["research"]["queries"] for _, options in calls)
    prepared = read_records(output / "processed" / "posts.csv")
    assert len(prepared) == 4
    assert {record.platform for record in prepared} == set(PUBLIC_PLATFORMS)
    assert all(record.gold_label == "" for record in prepared)
    report = rows(output / "collection_report.csv")
    assert len(report) == 2
    assert all(row["status"] == "success" for row in report)
    for platform in PUBLIC_PLATFORMS:
        raw = output / "raw" / f"{platform}.csv"
        assert len(read_records(raw)) == 2
        assert set(rows(raw)[0]) == set(RECORD_FIELDS)
        assert not {"author", "handle", "user_id", "followers"}.intersection(rows(raw)[0])
        manifest = rows(raw.with_name(f"{platform}_collection_manifest.csv"))[0]
        assert manifest["platform"] == platform
        assert manifest["collected_records"] == "2"
    assert len(rows(output / "processed" / "preprocessing_audit.csv")) == 4
    summary = rows(output / "run_summary.csv")[0]
    assert summary["analysis_status"] == "not_requested"
    assert summary["prepared_records"] == "4"
    assert all(path.suffix == ".csv" for path in output.rglob("*") if path.is_file())


def test_public_cli_runs_real_analysis_at_activity_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {platform: FakePublicCollector(platform, 120) for platform in PUBLIC_PLATFORMS}
    install_public_collectors(monkeypatch, sources)
    output = tmp_path / "analyzed-public-run"
    assert main(["collect-public", "--output-dir", str(output)]) == 0

    prepared = read_records(output / "processed" / "posts.csv")
    assert len(prepared) == 240
    assert all(record.gold_label == "" for record in prepared)
    documents = rows(output / "analysis" / "analyzed_documents.csv")
    assert len(documents) == 240
    assert {row["platform"] for row in documents} == set(PUBLIC_PLATFORMS)
    assert all(row["gold_label"] == "" for row in documents)
    assert all(row["topic_status"] == "assigned" for row in documents)
    assert {row["sentiment_label"] for row in documents} == {"positive", "negative", "neutral"}
    terms = rows(output / "analysis" / "topic_terms.csv")
    assert {row["topic_id"] for row in terms} == {"0", "1", "2"}
    assert rows(output / "analysis" / "topic_metrics.csv")
    assert rows(output / "analysis" / "sentiment_summary.csv")
    assert rows(output / "analysis" / "proximity_pairs.csv")
    summary = rows(output / "run_summary.csv")[0]
    assert summary["analysis_status"] == "success"
    assert summary["prepared_records"] == "240"
    assert summary["sources_with_records"] == "2"
    assert summary["activity_record_minimum_met"] == "True"
    assert all(source.closed for source in sources.values())
    assert all(path.suffix == ".csv" for path in output.rglob("*") if path.is_file())


def test_public_api_failure_preserves_other_source_without_creating_fake_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "bluesky": FakePublicCollector("bluesky", 3),
        "mastodon": FakePublicCollector("mastodon", 0, fail=True),
    }
    install_public_collectors(monkeypatch, sources)
    output = tmp_path / "partial-public-run"
    assert main(["collect-public", "--output-dir", str(output), "--skip-analysis"]) == 0

    prepared = read_records(output / "processed" / "posts.csv")
    assert len(prepared) == 3
    assert {record.platform for record in prepared} == {"bluesky"}
    report = {row["platform"]: row for row in rows(output / "collection_report.csv")}
    assert report["bluesky"]["status"] == "success"
    assert report["mastodon"]["status"] == "failed"
    assert report["mastodon"]["record_count"] == "0"
    assert not (output / "raw" / "mastodon.csv").exists()
    assert all(source.closed for source in sources.values())
