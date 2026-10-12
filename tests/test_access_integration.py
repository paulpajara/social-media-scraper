"""Offline integration of access checks, collection failures and CSV checkpoints."""

import csv
from unittest.mock import Mock

import pytest

from ph_sentiment import cli, collectors
from ph_sentiment.collectors.base import HTTPClient
from ph_sentiment.collectors.threads import ThreadsCollector
from ph_sentiment.collectors.x import XCollector
from ph_sentiment.config import load_config
from ph_sentiment.csvio import read_records, write_csv
from ph_sentiment.errors import ConfigurationError, HTTPCollectionError
from ph_sentiment.live import collector_config, live_collect


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "synthetic-threads-secret")
    monkeypatch.setenv("X_BEARER_TOKEN", "synthetic-x-secret")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)


@pytest.mark.parametrize("partial", [False, True])
def test_failed_x_query_preserves_diagnostics_and_any_received_records(
    tmp_path, monkeypatch, partial
):
    config = load_config()
    config["x"].update(max_pages=2, max_records=10)
    client = Mock()
    responses = []
    if partial:
        responses.append(
            {
                "data": [
                    {
                        "id": "123",
                        "text": "Government services",
                        "created_at": "2026-06-01T00:00:00Z",
                    }
                ],
                "meta": {"next_token": "next"},
            }
        )
    responses.append(HTTPCollectionError(402, "credits_depleted", "X API credits depleted."))
    client.request_json.side_effect = responses
    collector = XCollector(collector_config("x", config), client)
    monkeypatch.setattr(collectors, "make_collector", lambda *args: collector)
    result = live_collect(config, tmp_path, ["x"], skip_analysis=True, progress=lambda _: None)
    report = csv_rows(result.report_path)[0]
    assert report["status"] == ("partial" if partial else "failed")
    assert report["http_status"] == "402"
    assert report["error_category"] == "credits_depleted"
    assert report["diagnostics_path"] == str(tmp_path / "raw/x_query_diagnostics.csv")
    diagnostic = csv_rows(tmp_path / "raw/x_query_diagnostics.csv")[0]
    assert diagnostic["status"] == report["status"]
    assert diagnostic["api_records"] == str(int(partial))
    assert diagnostic["retained_records"] == str(int(partial))
    assert result.record_count == int(partial)
    if partial:
        assert len(read_records(result.raw_paths[0])) == 1
    else:
        assert not (tmp_path / "raw/x.csv").exists()
    assert "synthetic-x-secret" not in result.report_path.read_text(encoding="utf-8-sig")


def test_empty_threads_api_has_explanatory_csv_and_nonzero_single_collect_exit(
    tmp_path, monkeypatch, capsys
):
    client = Mock()
    client.request_json.side_effect = [
        {
            "data": {
                "is_valid": True,
                "scopes": ["threads_basic", "threads_keyword_search"],
                "expires_at": 0,
            }
        },
        *({"data": []} for _ in range(4)),
    ]
    monkeypatch.setattr(
        collectors, "make_collector", lambda platform, config: ThreadsCollector(config, client)
    )
    output = tmp_path / "threads.csv"
    assert cli.main(["collect", "--platform", "threads", "--output", str(output)]) == 2
    assert read_records(output) == []
    diagnostics = csv_rows(tmp_path / "threads_query_diagnostics.csv")
    assert len(diagnostics) == 4
    assert all(row["status"] == "empty_api" and row["api_records"] == "0" for row in diagnostics)
    assert "before local filtering" in capsys.readouterr().err


def test_failed_construction_does_not_link_old_diagnostic_csv(tmp_path, monkeypatch):
    old = tmp_path / "raw/x_query_diagnostics.csv"
    write_csv(old, [{"old": "previous-run"}], ["old"])

    def fail(*args):
        raise ConfigurationError("Invalid X settings.")

    monkeypatch.setattr(collectors, "make_collector", fail)
    result = live_collect(load_config(), tmp_path, ["x"], overwrite=True, progress=lambda _: None)
    assert csv_rows(result.report_path)[0]["diagnostics_path"] == ""


def test_access_cli_returns_blocked_status_and_safe_csv(tmp_path, monkeypatch):
    def response(self, method, url, **kwargs):
        if "debug_token" in url:
            return {
                "data": {
                    "is_valid": True,
                    "scopes": ["threads_basic", "threads_keyword_search"],
                    "expires_at": 0,
                    "user_id": "private-user",
                }
            }
        assert url == "https://api.x.com/2/usage/credits"
        return {"data": {"total_balance": 0}}

    monkeypatch.setattr(HTTPClient, "request_json", response)
    output = tmp_path / "access.csv"
    assert cli.main(["check-access", "--output", str(output)]) == 2
    assert [row["status"] for row in csv_rows(output)] == ["scope_granted", "blocked"]
    contents = output.read_text(encoding="utf-8-sig")
    assert "synthetic" not in contents and "private-user" not in contents


def test_access_cli_protects_existing_report_before_network(tmp_path, monkeypatch):
    output = write_csv(tmp_path / "access.csv", [{"original": "preserve"}], ["original"])
    before = output.read_bytes()
    client_call = Mock(side_effect=AssertionError("Unexpected request"))
    monkeypatch.setattr(HTTPClient, "request_json", client_call)
    assert cli.main(["check-access", "--output", str(output)]) == 2
    assert output.read_bytes() == before
    client_call.assert_not_called()


def test_single_collect_protects_diagnostic_csv_before_network(tmp_path, monkeypatch):
    output = tmp_path / "threads.csv"
    diagnostic = write_csv(
        tmp_path / "threads_query_diagnostics.csv", [{"original": "preserve"}], ["original"]
    )
    before = diagnostic.read_bytes()
    factory = Mock(side_effect=AssertionError("Unexpected collection"))
    monkeypatch.setattr(collectors, "make_collector", factory)
    assert cli.main(["collect", "--platform", "threads", "--output", str(output)]) == 2
    assert diagnostic.read_bytes() == before
    factory.assert_not_called()
