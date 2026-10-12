"""Offline Threads access checks distinguish upstream emptiness from local filtering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import Mock

import pytest

from ph_sentiment.collectors.threads import ThreadsCollector
from ph_sentiment.errors import AccessCollectionError, CollectionError, ConfigurationError


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "offline-test-token")
    return {
        "queries": ["government"],
        "start_date": "2026-01-01",
        "end_date": "2027-01-01",
        "max_records": 10,
        "max_pages": 2,
        "include_comments": False,
        "settings": {},
    }


def token_data(**overrides: Any) -> dict[str, Any]:
    return {
        "data": {
            "is_valid": True,
            "scopes": ["threads_basic", "threads_keyword_search"],
            "expires_at": 0,
            **overrides,
        }
    }


def post(identifier: str = "1", **overrides: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "text": "The government should improve public services.",
        "timestamp": "2026-04-01T00:00:00Z",
        "permalink": "https://www.threads.com/@example/post/example",
        **overrides,
    }


def client_with(*responses: Any) -> Mock:
    client = Mock()
    client.request_json.side_effect = list(responses)
    return client


def test_valid_token_preflight_retains_posts_and_respects_budget(config: dict[str, Any]) -> None:
    client = client_with(token_data(), {"data": [post()]})
    collector = ThreadsCollector(config, client)

    records = collector.collect()

    assert len(records) == 1
    debug_call, search_call = client.request_json.call_args_list
    assert debug_call.args == ("GET", "https://graph.threads.net/debug_token")
    assert debug_call.kwargs["params"] == {"input_token": "offline-test-token"}
    assert debug_call.kwargs["headers"]["Authorization"] == "Bearer offline-test-token"
    assert search_call.kwargs["params"]["limit"] == 10
    assert collector.query_diagnostics[0]["api_records"] == 1
    assert collector.query_diagnostics[0]["retained_records"] == 1
    assert collector.query_diagnostics[0]["status"] == "success"


@pytest.mark.parametrize("scopes", [[], ["threads_basic"], ["threads_keyword_search"]])
def test_missing_search_scopes_fail_before_keyword_request(
    config: dict[str, Any], scopes: list[str]
) -> None:
    client = client_with(token_data(scopes=scopes))
    collector = ThreadsCollector(config, client)

    with pytest.raises(AccessCollectionError, match="regenerate") as captured:
        collector.collect()

    assert captured.value.category == "permission_missing"
    assert client.request_json.call_count == 1
    assert collector.query_diagnostics == []


def test_invalid_token_fails_before_keyword_request(config: dict[str, Any]) -> None:
    client = client_with(token_data(is_valid=False))

    with pytest.raises(AccessCollectionError, match="invalid or expired") as captured:
        ThreadsCollector(config, client).collect()

    assert captured.value.category == "authentication"
    assert client.request_json.call_count == 1


def test_unavailable_debugger_preserves_collection_and_fixed_warning(
    config: dict[str, Any],
) -> None:
    client = client_with(CollectionError("upstream-secret-token"), {"data": [post()]})
    collector = ThreadsCollector(config, client)

    assert len(collector.collect()) == 1
    assert any(
        "Could not verify Threads token metadata" in warning for warning in collector.warnings
    )
    assert all("upstream-secret-token" not in warning for warning in collector.warnings)


def test_empty_api_is_distinguished_from_local_filters(config: dict[str, Any]) -> None:
    client = client_with(token_data(), {"data": []})
    collector = ThreadsCollector(config, client)

    assert collector.collect() == []
    diagnostic = collector.query_diagnostics[0]
    assert diagnostic["status"] == "empty_api"
    assert diagnostic["api_records"] == diagnostic["retained_records"] == 0
    assert "before local filtering" in diagnostic["message"]
    assert "do not prove" in diagnostic["message"]
    assert not any("missing permission" in warning for warning in collector.warnings)


def test_local_date_text_and_duplicate_filters_report_api_counts(config: dict[str, Any]) -> None:
    config["queries"] = ["government", "public services"]
    client = client_with(
        token_data(),
        {"data": [post()]},
        {
            "data": [
                post(),
                post("2", timestamp="2025-03-01T00:00:00Z"),
                post("3", text=""),
                post("invalid"),
            ]
        },
    )
    collector = ThreadsCollector(config, client)

    assert len(collector.collect()) == 1
    assert collector.query_diagnostics[1]["status"] == "filtered_empty"
    assert collector.query_diagnostics[1]["api_records"] == 4
    assert collector.query_diagnostics[1]["retained_records"] == 0
    assert client.request_json.call_args_list[2].kwargs["params"]["limit"] == 9


def test_api_failure_preserves_partial_query_counts(config: dict[str, Any]) -> None:
    client = client_with(
        token_data(),
        {
            "data": [post()],
            "paging": {
                "next": "https://graph.threads.net/v1.0/keyword_search?after=next",
                "cursors": {"after": "next"},
            },
        },
        CollectionError("safe failure"),
    )
    collector = ThreadsCollector(config, client)

    with pytest.raises(CollectionError, match="safe failure"):
        collector.collect()

    assert len(collector.records) == 1
    assert collector.query_diagnostics[0]["status"] == "partial"
    assert collector.query_diagnostics[0]["api_records"] == 1
    assert collector.query_diagnostics[0]["retained_records"] == 1


def test_near_expiry_warns_with_only_safe_timestamp(config: dict[str, Any]) -> None:
    expiry = datetime.now(UTC) + timedelta(hours=1)
    client = client_with(token_data(expires_at=int(expiry.timestamp())), {"data": []})
    collector = ThreadsCollector(config, client)

    collector.collect()

    assert any("Threads token expires at" in warning for warning in collector.warnings)
    assert all("offline-test-token" not in warning for warning in collector.warnings)


def test_expired_timestamp_stops_collection_even_when_debugger_marks_valid(config):
    expiry = int((datetime.now(UTC) - timedelta(seconds=1)).timestamp())
    client = client_with(token_data(expires_at=expiry))
    with pytest.raises(AccessCollectionError) as caught:
        ThreadsCollector(config, client).collect()
    assert caught.value.category == "authentication"
    assert client.request_json.call_count == 1


def test_preflight_can_be_disabled_explicitly(config: dict[str, Any]) -> None:
    config["settings"]["check_token"] = False
    client = client_with({"data": [post()]})

    assert len(ThreadsCollector(config, client).collect()) == 1
    assert client.request_json.call_count == 1
    assert "keyword_search" in client.request_json.call_args.args[1]


@pytest.mark.parametrize("setting", ["true", 1, None, []])
def test_preflight_setting_requires_boolean(config: dict[str, Any], setting: Any) -> None:
    config["settings"]["check_token"] = setting

    with pytest.raises(ConfigurationError, match="check_token must be true or false"):
        ThreadsCollector(config, client_with())
