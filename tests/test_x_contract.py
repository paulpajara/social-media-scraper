"""Regression checks for X's current search contract and full-text fallback."""

from unittest.mock import Mock

import pytest

from ph_sentiment.collectors.x import XCollector
from ph_sentiment.errors import HTTPCollectionError


@pytest.fixture
def archive_config(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "synthetic-test-token")
    return {
        "queries": ["Philippine government"],
        "start_date": "2026-01-01",
        "end_date": "2027-01-01",
        "max_records": 10,
        "max_pages": 1,
        "include_comments": False,
        "settings": {"search_mode": "archive"},
    }


def test_default_x_contract_requests_and_preserves_current_post_fields(archive_config):
    client = Mock()
    client.request_json.return_value = {
        "data": [
            {
                "id": "123",
                "text": "Truncated...",
                "created_at": "2026-02-01T00:00:00Z",
                "note_post": {"text": "Complete opinion about Philippine government services"},
            }
        ],
        "meta": {"result_count": 1},
    }
    records = XCollector(archive_config, client).collect()
    params = client.request_json.call_args.kwargs["params"]
    assert params["post.fields"] == "created_at,note_post"
    assert "tweet.fields" not in params
    assert records[0].text == "Complete opinion about Philippine government services"
    assert records[0].created_at == "2026-02-01T00:00:00Z"


@pytest.mark.parametrize("note", [None, {}, {"text": "  "}, {"text": 42}, "invalid-note"])
def test_x_invalid_or_empty_note_preserves_standard_post_text(archive_config, note):
    client = Mock()
    client.request_json.return_value = {
        "data": [
            {
                "id": "123",
                "text": "Public opinion about government services",
                "created_at": "2026-02-01T00:00:00Z",
                "note_post": note,
            }
        ]
    }
    records = XCollector(archive_config, client).collect()
    assert len(records) == 1
    assert records[0].text == "Public opinion about government services"


@pytest.mark.parametrize(
    "items,status",
    [
        ([], "empty_api"),
        (
            [
                {
                    "id": "123",
                    "text": "A post outside the study window",
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ],
            "filtered_empty",
        ),
    ],
)
def test_x_query_diagnostics_distinguish_api_empty_from_local_filtering(
    archive_config, items, status
):
    client = Mock()
    client.request_json.return_value = {"data": items}
    collector = XCollector(archive_config, client)
    assert collector.collect() == []
    assert collector.query_diagnostics[0]["status"] == status
    assert collector.query_diagnostics[0]["api_records"] == len(items)
    assert collector.query_diagnostics[0]["retained_records"] == 0


def test_x_failed_query_retains_safe_payment_diagnostic(archive_config):
    client = Mock()
    failure = HTTPCollectionError(402, "payment_required", "X API requires account credits.")
    client.request_json.side_effect = failure
    collector = XCollector(archive_config, client)
    with pytest.raises(HTTPCollectionError) as caught:
        collector.collect()
    assert caught.value is failure
    assert collector.query_diagnostics == [
        {
            "platform": "x",
            "query": "Philippine government",
            "api_records": 0,
            "retained_records": 0,
            "status": "failed",
            "message": "X API requires account credits.",
            "http_status": 402,
            "error_category": "payment_required",
        }
    ]


def test_x_query_diagnostics_preserve_partial_page_results(archive_config):
    archive_config["max_pages"] = 2
    client = Mock()
    client.request_json.side_effect = [
        {
            "data": [
                {
                    "id": "123",
                    "text": "Government services",
                    "created_at": "2026-02-01T00:00:00Z",
                }
            ],
            "meta": {"next_token": "page-two"},
        },
        HTTPCollectionError(402, "payment_required", "X API requires account credits."),
    ]
    collector = XCollector(archive_config, client)
    with pytest.raises(HTTPCollectionError):
        collector.collect()
    assert len(collector.records) == 1
    assert collector.query_diagnostics[0]["status"] == "partial"
    assert collector.query_diagnostics[0]["api_records"] == 1
    assert collector.query_diagnostics[0]["retained_records"] == 1


def test_x_record_budget_still_saves_successful_query_diagnostic(archive_config):
    archive_config["max_records"] = 1
    archive_config["queries"] = ["Philippine government", "gobyerno"]
    client = Mock()
    client.request_json.return_value = {
        "data": [
            {
                "id": "123",
                "text": "Government services",
                "created_at": "2026-02-01T00:00:00Z",
            }
        ],
        "meta": {"next_token": "page-two"},
    }
    collector = XCollector(archive_config, client)
    assert len(collector.collect()) == 1
    client.request_json.assert_called_once()
    assert len(collector.query_diagnostics) == 1
    assert collector.query_diagnostics[0]["status"] == "success"
