from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from ph_sentiment.collectors.base import HTTPClient
from ph_sentiment.collectors.threads import ThreadsCollector
from ph_sentiment.config import DEFAULT_CONFIG, load_config
from ph_sentiment.errors import HTTPCollectionError


@pytest.mark.parametrize(
    "status,category", [(401, "authentication"), (402, "payment_required"), (403, "permission")]
)
def test_http_access_failures_are_actionable_and_never_expose_response_bodies(status, category):
    response = Mock(status_code=status, headers={})
    response.json.return_value = {"detail": "sensitive-upstream-body"}
    session = Mock()
    session.request.return_value = response
    client = HTTPClient(session=session, min_interval=0)
    with pytest.raises(HTTPCollectionError) as caught:
        client.request_json(
            "GET",
            "https://api.x.com/2/tweets/search/all",
            headers={"Authorization": "Bearer private-test-token"},
        )
    assert caught.value.status_code == status
    assert caught.value.category == category
    assert "private-test-token" not in str(caught.value)
    assert "sensitive-upstream-body" not in str(caught.value)
    response.json.assert_called_once()
    response.close.assert_called_once()


@pytest.mark.parametrize(
    "host,status,payload,category",
    [
        (
            "api.x.com",
            402,
            {"title": "CreditsDepleted", "detail": "sensitive-account"},
            "credits_depleted",
        ),
        (
            "api.x.com",
            402,
            {"title": "PaymentRequired", "type": "https://api.x.com/2/problems/credits-depleted"},
            "credits_depleted",
        ),
        (
            "api.x.com",
            403,
            {"type": "https://api.x.com/2/problems/client-forbidden"},
            "endpoint_access",
        ),
        (
            "graph.threads.net",
            400,
            {"error": {"code": 190, "message": "private-test-token"}},
            "authentication",
        ),
        ("graph.threads.net", 403, {"error": {"code": 10}}, "permission_missing"),
        ("api.x.com", 402, {"type": []}, "payment_required"),
        ("api.x.com", 402, {"type": {}, "title": ["CreditsDepleted"]}, "payment_required"),
        ("api.x.com", 402, {"type": "https://attacker.test/credits-depleted"}, "payment_required"),
        ("graph.threads.net", 400, {"error": {"code": {"190": True}}}, "http_error"),
        ("graph.threads.net", 403, {"error": {"code": []}}, "permission"),
        ("graph.threads.net", 403, {"error": {"code": True}}, "permission"),
        ("api.x.com", 402, ValueError("private-test-token"), "payment_required"),
    ],
)
def test_access_error_codes_are_allowlisted_and_malformed_values_are_safe(
    host, status, payload, category
):
    response = Mock(status_code=status, headers={})
    if isinstance(payload, Exception):
        response.json.side_effect = payload
    else:
        response.json.return_value = payload
    session = Mock()
    session.request.return_value = response
    with pytest.raises(HTTPCollectionError) as caught:
        HTTPClient(session=session, min_interval=0).request_json("GET", f"https://{host}/test")
    assert caught.value.category == category
    assert "sensitive-account" not in str(caught.value)
    assert "private-test-token" not in str(caught.value)
    assert session.request.call_count == 1
    response.close.assert_called_once()


def test_threads_clamps_future_end_date_to_current_time(monkeypatch):
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "synthetic-test-token")
    config = load_config(DEFAULT_CONFIG)
    research = {
        **config["research"],
        "max_pages": 1,
        "queries": ["government"],
        "end_date": (datetime.now(UTC).date() + timedelta(days=365)).isoformat(),
        "settings": {**config["threads"], "check_token": False},
    }
    client = Mock()
    client.request_json.return_value = {"data": []}
    before = int(datetime.now(UTC).timestamp())
    assert ThreadsCollector(research, client).collect() == []
    after = int(datetime.now(UTC).timestamp())
    params = client.request_json.call_args.kwargs["params"]
    assert before <= params["until"] <= after


def test_threads_future_window_does_not_send_requests(monkeypatch):
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "synthetic-test-token")
    config = load_config(DEFAULT_CONFIG)
    research = {
        **config["research"],
        "start_date": "2098-01-01",
        "end_date": "2099-01-01",
        "settings": config["threads"],
    }
    client = Mock()
    assert ThreadsCollector(research, client).collect() == []
    client.request_json.assert_not_called()
