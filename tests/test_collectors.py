"""Mocked integration tests exercise API contracts without live credentials/data."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import requests

from ph_sentiment.collectors import make_collector
from ph_sentiment.collectors.base import HTTPClient, date_bounds
from ph_sentiment.collectors.facebook import FacebookCollector
from ph_sentiment.collectors.reddit import RedditCollector
from ph_sentiment.collectors.threads import ThreadsCollector
from ph_sentiment.collectors.x import XCollector
from ph_sentiment.errors import CollectionError, ConfigurationError


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200, headers: dict[str, str] | None = None):
        self.payload = payload
        self.status_code = status
        self.headers = headers or {}
        self.closed = False

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, deepcopy(kwargs)))
        assert self.responses, "Unexpected API request"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


def mocked_client(*payloads: Any, max_retries: int = 0) -> tuple[HTTPClient, FakeSession]:
    session = FakeSession(
        [
            payload if isinstance(payload, (FakeResponse, Exception)) else FakeResponse(payload)
            for payload in payloads
        ]
    )
    return HTTPClient(session=session, min_interval=0, max_retries=max_retries), session


@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "queries": ["government"],
        "start_date": "2026-01-01",
        "end_date": "2027-01-01",
        "max_records": 10,
        "max_pages": 3,
        "include_comments": False,
        "settings": {},
    }


@pytest.fixture(autouse=True)
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
        "X_BEARER_TOKEN",
        "THREADS_ACCESS_TOKEN",
        "FACEBOOK_ACCESS_TOKEN",
    ):
        monkeypatch.setenv(name, "test-credential")


def reddit_listing(items: list[dict[str, Any]], after: str | None = None) -> dict[str, Any]:
    return {"data": {"children": items, "after": after}}


def reddit_post(identifier: str, created: str, text: str = "government") -> dict[str, Any]:
    return {
        "kind": "t3",
        "data": {
            "id": identifier,
            "title": text,
            "selftext": "",
            "created_utc": datetime.fromisoformat(created).timestamp(),
        },
    }


def test_manila_calendar_dates_are_converted_to_utc() -> None:
    start, end = date_bounds("2026-01-01", "2027-01-01")
    assert start.isoformat() == "2025-12-31T16:00:00+00:00"
    assert end.isoformat() == "2026-12-31T16:00:00+00:00"


@pytest.mark.parametrize(
    "start,end",
    [
        ("2026-02-30", "2026-03-01"),
        ("2026-1-1", "2027-01-01"),
        ("2026-01-01", "2026-01-01"),
        (None, "2027-01-01"),
    ],
)
def test_invalid_collection_dates_fail_safely(start: Any, end: Any) -> None:
    with pytest.raises(ConfigurationError):
        date_bounds(start, end)


def test_http_retries_rate_limits_and_temporary_server_errors() -> None:
    session = FakeSession(
        [
            FakeResponse({}, 429, {"Retry-After": "2"}),
            FakeResponse({}, 503),
            FakeResponse({"data": []}),
        ]
    )
    sleeps: list[float] = []
    client = HTTPClient(session=session, max_retries=3, min_interval=0, sleep=sleeps.append)
    assert client.request_json("GET", "https://api.x.com/2/tweets/search/recent") == {"data": []}
    assert sleeps == [2, 2]
    assert len(session.calls) == 3
    assert all(
        call[2]["timeout"] == 30 and call[2]["allow_redirects"] is False for call in session.calls
    )


def test_long_retry_after_is_not_shortened() -> None:
    session = FakeSession([FakeResponse({}, 429, {"Retry-After": "120"})])
    client = HTTPClient(session=session, max_retries=3, min_interval=0, sleep=lambda _: None)
    with pytest.raises(CollectionError, match="longer wait"):
        client.request_json("GET", "https://api.x.com/2/tweets/search/recent")
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://api.x.com/2/tweets",
        "https://localhost/",
        "https://127.0.0.1/",
        "https://api.x.com.attacker.test/",
        "https://user:secret@api.x.com/",
        "https://api.x.com:444/",
        "https://api.x.com/#fragment",
    ],
)
def test_http_rejects_unsafe_urls_before_network(url: str) -> None:
    client, session = mocked_client()
    with pytest.raises(CollectionError, match="unsafe"):
        client.request_json("GET", url)
    assert session.calls == []


@pytest.mark.parametrize("status", [302, 401, 403, 400])
def test_http_never_includes_upstream_secrets_in_errors(status: int) -> None:
    client, session = mocked_client(FakeResponse({"error": "secret-token-and-request-url"}, status))
    with pytest.raises(CollectionError) as captured:
        client.request_json("GET", "https://graph.facebook.com/v25.0/123/posts")
    assert "secret-token" not in str(captured.value)
    assert len(session.calls) == 1


def test_http_network_retries_are_bounded_and_errors_redacted() -> None:
    client, session = mocked_client(
        requests.ConnectionError("secret-token"),
        requests.ConnectionError("secret-token"),
        max_retries=1,
    )
    client.sleep = lambda _: None
    with pytest.raises(CollectionError, match="bounded retries") as captured:
        client.request_json("GET", "https://graph.threads.net/v1.0/keyword_search")
    assert "secret-token" not in str(captured.value)
    assert len(session.calls) == 2


@pytest.mark.parametrize(
    "payload",
    [ValueError("secret"), {"error": {"message": "secret"}}, {"errors": [{"message": "secret"}]}],
)
def test_http_rejects_invalid_json_and_api_error_payloads(payload: Any) -> None:
    client, _ = mocked_client(FakeResponse(payload))
    with pytest.raises(CollectionError) as captured:
        client.request_json("GET", "https://api.x.com/2/tweets/search/recent")
    assert "secret" not in str(captured.value)


def test_http_partial_errors_require_explicit_opt_in() -> None:
    payload = {"data": [{"id": "101"}], "errors": [{"detail": "secret-token"}]}
    client, _ = mocked_client(payload)
    with pytest.raises(CollectionError) as captured:
        client.request_json("GET", "https://api.x.com/2/tweets/search/recent")
    assert "secret-token" not in str(captured.value)

    client, session = mocked_client(payload)
    assert (
        client.request_json(
            "GET", "https://api.x.com/2/tweets/search/recent", allow_partial_errors=True
        )
        == payload
    )
    assert "allow_partial_errors" not in session.calls[0][2]


@pytest.mark.parametrize(
    "payload",
    [
        {"errors": [{"detail": "secret-token"}]},
        {"data": [], "errors": [{"detail": "secret-token"}]},
        {"data": {"id": "101"}, "errors": [{"detail": "secret-token"}]},
        {"data": [{"id": "101"}], "error": {"message": "secret-token"}},
    ],
)
def test_http_partial_opt_in_keeps_error_only_and_graph_failures_fatal(payload: Any) -> None:
    client, _ = mocked_client(payload)
    with pytest.raises(CollectionError) as captured:
        client.request_json(
            "GET", "https://api.x.com/2/tweets/search/recent", allow_partial_errors=True
        )
    assert "secret-token" not in str(captured.value)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_http_partial_response_option_requires_boolean(value: Any) -> None:
    client, session = mocked_client()
    with pytest.raises(CollectionError, match="true or false"):
        client.request_json(
            "GET", "https://api.x.com/2/tweets/search/recent", allow_partial_errors=value
        )
    assert session.calls == []


def test_reddit_oauth_pagination_deduplication_and_date_filter(config: dict[str, Any]) -> None:
    config["queries"] = ["government", "Marcos"]
    config["settings"] = {"subreddits": ["Philippines"]}
    at_start = reddit_post("abc", "2025-12-31T16:00:00+00:00", "Government &amp; services")
    before_start = reddit_post("old", "2025-12-31T15:59:59+00:00")
    at_end = reddit_post("late", "2026-12-31T16:00:00+00:00")
    client, session = mocked_client(
        {"access_token": "oauth-token", "expires_in": 3600},
        reddit_listing([before_start, at_start, at_end], "t3_abc"),
        reddit_listing([reddit_post("def", "2026-01-02T00:00:00+00:00")]),
        reddit_listing([at_start]),
    )
    collector = RedditCollector(config, client)
    rows = collector.collect()
    assert [row.record_id for row in rows] == ["reddit:t3_abc", "reddit:t3_def"]
    assert rows[0].text == "Government & services"
    assert rows[0].created_at == "2025-12-31T16:00:00Z"
    assert session.calls[0][0:2] == ("POST", "https://www.reddit.com/api/v1/access_token")
    assert session.calls[0][2]["data"] == {"grant_type": "client_credentials"}
    assert session.calls[0][2]["auth"] == ("test-credential", "test-credential")
    assert session.calls[2][2]["params"]["after"] == "t3_abc"
    assert session.calls[1][2]["headers"]["Authorization"] == "Bearer oauth-token"


def test_reddit_comments_expand_omitted_children(config: dict[str, Any]) -> None:
    config["include_comments"] = True
    comment = {
        "kind": "t1",
        "data": {
            "id": "c1",
            "body": "Better services please",
            "created_utc": 1767312000,
            "replies": "",
        },
    }
    more = {"kind": "more", "data": {"children": ["c2", "c2"]}}
    comment2 = {"kind": "t1", "data": {"id": "c2", "body": "Improving", "created_utc": 1767312000}}
    client, session = mocked_client(
        {"access_token": "oauth-token", "expires_in": 3600},
        reddit_listing([reddit_post("abc", "2026-01-02T00:00:00+00:00")]),
        [reddit_listing([]), reddit_listing([comment, more])],
        {"json": {"errors": [], "data": {"things": [comment2]}}},
    )
    rows = RedditCollector(config, client).collect()
    assert [row.record_type for row in rows] == ["post", "comment", "comment"]
    assert session.calls[-1][1] == "https://oauth.reddit.com/api/morechildren"
    assert session.calls[-1][2]["params"]["children"] == "c2"
    assert session.calls[-1][2]["params"]["link_id"] == "t3_abc"


def test_reddit_token_is_renewed_after_expiration(config: dict[str, Any]) -> None:
    client, session = mocked_client(
        {"access_token": "first", "expires_in": 31}, {"access_token": "second", "expires_in": 3600}
    )
    now = [0.0]
    client.clock = lambda: now[0]
    collector = RedditCollector(config, client)
    assert collector._headers()["Authorization"] == "Bearer first"
    now[0] = 2
    assert collector._headers()["Authorization"] == "Bearer second"
    assert len(session.calls) == 2


def test_x_archive_pagination_date_window_and_minimum_page_size(config: dict[str, Any]) -> None:
    config["max_records"] = 2
    config["settings"] = {"search_mode": "archive"}
    client, session = mocked_client(
        {
            "data": [
                {"id": "101", "text": "Government services", "created_at": "2026-02-01T00:00:00Z"},
                {"id": "102", "text": "Too old", "created_at": "2025-12-31T15:59:59Z"},
            ],
            "meta": {"next_token": "next"},
        },
        {"data": [{"id": "103", "text": "Public health", "created_at": "2026-02-02T00:00:00Z"}]},
    )
    rows = XCollector(config, client).collect()
    assert [row.record_id for row in rows] == ["x:101", "x:103"]
    assert session.calls[0][1] == "https://api.x.com/2/tweets/search/all"
    assert session.calls[0][2]["params"]["max_results"] == 10
    assert session.calls[0][2]["params"]["tweet.fields"] == "created_at,note_tweet"
    assert session.calls[1][2]["params"]["next_token"] == "next"


def test_x_retains_partial_results_and_redacts_api_errors(config: dict[str, Any]) -> None:
    config["settings"] = {"search_mode": "archive"}
    client, session = mocked_client(
        {
            "data": [
                {"id": "101", "text": "Government services", "created_at": "2026-02-01T00:00:00Z"}
            ],
            "errors": [{"detail": "secret-token-and-request-url"}],
        }
    )
    collector = XCollector(config, client)
    assert [row.record_id for row in collector.collect()] == ["x:101"]
    assert any("partial results" in warning for warning in collector.warnings)
    assert all("secret-token" not in warning for warning in collector.warnings)
    assert "allow_partial_errors" not in session.calls[0][2]


def test_x_recent_refuses_unavailable_history(config: dict[str, Any]) -> None:
    config["start_date"] = "2006-01-01"
    client, session = mocked_client()
    with pytest.raises(ConfigurationError, match="last seven days"):
        XCollector(config, client)
    assert session.calls == []


def test_x_recent_uses_documented_endpoint(config: dict[str, Any]) -> None:
    now = datetime.now(UTC)
    config["start_date"] = (now - timedelta(days=2)).date().isoformat()
    config["end_date"] = (now + timedelta(days=1)).date().isoformat()
    client, session = mocked_client({"meta": {"result_count": 0}})
    assert XCollector(config, client).collect() == []
    assert session.calls[0][1] == "https://api.x.com/2/tweets/search/recent"
    assert (
        datetime.fromisoformat(session.calls[0][2]["params"]["end_time"].replace("Z", "+00:00"))
        < now
    )


def test_threads_paging_reconstructs_cursor_without_returned_token(config: dict[str, Any]) -> None:
    config["settings"] = {"search_type": "RECENT", "search_mode": "TAG"}
    client, session = mocked_client(
        {
            "data": [
                {
                    "id": "1",
                    "text": "Government services",
                    "timestamp": "2026-03-01T12:00:00+0000",
                    "permalink": "https://www.threads.com/@example/post/test",
                }
            ],
            "paging": {
                "cursors": {"after": "cursor"},
                "next": "https://graph.threads.net/v1.0/keyword_search?after=cursor&access_token=returned-secret",
            },
        },
        {"data": [{"id": "2", "text": "Too late", "timestamp": "2026-12-31T16:00:00Z"}]},
    )
    rows = ThreadsCollector(config, client).collect()
    assert [row.record_id for row in rows] == ["threads:1"]
    assert session.calls[1][1] == "https://graph.threads.net/v1.0/keyword_search"
    assert session.calls[1][2]["params"]["after"] == "cursor"
    assert "access_token" not in session.calls[1][2]["params"]
    assert session.calls[0][2]["params"]["search_mode"] == "TAG"
    assert session.calls[0][2]["headers"]["Authorization"] == "Bearer test-credential"


@pytest.mark.parametrize(
    "next_url",
    [
        "https://attacker.test/?access_token=secret",
        "http://graph.threads.net/v1.0/keyword_search?after=next",
        "https://graph.threads.net/v1.0/me?after=next",
    ],
)
def test_graph_refuses_unsafe_paging_links(config: dict[str, Any], next_url: str) -> None:
    client, session = mocked_client(
        {"data": [], "paging": {"next": next_url, "cursors": {"after": "next"}}}
    )
    with pytest.raises(CollectionError):
        ThreadsCollector(config, client).collect()
    assert len(session.calls) == 1


def test_graph_stops_repeated_cursors_and_reports_budget(config: dict[str, Any]) -> None:
    page = {
        "data": [],
        "paging": {
            "next": "https://graph.threads.net/v1.0/keyword_search?after=repeat",
            "cursors": {"after": "repeat"},
        },
    }
    client, session = mocked_client(page, page)
    collector = ThreadsCollector(config, client)
    collector.collect()
    assert len(session.calls) == 2
    assert any("repeated cursor" in warning for warning in collector.warnings)
    config["max_pages"] = 1
    client, _ = mocked_client(page)
    collector = ThreadsCollector(config, client)
    collector.collect()
    assert any("max_pages" in warning for warning in collector.warnings)


def test_facebook_literal_page_post_filter_and_comments(config: dict[str, Any]) -> None:
    config["settings"] = {"page_ids": ["123"], "graph_version": "v25.0"}
    config["include_comments"] = True
    client, session = mocked_client(
        {
            "data": [
                {
                    "id": "123_1",
                    "message": "GOVERNMENT services",
                    "created_time": "2026-04-01T12:00:00+0000",
                },
                {
                    "id": "123_2",
                    "message": "Unrelated topic",
                    "created_time": "2026-04-01T12:00:00+0000",
                },
            ]
        },
        {
            "data": [
                {
                    "id": "123_3",
                    "message": "Improve them please",
                    "created_time": "2026-04-02T12:00:00+0000",
                },
                {
                    "id": "123_4",
                    "message": "Wrong date",
                    "created_time": "2026-12-31T16:00:00+0000",
                },
            ]
        },
    )
    collector = FacebookCollector(config, client)
    rows = collector.collect()
    assert [row.record_id for row in rows] == ["facebook:123_1", "facebook:123_3"]
    assert [row.record_type for row in rows] == ["post", "comment"]
    assert session.calls[0][1] == "https://graph.facebook.com/v25.0/123/posts"
    assert session.calls[1][1] == "https://graph.facebook.com/v25.0/123_1/comments"
    assert all("from" not in call[2]["params"]["fields"] for call in session.calls)
    assert any("not unrestricted" in warning for warning in collector.warnings)


def test_facebook_requires_explicit_version_and_authorized_page_ids(config: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError, match="graph_version"):
        FacebookCollector(config)
    config["settings"] = {"graph_version": "v25.0", "page_ids": ["../../me"]}
    with pytest.raises(ConfigurationError, match="page_ids"):
        FacebookCollector(config)


def test_record_budget_is_global_across_queries(config: dict[str, Any]) -> None:
    config["max_records"] = 1
    config["queries"] = ["government", "services"]
    client, session = mocked_client(
        {
            "data": [
                {"id": "1", "text": "Government", "timestamp": "2026-03-01T00:00:00Z"},
                {"id": "2", "text": "Services", "timestamp": "2026-03-01T00:00:00Z"},
            ]
        }
    )
    assert len(ThreadsCollector(config, client).collect()) == 1
    assert len(session.calls) == 1


def test_missing_credentials_are_actionable_without_network(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("THREADS_ACCESS_TOKEN")
    client, session = mocked_client()
    with pytest.raises(ConfigurationError, match="THREADS_ACCESS_TOKEN"):
        ThreadsCollector(config, client).collect()
    assert session.calls == []


@pytest.mark.parametrize("collector_type", [XCollector, ThreadsCollector])
def test_unsupported_comment_collection_is_explicit(
    config: dict[str, Any], collector_type: Any
) -> None:
    config["include_comments"] = True
    config["settings"] = {"search_mode": "archive"} if collector_type is XCollector else {}
    with pytest.raises(ConfigurationError, match="include_comments=false"):
        collector_type(config)


def test_factory_rejects_unsupported_platform(config: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError, match="supported collector"):
        make_collector("unknown", config)


@pytest.mark.parametrize(
    "field_parameter,note_field", [("tweet.fields", "note_tweet"), ("post.fields", "note_post")]
)
def test_x_full_note_text_and_documented_field_spellings(
    config: dict[str, Any], field_parameter: str, note_field: str
) -> None:
    config["settings"] = {"search_mode": "archive", "fields_parameter": field_parameter}
    client, session = mocked_client(
        {
            "data": [
                {
                    "id": "101",
                    "text": "Truncated...",
                    "created_at": "2026-02-01T00:00:00Z",
                    note_field: {"text": "Complete government opinion with the entire explanation"},
                }
            ]
        }
    )
    rows = XCollector(config, client).collect()
    assert rows[0].text == "Complete government opinion with the entire explanation"
    assert session.calls[0][2]["params"][field_parameter] == f"created_at,{note_field}"
    assert session.calls[0][2]["params"]["sort_order"] == "recency"


def test_missing_invalid_and_naive_api_timestamps_are_skipped_with_warning(
    config: dict[str, Any],
) -> None:
    client, _ = mocked_client(
        {
            "data": [
                {"id": "1", "text": "government", "timestamp": None},
                {"id": "2", "text": "government", "timestamp": "not-a-date"},
                {"id": "3", "text": "government", "timestamp": "2026-02-01T00:00:00"},
            ]
        }
    )
    collector = ThreadsCollector(config, client)
    assert collector.collect() == []
    assert any("invalid publication timestamp" in warning for warning in collector.warnings)


def test_threads_provenance_url_removes_query_tokens(config: dict[str, Any]) -> None:
    client, _ = mocked_client(
        {
            "data": [
                {
                    "id": "1",
                    "text": "government",
                    "timestamp": "2026-02-01T00:00:00Z",
                    "permalink": "https://www.threads.com/@example/post/abc?access_token=secret#fragment",
                }
            ]
        }
    )
    rows = ThreadsCollector(config, client).collect()
    assert rows[0].source_url == "https://www.threads.com/@example/post/abc"


def test_reddit_comment_budget_reports_incomplete_tree(config: dict[str, Any]) -> None:
    config["include_comments"] = True
    config["max_pages"] = 1
    client, session = mocked_client(
        {"access_token": "oauth-token", "expires_in": 3600},
        reddit_listing([reddit_post("abc", "2026-01-02T00:00:00+00:00")]),
        [reddit_listing([]), reddit_listing([{"kind": "more", "data": {"children": ["c1"]}}])],
    )
    collector = RedditCollector(config, client)
    assert len(collector.collect()) == 1
    assert len(session.calls) == 3
    assert any("comment trees are incomplete" in warning for warning in collector.warnings)


@pytest.mark.parametrize(
    "settings",
    [
        {"timeout": 0},
        {"max_retries": -1},
        {"max_retries": True},
        {"min_interval": None},
        {"max_wait": "60"},
    ],
)
def test_invalid_http_settings_are_configuration_errors(settings: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        HTTPClient(**settings)


def test_http_ignores_nonfinite_retry_after(config: dict[str, Any]) -> None:
    sleeps: list[float] = []
    session = FakeSession(
        [FakeResponse({}, 429, {"Retry-After": "NaN"}), FakeResponse({"data": []})]
    )
    client = HTTPClient(session=session, min_interval=0, max_retries=1, sleep=sleeps.append)
    assert client.request_json("GET", "https://api.x.com/2/tweets/search/recent") == {"data": []}
    assert sleeps == [1]
