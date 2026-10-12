"""Public API contract tests use deterministic fixtures and never contact live services."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import pytest

from ph_sentiment.collectors.bluesky import SEARCH_ENDPOINT, BlueskyCollector
from ph_sentiment.collectors.mastodon import INSTANCE, MastodonCollector, plain_text
from ph_sentiment.config import PUBLIC_CONFIG, load_config
from ph_sentiment.errors import CollectionError, ConfigurationError
from ph_sentiment.live import collector_config

DID = "did:plc:abcdefghijklmnopqrstuvwx"


class PublicClient:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request_json(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        self.calls.append((method, endpoint, deepcopy(kwargs)))
        assert self.responses, "Unexpected API request"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "queries": ["Philippine government"],
        "start_date": "2026-01-01",
        "end_date": "2027-01-01",
        "max_records": 10,
        "max_pages": 3,
        "include_comments": False,
        "settings": {
            "hashtags": ["Philippines"],
            "government_terms": ["government", "marcos", "gobyerno"],
            "country_terms": ["philippines", "philippine", "pilipinas"],
        },
    }


def bluesky_post(identifier: str = "post1", **overrides: Any) -> dict[str, Any]:
    return {
        "uri": f"at://{DID}/app.bsky.feed.post/{identifier}",
        "record": {
            "text": "The Philippine government should improve public services.",
            "createdAt": "2026-02-01T00:00:00Z",
        },
        # Author/profile data is intentionally absent from every resulting Record.
        "author": {"handle": "private-fixture-handle", "description": "private-fixture-profile"},
        **overrides,
    }


def mastodon_post(identifier: str = "300", **overrides: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "created_at": "2026-02-01T00:00:00Z",
        "content": "<p>Philippine government services &amp; healthcare need improvement.</p>",
        "visibility": "public",
        "reblog": None,
        "spoiler_text": "",
        "url": f"https://mastodon.social/@public-fixture/{identifier}?tracking=discard#fragment",
        "account": {"display_name": "private-fixture-display"},
        **overrides,
    }


def test_bluesky_public_request_dates_pagination_and_global_dedup(config: dict[str, Any]) -> None:
    config["queries"].append("Marcos Philippines")
    first = bluesky_post()
    client = PublicClient(
        {"posts": [first], "cursor": "next-page"},
        {"posts": [first, bluesky_post("post2")]},
        {"posts": [first]},
    )
    collector = BlueskyCollector(config, client)
    rows = collector.collect()
    assert len(rows) == 2
    assert rows[0].record_id == f"bluesky:at://{DID}/app.bsky.feed.post/post1"
    assert rows[0].source_url == f"https://bsky.app/profile/{DID}/post/post1"
    assert "private-fixture" not in str(rows[0].to_dict())
    assert client.calls[0][0:2] == ("GET", SEARCH_ENDPOINT)
    assert client.calls[0][2]["params"] == {
        "q": "Philippine government",
        "sort": "latest",
        "limit": 10,
        "since": "2025-12-31T16:00:00Z",
        "until": collector.until.isoformat().replace("+00:00", "Z"),
    }
    assert client.calls[1][2]["params"]["cursor"] == "next-page"
    assert "cursor" not in client.calls[2][2]["params"]
    assert all("headers" not in call[2] for call in client.calls)
    assert collector.until <= datetime.now(UTC)


def test_bluesky_publication_window_is_half_open_and_malformed_items_skip(
    config: dict[str, Any],
) -> None:
    client = PublicClient(
        {
            "posts": [
                bluesky_post(
                    "first", record={"text": "Valid", "createdAt": "2025-12-31T16:00:00Z"}
                ),
                bluesky_post("old", record={"text": "Old", "createdAt": "2025-12-31T15:59:59Z"}),
                bluesky_post("end", record={"text": "Late", "createdAt": "2026-12-31T16:00:00Z"}),
                bluesky_post("badtime", record={"text": "Bad", "createdAt": "2026-01-01"}),
                bluesky_post("oversized", record={"text": "x" * 100_001}),
                bluesky_post("empty", record={"text": " "}),
                bluesky_post("badid", uri="at://localhost/app.bsky.feed.post/1"),
                bluesky_post("wrongcollection", uri=f"at://{DID}/app.bsky.actor.profile/self"),
                None,
            ]
        }
    )
    collector = BlueskyCollector(config, client)
    rows = collector.collect()
    assert len(rows) == 1
    assert rows[0].created_at == "2025-12-31T16:00:00Z"
    assert any("invalid publication timestamp" in warning for warning in collector.warnings)
    assert any("malformed" in warning for warning in collector.warnings)


@pytest.mark.parametrize("cursor", ["repeat", 123, "bad\nvalue", "x" * 4097])
def test_bluesky_stops_invalid_or_repeated_cursors(config: dict[str, Any], cursor: Any) -> None:
    page = {"posts": [bluesky_post()], "cursor": cursor}
    client = PublicClient(page, page)
    collector = BlueskyCollector(config, client)
    assert len(collector.collect()) == 1
    assert len(client.calls) == (2 if cursor == "repeat" else 1)
    assert any("cursor was invalid or repeated" in warning for warning in collector.warnings)


def test_bluesky_bounds_request_pages_and_record_budget(config: dict[str, Any]) -> None:
    config["max_pages"] = 1
    client = PublicClient({"posts": [bluesky_post()], "cursor": "next"})
    collector = BlueskyCollector(config, client)
    assert len(collector.collect()) == 1
    assert len(client.calls) == 1
    assert any("max_pages" in warning for warning in collector.warnings)
    config["max_records"] = 1
    config["queries"].append("Marcos")
    client = PublicClient({"posts": [bluesky_post(), bluesky_post("other")], "cursor": "next"})
    assert len(BlueskyCollector(config, client).collect()) == 1
    assert len(client.calls) == 1


def test_mastodon_public_html_visibility_relevance_and_numeric_pagination(
    config: dict[str, Any],
) -> None:
    client = PublicClient(
        [
            mastodon_post(
                "400", content="<p>Philippine government.</p><p>Good &amp; bad<br>services.</p>"
            ),
            mastodon_post("390", visibility="private"),
            mastodon_post("380", visibility="unlisted"),
            mastodon_post("370", reblog={"id": "400"}),
            mastodon_post("360", content="<p>Marcos was great at a concert in Spain.</p>"),
            mastodon_post("350", content="<p>Beautiful beaches #Philippines</p>"),
            mastodon_post("340", content="<p>Government of the Philippine islands</p>"),
        ],
        [
            mastodon_post(
                "330", spoiler_text="Philippine government", content="<p>Services improved.</p>"
            )
        ],
        [],
    )
    collector = MastodonCollector(config, client)
    rows = collector.collect()
    assert [row.record_id for row in rows] == [
        "mastodon:mastodon.social:400",
        "mastodon:mastodon.social:340",
        "mastodon:mastodon.social:330",
    ]
    assert rows[0].text == "Philippine government. Good & bad services."
    assert rows[0].source_url == "https://mastodon.social/@public-fixture/400"
    assert rows[0].query == "#Philippines"
    assert "private-fixture" not in str(rows[0].to_dict())
    assert client.calls[0][0:2] == ("GET", f"{INSTANCE}/api/v1/timelines/tag/Philippines")
    assert client.calls[1][2]["params"]["max_id"] == "340"
    assert client.calls[2][2]["params"]["max_id"] == "330"
    assert all("headers" not in call[2] for call in client.calls)
    assert any("relevance heuristic" in warning for warning in collector.warnings)


def test_mastodon_deduplicates_across_hashtags_and_bounds_global_records(
    config: dict[str, Any],
) -> None:
    config["settings"]["hashtags"] = ["Philippines", "Marcos"]
    client = PublicClient([mastodon_post()], [], [mastodon_post(), mastodon_post("200")], [])
    assert len(MastodonCollector(config, client).collect()) == 2
    config["max_records"] = 1
    client = PublicClient([mastodon_post(), mastodon_post("200")])
    assert len(MastodonCollector(config, client).collect()) == 1
    assert len(client.calls) == 1


@pytest.mark.parametrize("shared", ["bbm", " BBM ", "gobyerno", "  Government  "])
def test_mastodon_rejects_shared_country_and_government_evidence(config, shared):
    config["settings"]["government_terms"].extend(["bbm", "government"])
    config["settings"]["country_terms"].append(shared)
    client = PublicClient()
    with pytest.raises(ConfigurationError, match="separate country evidence"):
        MastodonCollector(config, client)
    assert client.calls == []


@pytest.mark.parametrize(
    "text",
    [
        "BBM entertainment and sports news #BBM",
        "Filipino adult entertainment and sports tags #BBM",
        "The gobyerno of Canada announced a budget #Gobyerno",
        "A university president discusses global AI education in the Philippines.",
    ],
)
def test_public_preset_excludes_acronym_foreign_and_corporate_president_collisions(text):
    config = collector_config("mastodon", load_config(PUBLIC_CONFIG))
    collector = MastodonCollector(config, PublicClient())
    collector._add_status(mastodon_post(content=f"<p>{text}</p>"), "Philippines")
    assert collector.records == []


def test_public_preset_keeps_explicit_philippine_governance_context():
    config = collector_config("mastodon", load_config(PUBLIC_CONFIG))
    collector = MastodonCollector(config, PublicClient())
    collector._add_status(
        mastodon_post(
            content="<p>President Marcos discusses government services in the Philippines.</p>"
        ),
        "Philippines",
    )
    assert len(collector.records) == 1


def test_mastodon_malformed_items_and_publication_window(config: dict[str, Any]) -> None:
    client = PublicClient(
        [
            mastodon_post("700", created_at="2025-12-31T16:00:00Z"),
            mastodon_post("600", created_at="2025-12-31T15:59:59Z"),
            mastodon_post("500", created_at="2026-12-31T16:00:00Z"),
            mastodon_post("400", created_at="bad"),
            mastodon_post("300", content=None),
            mastodon_post("200", content="x" * 200_001),
            mastodon_post("../bad"),
            None,
        ],
        [],
    )
    collector = MastodonCollector(config, client)
    rows = collector.collect()
    assert len(rows) == 1
    assert rows[0].created_at == "2025-12-31T16:00:00Z"
    assert any("invalid publication timestamp" in warning for warning in collector.warnings)
    assert any("malformed" in warning for warning in collector.warnings)


@pytest.mark.parametrize(
    "url",
    [
        "http://mastodon.social/@someone/1",
        "https://user:password@mastodon.social/@someone/1",
        "https://127.0.0.1/private",
        "https://localhost/private",
        "https://intranet.local/private",
        "https://mastodon.social:444/@someone/1",
        "https://mastodon.social/\ninvalid",
        None,
    ],
)
def test_mastodon_unsafe_provenance_is_discarded(config: dict[str, Any], url: Any) -> None:
    client = PublicClient([mastodon_post(url=url)], [])
    rows = MastodonCollector(config, client).collect()
    assert len(rows) == 1
    assert rows[0].source_url == ""


def test_mastodon_html_strips_scripts_and_keeps_entities_and_breaks() -> None:
    assert plain_text("<p>Gobyerno &amp; serbisyo.</p><script>secret()</script><br>Maganda.") == (
        "Gobyerno & serbisyo. Maganda."
    )


def test_mastodon_stops_nonadvancing_cursor_and_reports_budget(config: dict[str, Any]) -> None:
    page = [mastodon_post()]
    client = PublicClient(page, page)
    collector = MastodonCollector(config, client)
    assert len(collector.collect()) == 1
    assert len(client.calls) == 2
    assert any("did not advance" in warning for warning in collector.warnings)
    config["max_pages"] = 1
    collector = MastodonCollector(config, PublicClient(page))
    collector.collect()
    assert any("max_pages" in warning for warning in collector.warnings)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"hashtags": ["#Philippines"]},
        {"hashtags": ["../private"]},
        {"hashtags": ["Philippines"], "government_terms": []},
        {"hashtags": ["Philippines"], "government_terms": ["government"], "country_terms": []},
    ],
)
def test_mastodon_requires_safe_tags_and_explicit_relevance_terms(
    config: dict[str, Any], settings: dict[str, Any]
) -> None:
    config["settings"] = settings
    client = PublicClient()
    with pytest.raises(ConfigurationError):
        MastodonCollector(config, client)
    assert client.calls == []


@pytest.mark.parametrize("collector_type", [BlueskyCollector, MastodonCollector])
def test_public_collectors_require_false_include_comments(
    config: dict[str, Any], collector_type: Any
) -> None:
    config["include_comments"] = True
    client = PublicClient()
    with pytest.raises(ConfigurationError, match="include_comments=false"):
        collector_type(config, client)
    assert client.calls == []


@pytest.mark.parametrize(
    "collector_type,response", [(BlueskyCollector, []), (MastodonCollector, {"data": []})]
)
def test_public_collectors_reject_unexpected_response_shape(
    config: dict[str, Any], collector_type: Any, response: Any
) -> None:
    with pytest.raises(CollectionError, match="unexpected"):
        collector_type(config, PublicClient(response)).collect()


@pytest.mark.parametrize("collector_type", [BlueskyCollector, MastodonCollector])
def test_public_api_failure_preserves_already_collected_records(
    config: dict[str, Any], collector_type: Any
) -> None:
    first = (
        {"posts": [bluesky_post()], "cursor": "next"}
        if collector_type is BlueskyCollector
        else [mastodon_post()]
    )
    collector = collector_type(config, PublicClient(first, CollectionError("API unavailable")))
    with pytest.raises(CollectionError, match="unavailable"):
        collector.collect()
    assert len(collector.records) == 1
