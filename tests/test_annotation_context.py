"""Context retrieval is bounded and cannot substitute changed post text."""

from unittest.mock import Mock

import pytest

from ph_sentiment.annotation_context import ENDPOINT, fetch_context
from ph_sentiment.errors import CollectionError, DataValidationError
from ph_sentiment.schema import Record


def record(index=0, platform="bluesky"):
    uri = f"at://did:plc:abcdefghijklmnopqrstuvwx/app.bsky.feed.post/{index}"
    return Record(f"bluesky:{uri}", platform, "2026-09-01T00:00:00Z", "policy announcement")


def test_context_uses_documented_batches_and_omits_profiles():
    records = [record(i) for i in range(26)]
    client = Mock()
    post = {
        "uri": records[0].record_id.removeprefix("bluesky:"),
        "record": {"text": "Policy announcement"},
        "author": {"description": "private profile"},
        "embed": {
            "$type": "app.bsky.embed.external#view",
            "external": {"title": "Philippine policy", "description": "Public context"},
        },
    }
    client.request_json.side_effect = [{"posts": [post]}, {"posts": []}]
    rows = fetch_context(records, client)
    assert len(rows) == 26
    assert len(client.request_json.call_args_list) == 2
    assert client.request_json.call_args_list[0].args == ("GET", ENDPOINT)
    assert len(client.request_json.call_args_list[0].kwargs["params"]["uris"]) == 25
    assert rows[0]["snapshot_text_matches"] == "true"
    assert rows[0]["link_title"] == "Philippine policy"
    assert rows[1]["context_status"] == "unavailable"
    assert "private profile" not in str(rows)


def test_changed_or_failed_context_is_explicit():
    item = record()
    client = Mock()
    client.request_json.return_value = {
        "posts": [
            {"uri": item.record_id.removeprefix("bluesky:"), "record": {"text": "Changed post"}}
        ]
    }
    assert fetch_context([item], client)[0]["context_status"] == "changed_text"
    client.request_json.side_effect = CollectionError("Request failed")
    assert fetch_context([item], client)[0]["context_status"] == "request_failed"


def test_invalid_uris_or_duplicates_fail_before_requests():
    client = Mock()
    with pytest.raises(DataValidationError):
        fetch_context([record(), record()], client)
    with pytest.raises(DataValidationError):
        fetch_context([Record("bluesky:bad", "bluesky", "2026-09-01T00:00:00Z", "text")], client)
    client.request_json.assert_not_called()
