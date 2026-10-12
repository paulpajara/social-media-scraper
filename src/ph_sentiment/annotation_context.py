"""Fetch bounded public Bluesky post context without downloading linked websites.

Context supports annotation only. It never replaces the frozen analysis text,
and a changed or unavailable post must not be used to resolve its old meaning.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .collectors.base import HTTPClient, iso_utc
from .collectors.bluesky import POST_URI
from .errors import CollectionError, DataValidationError
from .preprocessing import clean_text
from .schema import Record

ENDPOINT = "https://api.bsky.app/xrpc/app.bsky.feed.getPosts"
CONTEXT_FIELDS = (
    "record_id",
    "platform",
    "context_status",
    "snapshot_text_matches",
    "current_text",
    "link_title",
    "link_description",
    "image_alt_text",
    "quoted_text",
    "collected_at",
    "context_method",
)


def _text(value: Any) -> str:
    return value[:10_000] if isinstance(value, str) else ""


def _embed_context(embed: Any) -> dict[str, str]:
    """Read only documented embed text, omitting profiles and media URLs."""
    fields = dict.fromkeys(("link_title", "link_description", "image_alt_text", "quoted_text"), "")
    if not isinstance(embed, dict):
        return fields
    kind = embed.get("$type", "")
    if kind == "app.bsky.embed.external#view":
        external = embed.get("external")
        if isinstance(external, dict):
            fields["link_title"] = _text(external.get("title"))
            fields["link_description"] = _text(external.get("description"))
    elif kind == "app.bsky.embed.images#view":
        images = embed.get("images")
        if isinstance(images, list):
            fields["image_alt_text"] = "\n".join(
                _text(item.get("alt")) for item in images[:4] if isinstance(item, dict)
            )[:10_000]
    elif kind == "app.bsky.embed.record#view":
        quoted = embed.get("record")
        if isinstance(quoted, dict) and quoted.get("$type") == "app.bsky.embed.record#viewRecord":
            value = quoted.get("value")
            if isinstance(value, dict):
                fields["quoted_text"] = _text(value.get("text"))
    elif kind == "app.bsky.embed.recordWithMedia#view":
        media = _embed_context(embed.get("media"))
        quotation = _embed_context(embed.get("record"))
        fields = {key: media[key] or quotation[key] for key in fields}
    return fields


def fetch_context(records: list[Record], client: HTTPClient) -> list[dict[str, str]]:
    """One CSV row per input; maximum 25 requested AT URIs per API batch.

    The supplied corpus must already be prepared. No credentials are needed.
    Batch failures are recorded with fixed safe status text and collection
    continues; unavailable or changed text cannot establish a reference label.
    """
    if len(records) > 10_000:
        raise DataValidationError("Annotation context is limited to 10,000 prepared records.")
    rows: dict[str, dict[str, str]] = {}
    requested: list[tuple[str, Record]] = []
    now = iso_utc(datetime.now(UTC))
    for record in records:
        record.validate()
        if record.record_id in rows:
            raise DataValidationError("Annotation context requires unique record IDs.")
        rows[record.record_id] = {
            "record_id": record.record_id,
            "platform": record.platform,
            "context_status": "not_requested",
            "snapshot_text_matches": "",
            "current_text": "",
            "link_title": "",
            "link_description": "",
            "image_alt_text": "",
            "quoted_text": "",
            "collected_at": now,
            "context_method": "post_text_only",
        }
        if record.platform != "bluesky":
            continue
        uri = record.record_id.removeprefix("bluesky:")
        if POST_URI.fullmatch(uri) is None:
            raise DataValidationError("Invalid Bluesky AT URI in annotation input.")
        requested.append((uri, record))
    for offset in range(0, len(requested), 25):
        batch = requested[offset : offset + 25]
        for _, record in batch:
            rows[record.record_id]["context_method"] = "app.bsky.feed.getPosts"
        try:
            payload = client.request_json(
                "GET", ENDPOINT, params={"uris": [uri for uri, _ in batch]}
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("posts"), list):
                raise CollectionError("Invalid public post context response.")
            posts = {
                item.get("uri"): item
                for item in payload["posts"]
                if isinstance(item, dict) and isinstance(item.get("uri"), str)
            }
        except CollectionError:
            for _, record in batch:
                rows[record.record_id]["context_status"] = "request_failed"
            continue
        for uri, record in batch:
            row = rows[record.record_id]
            post = posts.get(uri)
            content = post.get("record") if post else None
            if not isinstance(content, dict) or not isinstance(content.get("text"), str):
                row["context_status"] = "unavailable"
                continue
            current = _text(content["text"])
            matches = clean_text(current) == record.text
            row.update(_embed_context(post.get("embed")))
            row.update(
                context_status="available" if matches else "changed_text",
                snapshot_text_matches="true" if matches else "false",
                current_text=current,
            )
    return [rows[record.record_id] for record in records]
