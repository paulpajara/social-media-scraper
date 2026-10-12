"""Collect bounded public Bluesky search results without account credentials."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from ph_sentiment.collectors.base import Collector, iso_utc
from ph_sentiment.errors import CollectionError, ConfigurationError
from ph_sentiment.schema import Record

SEARCH_ENDPOINT = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
POST_URI = re.compile(
    r"at://(did:(?:plc:[a-z2-7]{24}|web:[A-Za-z0-9._%:-]{1,255}))"
    r"/app\.bsky\.feed\.post/([A-Za-z0-9._~:-]{1,512})"
)


class BlueskyCollector(Collector):
    platform = "bluesky"

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        super().__init__(config, client)
        if self.include_comments:
            raise ConfigurationError(
                "Bluesky collection supports search results only; set include_comments=false."
            )
        if any(len(query) > 512 for query in self.queries):
            raise ConfigurationError("Bluesky search queries must be at most 512 characters.")
        self.until = min(self.end, datetime.now(UTC))

    def _add_post(self, post: Any, query: str) -> None:
        if not isinstance(post, dict) or not isinstance(post.get("record"), dict):
            self.warn("Skipped a Bluesky result with malformed post metadata.")
            return
        uri = post.get("uri")
        matched = POST_URI.fullmatch(uri) if isinstance(uri, str) and len(uri) <= 1024 else None
        if matched is None or matched.group(2) in {".", ".."}:
            self.warn("Skipped a Bluesky post with an invalid public post identifier.")
            return
        record = post["record"]
        text = record.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 100_000:
            self.warn("Skipped a Bluesky post with missing or oversized text.")
            return
        created = self.timestamp_in_window(record.get("createdAt"))
        if created is None or created >= self.until:
            return
        did, record_key = matched.groups()
        # Derive a provenance link from the validated AT URI, never from profile metadata.
        source_url = (
            f"https://bsky.app/profile/{quote(did, safe=':')}/post/{quote(record_key, safe='')}"
        )
        self.add(uri, text, created.isoformat(), query, source_url=source_url)

    def collect(self) -> list[Record]:
        self.warn(
            "Bluesky public search is a bounded sample of indexed posts, not an exhaustive "
            "archive or a representative sample of Philippine residents. Search bounds use "
            "the API index timestamp; the local date filter uses the post publication timestamp."
        )
        if self.start >= self.until:
            self.warn("Bluesky collection dates begin in the future; no request was sent.")
            return self.records
        for query in self.queries:
            if self.full:
                break
            params: dict[str, Any] = {
                "q": query,
                "sort": "latest",
                "limit": min(100, self.max_records),
                "since": iso_utc(self.start),
                "until": iso_utc(self.until),
            }
            cursors: set[str] = set()
            for _ in range(self.max_pages):
                payload = self.client.request_json("GET", SEARCH_ENDPOINT, params=params)
                if not isinstance(payload, dict) or not isinstance(payload.get("posts"), list):
                    raise CollectionError("Bluesky returned an unexpected search response.")
                for post in payload["posts"]:
                    self._add_post(post, query)
                    if self.full:
                        return self.records
                cursor = payload.get("cursor")
                if not payload["posts"] or not cursor:
                    break
                if (
                    not isinstance(cursor, str)
                    or len(cursor) > 4096
                    or any(ord(character) < 32 for character in cursor)
                    or cursor in cursors
                ):
                    self.warn(
                        "Stopped Bluesky pagination because a cursor was invalid or repeated."
                    )
                    break
                cursors.add(cursor)
                params = {**params, "cursor": cursor}
            else:
                self.warn(
                    "Reached max_pages before exhausting Bluesky results; collection is bounded."
                )
        return self.records
