"""Collect X posts through documented recent or full-archive search."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from ph_sentiment.collectors.base import Collector, iso_utc, require_env
from ph_sentiment.errors import CollectionError, ConfigurationError
from ph_sentiment.schema import Record


class XCollector(Collector):
    platform = "x"

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        super().__init__(config, client)
        self.search_mode = self.settings.get("search_mode", "recent")
        if not isinstance(self.search_mode, str) or self.search_mode not in {"recent", "archive"}:
            raise ConfigurationError("X search_mode must be recent or archive.")
        # The official OpenAPI specification and quickstart use tweet.fields;
        # the newer web reference also documents post.fields. Keep the API
        # vocabulary configurable without accepting arbitrary request parameters.
        self.fields_parameter = self.settings.get("fields_parameter", "tweet.fields")
        if not isinstance(self.fields_parameter, str) or self.fields_parameter not in {
            "tweet.fields",
            "post.fields",
        }:
            raise ConfigurationError("X fields_parameter must be tweet.fields or post.fields.")
        if self._owns_client and self.search_mode == "archive":
            self.client.min_interval = max(self.client.min_interval, 1.05)
        if self.include_comments:
            raise ConfigurationError(
                "X collection supports search results only; set include_comments=false."
            )
        if self.search_mode == "recent" and self.start < datetime.now(UTC) - timedelta(days=7):
            raise ConfigurationError(
                "X recent search covers the last seven days. Use search_mode='archive' with eligible "
                "API access for earlier 2026 data, or select a recent date range."
            )

    def collect(self) -> list[Record]:
        headers = {"Authorization": f"Bearer {require_env('X_BEARER_TOKEN')}"}
        endpoint = "all" if self.search_mode == "archive" else "recent"
        self.warn(
            "X results depend on API entitlement, query operators, quotas and the configured page budget."
        )
        # Endpoints reject future end_time values. Retain the requested calendar
        # window for local filtering while requesting only existing posts.
        end = min(self.end, datetime.now(UTC) - timedelta(seconds=30))
        if self.start >= end:
            raise ConfigurationError(
                "X collection window must include time before the current moment."
            )
        for query in self.queries:
            cursor: str | None = None
            seen_cursors: set[str] = set()
            for _ in range(self.max_pages):
                if self.full:
                    return self.records
                params: dict[str, Any] = {
                    "query": query,
                    "start_time": iso_utc(self.start),
                    "end_time": iso_utc(end),
                    "max_results": max(10, min(100, self.max_records - len(self.records))),
                    self.fields_parameter: "created_at,note_tweet"
                    if self.fields_parameter == "tweet.fields"
                    else "created_at,note_post",
                    "sort_order": "recency",
                }
                if cursor:
                    params["next_token"] = cursor
                payload = self.client.request_json(
                    "GET",
                    f"https://api.x.com/2/tweets/search/{endpoint}",
                    allow_partial_errors=True,
                    headers=headers,
                    params=params,
                )
                if not isinstance(payload, dict) or not isinstance(payload.get("data", []), list):
                    raise CollectionError("X returned an unexpected search response.")
                if payload.get("errors"):
                    self.warn(
                        "X returned partial results with API errors; retained the available posts. "
                        "Some requested content may be unavailable."
                    )
                for item in payload.get("data", []):
                    if not isinstance(item, dict):
                        raise CollectionError("X returned an unexpected search item.")
                    identifier = item.get("id", "")
                    if not isinstance(identifier, str) or not re.fullmatch(r"\d{1,32}", identifier):
                        self.warn("Skipped an X post with an invalid identifier.")
                        continue
                    extended = item.get("note_tweet") or item.get("note_post")
                    text = extended.get("text") if isinstance(extended, dict) else None
                    self.add(
                        identifier,
                        text if isinstance(text, str) and text.strip() else item.get("text"),
                        item.get("created_at"),
                        query,
                        source_url=f"https://x.com/i/web/status/{identifier}",
                    )
                metadata = payload.get("meta") or {}
                if not isinstance(metadata, dict):
                    raise CollectionError("X returned invalid pagination metadata.")
                cursor = metadata.get("next_token")
                if self.full or not cursor:
                    break
                if not isinstance(cursor, str) or cursor in seen_cursors:
                    self.warn(
                        "Stopped X pagination because the API returned an invalid or repeated cursor."
                    )
                    break
                seen_cursors.add(cursor)
            else:
                self.warn("Reached max_pages for X search; collection is a bounded sample.")
        return self.records
