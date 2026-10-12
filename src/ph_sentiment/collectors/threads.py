"""Collect public Threads keyword-search results using the Threads Graph API."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ph_sentiment.collectors.base import Collector, require_env
from ph_sentiment.errors import AccessCollectionError, CollectionError, ConfigurationError
from ph_sentiment.schema import Record


class ThreadsCollector(Collector):
    platform = "threads"

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        super().__init__(config, client)
        self.search_type = self.settings.get("search_type", "RECENT")
        self.search_mode = self.settings.get("search_mode")
        self.check_token = self.settings.get("check_token", True)
        if not isinstance(self.check_token, bool):
            raise ConfigurationError("Threads check_token must be true or false.")
        if not isinstance(self.search_type, str) or self.search_type not in {"RECENT", "TOP"}:
            raise ConfigurationError("Threads search_type must be RECENT or TOP.")
        if self.search_mode is not None and (
            not isinstance(self.search_mode, str) or self.search_mode not in {"KEYWORD", "TAG"}
        ):
            raise ConfigurationError("Threads search_mode must be KEYWORD or TAG when supplied.")
        if self.include_comments:
            raise ConfigurationError(
                "Threads collection supports search results only; set include_comments=false."
            )

    def collect(self) -> list[Record]:
        token = require_env("THREADS_ACCESS_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}
        until = min(self.end, datetime.now(UTC))
        if self.start >= until:
            self.warn("Threads collection dates begin in the future; no request was sent.")
            return self.records
        if self.check_token:
            self._check_access(token)
        self.warn(
            "Threads public keyword search requires approved threads_keyword_search access. "
            "API ranking, retention and quotas can limit historical coverage."
        )
        for query in self.queries:
            if self.full:
                break
            params: dict[str, Any] = {
                "q": query,
                "search_type": self.search_type,
                "fields": "id,text,timestamp,permalink",
                "limit": min(100, self.max_records - len(self.records)),
                "since": int(self.start.timestamp()),
                "until": int(until.timestamp()),
            }
            if self.search_mode:
                params["search_mode"] = self.search_mode
            diagnostic: dict[str, Any] = {
                "platform": self.platform,
                "query": query,
                "api_records": 0,
                "retained_records": 0,
                "status": "failed",
                "message": "API collection request failed before this query completed.",
                "http_status": "",
                "error_category": "",
            }
            self.query_diagnostics.append(diagnostic)
            try:
                for items in self.graph_pages(
                    "https://graph.threads.net/v1.0/keyword_search", params, headers
                ):
                    diagnostic["api_records"] += len(items)
                    for item in items:
                        diagnostic["retained_records"] += int(self._retain_post(item, query))
            except CollectionError as error:
                diagnostic["status"] = "partial" if diagnostic["retained_records"] else "failed"
                diagnostic["http_status"] = getattr(error, "status_code", "")
                diagnostic["error_category"] = getattr(error, "category", "collection")
                raise
            if not diagnostic["api_records"]:
                diagnostic["status"] = "empty_api"
                diagnostic["message"] = (
                    "API returned no posts before local filtering; confirm public "
                    "threads_keyword_search App Review approval and query coverage. "
                    "Granted token scopes do not prove public-search approval."
                )
                self.warn(diagnostic["message"])
            elif not diagnostic["retained_records"]:
                diagnostic["status"] = "filtered_empty"
                diagnostic["message"] = (
                    "API returned posts, but none passed the collection date, text, "
                    "identifier and duplicate checks."
                )
                self.warn(diagnostic["message"])
            else:
                diagnostic["status"] = "success"
                diagnostic["message"] = "API posts were retained within the collection budget."
        return self.records

    def _retain_post(self, item: dict[str, Any], query: str) -> bool:
        """Validate identity and provenance before applying common record filters."""
        identifier = item.get("id", "")
        if not isinstance(identifier, str) or not re.fullmatch(r"\d{1,32}", identifier):
            self.warn("Skipped a Threads post with an invalid identifier.")
            return False
        permalink = item.get("permalink", "")
        if not isinstance(permalink, str):
            permalink = ""
        try:
            parsed = urlsplit(permalink)
            if (
                parsed.scheme != "https"
                or parsed.hostname
                not in {"threads.net", "www.threads.net", "threads.com", "www.threads.com"}
                or parsed.username
                or parsed.password
                or parsed.port not in (None, 443)
            ):
                permalink = ""
            else:
                # Provenance links never need query tokens or tracking parameters.
                permalink = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        except ValueError:
            permalink = ""
        return self.add(
            identifier,
            item.get("text"),
            item.get("timestamp"),
            query,
            source_url=permalink,
        )

    def _check_access(self, token: str) -> None:
        """Check granted token scopes without inferring public App Review approval."""
        from ph_sentiment.access import inspect_threads_token

        try:
            metadata = inspect_threads_token(self.client, token)
        except CollectionError:
            # The debugger may be unavailable to users who are not app testers.
            self.warn(
                "Could not verify Threads token metadata; public-search access remains unverified."
            )
            return
        if not metadata.is_valid or (
            metadata.expires_at is not None and metadata.expires_at <= datetime.now(UTC)
        ):
            raise AccessCollectionError(
                "authentication",
                "Threads token is invalid or expired; replace THREADS_ACCESS_TOKEN.",
            )
        if not {"threads_basic", "threads_keyword_search"}.issubset(metadata.scopes):
            raise AccessCollectionError(
                "permission_missing",
                "Threads token must grant threads_basic and threads_keyword_search. "
                "Add these permissions in the app authorization flow and regenerate "
                "THREADS_ACCESS_TOKEN; public search also requires App Review approval.",
            )
        if metadata.expires_at is not None and metadata.expires_at <= datetime.now(UTC) + timedelta(
            hours=24
        ):
            self.warn(
                f"Threads token expires at {metadata.expires_at.isoformat()}; "
                "refresh or replace it before another collection run."
            )
