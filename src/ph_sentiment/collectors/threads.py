"""Collect public Threads keyword-search results using the Threads Graph API."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ph_sentiment.collectors.base import Collector, require_env
from ph_sentiment.errors import ConfigurationError
from ph_sentiment.schema import Record


class ThreadsCollector(Collector):
    platform = "threads"

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        super().__init__(config, client)
        self.search_type = self.settings.get("search_type", "RECENT")
        self.search_mode = self.settings.get("search_mode")
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
        headers = {"Authorization": f"Bearer {require_env('THREADS_ACCESS_TOKEN')}"}
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
                "limit": 100,
                "since": int(self.start.timestamp()),
                "until": int(self.end.timestamp()),
            }
            if self.search_mode:
                params["search_mode"] = self.search_mode
            for items in self.graph_pages(
                "https://graph.threads.net/v1.0/keyword_search", params, headers
            ):
                for item in items:
                    identifier = item.get("id", "")
                    if not isinstance(identifier, str) or not re.fullmatch(r"\d{1,32}", identifier):
                        self.warn("Skipped a Threads post with an invalid identifier.")
                        continue
                    permalink = item.get("permalink", "")
                    if not isinstance(permalink, str):
                        permalink = ""
                    try:
                        parsed = urlsplit(permalink)
                        if (
                            parsed.scheme != "https"
                            or parsed.hostname
                            not in {
                                "threads.net",
                                "www.threads.net",
                                "threads.com",
                                "www.threads.com",
                            }
                            or parsed.username
                            or parsed.password
                            or parsed.port not in (None, 443)
                        ):
                            permalink = ""
                        else:
                            # Provenance links never need query tokens or tracking parameters.
                            permalink = urlunsplit(
                                (parsed.scheme, parsed.netloc, parsed.path, "", "")
                            )
                    except ValueError:
                        permalink = ""
                    self.add(
                        identifier,
                        item.get("text"),
                        item.get("timestamp"),
                        query,
                        source_url=permalink,
                    )
        return self.records
