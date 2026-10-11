"""Read authorized Facebook Page posts and their comments, filtering locally.

The Graph API does not offer unrestricted public-post keyword search. Page
content is a source-specific sample; Page-authored posts are distinct from
comments expressing public sentiment.
"""

from __future__ import annotations

import re
from typing import Any

from ph_sentiment.collectors.base import Collector, require_env
from ph_sentiment.errors import ConfigurationError
from ph_sentiment.schema import Record


class FacebookCollector(Collector):
    platform = "facebook"

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        super().__init__(config, client)
        self.graph_version = self.settings.get("graph_version", "")
        if not isinstance(self.graph_version, str) or not re.fullmatch(
            r"v\d{1,3}\.\d{1,2}", self.graph_version
        ):
            raise ConfigurationError(
                "Set Facebook graph_version explicitly to a version supported by your app."
            )
        self.page_ids = self.settings.get("page_ids", [])
        if (
            not isinstance(self.page_ids, list)
            or not self.page_ids
            or any(
                not isinstance(identifier, str) or not re.fullmatch(r"\d{1,32}", identifier)
                for identifier in self.page_ids
            )
        ):
            raise ConfigurationError(
                "Facebook page_ids must contain authorized numeric public Page IDs."
            )
        self.base_url = f"https://graph.facebook.com/{self.graph_version}"

    def _matched_query(self, text: Any) -> str | None:
        if not isinstance(text, str):
            return None
        folded = text.casefold()
        matches = [query for query in self.queries if query.casefold() in folded]
        return " | ".join(matches) if matches else None

    @staticmethod
    def _identifier(item: dict[str, Any]) -> str | None:
        identifier = item.get("id", "")
        return (
            identifier
            if isinstance(identifier, str) and re.fullmatch(r"\d{1,32}(?:_\d{1,32})?", identifier)
            else None
        )

    def _comments(self, post_id: str, query: str, headers: dict[str, str]) -> None:
        params = {
            "fields": "id,message,created_time",
            "limit": 100,
            "filter": "stream",
            "order": "chronological",
        }
        self.warn(
            "Facebook comments are limited to API-visible comments on matched Page posts; reply coverage may vary."
        )
        for items in self.graph_pages(f"{self.base_url}/{post_id}/comments", params, headers):
            for item in items:
                identifier = self._identifier(item)
                if identifier is None:
                    self.warn("Skipped a Facebook comment with an invalid identifier.")
                    continue
                self.add(
                    identifier,
                    item.get("message"),
                    item.get("created_time"),
                    query,
                    source_url=f"https://www.facebook.com/{identifier}",
                    record_type="comment",
                )

    def collect(self) -> list[Record]:
        headers = {"Authorization": f"Bearer {require_env('FACEBOOK_ACCESS_TOKEN')}"}
        self.warn(
            "Facebook collection reads configured authorized public Pages, not unrestricted public "
            "keyword search. Permissions or Page Public Content Access approval are required."
        )
        self.warn(
            "Facebook queries are literal case-insensitive phrases matched against Page post messages."
        )
        params = {
            "fields": "id,message,created_time",
            "limit": 100,
            "since": int(self.start.timestamp()),
            "until": int(self.end.timestamp()),
        }
        for page_id in self.page_ids:
            if self.full:
                break
            for items in self.graph_pages(f"{self.base_url}/{page_id}/posts", params, headers):
                for item in items:
                    if self.full:
                        break
                    query = self._matched_query(item.get("message"))
                    identifier = self._identifier(item)
                    if query is None or identifier is None:
                        continue
                    was_new = self.add(
                        identifier,
                        item.get("message"),
                        item.get("created_time"),
                        query,
                        source_url=f"https://www.facebook.com/{identifier}",
                    )
                    if self.include_comments and was_new and not self.full:
                        self._comments(identifier, query, headers)
        return self.records
