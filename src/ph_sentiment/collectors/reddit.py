"""Collect Reddit post-search results and optional comment trees via OAuth."""

from __future__ import annotations

import html
import math
import re
from collections import deque
from typing import Any

from ph_sentiment.collectors.base import Collector, require_env
from ph_sentiment.errors import CollectionError, ConfigurationError
from ph_sentiment.schema import Record


class RedditCollector(Collector):
    platform = "reddit"

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        super().__init__(config, client)
        self.subreddits = self.settings.get("subreddits", ["Philippines"])
        if (
            not isinstance(self.subreddits, list)
            or not self.subreddits
            or any(
                not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]{2,21}", name)
                for name in self.subreddits
            )
        ):
            raise ConfigurationError(
                "Reddit subreddits must be a nonempty list of valid subreddit names."
            )
        self._token = ""
        self._expires_at = 0.0
        self._user_agent = ""

    def _headers(self) -> dict[str, str]:
        if not self._token or self.client.clock() >= self._expires_at:
            self._user_agent = require_env("REDDIT_USER_AGENT")
            payload = self.client.request_json(
                "POST",
                "https://www.reddit.com/api/v1/access_token",
                data={"grant_type": "client_credentials"},
                auth=(require_env("REDDIT_CLIENT_ID"), require_env("REDDIT_CLIENT_SECRET")),
                headers={"User-Agent": self._user_agent},
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
                raise CollectionError("Reddit did not return an OAuth access token.")
            self._token = payload["access_token"]
            if (
                not self._token
                or not self._token.isascii()
                or any(c in self._token for c in ("\r", "\n", "\x00"))
            ):
                raise CollectionError("Reddit returned an invalid OAuth access token.")
            duration = payload.get("expires_in", 3600)
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise CollectionError("Reddit returned an invalid OAuth expiration.")
            self._expires_at = self.client.clock() + max(duration - 30, 1)
        return {"User-Agent": self._user_agent, "Authorization": f"Bearer {self._token}"}

    @staticmethod
    def _listing(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise CollectionError("Reddit returned an unexpected listing response.")
        data = payload["data"]
        children = data.get("children", [])
        if not isinstance(children, list) or any(not isinstance(item, dict) for item in children):
            raise CollectionError("Reddit returned an unexpected listing item.")
        after = data.get("after")
        if after is not None and not isinstance(after, str):
            raise CollectionError("Reddit returned an invalid listing cursor.")
        return children, after

    def _comments(self, post_id: str, query: str) -> None:
        payload = self.client.request_json(
            "GET",
            f"https://oauth.reddit.com/comments/{post_id}",
            headers=self._headers(),
            params={"limit": 100, "sort": "new", "raw_json": 1},
        )
        if not isinstance(payload, list) or len(payload) < 2:
            raise CollectionError("Reddit returned an unexpected comment-tree response.")
        children, _ = self._listing(payload[1])
        nodes = deque(children)
        additional: deque[str] = deque()
        fetched_ids: set[str] = set()
        requests_made = 1
        while (nodes or additional) and not self.full:
            while nodes and not self.full:
                node = nodes.popleft()
                data = node.get("data", {})
                if not isinstance(data, dict):
                    raise CollectionError("Reddit returned an unexpected comment item.")
                if node.get("kind") == "more":
                    more_ids = data.get("children", [])
                    if not isinstance(more_ids, list):
                        raise CollectionError(
                            "Reddit returned invalid omitted-comment identifiers."
                        )
                    for identifier in more_ids:
                        if (
                            isinstance(identifier, str)
                            and re.fullmatch(r"[a-z0-9]{1,32}", identifier)
                            and identifier not in fetched_ids
                        ):
                            fetched_ids.add(identifier)
                            additional.append(identifier)
                    continue
                if node.get("kind") != "t1":
                    continue
                identifier = data.get("id", "")
                if not isinstance(identifier, str) or not re.fullmatch(
                    r"[a-z0-9]{1,32}", identifier
                ):
                    self.warn("Skipped a Reddit comment with an invalid identifier.")
                    continue
                body = data.get("body", "")
                if isinstance(body, str):
                    self.add(
                        f"t1_{identifier}" if identifier else "",
                        html.unescape(body),
                        data.get("created_utc"),
                        query,
                        record_type="comment",
                        source_url=f"https://www.reddit.com/comments/{post_id}/_/{identifier}/",
                    )
                replies = data.get("replies")
                if replies:
                    replies_children, _ = self._listing(replies)
                    nodes.extend(replies_children)
            if not additional or self.full:
                break
            if requests_made >= self.max_pages:
                self.warn(
                    "Reached the per-post comment request budget; Reddit comment trees are incomplete."
                )
                break
            batch = [additional.popleft() for _ in range(min(100, len(additional)))]
            more = self.client.request_json(
                "GET",
                "https://oauth.reddit.com/api/morechildren",
                headers=self._headers(),
                params={
                    "link_id": f"t3_{post_id}",
                    "children": ",".join(batch),
                    "api_type": "json",
                    "raw_json": 1,
                    "sort": "new",
                },
            )
            requests_made += 1
            try:
                things = more["json"]["data"]["things"]
                if more["json"].get("errors") or not isinstance(things, list):
                    raise ValueError
                if any(not isinstance(item, dict) for item in things):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise CollectionError(
                    "Reddit returned an unexpected omitted-comment response."
                ) from None
            nodes.extend(things)

    def collect(self) -> list[Record]:
        self.warn(
            "Reddit search is a bounded API sample, not a complete historical archive or comment keyword search."
        )
        for subreddit in self.subreddits:
            for query in self.queries:
                after: str | None = None
                seen_cursors: set[str] = set()
                for _ in range(self.max_pages):
                    if self.full:
                        return self.records
                    params: dict[str, Any] = {
                        "q": query,
                        "restrict_sr": "true",
                        "sort": "new",
                        "type": "link",
                        "t": "all",
                        "limit": min(100, self.max_records - len(self.records)),
                        "raw_json": 1,
                    }
                    if after:
                        params["after"] = after
                    payload = self.client.request_json(
                        "GET",
                        f"https://oauth.reddit.com/r/{subreddit}/search",
                        headers=self._headers(),
                        params=params,
                    )
                    children, next_after = self._listing(payload)
                    for node in children:
                        if self.full:
                            break
                        if node.get("kind") != "t3" or not isinstance(node.get("data"), dict):
                            continue
                        item = node["data"]
                        identifier = item.get("id", "")
                        if not isinstance(identifier, str) or not re.fullmatch(
                            r"[a-z0-9]{1,32}", identifier
                        ):
                            self.warn("Skipped a Reddit post with an invalid identifier.")
                            continue
                        if self.timestamp_in_window(item.get("created_utc")) is None:
                            continue
                        title, body = item.get("title", ""), item.get("selftext", "")
                        title = title if isinstance(title, str) else ""
                        body = (
                            body
                            if isinstance(body, str) and body not in {"[removed]", "[deleted]"}
                            else ""
                        )
                        was_new = self.add(
                            f"t3_{identifier}",
                            html.unescape(f"{title}\n{body}".strip()),
                            item.get("created_utc"),
                            query,
                            source_url=f"https://www.reddit.com/comments/{identifier}/",
                        )
                        if self.include_comments and was_new and not self.full:
                            self._comments(identifier, query)
                    if self.full or not next_after:
                        break
                    if next_after in seen_cursors:
                        self.warn("Stopped Reddit pagination because the API repeated a cursor.")
                        break
                    seen_cursors.add(next_after)
                    after = next_after
                else:
                    self.warn(
                        "Reached max_pages for Reddit search; collection is a bounded sample."
                    )
        return self.records
