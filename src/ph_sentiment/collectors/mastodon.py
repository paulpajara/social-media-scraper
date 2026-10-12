"""Collect public hashtag timelines visible on mastodon.social without credentials."""

from __future__ import annotations

import ipaddress
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ph_sentiment.collectors.base import Collector
from ph_sentiment.errors import CollectionError, ConfigurationError
from ph_sentiment.schema import Record

INSTANCE = "https://mastodon.social"


class _PlainTextParser(HTMLParser):
    """Preserve readable paragraph breaks without retaining executable HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        elif not self.hidden and tag in {"p", "br", "div", "li", "blockquote"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        elif not self.hidden and tag in {"p", "div", "li", "blockquote"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def plain_text(value: str) -> str:
    parser = _PlainTextParser()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


def _public_source_url(value: Any) -> str:
    """Retain public HTTPS provenance links, discarding credentials and trackers."""
    if (
        not isinstance(value, str)
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        return ""
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", hostname)
            or hostname.endswith((".local", ".internal", ".test", ".localhost"))
        ):
            return ""
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            return urlunsplit(("https", parsed.netloc, parsed.path, "", ""))
        return ""
    except ValueError:
        return ""


class MastodonCollector(Collector):
    platform = "mastodon"

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        super().__init__(config, client)
        if self.include_comments:
            raise ConfigurationError(
                "Mastodon collection supports public hashtag results only; "
                "set include_comments=false."
            )
        self.hashtags = self.settings.get("hashtags")
        if (
            not isinstance(self.hashtags, list)
            or not 1 <= len(self.hashtags) <= 50
            or any(
                not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,100}", tag)
                for tag in self.hashtags
            )
        ):
            raise ConfigurationError(
                "Mastodon hashtags must contain 1-50 names using letters, numbers or "
                "underscores, without #."
            )
        self.government_pattern = self._term_pattern("government_terms")
        self.country_pattern = self._term_pattern("country_terms")

    def _term_pattern(self, name: str) -> re.Pattern[str]:
        terms = self.settings.get(name)
        if (
            not isinstance(terms, list)
            or not 1 <= len(terms) <= 100
            or any(
                not isinstance(term, str) or not term.strip() or len(term) > 200 for term in terms
            )
        ):
            raise ConfigurationError(f"Mastodon {name} must be a nonempty list of short terms.")
        escaped = (re.escape(" ".join(term.split())) for term in terms)
        return re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE)

    def _add_status(self, item: Any, hashtag: str) -> None:
        if not isinstance(item, dict):
            self.warn("Skipped a Mastodon status with malformed metadata.")
            return
        if item.get("visibility") != "public" or item.get("reblog") is not None:
            return
        identifier, content = item.get("id"), item.get("content")
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9]{1,32}", identifier):
            self.warn("Skipped a Mastodon status with an invalid identifier.")
            return
        spoiler = item.get("spoiler_text", "")
        if (
            not isinstance(content, str)
            or not isinstance(spoiler, str)
            or len(content) + len(spoiler) > 200_000
        ):
            self.warn("Skipped a Mastodon status with missing or oversized text.")
            return
        # spoiler_text is already plain text; only content contains HTML.
        text = " ".join(f"{spoiler} {plain_text(content)}".split())
        if not text or len(text) > 100_000:
            self.warn("Skipped a Mastodon status with empty or oversized text.")
            return
        if not self.government_pattern.search(text) or not self.country_pattern.search(text):
            return
        self.add(
            f"mastodon.social:{identifier}",
            text,
            item.get("created_at"),
            f"#{hashtag}",
            source_url=_public_source_url(item.get("url")),
        )

    def collect(self) -> list[Record]:
        self.warn(
            "Mastodon data is limited to public hashtag posts visible to mastodon.social; "
            "federation and retention limit coverage. Government/country term matches are "
            "a relevance heuristic that needs human review, not a representative sample."
        )
        for hashtag in self.hashtags:
            if self.full:
                break
            params = {"limit": min(40, self.max_records)}
            previous_cursor: int | None = None
            for _ in range(self.max_pages):
                payload = self.client.request_json(
                    "GET", f"{INSTANCE}/api/v1/timelines/tag/{hashtag}", params=params
                )
                if not isinstance(payload, list):
                    raise CollectionError("Mastodon returned an unexpected hashtag response.")
                if not payload:
                    break
                page_ids: list[int] = []
                for item in payload:
                    self._add_status(item, hashtag)
                    if self.full:
                        return self.records
                    identifier = item.get("id") if isinstance(item, dict) else None
                    if isinstance(identifier, str) and re.fullmatch(r"[0-9]{1,32}", identifier):
                        page_ids.append(int(identifier))
                if not page_ids:
                    self.warn("Stopped Mastodon pagination because no valid cursor was available.")
                    break
                cursor = min(page_ids)
                if previous_cursor is not None and cursor >= previous_cursor:
                    self.warn("Stopped Mastodon pagination because the cursor did not advance.")
                    break
                previous_cursor = cursor
                # Numeric IDs keep requests on the fixed instance; never follow Link headers.
                params = {**params, "max_id": str(cursor)}
            else:
                self.warn(
                    "Reached max_pages before exhausting Mastodon results; collection is bounded."
                )
        return self.records
