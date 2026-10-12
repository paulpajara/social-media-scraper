"""Shared, bounded HTTP transport and collector utilities.

Only documented platform APIs are contacted. Credentials stay in headers;
API error bodies and request URLs are never logged.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import requests

from ph_sentiment.errors import (
    CollectionError,
    ConfigurationError,
    DataValidationError,
    HTTPCollectionError,
)
from ph_sentiment.preprocessing import date_window
from ph_sentiment.schema import Record, parse_timestamp

API_HOSTS = frozenset(
    {
        "api.x.com",
        "graph.threads.net",
        "api.bsky.app",
        "mastodon.social",
    }
)


def date_bounds(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """Interpret dates as inclusive/exclusive midnight in the Manila calendar."""
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start_date) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", end_date
        ):
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigurationError("Collection dates must use valid YYYY-MM-DD values.") from None
    return date_window(start_date, end_date)


def utc_timestamp(value: Any) -> datetime:
    """Parse documented epoch or timezone-aware ISO timestamps without guessing."""
    if isinstance(value, bool):
        raise ValueError("Invalid timestamp")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC)
    if not isinstance(value, str):
        raise ValueError("Invalid timestamp")
    try:
        return parse_timestamp(value)
    except DataValidationError:
        raise ValueError("Invalid API timestamp") from None


def iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def require_env(name: str) -> str:
    """Name missing credentials, never expose their values."""
    value = os.environ.get(name, "").strip()
    if (
        not value
        or not value.isascii()
        or any(character in value for character in ("\r", "\n", "\x00"))
    ):
        raise ConfigurationError(f"Set a valid {name} environment variable before collecting.")
    return value


def api_url(url: str) -> str:
    """Reject redirects, arbitrary hosts, credentials in URLs, and insecure transport."""
    if not isinstance(url, str):
        raise CollectionError("Refused an unsafe API or pagination URL.")
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname in API_HOSTS
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
            and not any(ord(character) < 32 for character in url)
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise CollectionError("Refused an unsafe API or pagination URL.")
    return url


class HTTPClient:
    """Reusable Requests session with timeouts, bounded retries and safe failures.

    max_retries counts retries after the first attempt. Retry-After values above
    max_wait cause a clear failure instead of retrying before the server allows.
    Inject session/sleep/clock to run tests without network requests or delays.
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        timeout: float = 30,
        max_retries: int = 3,
        max_wait: float = 60,
        min_interval: float = 0.65,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 300
            or isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 10
            or isinstance(min_interval, bool)
            or not isinstance(min_interval, (int, float))
            or not 0 <= min_interval <= 60
            or isinstance(max_wait, bool)
            or not isinstance(max_wait, (int, float))
            or not 0 < max_wait <= 300
        ):
            raise ConfigurationError("Invalid HTTP timeout, retry or pacing settings.")
        self.session = session if session is not None else requests.Session()
        if session is None:
            # Avoid sending bearer credentials through an unexpected inherited proxy.
            self.session.trust_env = False
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.min_interval = min_interval
        self.sleep = sleep
        self.clock = clock
        self._last_request: float | None = None

    def close(self) -> None:
        self.session.close()

    def _wait(self, headers: Mapping[str, str], attempt: int) -> None:
        delay = min(2**attempt, self.max_wait)
        retry_after = headers.get("Retry-After")
        if retry_after:
            try:
                seconds = float(retry_after)
                if not math.isfinite(seconds):
                    raise ValueError
                delay = max(seconds, 0)
            except (TypeError, ValueError):
                try:
                    retry_date = parsedate_to_datetime(retry_after)
                    delay = max((retry_date - datetime.now(UTC)).total_seconds(), 0)
                except (TypeError, ValueError, OverflowError):
                    pass
        if delay > self.max_wait:
            raise CollectionError(
                "API rate limit requires a longer wait; retry this collection later."
            )
        self.sleep(delay)

    def request_json(
        self, method: str, url: str, *, allow_partial_errors: bool = False, **kwargs: Any
    ) -> Any:
        api_url(url)
        if not isinstance(method, str) or method not in {"GET", "POST"}:
            raise CollectionError("Unsupported collection HTTP method.")
        if not isinstance(allow_partial_errors, bool):
            raise CollectionError("Partial-response handling must be true or false.")
        if {"timeout", "allow_redirects", "verify"}.intersection(kwargs):
            raise CollectionError("Collection transport safety options cannot be overridden.")
        for attempt in range(self.max_retries + 1):
            if self._last_request is not None:
                remaining = self.min_interval - (self.clock() - self._last_request)
                if remaining > 0:
                    self.sleep(remaining)
            self._last_request = self.clock()
            try:
                response = self.session.request(
                    method, url, timeout=self.timeout, allow_redirects=False, **kwargs
                )
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise CollectionError("API connection failed after bounded retries.") from None
                self._wait({}, attempt)
                continue
            status = response.status_code
            headers = response.headers
            if status == 429 or status in {500, 502, 503, 504}:
                response.close()
                if attempt == self.max_retries:
                    raise CollectionError(f"API returned HTTP {status} after bounded retries.")
                self._wait(headers, attempt)
                continue
            if status in {401, 403}:
                response.close()
                raise HTTPCollectionError(
                    status,
                    "authentication" if status == 401 else "permission",
                    f"API returned HTTP {status}; check token validity, app approval and read permissions.",
                )
            if status == 402:
                response.close()
                raise HTTPCollectionError(
                    status,
                    "payment_required",
                    "API returned HTTP 402 Payment Required; check API credits, billing, and endpoint access.",
                )
            if not 200 <= status < 300:
                response.close()
                raise HTTPCollectionError(
                    status,
                    "http_error",
                    f"API returned HTTP {status}; check collection settings and API access.",
                )
            try:
                payload = response.json()
            except (ValueError, requests.RequestException):
                raise CollectionError("API returned an invalid JSON response.") from None
            finally:
                response.close()
            if isinstance(payload, dict) and (
                payload.get("error")
                or (
                    payload.get("errors")
                    and not (
                        allow_partial_errors
                        and isinstance(payload.get("data"), list)
                        and payload["data"]
                    )
                )
            ):
                raise CollectionError(
                    "API reported an error; check read permissions, quota and query settings."
                )
            return payload
        raise CollectionError("API request did not complete.")


class Collector:
    """Common date filtering, uniqueness and budget controls for each source."""

    platform = ""

    def __init__(self, config: dict[str, Any], client: HTTPClient | None = None) -> None:
        self.config = config
        self.settings = config.get("settings", {})
        if not isinstance(self.settings, dict):
            raise ConfigurationError("Collector settings must be a table.")
        self.queries = config.get("queries", [])
        if (
            not isinstance(self.queries, list)
            or not self.queries
            or any(not isinstance(query, str) or not query.strip() for query in self.queries)
        ):
            raise ConfigurationError("Collection requires at least one nonempty query.")
        self.start, self.end = date_bounds(config.get("start_date", ""), config.get("end_date", ""))
        self.max_records = self._integer(config, "max_records", 1000, 1, 1_000_000)
        self.max_pages = self._integer(config, "max_pages", 20, 1, 1000)
        self.include_comments = config.get("include_comments", False)
        if not isinstance(self.include_comments, bool):
            raise ConfigurationError("include_comments must be true or false.")
        self._owns_client = client is None
        self.client = (
            client
            if client is not None
            else HTTPClient(
                timeout=config.get("request_timeout", 30), max_retries=config.get("max_retries", 3)
            )
        )
        self.warnings: list[str] = []
        self.records: list[Record] = []
        self._seen: set[str] = set()
        self.collected_at = iso_utc(datetime.now(UTC))

    @staticmethod
    def _integer(config: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ConfigurationError(f"{key} must be an integer from {minimum} to {maximum}.")
        return value

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def full(self) -> bool:
        return len(self.records) >= self.max_records

    def timestamp_in_window(self, value: Any) -> datetime | None:
        try:
            created = utc_timestamp(value)
        except (ValueError, OverflowError, OSError):
            self.warn("Skipped an API item with a missing or invalid publication timestamp.")
            return None
        return created if self.start <= created < self.end else None

    def add(
        self,
        identifier: Any,
        text: Any,
        timestamp: Any,
        query: str,
        *,
        source_url: str = "",
        record_type: str = "post",
    ) -> bool:
        if self.full or not isinstance(identifier, str) or not identifier:
            return False
        if (
            not isinstance(text, str)
            or not text.strip()
            or text.strip() in {"[removed]", "[deleted]"}
        ):
            return False
        created = self.timestamp_in_window(timestamp)
        if created is None:
            return False
        record_id = f"{self.platform}:{identifier}"
        if record_id in self._seen:
            return False
        record = Record(
            record_id=record_id,
            platform=self.platform,
            created_at=iso_utc(created),
            text=text.strip(),
            source_url=source_url,
            query=query,
            record_type=record_type,
            collected_at=self.collected_at,
        )
        record.validate()
        self.records.append(record)
        self._seen.add(record_id)
        return True

    def graph_pages(
        self, endpoint: str, params: dict[str, Any], headers: dict[str, str]
    ) -> Iterator[list[dict[str, Any]]]:
        """Reconstruct Graph requests from a cursor, never follow returned token URLs."""
        expected = urlsplit(api_url(endpoint))
        seen_cursors: set[str] = set()
        page_params = dict(params)
        for _ in range(self.max_pages):
            payload = self.client.request_json("GET", endpoint, params=page_params, headers=headers)
            if not isinstance(payload, dict) or not isinstance(payload.get("data", []), list):
                raise CollectionError("API returned an unexpected collection response.")
            items = payload.get("data", [])
            if any(not isinstance(item, dict) for item in items):
                raise CollectionError("API returned an unexpected collection item.")
            yield items
            if self.full:
                return
            paging = payload.get("paging") or {}
            if not isinstance(paging, dict):
                raise CollectionError("API returned invalid pagination metadata.")
            next_url = paging.get("next")
            if not next_url:
                return
            if not isinstance(next_url, str):
                raise CollectionError("API returned invalid pagination metadata.")
            parsed = urlsplit(api_url(next_url))
            if (parsed.hostname, parsed.path) != (expected.hostname, expected.path):
                raise CollectionError("Refused a pagination URL outside the requested endpoint.")
            cursors = paging.get("cursors") or {}
            if not isinstance(cursors, dict):
                raise CollectionError("API returned invalid pagination cursors.")
            cursor = cursors.get("after") or parse_qs(parsed.query).get("after", [None])[0]
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                self.warn(
                    "Stopped pagination because the API returned a missing or repeated cursor."
                )
                return
            seen_cursors.add(cursor)
            page_params = {**params, "after": cursor}
        self.warn(
            "Reached max_pages before exhausting the API results; collection is a bounded sample."
        )

    def collect(self) -> list[Record]:
        raise NotImplementedError
