"""Read-only access diagnostics; never export tokens, identities or error bodies."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .collectors.base import HTTPClient, require_env
from .csvio import write_csv
from .errors import CollectionError, ConfigurationError, PipelineError

ACCESS_PLATFORMS = ("threads", "x")
ACCESS_FIELDS = (
    "platform",
    "check",
    "status",
    "http_status",
    "error_category",
    "message",
    "threads_basic",
    "threads_keyword_search",
    "expires_at",
    "credit_balance_usd",
)


@dataclass(frozen=True, slots=True)
class TokenMetadata:
    is_valid: bool
    scopes: tuple[str, ...]
    expires_at: datetime | None


def inspect_threads_token(client: HTTPClient, token: str) -> TokenMetadata:
    """A Threads tester token can inspect itself; no app secret is required.

    The debugger requires input_token as a query parameter. Send it only to the
    fixed TLS endpoint and keep requests exceptions, URLs and metadata private.
    Scope presence does not establish App Review approval for public search.
    """
    payload = client.request_json(
        "GET",
        "https://graph.threads.net/debug_token",
        headers={"Authorization": f"Bearer {token}"},
        params={"input_token": token},
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("is_valid"), bool):
        raise CollectionError("Threads returned unverifiable token metadata.")
    # Invalid-token responses may legitimately omit scope and expiration fields.
    if not data["is_valid"]:
        return TokenMetadata(False, (), None)
    scopes, expiry = data.get("scopes"), data.get("expires_at")
    if (
        not isinstance(scopes, list)
        or any(not isinstance(scope, str) for scope in scopes)
        or isinstance(expiry, bool)
        or not isinstance(expiry, int)
        or not 0 <= expiry <= 253402300799
    ):
        raise CollectionError("Threads returned unverifiable token permission or expiry metadata.")
    expires_at = datetime.fromtimestamp(expiry, UTC) if expiry else None
    return TokenMetadata(data["is_valid"], tuple(scopes), expires_at)


def inspect_x_credits(client: HTTPClient, token: str) -> float:
    """Read the documented USD balance; no post-search request or purchase."""
    import math

    payload = client.request_json(
        "GET",
        "https://api.x.com/2/usage/credits",
        headers={"Authorization": f"Bearer {token}"},
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    balance = data.get("total_balance") if isinstance(data, dict) else None
    if isinstance(balance, bool) or not isinstance(balance, (float, int)):
        raise CollectionError("X returned unverifiable credit balance metadata.")
    try:
        value = float(balance)
    except OverflowError:
        raise CollectionError("X returned unverifiable credit balance metadata.") from None
    if not math.isfinite(value) or value < 0:
        raise CollectionError("X returned unverifiable credit balance metadata.")
    return value


def check_access(
    output: Path,
    platforms: Sequence[str] = ACCESS_PLATFORMS,
    *,
    client_factory: Callable[[], HTTPClient] = HTTPClient,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Check each credential independently and checkpoint safe CSV diagnostics."""
    selected = list(dict.fromkeys(platforms))
    if not selected or any(platform not in ACCESS_PLATFORMS for platform in selected):
        raise ConfigurationError("Access checks support Threads and X only.")
    rows: list[dict[str, Any]] = []
    for platform in selected:
        row = dict.fromkeys(ACCESS_FIELDS, "")
        row.update(
            platform=platform, check="token_metadata" if platform == "threads" else "credits"
        )
        client = None
        try:
            token = require_env(
                "THREADS_ACCESS_TOKEN" if platform == "threads" else "X_BEARER_TOKEN"
            )
            client = client_factory()
            if platform == "threads":
                metadata = inspect_threads_token(client, token)
                row.update(
                    http_status=200,
                    threads_basic="threads_basic" in metadata.scopes,
                    threads_keyword_search="threads_keyword_search" in metadata.scopes,
                    expires_at=metadata.expires_at.isoformat() if metadata.expires_at else "",
                )
                if not metadata.is_valid or (
                    metadata.expires_at and metadata.expires_at <= datetime.now(UTC)
                ):
                    row.update(
                        status="blocked",
                        error_category="authentication",
                        message="Threads token is invalid or expired; generate a new token locally.",
                    )
                elif not row["threads_basic"] or not row["threads_keyword_search"]:
                    row.update(
                        status="blocked",
                        error_category="permission_missing",
                        message="Authorize threads_basic and threads_keyword_search and regenerate the token. Public search also requires App Review approval.",
                    )
                else:
                    row.update(
                        status="scope_granted",
                        message="Required token scopes are present. Confirm public threads_keyword_search App Review approval; scopes alone do not prove public-search access.",
                    )
            else:
                balance = inspect_x_credits(client, token)
                row.update(http_status=200, credit_balance_usd=balance)
                if balance <= 0:
                    row.update(
                        status="blocked",
                        error_category="credits_depleted",
                        message="X API credit balance is zero. Check Billing/Credits in the X Developer Console before retrying search.",
                    )
                else:
                    row.update(
                        status="credits_available",
                        message="X credits are available. Endpoint entitlement and spending limits still apply; search reads can incur charges.",
                    )
        except PipelineError as exc:
            row.update(
                status="missing_credentials" if client is None else "unverified",
                http_status=getattr(exc, "status_code", ""),
                error_category=getattr(exc, "category", ""),
                message=str(exc),
            )
        finally:
            if client is not None:
                client.close()
        rows.append(row)
        write_csv(output, rows, ACCESS_FIELDS)
        progress(f"{platform}: {row['status']}. {row['message']}")
    return rows
