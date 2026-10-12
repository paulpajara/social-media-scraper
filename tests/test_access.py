"""Read-only credential diagnostics use mocked API responses and safe CSV output."""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from ph_sentiment.access import (
    ACCESS_FIELDS,
    TokenMetadata,
    check_access,
    inspect_threads_token,
    inspect_x_credits,
)
from ph_sentiment.errors import CollectionError, ConfigurationError, HTTPCollectionError


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THREADS_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)


def client_with(response: Any) -> Mock:
    client = Mock()
    if isinstance(response, Exception):
        client.request_json.side_effect = response
    else:
        client.request_json.return_value = response
    return client


def threads_data(**overrides: Any) -> dict[str, Any]:
    return {
        "data": {
            "is_valid": True,
            "scopes": ["threads_basic", "threads_keyword_search"],
            "expires_at": 0,
            **overrides,
        }
    }


def enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "offline-threads-secret")
    monkeypatch.setenv("X_BEARER_TOKEN", "offline-x-secret")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(ACCESS_FIELDS)
        return list(reader)


def test_threads_metadata_preserves_scopes_and_timezone_aware_expiry() -> None:
    expiry = 2_000_000_000
    client = client_with(threads_data(expires_at=expiry, scopes=["threads_basic"]))

    metadata = inspect_threads_token(client, "offline-token")

    assert metadata == TokenMetadata(True, ("threads_basic",), datetime.fromtimestamp(expiry, UTC))
    client.request_json.assert_called_once_with(
        "GET",
        "https://graph.threads.net/debug_token",
        headers={"Authorization": "Bearer offline-token"},
        params={"input_token": "offline-token"},
    )


def test_threads_invalid_token_may_omit_scope_and_expiry_metadata() -> None:
    client = client_with({"data": {"is_valid": False}})

    assert inspect_threads_token(client, "offline-token") == TokenMetadata(False, (), None)


def test_threads_zero_expiry_is_not_an_expired_epoch() -> None:
    assert inspect_threads_token(client_with(threads_data()), "offline-token").expires_at is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"data": []},
        {"data": {}},
        {"data": {"is_valid": "true"}},
        {"data": {"is_valid": 1}},
    ],
)
def test_threads_requires_verified_boolean_validity(payload: Any) -> None:
    with pytest.raises(CollectionError, match="unverifiable token metadata"):
        inspect_threads_token(client_with(payload), "offline-token")


@pytest.mark.parametrize(
    "scopes", [None, "threads_basic", (), [None], [1], {"threads_basic": True}]
)
def test_threads_rejects_malformed_scope_metadata(scopes: Any) -> None:
    with pytest.raises(CollectionError, match="permission or expiry metadata"):
        inspect_threads_token(client_with(threads_data(scopes=scopes)), "offline-token")


@pytest.mark.parametrize("expiry", [None, True, False, -1, "100", 100.0, 253_402_300_800, 10**1000])
def test_threads_rejects_malformed_or_out_of_range_expiry(expiry: Any) -> None:
    with pytest.raises(CollectionError, match="permission or expiry metadata"):
        inspect_threads_token(client_with(threads_data(expires_at=expiry)), "offline-token")


def test_threads_valid_token_with_missing_metadata_is_unverified() -> None:
    with pytest.raises(CollectionError, match="permission or expiry metadata"):
        inspect_threads_token(client_with({"data": {"is_valid": True}}), "offline-token")


@pytest.mark.parametrize("balance", [0, 1, 1.25])
def test_x_credit_balance_contract_is_read_only(balance: float) -> None:
    client = client_with({"data": {"total_balance": balance}})

    assert inspect_x_credits(client, "offline-token") == float(balance)
    client.request_json.assert_called_once_with(
        "GET",
        "https://api.x.com/2/usage/credits",
        headers={"Authorization": "Bearer offline-token"},
    )


@pytest.mark.parametrize(
    "balance",
    [None, True, False, "1.25", [], {}, -1, float("nan"), float("inf"), -float("inf"), 10**1000],
)
def test_x_rejects_unverifiable_balances_with_safe_errors(balance: Any) -> None:
    with pytest.raises(CollectionError, match="unverifiable credit balance metadata"):
        inspect_x_credits(client_with({"data": {"total_balance": balance}}), "offline-token")


@pytest.mark.parametrize(
    "payload", [None, [], {}, {"data": []}, {"data": {}}, {"data": {"balance": 10}}]
)
def test_x_requires_documented_balance_wrapper(payload: Any) -> None:
    with pytest.raises(CollectionError, match="unverifiable credit balance metadata"):
        inspect_x_credits(client_with(payload), "offline-token")


def test_missing_credentials_do_not_create_clients_or_send_requests(tmp_path: Path) -> None:
    factory = Mock()
    output = tmp_path / "access.csv"

    rows = check_access(output, client_factory=factory, progress=lambda _: None)

    assert [row["status"] for row in rows] == ["missing_credentials", "missing_credentials"]
    assert [row["platform"] for row in rows] == ["threads", "x"]
    assert len(read_csv(output)) == 2
    factory.assert_not_called()


def test_sources_are_independent_and_checkpointed_before_the_next_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable(monkeypatch)
    output = tmp_path / "access.csv"
    threads_client = client_with(
        HTTPCollectionError(403, "permission", "Threads permission denied.")
    )
    x_client = client_with({"data": {"total_balance": 2.50}})
    clients = [threads_client, x_client]
    checkpoint_sizes: list[int] = []

    def factory() -> Mock:
        if output.exists():
            checkpoint_sizes.append(len(read_csv(output)))
        return clients.pop(0)

    progress: list[str] = []
    rows = check_access(output, client_factory=factory, progress=progress.append)

    assert checkpoint_sizes == [1]
    assert rows[0]["status"] == "unverified"
    assert rows[0]["http_status"] == 403
    assert rows[0]["error_category"] == "permission"
    assert rows[1]["status"] == "credits_available"
    assert rows[1]["credit_balance_usd"] == 2.5
    assert len(progress) == len(read_csv(output)) == 2
    threads_client.close.assert_called_once_with()
    x_client.close.assert_called_once_with()


def test_empty_x_balance_is_a_blocker_not_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("X_BEARER_TOKEN", "offline-x-secret")
    client = client_with({"data": {"total_balance": 0}})

    rows = check_access(
        tmp_path / "access.csv", ["x"], client_factory=lambda: client, progress=lambda _: None
    )

    assert rows[0]["status"] == "blocked"
    assert rows[0]["error_category"] == "credits_depleted"
    assert rows[0]["credit_balance_usd"] == 0
    assert client.request_json.call_count == 1
    client.close.assert_called_once_with()


def test_threads_granted_scopes_keep_public_approval_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "offline-threads-secret")
    client = client_with(
        threads_data(
            application="private-app-identity",
            user_id="private-user-identity",
            scopes=["threads_basic", "threads_keyword_search", "private-upstream-value"],
            access_token="response-secret-token",
        )
    )
    output = tmp_path / "access.csv"

    rows = check_access(output, ["threads"], client_factory=lambda: client)

    assert rows[0]["status"] == "scope_granted"
    assert rows[0]["threads_basic"] is rows[0]["threads_keyword_search"] is True
    assert "do not prove" in rows[0]["message"]
    exposed = capsys.readouterr().out + output.read_text(encoding="utf-8-sig")
    for secret in (
        "offline-threads-secret",
        "private-app-identity",
        "private-user-identity",
        "private-upstream-value",
        "response-secret-token",
    ):
        assert secret not in exposed
    client.close.assert_called_once_with()


@pytest.mark.parametrize("scopes", [[], ["threads_basic"], ["threads_keyword_search"]])
def test_threads_missing_scope_report_names_required_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scopes: list[str]
) -> None:
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "offline-threads-secret")
    client = client_with(threads_data(scopes=scopes))

    rows = check_access(
        tmp_path / "access.csv", ["threads"], client_factory=lambda: client, progress=lambda _: None
    )

    assert rows[0]["status"] == "blocked"
    assert rows[0]["error_category"] == "permission_missing"
    assert "regenerate" in rows[0]["message"]
    client.close.assert_called_once_with()


@pytest.mark.parametrize("metadata", [{"is_valid": False}, {"is_valid": True, "expires_at": 1}])
def test_threads_invalid_or_expired_token_report_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: dict[str, Any]
) -> None:
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "offline-threads-secret")
    client = client_with(threads_data(**metadata))

    rows = check_access(
        tmp_path / "access.csv", ["threads"], client_factory=lambda: client, progress=lambda _: None
    )

    assert rows[0]["status"] == "blocked"
    assert rows[0]["error_category"] == "authentication"
    client.close.assert_called_once_with()


def test_threads_valid_future_expiry_is_exported_as_utc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "offline-threads-secret")
    expiry = int((datetime.now(UTC) + timedelta(days=30)).timestamp())
    client = client_with(threads_data(expires_at=expiry))

    rows = check_access(
        tmp_path / "access.csv", ["threads"], client_factory=lambda: client, progress=lambda _: None
    )

    assert rows[0]["status"] == "scope_granted"
    assert rows[0]["expires_at"] == datetime.fromtimestamp(expiry, UTC).isoformat()


def test_malformed_metadata_is_reported_safely_and_other_source_still_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable(monkeypatch)
    first = client_with({"data": {"is_valid": "upstream-secret"}})
    second = client_with({"data": {"total_balance": 10**1000}})
    factory = Mock(side_effect=[first, second])
    output = tmp_path / "access.csv"

    rows = check_access(output, client_factory=factory, progress=lambda _: None)

    assert [row["status"] for row in rows] == ["unverified", "unverified"]
    assert "upstream-secret" not in output.read_text(encoding="utf-8-sig")
    first.close.assert_called_once_with()
    second.close.assert_called_once_with()


def test_duplicate_platform_selection_is_checked_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("X_BEARER_TOKEN", "offline-x-secret")
    factory = Mock(return_value=client_with({"data": {"total_balance": 1}}))

    rows = check_access(
        tmp_path / "access.csv", ["x", "x"], client_factory=factory, progress=lambda _: None
    )

    assert len(rows) == 1
    factory.assert_called_once_with()


@pytest.mark.parametrize("platforms", [[], ["reddit"], ["threads", "facebook"]])
def test_invalid_platform_selection_fails_before_requests_or_output(
    tmp_path: Path, platforms: list[str]
) -> None:
    output = tmp_path / "access.csv"
    factory = Mock()

    with pytest.raises(ConfigurationError, match="Threads and X only"):
        check_access(output, platforms, client_factory=factory)

    assert not output.exists()
    factory.assert_not_called()
