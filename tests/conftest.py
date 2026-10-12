"""Unit/integration fixtures are offline; real API checks run through the CLI."""

import pytest


@pytest.fixture(autouse=True)
def prevent_unmocked_network(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("Unexpected real HTTP request in a test; inject a mock collector or session.")

    monkeypatch.setattr("requests.sessions.Session.request", unexpected_request)
