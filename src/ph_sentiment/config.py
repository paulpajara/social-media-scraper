"""Validated TOML settings; API secrets are intentionally never read from TOML."""

import tomllib
from pathlib import Path

from .errors import ConfigurationError
from .preprocessing import date_window

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "research.example.toml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    try:
        with Path(path).open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(
            "Cannot read config. Supply a valid TOML file using --config."
        ) from exc
    research = config.get("research", {})
    if not isinstance(research, dict):
        raise ConfigurationError("[research] must be a TOML table.")
    date_window(research.get("start_date", "2026-01-01"), research.get("end_date", "2027-01-01"))
    queries = research.get("queries", [])
    if (
        not isinstance(queries, list)
        or not queries
        or any(
            not isinstance(query, str) or not query.strip() or len(query) > 1024
            for query in queries
        )
    ):
        raise ConfigurationError(
            "research.queries must be a nonempty list of query strings (<=1024 chars)."
        )
    for key, default, upper in [("max_records", 1000, 100_000), ("max_pages", 20, 1000)]:
        value = research.get(key, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ConfigurationError(f"research.{key} must be an integer from 1 to {upper}.")
        research[key] = value
    if type(research.get("include_comments", False)) is not bool:
        raise ConfigurationError("research.include_comments must be true or false.")
    research.setdefault("include_comments", False)
    research.setdefault("start_date", "2026-01-01")
    research.setdefault("end_date", "2027-01-01")
    config["research"] = research
    for name, section in config.items():
        if not isinstance(section, dict):
            raise ConfigurationError(f"[{name}] must be a TOML table.")
        if any(
            any(secret in key.lower() for secret in ("token", "secret", "password"))
            for key in section
        ):
            raise ConfigurationError("Store credentials in environment variables, not TOML.")
    analysis = config.get("analysis", {})
    defaults = {
        "minimum_records": 200,
        "n_topics": 3,
        "seed": 42,
        "max_features": 5000,
        "top_terms": 10,
        "max_pairs": 5000,
    }
    maximums = {
        "minimum_records": 100_000,
        "n_topics": 50,
        "seed": 2**32 - 1,
        "max_features": 50_000,
        "top_terms": 50,
        "max_pairs": 100_000,
    }
    for key, default in defaults.items():
        value = analysis.get(key, default)
        minimum = {"seed": 0, "n_topics": 3, "top_terms": 2, "minimum_records": 3}.get(key, 1)
        if type(value) is not int or not minimum <= value <= maximums[key]:
            raise ConfigurationError(
                f"analysis.{key} must be an integer from {minimum} to {maximums[key]}."
            )
        analysis[key] = value
    config["analysis"] = analysis
    return config
