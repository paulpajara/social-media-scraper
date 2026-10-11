import pytest

from ph_sentiment.config import DEFAULT_CONFIG, load_config
from ph_sentiment.errors import ConfigurationError


def test_example_config_has_activity_requirements():
    config = load_config(DEFAULT_CONFIG)
    assert config["analysis"]["minimum_records"] == 200
    assert config["analysis"]["n_topics"] >= 3


@pytest.mark.parametrize(
    "setting", ["max_records = -1", 'include_comments = "yes"', "queries = []"]
)
def test_invalid_research_configuration(tmp_path, setting):
    path = tmp_path / "bad.toml"
    base = '[research]\nstart_date = "2026-01-01"\nend_date = "2027-01-01"\n'
    if not setting.startswith("queries"):
        base += 'queries = ["government"]\n'
    path.write_text(base + setting, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_rejects_credentials_in_config(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        '[research]\nqueries = ["government"]\n[x]\naccess_token = "placeholder"', encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="environment"):
        load_config(path)
