"""Factory for documented API collectors; credential-free imports are safe."""

from typing import Any

from ph_sentiment.collectors.base import Collector, HTTPClient
from ph_sentiment.collectors.bluesky import BlueskyCollector
from ph_sentiment.collectors.facebook import FacebookCollector
from ph_sentiment.collectors.mastodon import MastodonCollector
from ph_sentiment.collectors.reddit import RedditCollector
from ph_sentiment.collectors.threads import ThreadsCollector
from ph_sentiment.collectors.x import XCollector
from ph_sentiment.errors import ConfigurationError

__all__ = ["Collector", "HTTPClient", "make_collector"]


def make_collector(platform: str, config: dict[str, Any]) -> Collector:
    classes = {
        "reddit": RedditCollector,
        "x": XCollector,
        "threads": ThreadsCollector,
        "facebook": FacebookCollector,
        "bluesky": BlueskyCollector,
        "mastodon": MastodonCollector,
    }
    try:
        collector_class = classes[platform]
    except (KeyError, TypeError):
        raise ConfigurationError(
            "Choose a supported collector: reddit, x, threads, facebook, bluesky, or mastodon."
        ) from None
    return collector_class(config)
