"""Behavioral tests for the multilingual heuristic and its limitations."""

from dataclasses import FrozenInstanceError

import pytest

from ph_sentiment.sentiment import LEXICON, SentimentClassifier


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The government service is good and helpful.", "positive"),
        ("The transport policy is terrible.", "negative"),
        ("The budget hearing is scheduled tomorrow.", "neutral"),
        ("Hindi maganda ang serbisyo.", "negative"),
        ("Hindi palpak ang serbisyo.", "positive"),
        ("Walang kurapsyon sa proyektong ito.", "positive"),
        ("Sobrang palpak ang programa.", "negative"),
        ("Napakaganda ng serbisyo.", "positive"),
        ("Walang kwenta ang programa.", "negative"),
        ("The service isn't good.", "negative"),
        ("The project is not bad.", "positive"),
        ("I cannot support this policy.", "negative"),
        ("It is not only good, it is helpful.", "positive"),
        ("Hindi lang maganda, mahusay din.", "positive"),
        ("Good, pero palpak ang implementation.", "negative"),
        ("Bad, but the new service is good.", "positive"),
        ("Marcos Duterte Philippine government.", "neutral"),
        ("@good https://example.com/great Budget discussion.", "neutral"),
        ("", "neutral"),
    ],
)
def test_classification_examples(text: str, expected: str) -> None:
    result = SentimentClassifier().predict(text)
    assert result.label == expected
    assert -1.0 <= result.score <= 1.0


def test_negation_does_not_cross_punctuation() -> None:
    result = SentimentClassifier().predict("Not mentioned. Good service.")
    assert result.label == "positive"
    assert "not" not in result.matched_terms[0]


def test_intensifiers_change_score_and_evidence_is_auditable() -> None:
    classifier = SentimentClassifier()
    ordinary = classifier.predict("Good service")
    intensified = classifier.predict("Very good service")
    assert intensified.score > ordinary.score
    assert ordinary.matched_terms == ("good:+1.000",)
    assert classifier.predict("hindi maganda").matched_terms == ("hindi maganda:-1.000",)


def test_model_and_lexicon_are_immutable() -> None:
    classifier = SentimentClassifier()
    assert classifier.version == "en-fil-taglish-2026.1"
    with pytest.raises(TypeError):
        LEXICON["marcos"] = 1.0  # type: ignore[index]
    result = classifier.predict("good")
    with pytest.raises(FrozenInstanceError):
        result.score = 0.0  # type: ignore[misc]
