"""A transparent, fixed English/Filipino/Taglish sentiment baseline.

The score describes matched language, not a probability or government approval.
Sarcasm, quotations, target attribution, and mixed sentiment need human review.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from .preprocessing import clean_text, tokenize

LEXICON_VERSION = "en-fil-taglish-2026.1"

# Curated polarity terms: no politician, party, or demographic identity cues.
LEXICON: Mapping[str, float] = MappingProxyType(
    {
        "accountable": 1.0,
        "achievement": 1.0,
        "appreciate": 1.0,
        "better": 1.0,
        "competent": 1.0,
        "effective": 1.0,
        "efficient": 1.0,
        "excellent": 1.5,
        "fair": 1.0,
        "good": 1.0,
        "great": 1.5,
        "helpful": 1.0,
        "honest": 1.0,
        "improved": 1.0,
        "improvement": 1.0,
        "love": 1.0,
        "progress": 1.0,
        "reliable": 1.0,
        "successful": 1.0,
        "support": 1.0,
        "transparent": 1.0,
        "trustworthy": 1.0,
        "maayos": 1.0,
        "mabuti": 1.0,
        "magaling": 1.0,
        "maganda": 1.0,
        "mahusay": 1.5,
        "makatarungan": 1.0,
        "matapat": 1.0,
        "nakatulong": 1.0,
        "pagbuti": 1.0,
        "suporta": 1.0,
        "tagumpay": 1.0,
        "tiwala": 1.0,
        "tumutulong": 1.0,
        "abusive": -1.5,
        "awful": -1.5,
        "bad": -1.0,
        "corrupt": -1.5,
        "corruption": -1.5,
        "disappointing": -1.0,
        "dishonest": -1.0,
        "failed": -1.0,
        "failure": -1.0,
        "hate": -1.0,
        "incompetent": -1.5,
        "ineffective": -1.0,
        "inefficient": -1.0,
        "poor": -1.0,
        "terrible": -1.5,
        "unfair": -1.0,
        "useless": -1.0,
        "worse": -1.0,
        "worst": -1.5,
        "abuso": -1.5,
        "bulok": -1.5,
        "kawawa": -1.0,
        "kurakot": -1.5,
        "kurapsyon": -1.5,
        "mahina": -1.0,
        "masama": -1.0,
        "nakakadismaya": -1.0,
        "nakakainis": -1.0,
        "palpak": -1.5,
        "pangit": -1.0,
        "sablay": -1.0,
        "tiwali": -1.5,
        "walangkwenta": -1.5,
        "❤": 1.0,
        "❤️": 1.0,
        "👍": 1.0,
        "👎": -1.0,
    }
)

_NEGATORS = frozenset(
    {"not", "no", "never", "neither", "cannot", "hindi", "di", "wala", "walang", "huwag"}
)
_INTENSIFIERS = MappingProxyType(
    {
        "very": 1.5,
        "really": 1.3,
        "extremely": 1.8,
        "so": 1.3,
        "sobrang": 1.8,
        "sobra": 1.5,
        "napaka": 1.8,
        "talagang": 1.5,
        "medyo": 0.6,
        "slightly": 0.6,
        "somewhat": 0.6,
    }
)
_CONTRASTS = frozenset({"but", "however", "pero", "ngunit", "subalit"})
_CLAUSE_BOUNDARY = re.compile(r"[.!?;,:]+")
_CONTRACTIONS = re.compile(r"\b(\w+)n['’]t\b", re.IGNORECASE)
_WALANG_KWENTA = re.compile(r"\bwalang\s+kwenta\b")


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Bounded heuristic score and an audit trail of scored language."""

    label: str
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SentimentClassifier:
    """An immutable baseline; there are no learned or tunable parameters."""

    version: ClassVar[str] = LEXICON_VERSION

    def predict(self, text: str) -> SentimentResult:
        """Score each clause with a three-token negation window.

        Negation resets at punctuation and contrast markers. The final contrast
        clause receives greater weight. Intensifiers affect the next cue only.
        """
        normalized = clean_text(text)
        normalized = _CONTRACTIONS.sub(r"\1 not", normalized)
        normalized = _WALANG_KWENTA.sub("walangkwenta", normalized)
        clauses = [tokenize(part) for part in _CLAUSE_BOUNDARY.split(normalized)]
        flat_tokens = [token for clause in clauses for token in clause]
        contrasts = [i for i, token in enumerate(flat_tokens) if token in _CONTRASTS]
        last_contrast = contrasts[-1] if contrasts else None
        total = 0.0
        evidence: list[str] = []
        absolute_position = 0

        for tokens in clauses:
            boundary = 0
            for index, token in enumerate(tokens):
                position = absolute_position + index
                if token in _CONTRASTS:
                    boundary = index + 1
                    continue
                base = LEXICON.get(token)
                prefix_multiplier = 1.0
                # Productive Filipino intensification, e.g. "napakaganda".
                if base is None and token.startswith("napaka"):
                    stem = token[len("napaka") :]
                    base = LEXICON.get(stem, LEXICON.get(f"ma{stem}"))
                    prefix_multiplier = 1.8
                if base is None:
                    continue
                start = max(boundary, index - 3)
                preceding = tokens[start:index]
                negations = [
                    word
                    for offset, word in enumerate(preceding)
                    if word in _NEGATORS
                    and not (
                        offset + 1 < len(preceding)
                        and preceding[offset + 1] in {"only", "lang", "lamang"}
                    )
                ]
                value = base * prefix_multiplier
                if len(negations) % 2:
                    value = -value
                if preceding:
                    value *= _INTENSIFIERS.get(preceding[-1], 1.0)
                if last_contrast is not None:
                    value *= 0.5 if position < last_contrast else 1.5
                total += value
                cue = " ".join([*negations, token])
                evidence.append(f"{cue}:{value:+.3f}")
            absolute_position += len(tokens)

        score = math.tanh(total / 3.0)
        label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        return SentimentResult(label, score, tuple(evidence))
