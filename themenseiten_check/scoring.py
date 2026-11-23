"""Relate each material to the topic it is shown under, and decide what stays.

The query side is the topic page itself: title, keywords and -- selectable --
its description, its compendium text, or both (``TOPIC_TEXTS``).
The document side is what a material says about itself. A reranker scores each
pair; everything at or above the threshold is kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .collect import Item, TopicPage

#: Cross-encoders truncate at their own token limit; this keeps the input sane
#: for embedding APIs too and keeps one runaway description from dominating.
_MAX_CHARS = 1500
#: The topic side may run longer: it carries the compendium text as well.
_MAX_QUERY_CHARS = 2500


class Reranker(Protocol):
    name: str
    #: A sensible cut-off on this backend's own score scale.
    default_threshold: float

    async def score(self, query: str, documents: list[str]) -> list[float]:
        """One relevance score per document, same order, higher is better."""
        ...


@dataclass(frozen=True)
class Scored:
    item: Item
    score: float
    keep: bool
    #: 1-based position on the original page, to show how far it moved.
    original_rank: int


#: Which text of the topic page the materials are measured against. The
#: description is the default: short and written for the page. "headings" are the
#: compendium's h1-h3 headings -- its outline, short enough for a cross-encoder.
#: The full compendium text is longer and broader -- more vocabulary, also more
#: that is not the topic. "both" is description plus full compendium.
TOPIC_TEXTS = ("description", "headings", "compendium", "both")
DEFAULT_TOPIC_TEXT = "description"


def topic_texts(topic: TopicPage, source: str = DEFAULT_TOPIC_TEXT) -> tuple[list[str], str]:
    """The chosen texts, and which one was actually used.

    When the chosen text is empty the other one stands in -- title and
    keywords alone are too thin to measure against -- and the second value
    says so, so the caller can show it.

    Raises:
        ValueError: for a source not in ``TOPIC_TEXTS``.
    """
    if source not in TOPIC_TEXTS:
        raise ValueError(f"unknown topic text {source!r}, expected one of {TOPIC_TEXTS}")
    available = {"description": topic.description, "compendium": topic.compendium,
                 "headings": ", ".join(topic.compendium_headings)}
    wanted = ["description", "compendium"] if source == "both" else [source]
    used = [name for name in wanted if available[name]]
    if not used:
        # Stand-in, shortest first: what was asked for was short.
        used = [name for name in ("description", "headings", "compendium") if available[name]][:1]
    label = "both" if len(used) == 2 else (used[0] if used else "none")
    return [available[name] for name in used], label


def topic_query(topic: TopicPage, source: str = DEFAULT_TOPIC_TEXT) -> str:
    parts = [f"Thema: {topic.title}"]
    if topic.keywords:
        parts.append("Schlagworte: " + ", ".join(topic.keywords))
    parts.extend(topic_texts(topic, source)[0])
    return "\n".join(parts)[:_MAX_QUERY_CHARS]


def item_text(item: Item) -> str:
    parts = [item.title]
    if item.description:
        parts.append(item.description)
    if item.keywords:
        parts.append("Schlagworte: " + ", ".join(item.keywords))
    if item.subjects:
        parts.append("Fach: " + ", ".join(item.subjects))
    return "\n".join(parts)[:_MAX_CHARS]


async def rerank(topic: TopicPage, reranker: Reranker, threshold: float | None = None,
                 source: str = DEFAULT_TOPIC_TEXT) -> list[Scored]:
    """Score every item against the topic, best first.

    Ties keep the original page order, so two runs over the same page with the
    same scores produce the same list.
    """
    if not topic.items:
        return []
    cut = reranker.default_threshold if threshold is None else threshold
    if hasattr(reranker, "score_page"):  # needs the whole page, e.g. for IDF
        scores = await reranker.score_page(topic, list(topic.items), source)
    else:
        scores = await reranker.score(topic_query(topic, source),
                                      [item_text(i) for i in topic.items])
    if len(scores) != len(topic.items):
        raise RuntimeError(f"{reranker.name} returned {len(scores)} scores "
                           f"for {len(topic.items)} items")
    scored = [Scored(item, float(s), float(s) >= cut, rank)
              for rank, (item, s) in enumerate(zip(topic.items, scores), start=1)]
    return sorted(scored, key=lambda s: (-s.score, s.original_rank))
