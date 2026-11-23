"""Relate each material to the topic it is shown under, and decide what stays.

The query side is the topic page itself: title, description, keywords and the
compendium text -- the last two parts say what the editors mean by the topic;
a portal description is often generic ("Hier findest du ... Material für Physik").
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


def topic_query(topic: TopicPage) -> str:
    parts = [f"Thema: {topic.title}"]
    if topic.description:
        parts.append(topic.description)
    if topic.keywords:
        parts.append("Schlagworte: " + ", ".join(topic.keywords))
    if topic.compendium:
        parts.append(topic.compendium)
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


async def rerank(topic: TopicPage, reranker: Reranker,
                 threshold: float | None = None) -> list[Scored]:
    """Score every item against the topic, best first.

    Ties keep the original page order, so two runs over the same page with the
    same scores produce the same list.
    """
    if not topic.items:
        return []
    cut = reranker.default_threshold if threshold is None else threshold
    if hasattr(reranker, "score_page"):  # needs the whole page, e.g. for IDF
        scores = await reranker.score_page(topic, list(topic.items))
    else:
        scores = await reranker.score(topic_query(topic), [item_text(i) for i in topic.items])
    if len(scores) != len(topic.items):
        raise RuntimeError(f"{reranker.name} returned {len(scores)} scores "
                           f"for {len(topic.items)} items")
    scored = [Scored(item, float(s), float(s) >= cut, rank)
              for rank, (item, s) in enumerate(zip(topic.items, scores), start=1)]
    return sorted(scored, key=lambda s: (-s.score, s.original_rank))
