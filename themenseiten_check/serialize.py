"""A scored topic page as JSON for the web app.

Scores travel with every material; ``keep`` does not. The browser applies the
threshold itself, so moving the slider needs no second scoring run.
"""

from __future__ import annotations

from typing import Any

from .collect import TopicPage
from .scoring import DEFAULT_TOPIC_TEXT, Scored, topic_texts


def page_as_json(topic: TopicPage, scored: list[Scored], *, backend: str,
                 threshold: float, repository: str,
                 topic_text: str = DEFAULT_TOPIC_TEXT) -> dict[str, Any]:
    by_id = {s.item.id: s for s in scored}
    return {
        "topic": {"id": topic.id, "title": topic.title, "description": topic.description,
                  "keywords": list(topic.keywords), "url": topic.url,
                  "compendium": topic.compendium,
                  "headings": list(topic.compendium_headings),
                  "description_source": topic.description_source},
        "items": {s.item.id: {"title": s.item.title, "description": s.item.description,
                              "keywords": list(s.item.keywords),
                              "subjects": list(s.item.subjects), "url": s.item.url,
                              "score": round(s.score, 4)}
                  for s in scored},
        # Only ids the scorer saw -- a lane never points at a material without a score.
        "lanes": [{"id": lane.id, "heading": lane.heading, "kind": lane.kind,
                   "widgets": list(lane.widgets),
                   "item_ids": [i for i in lane.item_ids if i in by_id]}
                  for lane in topic.lanes],
        "notes": list(topic.notes),
        "backend": backend,
        # What was asked for, and what was used -- they differ when the
        # chosen text is empty and the other one stood in.
        "topic_text": {"chosen": topic_text, "used": topic_texts(topic, topic_text)[1]},
        "threshold": threshold,
        "repository": repository,
    }
