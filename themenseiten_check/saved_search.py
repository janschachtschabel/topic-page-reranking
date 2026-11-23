"""Run a widget's saved search -- as an approximation, and saying so.

The edu-sharing client deliberately does not run these: the stored filters
carry ``virtual:`` fields the metadata set does not know. Here they are run
anyway because on a topic page they are often where most of the visible
content comes from. What makes the result an approximation is reported, never
hidden: every ``virtual:`` filter dropped, every filter the repository ignored
or could not resolve, and the cap on hits.

The shape of ``propertyFilters`` is not documented by the client. Only the
plain form ``{property: value | [values]}`` is used; anything else is skipped
and named.
"""

from __future__ import annotations

from typing import Any

from edusharing import AsyncRepository, EduSharingError

#: How many hits of one saved search are read -- about what a swimlane shows.
DEFAULT_LIMIT = 24


def split_filters(filters: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    """Usable filters, and the names of the ones that had to be dropped."""
    usable: dict[str, list[str]] = {}
    dropped: list[str] = []
    for prop, value in filters.items():
        if prop.startswith("virtual:"):
            dropped.append(prop)
        elif isinstance(value, str) and value:
            usable[prop] = [value]
        elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            usable[prop] = list(value)
        else:
            dropped.append(prop)
    return usable, dropped


async def run_saved_search(repo: AsyncRepository, search: dict[str, Any], *,
                           limit: int = DEFAULT_LIMIT,
                           properties: tuple[str, ...] = ()) -> tuple[list[dict], list[str]]:
    """Hits of one saved search, and what makes them an approximation.

    Never raises for a refused search: one bad widget must not cost the page.
    """
    usable, dropped = split_filters(search.get("filters") or {})
    notes = [f"Filter nicht angewandt: {', '.join(dropped)}"] if dropped else []
    try:
        answer = await repo.flows.search(search.get("text") or None, raw_filters=usable or None,
                                         limit=limit, properties=properties)
    except EduSharingError as exc:
        return [], notes + [f"Suche abgelehnt ({type(exc).__name__})"]
    lost = [str(u) for u in answer.get("unresolved") or []] + list(answer.get("ignored") or [])
    if lost:
        notes.append(f"vom Repositorium nicht angewandt: {', '.join(lost)}")
    if answer.get("total", 0) > len(answer["hits"]):
        notes.append(f"erste {len(answer['hits'])} von {answer['total']} Treffern")
    return answer["hits"], notes
