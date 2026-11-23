"""Find topic pages by text through the repository, not through the portal.

A WLO topic page is a collection, so ``flows.find_collections`` finds them --
with subject narrowing (applied locally by the client) and two extra
properties that say what a page has to offer: a page-builder layout and a
compendium text.

One leg of the collection search returns hits without properties; for those
both markers are unknown (``None``), not false -- the same honesty the client's
``find_pages`` keeps with its ``checked`` count.
"""

from __future__ import annotations

from typing import Any

from edusharing import AsyncRepository

from .collect import KEYWORD_PROP

PAGE_REF = "ccm:page_config_ref"
#: Every node record carries ``cm:name``; asked for alongside, it tells a hit
#: with properties (judgeable) from one of the property-less leg.
_PROBE = "cm:name"


async def find_topic_pages(repo: AsyncRepository, text: str, *, subject: str = "",
                           limit: int = 25) -> dict[str, Any]:
    """Collections matching ``text``, optionally within one subject.

    Returns:
        ``{query, subject, hits, total, total_is_lower_bound, unresolved,
        warnings}``; each hit ``{id, title, url, description, has_layout,
        has_compendium}``, the two markers ``None`` where the hit carried no
        properties to judge by.
    """
    compendium = repo.metadata_profile.compendium_property
    props = tuple(p for p in (_PROBE, PAGE_REF, compendium, KEYWORD_PROP) if p)
    kwargs: dict[str, Any] = {"limit": limit, "properties": props}
    if subject:
        kwargs["subject"] = subject
    found = await repo.flows.find_collections(text, **kwargs)
    hits = []
    for hit in found["hits"]:
        fields = hit.get("fields") or {}
        judged = bool(fields.get(_PROBE))
        hits.append({
            "id": hit["id"], "title": hit["title"], "url": hit.get("url") or "",
            "description": (hit.get("description") or "")[:300],
            "has_layout": bool(fields.get(PAGE_REF)) if judged else None,
            "has_compendium": (bool(fields.get(compendium)) if judged and compendium else None),
        })
    return {
        "query": text, "subject": subject, "hits": hits,
        "total": found.get("total", len(hits)),
        "total_is_lower_bound": found.get("total_is_lower_bound", False),
        # ``{field, value, suggestions}`` -- a subject the vocabulary does not
        # know was NOT applied; the suggestions say what was probably meant.
        "unresolved": list(found.get("unresolved") or []),
        "warnings": list(found.get("warnings") or []),
    }
