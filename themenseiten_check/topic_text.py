"""The two texts of a topic page: where they live, and how to read them plainly.

**The description is read the way the WLO MCP server reads it**
(``topic-page-structure.ts``, ``getTopicPageContent``): the collection's
``cclom:general_description`` first, then ``collection.description`` -- the
collection object the node's metadata answer carries -- which is what the page
renders as its intro above the swimlanes. ``describe`` alone only sees the
properties; on staging the Optik collection had none there (seen 2026-11-23).
``cm:description`` comes last, as the property the client's write path uses.

**The compendium text is Markdown** -- an ``::: inhaltsverzeichnis`` block,
headings, anchor links. ``plain_text`` keeps the words and drops the markup and
the table of contents, so neither the display nor the n-grams see ``#teil-1``.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any

from edusharing import AsyncRepository, EduSharingError


def _first(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return value.strip() if isinstance(value, str) else ""


async def description_of(repo: AsyncRepository, collection_id: str,
                         properties: dict[str, Any], notes: list[str]) -> tuple[str, str]:
    """The topic page's description and where it was found ('' and '' if nowhere).

    ``properties`` are the collection's raw properties, already fetched by
    ``describe``; the node is read again only when they hold no description.
    """
    text = plain_text(_first(properties.get("cclom:general_description")))
    if text:
        return text, "cclom:general_description"
    try:
        node = await repo.node(collection_id)
    except EduSharingError as exc:
        notes.append(f"Beschreibung der Sammlung nicht lesbar ({type(exc).__name__}).")
        node = None
    collection = (node.raw.get("collection") if node is not None else None) or {}
    text = plain_text(_first(collection.get("description")) if isinstance(collection, dict) else "")
    if text:
        return text, "collection.description"
    text = plain_text(_first(properties.get("cm:description")))
    return (text, "cm:description") if text else ("", "")


_TOC_LINE = re.compile(r"^\s*[-*+]\s*\[[^\]]*\]\(#[^)]*\)\s*$")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def plain_text(text: str) -> str:
    """Markdown or HTML to plain prose: words kept, markup and table of contents dropped."""
    lines = []
    for line in unescape(re.sub(r"<[^>]+>", " ", text or "")).splitlines():
        if line.strip().startswith(":::") or _TOC_LINE.match(line):
            continue
        line = _LINK.sub(r"\1", line)
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)          # headings
        line = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", line)    # list markers
        line = re.sub(r"[*_`]{1,3}", "", line).replace("\\", "")
        if line.strip().lower() == "inhaltsverzeichnis":
            continue
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
