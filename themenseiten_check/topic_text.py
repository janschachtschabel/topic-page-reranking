"""The two texts of a topic page: where they live, and how to read them plainly.

**The description is not where ``describe`` looks.** For materials it sits in
``cclom:general_description``/``cm:description``; a collection's description is
written into the ``collection`` object of the collection API (the client's
``collections.update`` documents it: as ``properties["cm:description"]`` it is
silently dropped). On staging the Optik collection answered ``describe`` with an
empty description while its compendium text was full (seen 2026-11-23). So the
collection endpoint is asked first; the properties stay as a fallback.

**The compendium text is Markdown** -- an ``::: inhaltsverzeichnis`` block,
headings, anchor links. ``plain_text`` keeps the words and drops the markup and
the table of contents, so neither the display nor the n-grams see ``#teil-1``.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any

from edusharing import AsyncRepository, EduSharingError
from edusharing.urls import path_segment


async def collection_description(repo: AsyncRepository, collection_id: str,
                                 notes: list[str]) -> str:
    """``collection.description`` from the collection endpoint; empty if absent or refused."""
    try:
        answer = await repo.raw.json(
            "GET", f"/collection/v1/collections/-home-/{path_segment(collection_id)}")
    except EduSharingError as exc:
        notes.append(f"Beschreibung der Sammlung nicht lesbar ({type(exc).__name__}).")
        return ""
    node: Any = answer.get("collection") if isinstance(answer, dict) else None
    inner = node.get("collection") if isinstance(node, dict) else None
    value = inner.get("description") if isinstance(inner, dict) else None
    return plain_text(value) if isinstance(value, str) else ""


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
