"""Read a WLO topic page (Themenseite) the way its visitor sees it: in lanes.

A topic page is a collection. Its content comes from three places, each read
the way the edu-sharing client documents it:

* the page builder's swimlanes -- widgets with a fixed node list
  (``sortedNodeIds``) and, optionally and approximated, widgets with a saved
  search (see ``saved_search``);
* the collection's own materials (``flows.collection_contents``) -- these are
  *references*; ``original_id`` names the record behind one;
* optionally the materials of its direct sub-collections.

Each place becomes a ``Lane``. A material may sit in several lanes; it is
listed in each, but ``TopicPage.items`` holds it once, so it is scored once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from edusharing import AsyncRepository, EduSharingError

from .saved_search import run_saved_search
from .topic_text import description_of, headings, plain_text

KEYWORD_PROP = "cclom:general_keyword"
#: Page size for collection listings and describe_many (its own cap is 50).
_PAGE = 50


@dataclass(frozen=True)
class Item:
    """One material shown on the topic page."""

    id: str
    title: str
    description: str
    keywords: tuple[str, ...]
    subjects: tuple[str, ...]
    url: str
    #: The lane it was first seen in, in words a reader understands.
    source: str


@dataclass(frozen=True)
class Lane:
    """One section of the page, with its materials in page order."""

    id: str
    heading: str
    #: ``seite`` (page builder), ``sammlung`` or ``untersammlung``.
    kind: str
    item_ids: tuple[str, ...]
    #: One line per widget: what it is and how much of it could be evaluated.
    widgets: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopicPage:
    id: str
    title: str
    description: str
    keywords: tuple[str, ...]
    url: str
    #: Every material once, in first-seen order. What gets scored.
    items: tuple[Item, ...]
    lanes: tuple[Lane, ...] = ()
    #: Everything that makes the picture incomplete: caps, refusals, approximations.
    notes: tuple[str, ...] = field(default=())
    #: The collection's compendium text (``ccm:oeh_collection_compendium_text`` on
    #: WLO) -- usually far more telling about the topic than the description.
    #: Read from the metadata profile, so another metadata set can name its own.
    compendium: str = ""
    #: The h1-h3 headings of the compendium text, scaffolding removed -- the
    #: short form of the topic for a model that cannot read the whole text.
    compendium_headings: tuple[str, ...] = ()
    #: Where the description was found -- shown, so it can be checked against
    #: other tools reading the same collection. Empty when none was found.
    description_source: str = ""
    #: Subject labels of the collection (``fields.subject`` of ``describe``).
    subjects: tuple[str, ...] = ()


def _item(raw: dict[str, Any], source: str) -> Item:
    """Build an Item from a hit dict or a describe dict -- both share the keys used here."""
    fields = raw.get("fields") or {}
    keywords = raw.get("keywords") or fields.get(KEYWORD_PROP) or []
    return Item(
        id=raw.get("original_id") or raw["id"],
        title=(raw.get("title") or "").strip() or "(ohne Titel)",
        description=(raw.get("description") or "").strip(),
        keywords=tuple(keywords),
        subjects=tuple(fields.get("subject") or ()),
        url=raw.get("url") or "",
        source=source,
    )


def _compendium(repo: AsyncRepository, meta: dict[str, Any]) -> str:
    """The raw compendium text (Markdown) from an already fetched description."""
    prop = repo.metadata_profile.compendium_property
    value = (meta.get("properties") or {}).get(prop) if prop else None
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, str) else ""


class _Builder:
    """Collects lanes and items; keeps each material once in ``items``."""

    def __init__(self) -> None:
        self.items: dict[str, Item] = {}
        self.lanes: list[Lane] = []
        self.notes: list[str] = []

    def add_lane(self, heading: str, kind: str, raws: list[dict],
                 widgets: tuple[str, ...] = ()) -> None:
        ids: list[str] = []
        for raw in raws:
            item = _item(raw, heading)
            self.items.setdefault(item.id, item)
            if item.id not in ids:
                ids.append(item.id)
        self.lanes.append(Lane(f"lane-{len(self.lanes) + 1}", heading, kind,
                               tuple(ids), widgets))


async def _materials(repo: AsyncRepository, collection_id: str, budget: int,
                     notes: list[str], label: str) -> tuple[list[dict], list[dict]]:
    """All materials of one collection up to ``budget``, plus its sub-collections."""
    materials: list[dict] = []
    subcollections: list[dict] = []
    offset = 0
    while len(materials) < budget:
        page = await repo.flows.collection_contents(
            collection_id, limit=min(_PAGE, budget - len(materials)),
            offset=offset, properties=(KEYWORD_PROP,))
        if offset == 0:
            subcollections = page["collections"]
            if page["collections_truncated"]:
                notes.append(f"{label}: Liste der Untersammlungen gekürzt.")
        materials.extend(page["materials"])
        offset += page["returned_materials"]
        if page["returned_materials"] == 0 or offset >= page["total_materials"]:
            return materials, subcollections
    notes.append(f"{label}: nach {budget} Inhalten abgebrochen (Obergrenze).")
    return materials, subcollections


async def _describe(repo: AsyncRepository, ids: list[str],
                    notes: list[str]) -> dict[str, dict]:
    """describe_many in chunks; missing nodes become one note."""
    found: dict[str, dict] = {}
    missing = 0
    for start in range(0, len(ids), _PAGE):
        many = await repo.flows.describe_many(ids[start:start + _PAGE])
        found.update({n["id"]: n for n in many["nodes"]})
        missing += len(many["failed"])
    if missing:
        notes.append(f"{missing} Widget-Inhalte nicht mehr abrufbar.")
    return found


async def _page_lanes(repo: AsyncRepository, collection_id: str, builder: _Builder,
                      run_searches: bool) -> None:
    """The page builder's swimlanes, in page order."""
    try:
        page = await repo.flows.page(collection_id, resolve_widgets=True)
    except EduSharingError as exc:
        builder.notes.append(f"Seitenlayout nicht lesbar ({type(exc).__name__}).")
        return
    if page["rendered"] is None:
        builder.notes.append("Keine Seitenlayout-Variante: nur Sammlungsinhalte ausgewertet.")
        return
    if page["truncated"]:
        builder.notes.append(
            f"Seitenlayout nur teilweise gelesen ({', '.join(page['truncated_by'])}).")

    fixed = [i for lane in page["swimlanes"] for w in lane["items"] for i in w.get("node_ids", [])]
    described = await _describe(repo, list(dict.fromkeys(fixed)), builder.notes)
    searches = [w["search"] for lane in page["swimlanes"] for w in lane["items"]
                if "search" in w] if run_searches else []
    results = iter(await asyncio.gather(
        *(run_saved_search(repo, s, properties=(KEYWORD_PROP,)) for s in searches)))

    for lane in page["swimlanes"]:
        raws: list[dict] = []
        widgets: list[str] = []
        for w in lane["items"]:
            name = w["widget"]
            if "node_ids" in w:
                listed = [described[i] for i in w["node_ids"] if i in described]
                raws.extend(listed)
                widgets.append(f"{name}: feste Liste, {len(listed)} Inhalte")
            elif "search" in w:
                text = w["search"]["text"] or "ohne Suchbegriff"
                if run_searches:
                    hits, why = next(results)
                    raws.extend(hits)
                    extra = f" ({'; '.join(why)})" if why else ""
                    widgets.append(f"{name}: gespeicherte Suche „{text}“, angenähert, "
                                   f"{len(hits)} Treffer{extra}")
                else:
                    widgets.append(f"{name}: gespeicherte Suche „{text}“, nicht ausgeführt")
            elif "unreachable" in w or "unreadable" in w:
                widgets.append(f"{name}: nicht lesbar")
            else:
                widgets.append(f"{name}: kein auswertbarer Inhalt")
        builder.add_lane(lane["heading"] or "(ohne Überschrift)", "seite", raws, tuple(widgets))


async def load_topic_page(repo: AsyncRepository, collection_id: str, *,
                          max_items: int = 200,
                          include_subcollections: bool = False,
                          include_widgets: bool = True,
                          run_searches: bool = False) -> TopicPage:
    """Read the topic page behind ``collection_id``.

    Lanes come in page order, then the collection's own materials, then each
    sub-collection. ``max_items`` caps the collection listings, not the widgets.

    Raises:
        NotFoundError: when no collection carries this id.
    """
    builder = _Builder()
    meta = await repo.flows.describe(collection_id)

    if include_widgets:
        await _page_lanes(repo, collection_id, builder, run_searches)

    materials, subs = await _materials(repo, collection_id, max_items, builder.notes, "Sammlung")
    builder.add_lane("Inhalte der Sammlung", "sammlung", materials)

    if include_subcollections:
        left = max_items - len(materials)
        for sub in subs:
            if left <= 0:
                builder.notes.append("Untersammlungen wegen der Obergrenze nicht vollständig gelesen.")
                break
            found, _ = await _materials(repo, sub["id"], left, builder.notes, sub["title"])
            left -= len(found)
            builder.add_lane(f"Untersammlung „{sub['title']}“", "untersammlung", found)

    description, source = await description_of(
        repo, collection_id, meta.get("properties") or {}, builder.notes)
    if not description:
        # Said, because the portal may still show a text: then it lives in
        # WordPress, not in the repository, and this is where to look next.
        builder.notes.append("Keine Beschreibung im Repositorium gefunden (geprüft: "
                             "cclom:general_description, collection.description, cm:description).")
    compendium = _compendium(repo, meta)
    return TopicPage(
        id=meta["id"], title=meta["title"],
        description=description, description_source=source,
        keywords=tuple(meta.get("keywords") or ()), url=meta.get("url") or "",
        compendium=plain_text(compendium),
        compendium_headings=tuple(headings(compendium)),
        subjects=tuple((meta.get("fields") or {}).get("subject") or ()),
        items=tuple(builder.items.values()), lanes=tuple(builder.lanes),
        notes=tuple(builder.notes))
