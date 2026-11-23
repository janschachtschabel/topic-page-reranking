"""Resolve a WLO portal page (WordPress) to the collection behind it.

The topic pages visitors see live in WordPress, e.g.
``https://wordpress.staging.openeduhub.net/portal/physik-optik/`` -- a slug,
no node id. The page's HTML does reference the collection somewhere, but the
markup is the theme's and not a contract. So nothing here depends on where in
the HTML the id sits:

1. collect every UUID in the HTML,
2. ask the repository which of them are collections,
3. keep the one whose title equals the page's ``<h1>``.

When that is not unique, a collection search by the title is tried; when it is
still not unique, the caller gets the candidates to choose from rather than a
guess.

Only hosts on an allow-list are fetched, over https, redirects re-checked at
every hop -- the URL comes from a visitor and must not turn the server into a
proxy for anything else.
"""

from __future__ import annotations

import re
from html import unescape
from urllib.parse import urljoin, urlparse

import httpx

from edusharing import AsyncRepository
from edusharing.agent import check_url

from .urls import UUID, collection_id_from

DEFAULT_HOSTS = ("wordpress.staging.openeduhub.net", "wirlernenonline.de",
                 "www.wirlernenonline.de")
_MAX_BYTES = 3_000_000
_MAX_REDIRECTS = 3
_MAX_CANDIDATES = 50


class NeedsChoice(Exception):
    """More than one collection fits; the visitor has to pick."""

    def __init__(self, title: str, choices: list[dict[str, str]]) -> None:
        super().__init__(f"Mehrere Sammlungen passen zu „{title}“.")
        self.choices = choices


def _allowed(url: str, hosts: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() in hosts


async def fetch_html(url: str, hosts: tuple[str, ...],
                     client: httpx.AsyncClient | None = None) -> str:
    """The page's HTML, capped in size, never leaving the allow-listed hosts.

    Raises:
        ValueError: for a host off the list, too many redirects, a page too
            large or an HTTP error -- all things the visitor can act on.
    """
    own = client is None
    client = client or httpx.AsyncClient(timeout=10.0, follow_redirects=False)
    try:
        for _ in range(_MAX_REDIRECTS + 1):
            if not _allowed(url, hosts):
                raise ValueError(f"Seiten von {urlparse(url).hostname or url!r} werden nicht "
                                 f"abgerufen. Erlaubt: {', '.join(hosts)}.")
            check_url(url)
            async with client.stream("GET", url, headers={"Accept": "text/html"}) as res:
                if res.is_redirect:
                    url = urljoin(url, res.headers.get("location", ""))
                    continue
                if res.status_code >= 400:
                    raise ValueError(f"Die Portalseite antwortet mit HTTP {res.status_code}.")
                body = bytearray()
                async for chunk in res.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_BYTES:
                        raise ValueError("Die Portalseite ist zu groß.")
                return body.decode(res.encoding or "utf-8", errors="replace")
        raise ValueError("Zu viele Weiterleitungen.")
    finally:
        if own:
            await client.aclose()


def page_title(html: str) -> str:
    """The first ``<h1>``, else the ``<title>`` up to its site suffix."""
    for pattern in (r"<h1[^>]*>(.*?)</h1>", r"<title[^>]*>(.*?)</title>"):
        if found := re.search(pattern, html, re.I | re.S):
            text = unescape(re.sub(r"<[^>]+>", " ", found.group(1)))
            text = re.split(r"\s[|–-]\s", " ".join(text.split()))[0]
            if text:
                return text
    return ""


def uuid_candidates(html: str) -> list[str]:
    """Every UUID in the page, first occurrence first."""
    return list(dict.fromkeys(m.lower() for m in UUID.findall(html)))[:_MAX_CANDIDATES]


def _is_collection(node: dict) -> bool:
    """A collection is a ``ccm:map`` that carries collection metadata.

    Checked loosely on purpose -- aspect and property names are not something
    this module should pin down more tightly than it can verify.
    """
    if node.get("type") != "ccm:map":
        return False
    marks = list(node.get("aspects") or []) + list((node.get("properties") or {}).keys())
    return any("collection" in str(m).lower() for m in marks)


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


async def resolve(repo: AsyncRepository, text: str, *, hosts: tuple[str, ...] = DEFAULT_HOSTS,
                  client: httpx.AsyncClient | None = None) -> str:
    """Collection id for a bare id, an edu-sharing URL or a portal page URL.

    Raises:
        ValueError: when nothing points to a collection.
        NeedsChoice: when several collections fit equally well.
    """
    try:
        return collection_id_from(text)
    except ValueError:
        if not urlparse(text.strip()).scheme:
            raise

    html = await fetch_html(text.strip(), hosts, client)
    title = page_title(html)
    found = await repo.flows.describe_many(uuid_candidates(html))
    collections = [n for n in found["nodes"] if _is_collection(n)]
    exact = [n for n in collections if title and _same(n["title"], title)]
    if len(exact) == 1:
        return exact[0]["id"]
    if not exact and len(collections) == 1:
        return collections[0]["id"]

    if not exact and title:
        hits = (await repo.flows.find_collections(title, limit=10))["hits"]
        exact = [h for h in hits if _same(h["title"], title)]
        if len(exact) == 1:
            return exact[0]["id"]

    choices = exact or collections
    if not choices:
        raise ValueError(f"Auf der Portalseite „{title or text}“ ist keine Sammlung zu finden.")
    raise NeedsChoice(title, [{"id": c["id"], "title": c["title"], "url": c.get("url") or ""}
                              for c in choices[:10]])
