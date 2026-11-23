"""Resolving WordPress portal pages to their collection."""

import asyncio

import httpx
import pytest

from themenseiten_check.portal import (DEFAULT_HOSTS, NeedsChoice, fetch_html, page_title,
                                       resolve, uuid_candidates)

PORTAL = "https://wordpress.staging.openeduhub.net/portal/physik-optik/"
OPTIK = "aaaaaaaa-1111-2222-3333-444444444444"
PHYSIK = "bbbbbbbb-1111-2222-3333-444444444444"
GEO = "cccccccc-1111-2222-3333-444444444444"
IMG = "dddddddd-1111-2222-3333-444444444444"

HTML = f"""<html><head><title>Optik – WirLernenOnline</title></head><body>
<nav><a href="/portal/physik/">Physik</a></nav>
<div class="wlo-header"><p>Themenseite im Fachportal Physik</p><h1>Optik</h1></div>
<script>var portal = {{"collectionId": "{OPTIK}", "parent": "{PHYSIK}"}};</script>
<a href="/portal/geometrische-optik/" data-id="{GEO}">Geometrische Optik (123)</a>
<img src="https://repo/preview?nodeId={IMG}"><span>{OPTIK.upper()}</span></body></html>"""


def collection(nid, title):
    return {"id": nid, "title": title, "type": "ccm:map", "aspects": ["ccm:collection"],
            "properties": {}, "url": f"https://repo/{nid}"}


NODES = {OPTIK: collection(OPTIK, "Optik"), PHYSIK: collection(PHYSIK, "Physik"),
         GEO: collection(GEO, "Geometrische Optik"),
         IMG: {"id": IMG, "title": "Bild", "type": "ccm:io", "aspects": [], "properties": {}}}


class Flows:
    def __init__(self, nodes, hits=()):
        self.nodes, self.hits, self.asked = nodes, list(hits), None

    async def describe_many(self, ids):
        self.asked = list(ids)
        return {"nodes": [self.nodes[i] for i in ids if i in self.nodes], "failed": []}

    async def find_collections(self, text, limit):
        return {"hits": self.hits}


class Repo:
    def __init__(self, flows):
        self.flows = flows


def client(pages):
    def handler(request):
        url = str(request.url)
        if url not in pages:
            return httpx.Response(404)
        status, body = pages[url]
        if status in (301, 302):
            return httpx.Response(status, headers={"location": body})
        return httpx.Response(status, text=body, headers={"content-type": "text/html; charset=utf-8"})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


def run(coro):
    return asyncio.run(coro)


def test_page_title_and_candidates():
    assert page_title(HTML) == "Optik"
    assert page_title("<title>Farben | WLO</title>") == "Farben"
    assert uuid_candidates(HTML) == [OPTIK, PHYSIK, GEO, IMG]  # deduplicated, case folded


def test_resolves_portal_page_by_title_among_collections():
    flows = Flows(NODES)
    got = run(resolve(Repo(flows), PORTAL, client=client({PORTAL: (200, HTML)})))
    assert got == OPTIK
    assert flows.asked == [OPTIK, PHYSIK, GEO, IMG]


def test_follows_redirect_on_allowed_host():
    target = PORTAL.rstrip("/")
    pages = {target: (301, PORTAL), PORTAL: (200, HTML)}
    assert run(resolve(Repo(Flows(NODES)), target, client=client(pages))) == OPTIK


def test_bare_id_and_edusharing_url_skip_the_fetch():
    assert run(resolve(Repo(Flows({})), f"https://x/?id={OPTIK}", client=client({}))) == OPTIK


def test_falls_back_to_collection_search_when_page_names_no_matching_collection():
    html = f"<h1>Optik</h1><a data-id='{PHYSIK}'></a><a data-id='{GEO}'></a>"
    flows = Flows(NODES, hits=[{"id": OPTIK, "title": "Optik"}, {"id": GEO, "title": "Geometrische Optik"}])
    assert run(resolve(Repo(flows), PORTAL, client=client({PORTAL: (200, html)}))) == OPTIK


def test_ambiguity_is_handed_back_not_guessed():
    html = f"<h1>Optik</h1><a data-id='{PHYSIK}'></a><a data-id='{GEO}'></a>"
    flows = Flows(NODES, hits=[{"id": "1", "title": "Optik"}, {"id": "2", "title": "Optik"}])
    with pytest.raises(NeedsChoice) as caught:
        run(resolve(Repo(flows), PORTAL, client=client({PORTAL: (200, html)})))
    assert [c["id"] for c in caught.value.choices] == ["1", "2"]


def test_no_collection_at_all():
    with pytest.raises(ValueError, match="keine Sammlung"):
        run(resolve(Repo(Flows({}, hits=[])), PORTAL, client=client({PORTAL: (200, "<h1>Leer</h1>")})))


@pytest.mark.parametrize("url", [
    "https://evil.example/portal/x/",
    "http://wordpress.staging.openeduhub.net/portal/physik-optik/",
])
def test_refuses_hosts_off_the_list_and_plain_http(url):
    with pytest.raises(ValueError, match="nicht abgerufen"):
        run(fetch_html(url, DEFAULT_HOSTS, client({})))


def test_redirect_off_the_list_is_refused():
    pages = {PORTAL: (302, "http://169.254.169.254/latest/meta-data/")}
    with pytest.raises(ValueError, match="nicht abgerufen"):
        run(fetch_html(PORTAL, DEFAULT_HOSTS, client(pages)))


def test_http_error_and_size_cap():
    with pytest.raises(ValueError, match="HTTP 404"):
        run(fetch_html(PORTAL, DEFAULT_HOSTS, client({})))
    with pytest.raises(ValueError, match="zu groß"):
        run(fetch_html(PORTAL, DEFAULT_HOSTS, client({PORTAL: (200, "x" * 3_100_000)})))
