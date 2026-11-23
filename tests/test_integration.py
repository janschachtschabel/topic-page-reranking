"""One pass through the real edu-sharing client: HTTP mock -> library flows ->
load_topic_page -> rerank -> HTML. Response shapes follow the client's own
integration tests (tests/test_integration_sync.py there)."""

import asyncio
import json

import httpx

from edusharing import AsyncRepository
from themenseiten_check.collect import load_topic_page
from themenseiten_check.report import render
from themenseiten_check.scoring import rerank

REPO = "https://repo.test/edu-sharing"
FACH = "http://w3id.org/openeduhub/vocabs/discipline/460"


def _ref(i):
    return f"workspace://SpacesStore/{i}"


def _io(nid, title, desc, original=None):
    node = {"ref": {"id": nid}, "title": title, "type": "ccm:io", "mimetype": "text/html",
            "content": {"hash": None},
            "properties": {"cclom:title": [title], "cclom:general_description": [desc],
                           "cclom:general_keyword": ["Optik"], "ccm:taxonid": [FACH],
                           "ccm:taxonid_DISPLAYNAME": ["Physik"]}}
    if original:
        node["originalId"] = original
    return node


def _map(nid, **props):
    props = {"cclom:title": [nid], "cm:name": [nid], **props}
    return {"ref": {"id": nid}, "type": "ccm:map", "title": props["cclom:title"][0],
            "name": nid, "properties": props,
            "access": ["Read"]}


NODES = {
    "col": _map("col", **{"cclom:title": ["Optik"],
                          "cm:description": ["Licht, Brechung und Linsen"],
                          "cclom:general_description": ["Licht, Brechung und Linsen"],
                          "ccm:oeh_collection_compendium_text": ["Die Optik untersucht das Licht."],
                          "ccm:page_config_ref": [_ref("folder")]}),
    "folder": _map("folder", **{"ccm:page_config": [json.dumps(
        {"variants": [_ref("var")], "default": _ref("var")})]}),
    "var": _map("var", **{"ccm:page_variant_is_template": ["false"],
                          "ccm:page_variant_config": [json.dumps({"structure": {"swimlanes": [
                              {"heading": "Empfohlen", "type": "container",
                               "grid": [{"item": "wlo-collection-chips", "nodeId": "w1"},
                                        {"item": "wlo-search", "nodeId": "w2"}]}]}})]}),
    "w1": _map("w1", **{"ccm:widget_config": [json.dumps({"sortedNodeIds": [_ref("n9")]})]}),
    "w2": _map("w2", **{"ccm:widget_config": [json.dumps({"searchText": "Optik",
                                                            "propertyFilters": {}})]}),
    "n9": _io("n9", "Linsen im Alltag", "Sammellinse und Zerstreuungslinse"),
}
MATERIALS = [_io("r1", "Bruchrechnen üben", "Brüche kürzen", original="o1"),
             _io("r2", "Brechung von Licht", "Licht an Grenzflächen", original="o2")]


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/col/children/collections"):
        return httpx.Response(200, json={"collections": [], "pagination": {"total": 0}})
    if path.endswith("/col/children") and request.url.params.get("filter") == "files":
        return httpx.Response(200, json={"nodes": MATERIALS, "pagination": {
            "total": len(MATERIALS), "from": 0, "count": len(MATERIALS)}})
    if path.endswith("/folder/children"):
        return httpx.Response(200, json={"nodes": [NODES["var"]],
                                         "pagination": {"total": 1, "from": 0, "count": 1}})
    if path.endswith("/metadata"):
        nid = path.rsplit("/", 2)[-2]
        if nid in NODES:
            return httpx.Response(200, json={"node": NODES[nid]})
        return httpx.Response(404, json={"error": "DAOMissingException", "message": nid})
    if path.endswith("/parents") or "/usage" in path:
        return httpx.Response(200, json={"nodes": []})
    raise AssertionError(f"not mocked: {request.method} {request.url}")


class Keyword:
    name, default_threshold = "keyword", 0.5

    async def score(self, query, documents):
        return [0.9 if ("licht" in d.lower() or "linse" in d.lower()) else 0.05
                for d in documents]


def test_end_to_end_through_the_client():
    async def go():
        async with AsyncRepository(
                REPO, metadataset="mds_oeh",
                client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as repo:
            return await load_topic_page(repo, "col")

    topic = asyncio.run(go())
    assert topic.title == "Optik"
    assert "Brechung und Linsen" in topic.description
    assert topic.compendium == "Die Optik untersucht das Licht."
    assert [(lane.heading, lane.item_ids) for lane in topic.lanes] == [
        ("Empfohlen", ("n9",)), ("Inhalte der Sammlung", ("o1", "o2"))]
    assert "gespeicherte Suche „Optik“, nicht ausgeführt" in topic.lanes[0].widgets[1]
    assert topic.items[1].subjects == ("Physik",)
    assert "Optik" in topic.items[1].keywords

    scored = asyncio.run(rerank(topic, Keyword()))
    assert [s.item.title for s in scored if not s.keep] == ["Bruchrechnen üben"]
    html = render(topic, scored, backend="keyword", threshold=0.5, repository=REPO)
    assert "1 von 3</strong>" in html


SEARCH_HITS = [_io("s1", "Licht und Farben", "Spektrum"), _io("s2", "Römisches Reich", "Antike")]


def search_handler(request: httpx.Request) -> httpx.Response:
    if "/search/v1" in request.url.path:
        sent = json.loads(request.content)
        search_handler.sent = sent
        return httpx.Response(200, json={"nodes": SEARCH_HITS, "pagination": {
            "total": 2, "from": 0, "count": 2}, "facets": []})
    return handler(request)


def test_saved_search_through_the_client(monkeypatch):
    monkeypatch.setitem(NODES, "w2", _map("w2", **{"ccm:widget_config": [json.dumps({
        "searchText": "Licht", "propertyFilters": {"ccm:taxonid": [FACH], "virtual:foo": ["x"]}})]}))

    async def go():
        async with AsyncRepository(
                REPO, metadataset="mds_oeh",
                client=httpx.AsyncClient(transport=httpx.MockTransport(search_handler))) as repo:
            return await load_topic_page(repo, "col", run_searches=True)

    topic = asyncio.run(go())
    assert topic.lanes[0].item_ids == ("n9", "s1", "s2")
    assert "virtual:foo" in topic.lanes[0].widgets[1]
    sent = json.dumps(search_handler.sent)
    assert FACH in sent and "virtual:foo" not in sent


def test_find_topic_pages_through_the_client():
    from themenseiten_check.search import find_topic_pages

    with_page = _map("p1", **{"cclom:title": ["Optik"], "ccm:page_config_ref": [_ref("f")],
                              "ccm:oeh_collection_compendium_text": ["Text"]})
    plain = _map("p2", **{"cclom:title": ["Optik (Chemie)"]})
    blind = {"ref": {"id": "p3"}, "type": "ccm:map", "title": "Optik alt", "properties": {}}

    def search(request):
        if request.url.path.endswith("/collections") and request.method == "POST":
            return httpx.Response(200, json={"nodes": [with_page, plain],
                                             "pagination": {"total": 2, "from": 0, "count": 2}})
        if "/collection/v1/collections/-home-/search" in str(request.url):
            return httpx.Response(200, json={"collections": [blind]})
        raise AssertionError(f"not mocked: {request.method} {request.url}")

    async def go():
        async with AsyncRepository(
                REPO, metadataset="mds_oeh",
                client=httpx.AsyncClient(transport=httpx.MockTransport(search))) as repo:
            return await find_topic_pages(repo, "Optik")

    found = asyncio.run(go())
    marks = {h["id"]: (h["has_layout"], h["has_compendium"]) for h in found["hits"]}
    assert marks == {"p1": (True, True), "p2": (False, False), "p3": (None, None)}
