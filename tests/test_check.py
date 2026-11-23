"""Offline tests. The fake repository returns the dict shapes the
edu-sharing client documents for its flows (collection_contents, describe,
describe_many, page with resolve_widgets=True)."""

import asyncio
import re

import pytest

from themenseiten_check.collect import load_topic_page
from themenseiten_check.report import render
from themenseiten_check.scoring import item_text, rerank, topic_query


def hit(i, title, desc="", kw=(), subject=("Physik",), ref=True):
    return {"id": f"ref-{i}" if ref else f"n{i}", "original_id": f"n{i}" if ref else None,
            "title": title, "description": desc, "url": f"https://example.org/{i}",
            "fields": {"subject": list(subject), "cclom:general_keyword": list(kw)}}


class FakeFlows:
    def __init__(self, contents, subs=None, page=None, nodes=None):
        self.contents, self.subs = contents, subs or {}
        self._page, self.nodes = page, nodes or {}
        self.calls = []

    async def describe(self, cid):
        return {"id": cid, "title": "Optik", "url": "https://wlo/optik",
                "description": "Licht, Reflexion, Brechung und Linsen.",
                "keywords": ["Licht", "Linse"],
                "properties": {"ccm:oeh_collection_compendium_text": ["Optik ist die Lehre vom Licht."],
                               "cclom:general_description": ["Licht, Reflexion, Brechung und Linsen."]}}

    async def collection_contents(self, cid, *, limit, offset, properties):
        self.calls.append((cid, limit, offset))
        mats = self.contents.get(cid, [])
        chunk = mats[offset:offset + limit]
        return {"id": cid, "materials": chunk, "collections": self.subs.get(cid, []),
                "total_materials": len(mats), "returned_materials": len(chunk),
                "collections_truncated": False}

    async def page(self, cid, *, resolve_widgets):
        return self._page or {"rendered": None, "swimlanes": [], "truncated": False,
                              "truncated_by": []}

    async def search(self, text, *, raw_filters, limit, properties):
        self.searched = (text, raw_filters)
        return {"hits": [hit(70, "Licht im Alltag")], "total": 40,
                "unresolved": [], "ignored": []}

    async def describe_many(self, ids):
        found = [self.nodes[i] for i in ids if i in self.nodes]
        failed = [{"id": i, "reason": "NotFoundError"} for i in ids if i not in self.nodes]
        return {"nodes": found, "failed": failed}


class FakeProfile:
    compendium_property = "ccm:oeh_collection_compendium_text"


class FakeNode:
    def __init__(self, raw):
        self.raw = raw


class FakeNodes:
    """``repo.node``: the collection object with a description, or a refusal."""

    def __init__(self, description=None, error=None):
        self.description, self.error = description, error

    async def __call__(self, node_id):
        if self.error:
            raise self.error
        coll = {} if self.description is None else {"description": self.description}
        return FakeNode({"ref": {"id": node_id}, "collection": coll})


class FakeRepo:
    metadata_profile = FakeProfile()

    def __init__(self, flows, node=None):
        self.flows = flows
        self.node = node or FakeNodes()


class FixedReranker:
    """Scores by whether the document mentions a word of the topic."""
    name, default_threshold = "fake", 0.5

    async def score(self, query, documents):
        words = {"licht", "linse", "brechung"}
        return [0.9 if words & set(re.findall(r"\w+", d.lower())) else 0.1 for d in documents]


def run(coro):
    return asyncio.run(coro)


def test_pages_through_materials_and_dedupes():
    mats = [hit(i, f"Licht {i}") for i in range(120)] + [hit(5, "Licht 5 doppelt")]
    flows = FakeFlows({"c": mats})
    topic = run(load_topic_page(FakeRepo(flows), "c", max_items=500))
    assert [c[2] for c in flows.calls] == [0, 50, 100]
    assert len(topic.items) == 120
    assert topic.items[5].title == "Licht 5"          # first occurrence wins, order kept
    assert topic.items[0].id == "n0"                    # original id, not the reference


def test_max_items_cap_is_reported():
    flows = FakeFlows({"c": [hit(i, "x") for i in range(30)]})
    topic = run(load_topic_page(FakeRepo(flows), "c", max_items=10))
    assert len(topic.items) == 10
    assert any("Obergrenze" in n for n in topic.notes)


PAGE = {"rendered": {"id": "v"}, "truncated": False, "truncated_by": [],
        "swimlanes": [{"heading": "Tipps", "type": "container", "items": [
            {"widget": "list", "node_id": "w1", "node_ids": ["n50", "gone"]},
            {"widget": "search", "node_id": "w2",
             "search": {"text": "Optik", "filters": {"virtual:x": ["a"], "ccm:taxonid": ["u"]}}},
            {"widget": "wlo-editorial-members", "node_id": None},
        ]}]}
NODES = {"n50": {"id": "n50", "title": "Linsen erklärt", "description": "",
                 "keywords": ["Linse"], "fields": {}, "url": "", "original_id": None}}


def _flows():
    return FakeFlows({"c": [hit(1, "Licht"), hit(50, "Linsen erklärt", ref=False)],
                      "s": [hit(2, "Brechung")]},
                     subs={"c": [{"id": "s", "title": "Brechung"}]}, page=PAGE, nodes=NODES)


def test_lanes_follow_the_page_then_collection_then_subcollections():
    flows = _flows()
    topic = run(load_topic_page(FakeRepo(flows), "c", include_subcollections=True))
    assert [(l.heading, l.kind, l.item_ids) for l in topic.lanes] == [
        ("Tipps", "seite", ("n50",)),
        ("Inhalte der Sammlung", "sammlung", ("n1", "n50")),
        ("Untersammlung „Brechung“", "untersammlung", ("n2",))]
    # a material in two lanes is scored once
    assert [i.id for i in topic.items] == ["n50", "n1", "n2"]
    widgets = topic.lanes[0].widgets
    assert widgets[0] == "list: feste Liste, 1 Inhalte"
    assert "nicht ausgeführt" in widgets[1]
    assert widgets[2].endswith("kein auswertbarer Inhalt")
    assert any("nicht mehr abrufbar" in n for n in topic.notes)
    assert not hasattr(flows, "searched")


def test_saved_searches_run_approximated_when_asked():
    flows = _flows()
    topic = run(load_topic_page(FakeRepo(flows), "c", run_searches=True))
    assert flows.searched == ("Optik", {"ccm:taxonid": ["u"]})
    assert topic.lanes[0].item_ids == ("n50", "n70")
    line = topic.lanes[0].widgets[1]
    assert "angenähert, 1 Treffer" in line and "virtual:x" in line and "erste 1 von 40" in line


def test_texts_carry_description_and_keywords():
    flows = FakeFlows({"c": [hit(1, "Spiegel", "Reflexion am Spiegel", kw=["Spiegel"])]})
    topic = run(load_topic_page(FakeRepo(flows), "c", include_widgets=False))
    assert "Brechung und Linsen" in topic_query(topic)
    assert "Schlagworte: Licht, Linse" in topic_query(topic)
    assert topic.compendium == "Optik ist die Lehre vom Licht."
    assert topic.compendium not in topic_query(topic)            # default: description
    assert topic.compendium in topic_query(topic, "compendium")
    assert "Brechung und Linsen" not in topic_query(topic, "compendium")
    both = topic_query(topic, "both")
    assert topic.compendium in both and "Brechung und Linsen" in both
    text = item_text(topic.items[0])
    assert "Reflexion am Spiegel" in text and "Schlagworte: Spiegel" in text and "Fach: Physik" in text


def test_rerank_orders_filters_and_keeps_original_rank():
    mats = [hit(1, "Bruchrechnen"), hit(2, "Licht und Schatten"), hit(3, "Linse bauen")]
    topic = run(load_topic_page(FakeRepo(FakeFlows({"c": mats})), "c", include_widgets=False))
    scored = run(rerank(topic, FixedReranker()))
    assert [s.item.title for s in scored] == ["Licht und Schatten", "Linse bauen", "Bruchrechnen"]
    assert [s.keep for s in scored] == [True, True, False]
    assert scored[-1].original_rank == 1


def test_rerank_rejects_wrong_score_count():
    class Broken(FixedReranker):
        async def score(self, q, d):
            return [1.0]
    topic = run(load_topic_page(FakeRepo(FakeFlows({"c": [hit(1, "a"), hit(2, "b")]})), "c"))
    with pytest.raises(RuntimeError):
        run(rerank(topic, Broken()))


def test_report_escapes_and_shows_both_sides():
    mats = [hit(1, "<script>alert(1)</script> Licht"), hit(2, "Bruchrechnen")]
    mats[1]["url"] = "javascript:alert(1)"
    topic = run(load_topic_page(FakeRepo(FakeFlows({"c": mats})), "c"))
    html = render(topic, run(rerank(topic, FixedReranker())), backend="fake",
                  threshold=0.5, repository="https://staging")
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert "javascript:" not in html
    assert "1 von 2</strong>" in html
    assert "Nach Reranking (1)" in html and "Aussortiert (1)" in html



def test_topic_texts_fall_back_and_say_so():
    from themenseiten_check.collect import TopicPage
    from themenseiten_check.scoring import topic_texts
    full = TopicPage("t", "T", "kurz", (), "", (), compendium="lang")
    assert topic_texts(full) == (["kurz"], "description")
    assert topic_texts(full, "both") == (["kurz", "lang"], "both")
    only_comp = TopicPage("t", "T", "", (), "", (), compendium="lang")
    assert topic_texts(only_comp, "description") == (["lang"], "compendium")
    assert topic_texts(TopicPage("t", "T", "", (), "", ()), "both") == ([], "none")
    with pytest.raises(ValueError):
        topic_texts(full, "x")



class NoDescriptionProps(FakeFlows):
    async def describe(self, cid):
        return {"id": cid, "title": "Optik", "url": "", "keywords": [],
                "properties": {"cm:description": ["aus cm:description"]}}


def test_description_order_follows_the_mcp_server():
    # 1. cclom:general_description wins when present
    topic = run(load_topic_page(FakeRepo(FakeFlows({"c": []}), FakeNodes("aus dem DTO")), "c"))
    assert (topic.description, topic.description_source) == (
        "Licht, Reflexion, Brechung und Linsen.", "cclom:general_description")
    # 2. otherwise collection.description, cleaned of HTML
    topic = run(load_topic_page(FakeRepo(NoDescriptionProps({"c": []}),
                                         FakeNodes("<p>Alles über <b>Licht</b></p>")), "c"))
    assert topic.description_source == "collection.description"
    assert "Licht" in topic.description and "<" not in topic.description
    # 3. cm:description last
    topic = run(load_topic_page(FakeRepo(NoDescriptionProps({"c": []})), "c"))
    assert (topic.description, topic.description_source) == ("aus cm:description", "cm:description")


def test_refused_node_read_is_noted():
    from edusharing import EduSharingError
    repo = FakeRepo(NoDescriptionProps({"c": []}), FakeNodes(error=EduSharingError("403")))
    topic = run(load_topic_page(repo, "c"))
    assert topic.description == "aus cm:description"
    assert any("Beschreibung der Sammlung nicht lesbar" in n for n in topic.notes)


def test_missing_description_is_noted():
    class NoDescription(FakeFlows):
        async def describe(self, cid):
            return {"id": cid, "title": "Optik", "url": "", "keywords": [], "properties": {}}
    topic = run(load_topic_page(FakeRepo(NoDescription({"c": []})), "c"))
    assert topic.description == ""
    assert any("Keine Beschreibung im Repositorium" in n for n in topic.notes)


def test_plain_text_drops_markdown_and_table_of_contents():
    from themenseiten_check.topic_text import plain_text
    md = ("::: inhaltsverzeichnis\n## Inhaltsverzeichnis\n\n"
          "- [Kompendialer Text](#kompendialer-text)\n- [Teil 1](#teil-1)\n:::\n"
          "## 1\\. Themendefinition\nDie **Optik** ist die Lehre vom [Licht](https://x.org).\n")
    out = plain_text(md)
    assert "Inhaltsverzeichnis" not in out and "#" not in out and "](" not in out
    assert "Die Optik ist die Lehre vom Licht." in out and "1. Themendefinition" in out
