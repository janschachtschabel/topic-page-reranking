"""The web side: URL parsing, JSON shape, the Vercel endpoint."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from themenseiten_check.collect import Item, Lane, TopicPage
from themenseiten_check.scoring import rerank
from themenseiten_check.serialize import page_as_json
from themenseiten_check.urls import collection_id_from

ID = "4e3a9c1d-2b7f-4f0e-9a51-8c2d6e7f1a23"
PARENT = "11111111-2222-3333-4444-555555555555"


@pytest.mark.parametrize("text", [
    ID, f"  {ID.upper()} ",
    f"https://repository.staging.openeduhub.net/edu-sharing/components/collections?id={ID}&scope=x",
    f"https://staging.wirlernenonline.de/themenseite/{PARENT}/{ID}",
    f"https://example.org/x/{PARENT}?nodeId={ID}",
])
def test_collection_id_from(text):
    assert collection_id_from(text) == ID


def test_collection_id_from_rejects_text_without_uuid():
    with pytest.raises(ValueError):
        collection_id_from("https://wirlernenonline.de/optik")


def _topic():
    a = Item("a", "Licht", "", (), (), "https://x/a", "L")
    b = Item("b", "Brüche", "", (), (), "", "L")
    return TopicPage("c", "Optik", "Licht und Linsen", ("Licht",), "", (a, b),
                     lanes=(Lane("lane-1", "Tipps", "seite", ("a", "b"), ("w: feste Liste",)),),
                     notes=("Hinweis",))


class Half:
    name, default_threshold = "half", 0.5

    async def score(self, query, documents):
        return [0.9, 0.1][:len(documents)]


def test_page_as_json_carries_scores_not_verdicts():
    topic = _topic()
    out = page_as_json(topic, asyncio.run(rerank(topic, Half())), backend="half",
                       threshold=0.5, repository="r")
    assert out["lanes"][0]["item_ids"] == ["a", "b"]
    assert out["items"]["b"]["score"] == 0.1 and "keep" not in out["items"]["b"]
    assert out["topic"]["description"] == "Licht und Linsen"
    json.dumps(out)


def _load_app(monkeypatch, **env):
    for key in ("DEMO_PASSWORD",):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    path = Path(__file__).resolve().parent.parent / "api" / "index.py"
    spec = importlib.util.spec_from_file_location("analyze_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRepo:
    url = "https://staging"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_endpoint_end_to_end(monkeypatch):
    mod = _load_app(monkeypatch)
    seen = {}

    async def fake_load(repo, cid, **kw):
        seen.update(cid=cid, **kw)
        return _topic()

    monkeypatch.setattr(mod, "_repository", FakeRepo)
    monkeypatch.setattr(mod, "load_topic_page", fake_load)
    monkeypatch.setattr(mod, "TextStatsReranker", Half)

    res = TestClient(mod.app).post("/api/analyze", json={"url": f"https://x/?id={ID}"})
    assert res.status_code == 200
    assert seen["cid"] == ID and seen["run_searches"] is True
    assert res.json()["items"]["a"]["score"] == 0.9


def test_endpoint_password_and_bad_input(monkeypatch):
    mod = _load_app(monkeypatch, DEMO_PASSWORD="geheim")
    client = TestClient(mod.app)
    assert client.post("/api/analyze", json={"url": ID}).status_code == 401
    bad = client.post("/api/analyze", json={"url": "ohne id"}, headers={"X-Demo-Password": "geheim"})
    assert bad.status_code == 400 and "UUID" in bad.json()["error"]
    assert client.post("/api/analyze", json={"url": ID, "max_items": 0},
                       headers={"X-Demo-Password": "geheim"}).status_code == 422


def test_endpoint_scores_with_text_statistics(monkeypatch):
    mod = _load_app(monkeypatch)

    async def fake_load(repo, cid, **kw):
        return _topic()

    monkeypatch.setattr(mod, "_repository", FakeRepo)
    monkeypatch.setattr(mod, "load_topic_page", fake_load)
    res = TestClient(mod.app).post("/api/analyze", json={"url": ID})
    assert res.status_code == 200
    items = res.json()["items"]
    assert items["a"]["score"] > items["b"]["score"]
    assert res.json()["backend"].startswith("Textstatistik")
    assert res.json()["topic_text"] == {"chosen": "description", "used": "description"}
    res = TestClient(mod.app).post("/api/analyze", json={"url": ID, "topic_text": "compendium"})
    assert res.json()["topic_text"] == {"chosen": "compendium", "used": "description"}  # none: stand-in
    assert TestClient(mod.app).post("/api/analyze",
                                    json={"url": ID, "topic_text": "x"}).status_code == 422


def test_config_reports_password_and_cross_encoder(monkeypatch, tmp_path):
    from themenseiten_check import onnx_reranker
    monkeypatch.setattr(onnx_reranker, "MODEL_DIR", tmp_path)       # no model here
    cfg = TestClient(_load_app(monkeypatch).app).get("/api/config").json()
    assert cfg["password"] is False
    assert cfg["crossencoder"]["available"] is False
    assert "Modell nicht ausgeliefert" in cfg["crossencoder"]["reason"]
    mod = _load_app(monkeypatch, DEMO_PASSWORD="x")
    assert TestClient(mod.app).get("/api/config").json()["password"] is True


@pytest.mark.parametrize("backend", ["crossencoder", "hybrid"])
def test_cross_encoder_request_without_model_is_refused(monkeypatch, tmp_path, backend):
    from themenseiten_check import onnx_reranker
    monkeypatch.setattr(onnx_reranker, "MODEL_DIR", tmp_path)
    mod = _load_app(monkeypatch)

    async def fake_load(repo, cid, **kw):
        return _topic()

    monkeypatch.setattr(mod, "_repository", FakeRepo)
    monkeypatch.setattr(mod, "load_topic_page", fake_load)
    res = TestClient(mod.app).post("/api/analyze", json={"url": ID, "backend": backend})
    assert res.status_code == 503 and "Cross-Encoder nicht verfügbar" in res.json()["error"]


def test_endpoint_hands_back_choices(monkeypatch):
    mod = _load_app(monkeypatch)

    async def ambiguous(repo, text, hosts):
        raise mod.NeedsChoice("Optik", [{"id": "1", "title": "Optik", "url": ""}])

    monkeypatch.setattr(mod, "_repository", FakeRepo)
    monkeypatch.setattr(mod, "resolve", ambiguous)
    res = TestClient(mod.app).post("/api/analyze", json={"url": "https://wordpress.staging.openeduhub.net/portal/x/"})
    assert res.status_code == 409 and res.json()["choices"][0]["id"] == "1"


def test_endpoint_rejects_foreign_portal_host(monkeypatch):
    mod = _load_app(monkeypatch)
    monkeypatch.setattr(mod, "_repository", FakeRepo)
    res = TestClient(mod.app).post("/api/analyze", json={"url": "https://evil.example/x"})
    assert res.status_code == 400 and "nicht abgerufen" in res.json()["error"]


def test_search_endpoint(monkeypatch):
    mod = _load_app(monkeypatch)
    seen = {}

    async def fake_find(repo, text, subject):
        seen.update(text=text, subject=subject)
        return {"hits": [{"id": "1", "title": "Optik"}]}

    monkeypatch.setattr(mod, "_repository", FakeRepo)
    monkeypatch.setattr(mod, "find_topic_pages", fake_find)
    client = TestClient(mod.app)
    res = client.post("/api/search", json={"text": " Optik ", "subject": "Physik"})
    assert res.status_code == 200 and res.json()["hits"][0]["id"] == "1"
    assert seen == {"text": "Optik", "subject": "Physik"}
    assert client.post("/api/search", json={"text": ""}).status_code == 422


def test_root_serves_the_page(monkeypatch):
    mod = _load_app(monkeypatch)
    res = TestClient(mod.app).get("/")
    assert res.status_code == 200 and "text/html" in res.headers["content-type"]
    assert "Themenseiten-Check" in res.text


def test_unknown_route_names_the_path(monkeypatch):
    mod = _load_app(monkeypatch)
    res = TestClient(mod.app).post("/api/index", json={})
    assert res.status_code == 404 and "/api/index" in res.json()["error"]


def test_vercel_config_has_no_path_changing_rewrite():
    config = json.loads((Path(__file__).resolve().parent.parent / "vercel.json").read_text())
    assert "rewrites" not in config
