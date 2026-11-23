"""The ONNX cross-encoder path, with a toy model built here.

The real model cannot be downloaded in the test environment; a toy with the
same inputs (input_ids, attention_mask, token_type_ids -> logits [batch, 1])
exercises everything around it: quantisation, sharding under a size limit,
loading sharded weights, pair encoding with the query cap, padding, scoring.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
tokenizers = pytest.importorskip("tokenizers")
np = pytest.importorskip("numpy")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_model  # noqa: E402

from themenseiten_check import onnx_reranker  # noqa: E402
from themenseiten_check.collect import Item, TopicPage  # noqa: E402

WORDS = ("optik licht linse linsen spiegel brechung farben auge physik bruchrechnen pizza "
         "krieg geschichte mathematik klasse und mit der die im versuch").split()
FITTING = {"licht", "linse", "linsen", "spiegel", "brechung", "farben", "auge", "optik"}


def make_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({w: i + 4 for i, w in enumerate(WORDS)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok.save(str(path))


def make_model(path: Path, vocab: int = 2000, dim: int = 4) -> None:
    """logits = sum over tokens of the second segment of emb[token] . w"""
    rng = np.random.default_rng(0)
    emb = (rng.normal(size=(vocab, dim)) * 0.01).astype(np.float32)
    for w in FITTING:
        emb[WORDS.index(w) + 4, 0] = 1.0
    emb[0] = 0.0
    emb2 = (rng.normal(size=(vocab, dim)) * 0.01).astype(np.float32)   # a second big tensor
    w = np.array([[2.0], [0.1], [0.1], [0.1]], dtype=np.float32)
    nodes = [
        helper.make_node("Gather", ["emb", "input_ids"], ["x1"]),
        helper.make_node("Gather", ["emb2", "input_ids"], ["x2"]),
        helper.make_node("Add", ["x1", "x2"], ["x"]),
        helper.make_node("Cast", ["token_type_ids"], ["tt"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["tt", "axis2"], ["tt3"]),
        helper.make_node("Mul", ["x", "tt3"], ["xb"]),              # only the material segment counts
        helper.make_node("MatMul", ["xb", "w"], ["s"]),
        helper.make_node("ReduceSum", ["s", "axis1"], ["logits"], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "toy",
        [helper.make_tensor_value_info(n, TensorProto.INT64, ["b", "s"])
         for n in ("input_ids", "attention_mask", "token_type_ids")],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 1])],
        [numpy_helper.from_array(emb, "emb"), numpy_helper.from_array(emb2, "emb2"),
         numpy_helper.from_array(w, "w"),
         numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
         numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save_model(model, str(path))


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("model")
    make_model(base / "fp32.onnx")
    prepare_model.quantize(base / "fp32.onnx", base / "int8.onnx")
    out = base / "model"
    files = prepare_model.shard(base / "int8.onnx", out, max_bytes=12_000, min_tensor_bytes=1_000)
    make_tokenizer(out / "tokenizer.json")
    (out / "manifest.json").write_text(json.dumps({"model": "toy", "quantization": "int8"}))
    return base, out, files


def test_shards_stay_under_the_limit_and_load(model_dir):
    base, out, files = model_dir
    shards = [f for f in files if f.name.startswith("weights-")]
    assert len(shards) >= 2 and all(f.stat().st_size <= 12_000 for f in shards)
    assert all(f.name.isascii() and ":" not in f.name for f in files)
    feed = {"input_ids": np.array([[2, 5, 3, 6, 7, 3]], dtype=np.int64),
            "attention_mask": np.ones((1, 6), dtype=np.int64),
            "token_type_ids": np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int64)}
    fp32 = ort.InferenceSession(str(base / "fp32.onnx")).run(None, feed)[0]
    int8 = ort.InferenceSession(str(out / "model.onnx")).run(None, feed)[0]
    assert np.allclose(fp32, int8, atol=0.05)


def test_a_tensor_larger_than_a_file_is_refused(model_dir, tmp_path):
    base, _, _ = model_dir
    with pytest.raises(ValueError, match="größer als eine Datei"):
        prepare_model.shard(base / "fp32.onnx", tmp_path / "m", max_bytes=10_000, min_tensor_bytes=1_000)


def test_unavailable_names_the_missing_model(tmp_path):
    assert "Modell nicht ausgeliefert" in onnx_reranker.unavailable(tmp_path)
    with pytest.raises(RuntimeError):
        onnx_reranker.OnnxCrossEncoder(tmp_path)


def test_pairs_cap_the_query_and_the_total(model_dir, monkeypatch):
    from tokenizers import Tokenizer
    _, out, _ = model_dir
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    monkeypatch.setattr(onnx_reranker, "QUERY_TOKENS", 5)
    monkeypatch.setattr(onnx_reranker, "MAX_TOKENS", 12)
    query = " ".join(["licht"] * 50)                 # a compendium-sized topic text
    enc = onnx_reranker.encode_pairs(tok, query, ["linse " * 40, "auge"])
    first = enc[0]
    assert first.type_ids.count(0) == 5 + 2          # 5 query tokens + [CLS] [SEP]
    assert len(first.ids) <= 12 and len(enc[1].ids) == 5 + 2 + 1 + 1


def test_cross_encoder_scores_a_page(model_dir):
    _, out, _ = model_dir
    p = ("Physik",)
    items = [Item("1", "Linsen im Versuch", "Licht und Brechung", ("Optik",), p, "", ""),
             Item("2", "Spiegel und Farben", "", (), p, "", ""),
             Item("3", "Das Auge", "Licht", (), p, "", ""),
             Item("4", "Bruchrechnen mit Pizza", "Klasse", (), ("Mathematik",), "", ""),
             Item("5", "Der Krieg", "Geschichte", (), p, "", "")]
    topic = TopicPage("t", "Optik", "Licht und Linsen", ("Licht",), "", tuple(items), subjects=p)
    reranker = onnx_reranker.OnnxCrossEncoder(out)
    assert reranker.name == "Cross-Encoder (toy, int8)"
    scores = asyncio.run(reranker.score_page(topic, items))
    assert min(scores[:3]) > max(scores[3:])
    assert scores[3] <= 0.4                          # disjoint subject capped
    hybrid = asyncio.run(onnx_reranker.HybridReranker(out).score_page(topic, items))
    assert min(hybrid[:3]) > max(hybrid[3:])


def test_page_scale_centres_the_median():
    assert onnx_reranker.page_scale([-1.0, -2.0, -3.0, -9.0]) == [0.75, 0.5, 0.25, 0.0]
    assert onnx_reranker.page_scale([-4.0, -4.0]) == [0.5, 0.5]


def test_prepare_compare_reports_both_precisions(model_dir):
    base, out, _ = model_dir
    rows = prepare_model.compare(base / "fp32.onnx", out)
    assert len(rows) == len(prepare_model.SAMPLE)
    assert all(abs(r["fp32"] - r["int8"]) < 0.1 for r in rows)


def test_headings_as_topic_text():
    from themenseiten_check.scoring import topic_texts
    from themenseiten_check.topic_text import headings
    md = ("::: inhaltsverzeichnis\n## Inhaltsverzeichnis\n- [Teil 1](#t)\n:::\n"
          "# Kompendialer Text \"Optik\"\n## Teil 1: Fachlicher Teil\n### 1\\. Themendefinition\n"
          "### Geometrische Optik\n#### Zu tief\n## [Farben](#farben)\n")
    assert headings(md) == ["Geometrische Optik", "Farben"]
    topic = TopicPage("t", "Optik", "", (), "", (), compendium="lang",
                      compendium_headings=("Geometrische Optik", "Farben"))
    assert topic_texts(topic, "headings") == (["Geometrische Optik, Farben"], "headings")
    assert topic_texts(topic, "description") == (["Geometrische Optik, Farben"], "headings")
