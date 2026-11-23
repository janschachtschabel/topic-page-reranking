"""The shipped model itself, on the example pages -- skipped where model/ is absent.

Guards the measured numbers (onnx_reranker module docstring) against a
re-quantised or replaced model that silently ranks worse.
"""

import asyncio

import pytest

from eval_pages import PAGES
from themenseiten_check import onnx_reranker
from themenseiten_check.scoring import rerank

pytestmark = pytest.mark.skipif(bool(onnx_reranker.unavailable()), reason="model/ not shipped here")


def _auc(scored, fits):
    pos = [s.score for s in scored if fits[s.item.id]]
    neg = [s.score for s in scored if not fits[s.item.id]]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


@pytest.mark.parametrize("cls, floor", [("OnnxCrossEncoder", 0.8), ("HybridReranker", 0.9)])
def test_real_model_ranks_the_example_pages(cls, floor):
    reranker = getattr(onnx_reranker, cls)()
    aucs = [_auc(asyncio.run(rerank(topic, reranker)), fits) for topic, fits in PAGES.values()]
    assert min(aucs) >= floor, aucs          # measured 0.82 / 0.92 at the lowest


def test_manifest_label_survives_a_cp1252_manifest(tmp_path):
    (tmp_path / "manifest.json").write_bytes('{"model": "x/Französisch", "quantization": "int8 q"}'.encode("cp1252"))
    assert onnx_reranker._manifest_label(tmp_path).startswith("Franz")
