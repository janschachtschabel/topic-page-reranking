"""A small cross-encoder that fits a Vercel function: ONNX, int8, no torch.

The model is ``cross-encoder/msmarco-MiniLM-L12-en-de-v1`` (German/English),
quantised to int8 by ``scripts/prepare_model.py`` and shipped in ``model/``:
``model.onnx`` plus weight shards, each under GitHub's 100 MiB per-file limit,
and the Hugging Face ``tokenizer.json``. Runtime needs ``onnxruntime``,
``tokenizers`` and ``numpy`` -- about 160 MB, with the model about 280 MB, under
Vercel's 500 MB for Python functions.

**What the model reads.** A cross-encoder judges one (topic, material) pair at
a time and sees 256 tokens of it here -- enough for a material's title,
description and keywords, far too little for a compendium of many pages. The
topic side is therefore capped at ``QUERY_TOKENS``; with the full compendium
text chosen, only its first lines would count. The compendium's headings
("Überschriften") or the description are what it should be given.

**What the score means.** msmarco cross-encoders were trained on search
queries against passages; for "topic vs. material" their logits sit far below
zero (-1 to -11 on the example pages) and shift from page to page. So the
logits are placed on the page's own scale: the median material gets 0.5, two
median absolute deviations above it 1.0, two below it 0.0. A material of a
disjoint subject is multiplied by 0.4.

**Measured** with the shipped int8 model on the example pages
(tests/eval_pages.py, description as topic text): ranking AUC 0.88 / 0.82 /
0.82 without, 0.93 / 0.91 / 0.82 with the subject cap -- behind the text
statistics on the same pages (1.00 each). Threshold 0.25 keeps 29 of 33
fitting materials and drops 13 of 18 misfits; at 0.45, the best F1, it would
drop 10 fitting ones as well -- too many for a review tool. The full compendium text as query was worse
still (AUC 0.75 / 0.83 / 0.57): the model sees only its first lines. The
combination with the text statistics (``HybridReranker``) reached AUC 1.00 /
1.00 / 0.92 and, at 0.40, keeps 32 of 33 and drops 16 of 18. The example pages were written while tuning the text statistics,
so they favour it; real pages, marked in the app, decide.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from .collect import Item, TopicPage
from .scoring import DEFAULT_TOPIC_TEXT, item_text, topic_texts
from .textstats import DISJOINT_SUBJECT, subjects_agree

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
#: Tokens the topic side may take of the pair; the material gets the rest.
QUERY_TOKENS = 96
MAX_TOKENS = 256
BATCH = 32

_cache: dict[Path, tuple[Any, Any, list[str]]] = {}
_lock = threading.Lock()


def unavailable(model_dir: Path | None = None) -> str:
    """Why the model cannot run here, or '' when it can."""
    model_dir = model_dir or MODEL_DIR
    if not (model_dir / "model.onnx").is_file() or not (model_dir / "tokenizer.json").is_file():
        return ("Modell nicht ausgeliefert: model/ enthält kein model.onnx und tokenizer.json "
                "(scripts/prepare_model.py bzw. die GitHub-Action „Modell vorbereiten“).")
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError as exc:
        return f"Laufzeit fehlt: {exc.name} ist nicht installiert."
    return ""


def _load(model_dir: Path) -> tuple[Any, Any, list[str]]:
    """Session, tokenizer and input names -- loaded once per warm instance."""
    with _lock:
        if model_dir not in _cache:
            _cache[model_dir] = _open(model_dir)
        return _cache[model_dir]


def _open(model_dir: Path) -> tuple[Any, Any, list[str]]:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_dir / "model.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    # Truncation and padding are done per pair below, never configured on
    # the shared tokenizer: Fluid compute runs requests concurrently in one
    # instance, and tokenizer settings are global state.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return session, tokenizer, [i.name for i in session.get_inputs()]


def encode_pairs(tokenizer: Any, query: str, docs: list[str]) -> list[Any]:
    """Model inputs for (query, doc) pairs: query capped at ``QUERY_TOKENS``,
    doc cut to what is left of ``MAX_TOKENS``, special tokens added by the
    model's own post-processor. Thread-safe: no tokenizer setting is changed."""
    processor = tokenizer.post_processor
    specials = processor.num_special_tokens_to_add(True) if processor else 0
    q = tokenizer.encode(query, add_special_tokens=False)
    q.truncate(QUERY_TOKENS)
    room = MAX_TOKENS - len(q.ids) - specials
    pairs = []
    for doc in docs:
        d = tokenizer.encode(doc, add_special_tokens=False)
        d.truncate(max(room, 1))
        pairs.append(tokenizer.post_process(q, d, add_special_tokens=True))
    return pairs


def pad_token_id(tokenizer: Any) -> int:
    for token in ("<pad>", "[PAD]"):
        found = tokenizer.token_to_id(token)
        if found is not None:
            return found
    return 0


def cross_query(topic: TopicPage, source: str) -> str:
    parts = [topic.title]
    if topic.keywords:
        parts.append(", ".join(topic.keywords))
    parts.extend(topic_texts(topic, source)[0])
    return ". ".join(p.strip().rstrip(".") for p in parts if p.strip())


def _manifest_label(model_dir: Path) -> str:
    """'<model>, <quantization>' from manifest.json -- only a label, so it never fails.

    Read as bytes and decoded leniently: a manifest written on Windows without
    an explicit encoding is cp1252 (seen 2026-11-23: "Französische" in the
    sample pairs broke a strict UTF-8 read and with it the whole reranker).
    """
    try:
        meta = json.loads((model_dir / "manifest.json").read_bytes().decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        meta = {}
    model = str(meta.get("model", "ONNX")).rsplit("/", 1)[-1]
    return f"{model}, {meta.get('quantization', 'int8').split(' ')[0]}"


def page_scale(logits: list[float]) -> list[float]:
    """Logits onto the page's own 0..1 scale: median 0.5, +-2 MAD to 1 and 0."""
    ordered = sorted(logits)
    median = ordered[len(ordered) // 2]
    mad = sorted(abs(x - median) for x in logits)[len(logits) // 2] or 1.0
    return [min(1.0, max(0.0, 0.5 + 0.25 * (x - median) / mad)) for x in logits]


class OnnxCrossEncoder:
    """Scores a page with the shipped int8 cross-encoder."""

    #: Balanced on the example pages: 29/33 fitting kept, 13/18 misfits dropped.
    default_threshold = 0.25

    def __init__(self, model_dir: Path | None = None) -> None:
        model_dir = model_dir or MODEL_DIR
        reason = unavailable(model_dir)
        if reason:
            raise RuntimeError(reason)
        self._dir = model_dir
        self.name = f"Cross-Encoder ({_manifest_label(model_dir)})"

    def _run(self, query: str, docs: list[str]) -> list[float]:
        import numpy as np

        session, tokenizer, inputs = _load(self._dir)
        pad = pad_token_id(tokenizer)
        logits: list[float] = []
        for start in range(0, len(docs), BATCH):
            encs = encode_pairs(tokenizer, query, docs[start:start + BATCH])
            width = max(len(e.ids) for e in encs)
            fill = lambda seq, value: list(seq) + [value] * (width - len(seq))  # noqa: E731
            feed = {
                "input_ids": np.array([fill(e.ids, pad) for e in encs], dtype=np.int64),
                "attention_mask": np.array([fill(e.attention_mask, 0) for e in encs], dtype=np.int64),
                "token_type_ids": np.array([fill(e.type_ids, 0) for e in encs], dtype=np.int64),
            }
            out = session.run(None, {k: v for k, v in feed.items() if k in inputs})[0]
            logits.extend(float(row[0]) for row in out.reshape(len(encs), -1))
        return logits

    async def score_page(self, topic: TopicPage, items: list[Item],
                         source: str = DEFAULT_TOPIC_TEXT) -> list[float]:
        if not items:
            return []
        logits = await asyncio.to_thread(self._run, cross_query(topic, source),
                                         [item_text(i) for i in items])
        return [r * (1.0 if subjects_agree(topic.subjects, i.subjects) else DISJOINT_SUBJECT)
                for r, i in zip(page_scale(logits), items)]


class HybridReranker:
    """Text statistics and cross-encoder, half each.

    The two err differently: the statistics see shared word fragments and the
    page's cohesion, the model sees meaning across different words. On the
    example pages the mix ranked better than the model alone (mean AUC 0.97 vs
    0.84), not better than the statistics alone.
    """

    #: Best mean F1 on the example pages (0.91): 32/33 kept, 16/18 dropped.
    default_threshold = 0.4

    def __init__(self, model_dir: Path | None = None) -> None:
        from .textstats import TextStatsReranker
        self._stats = TextStatsReranker()
        self._model = OnnxCrossEncoder(model_dir)
        self.name = f"Kombination: Textstatistik + {self._model.name}"

    async def score_page(self, topic: TopicPage, items: list[Item],
                         source: str = DEFAULT_TOPIC_TEXT) -> list[float]:
        stats = await self._stats.score_page(topic, items, source)
        model = await self._model.score_page(topic, items, source)
        return [0.5 * a + 0.5 * b for a, b in zip(stats, model)]
