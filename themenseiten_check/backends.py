"""The model-based reranker, for local use.

``CrossEncoderReranker`` reads topic and material *together* -- a real
reranker, multilingual, runs locally, free, no data leaves the machine. Too
large for a serverless function (torch), so the web app uses
``textstats.TextStatsReranker`` instead; the CLI offers both.

Its default threshold is a starting point on its own 0..1 scale, not a
calibrated value -- tune it per subject with ``--threshold``.
"""

from __future__ import annotations

import asyncio

DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    name = "cross-encoder"
    default_threshold = 0.3

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise SystemExit(
                "Das lokale Reranking braucht sentence-transformers: "
                "pip install -e '.[local]' -- oder --backend textstats nutzen.") from exc
        self.name = f"cross-encoder ({model_name})"
        self._model = CrossEncoder(model_name, max_length=512)

    async def score(self, query: str, documents: list[str]) -> list[float]:
        # With one output label, predict() applies a sigmoid: scores are 0..1.
        pairs = [(query, doc) for doc in documents]
        scores = await asyncio.to_thread(self._model.predict, pairs, batch_size=16)
        return [float(s) for s in scores]
