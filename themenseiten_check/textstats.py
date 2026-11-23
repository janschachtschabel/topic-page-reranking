"""Text statistics instead of a model: character n-gram TF-IDF plus subject agreement.

Free, no key, no download, pure Python -- so it runs in a Vercel function,
where a cross-encoder cannot (torch exceeds the function size).

Why character n-grams and not whole words: German compounds and inflection.
"Linsen" and "Sammellinse" share no word but share "lin", "ins", "nse", "linse";
"Licht" sits inside "Lichtbrechung". Word matching -- the rule the client's
``edusharing.ranking`` and wlo-mcp-sc's reranker use -- catches the second and
misses the first.

What the wlo-mcp-sc reranker contributed and what it did not:

* kept: fields weigh differently (title over keywords over description), and
  stopwords carry no signal (the client's ``GERMAN`` profile);
* left out: its metadata-quality bonus (license, preview, publisher). That
  answers "is this a good record?", not "does it belong to this topic?" -- it
  would lift a well-kept off-topic record above a sparse fitting one.

Added: **subject agreement.** A material whose subject labels exist and share
none with the topic's (Mathematik on an Optik page under Physik) is the most
common misassignment and needs no text statistics to be seen; its score is
multiplied by 0.4. Agreement adds nothing: "Physik" does not make a mechanics
worksheet fit an optics page.

The score is relative to the page's **median** similarity, capped at 1: 1.0
means "as close to the topic as a typical material of this page", 0.5 "half as
close". The median rather than the best hit (which wlo-mcp-sc normalises
against) because one keyword-heavy outlier would otherwise squash every other
score. The premise is that most of a page fits -- on a page where most does
not, the scale shifts with it. That is the price of having no absolute
yardstick without a model.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from edusharing.language import GERMAN

from .collect import Item, TopicPage

#: Field weights. Topic: the description is often portal boilerplate
#: ("Hier findest du ... Material für Physik"), the compendium says what is meant.
#: Sub-topic titles were tried as topic vocabulary and dropped: their suffixes
#: ("Geometrische") matched unrelated words ("Elektrischer") on the example set.
TOPIC_FIELDS = {"title": 3.0, "keywords": 2.0, "compendium": 1.5, "description": 0.5}
ITEM_FIELDS = {"title": 2.0, "keywords": 1.5, "description": 1.0}
#: 4- and 5-grams: 3-grams ("ung", "ich") match across unrelated words and
#: blurred the separation on the Optik example set (see tests).
NGRAMS = (4, 5)
#: Factor for a material whose subjects share none with the topic's.
DISJOINT_SUBJECT = 0.4

_WORD = re.compile(r"[a-zäöüß]{3,}")


def normalise(text: str) -> str:
    """Lowercase, NFC, so a decomposed umlaut matches a composed one."""
    return unicodedata.normalize("NFC", text or "").lower()


def ngrams(text: str) -> Counter[str]:
    """Character n-grams of every non-stopword, each word padded so starts and ends count.

    No suffix stripping: tried, it lowered the ranking quality on the example
    set (AUC 0.97 -> 0.88) -- with short records the endings carry as much
    signal as noise.
    """
    grams: Counter[str] = Counter()
    for word in _WORD.findall(normalise(text)):
        if word in GERMAN.stopwords:
            continue
        padded = f"_{word}_"
        for n in NGRAMS:
            grams.update(padded[i:i + n] for i in range(len(padded) - n + 1))
    return grams


def _weighted(fields: dict[str, str], weights: dict[str, float]) -> Counter[str]:
    bag: Counter[str] = Counter()
    for name, weight in weights.items():
        for gram, count in ngrams(fields.get(name, "")).items():
            bag[gram] += weight * count
    return bag


def topic_bag(topic: TopicPage) -> Counter[str]:
    return _weighted({"title": topic.title, "keywords": " ".join(topic.keywords),
                      "compendium": topic.compendium, "description": topic.description},
                     TOPIC_FIELDS)


def item_bag(item: Item) -> Counter[str]:
    return _weighted({"title": item.title, "keywords": " ".join(item.keywords),
                      "description": item.description}, ITEM_FIELDS)


def _tfidf(bag: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    # Sublinear tf: the tenth "licht" says less than the first.
    vec = {g: (1 + math.log(c)) * idf[g] for g, c in bag.items() if c > 0}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {g: v / norm for g, v in vec.items()} if norm else {}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(g, 0.0) for g, v in a.items())


def subjects_agree(topic: tuple[str, ...], item: tuple[str, ...]) -> float:
    """1 when they share a subject or either side has none; 0 when both name disjoint ones."""
    if not topic or not item:
        return 1.0
    return 1.0 if {normalise(s) for s in topic} & {normalise(s) for s in item} else 0.0


class TextStatsReranker:
    """Scores a whole page at once: IDF needs every material of it."""

    name = "Textstatistik (Zeichen-n-Gramme, TF-IDF, Fach)"
    default_threshold = 0.5

    async def score_page(self, topic: TopicPage, items: list[Item]) -> list[float]:
        bags = [item_bag(i) for i in items]
        query = topic_bag(topic)
        docs = [query, *bags]
        df: Counter[str] = Counter()
        for bag in docs:
            df.update(bag.keys())
        idf = {g: math.log((1 + len(docs)) / (1 + d)) + 1 for g, d in df.items()}
        query_vec = _tfidf(query, idf)
        sims = [cosine(query_vec, _tfidf(b, idf)) for b in bags]
        typical = _median([x for x in sims if x > 0]) or 1.0
        return [min(1.0, sim / typical)
                * (1.0 if subjects_agree(topic.subjects, item.subjects) else DISJOINT_SUBJECT)
                for sim, item in zip(sims, items)]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
