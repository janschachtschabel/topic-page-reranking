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

The score has two halves, each relative to the page's **median** and capped at
1: closeness to the topic text, and **cohesion** -- closeness to all other
materials of the page together. 1.0 means "as close as a typical material of
this page". The median rather than the best hit (which wlo-mcp-sc normalises
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
from .scoring import DEFAULT_TOPIC_TEXT, topic_texts

#: Field weights. On the topic side the text is whichever ``scoring.topic_texts``
#: chose (description, compendium or both), each at the "text" weight.
#: Sub-topic titles were tried as topic vocabulary and dropped: their suffixes
#: ("Geometrische") matched unrelated words ("Elektrischer") on the example set.
TOPIC_FIELDS = {"title": 3.0, "keywords": 2.0, "text": 1.5}
ITEM_FIELDS = {"title": 2.0, "keywords": 1.5, "description": 1.0}
#: 4- and 5-grams: 3-grams ("ung", "ich") match across unrelated words and
#: blurred the separation on the Optik example set (see tests).
NGRAMS = (4, 5)
#: Share of the topic similarity in the score; the rest is cohesion with the
#: page. Equal shares scored best on the example pages (mean AUC 1.00 with the
#: description, against 0.975 for topic similarity alone and 0.96 at 0.3).
TOPIC_WEIGHT = 0.5
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


def topic_bag(topic: TopicPage, source: str = DEFAULT_TOPIC_TEXT) -> Counter[str]:
    return _weighted({"title": topic.title, "keywords": " ".join(topic.keywords),
                      "text": " ".join(topic_texts(topic, source)[0])}, TOPIC_FIELDS)


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


def cohesion(vecs: list[dict[str, float]]) -> list[float]:
    """Each material's similarity to all the OTHER materials of the page together.

    The page's own consensus as a second yardstick: misassigned materials are
    outliers among their neighbours even where they share words with the topic
    text ("Elektrischer Stromkreis" and the optics description both say
    "Physik"-ish things; the forty optics materials around it do not).
    Leave-one-out, so a material does not vote for itself.
    """
    total: Counter[str] = Counter()
    for vec in vecs:
        total.update(vec)
    out = []
    for vec in vecs:
        rest = {g: total[g] - vec.get(g, 0.0) for g in total}
        norm = math.sqrt(sum(x * x for x in rest.values()))
        out.append(sum(v * rest.get(g, 0.0) for g, v in vec.items()) / norm if norm else 0.0)
    return out


def _relative(values: list[float]) -> list[float]:
    """Divide by the median of the positive values, cap at 1."""
    typical = _median([x for x in values if x > 0]) or 1.0
    return [min(1.0, x / typical) for x in values]


class TextStatsReranker:
    """Scores a whole page at once: IDF and cohesion need every material of it."""

    name = "Textstatistik (Zeichen-n-Gramme, TF-IDF, Seitenzusammenhalt, Fach)"
    #: Tuned on the example pages (tests/eval_pages.py): best mean F1 for
    #: finding misfits at 0.60 with the description as topic text. The margin
    #: there is narrow (lowest fit 0.65, highest misfit 0.61), so treat it as a
    #: starting point and calibrate on marked real pages in the app.
    default_threshold = 0.6

    async def score_page(self, topic: TopicPage, items: list[Item],
                         source: str = DEFAULT_TOPIC_TEXT) -> list[float]:
        bags = [item_bag(i) for i in items]
        query = topic_bag(topic, source)
        docs = [query, *bags]
        df: Counter[str] = Counter()
        for bag in docs:
            df.update(bag.keys())
        idf = {g: math.log((1 + len(docs)) / (1 + d)) + 1 for g, d in df.items()}
        query_vec = _tfidf(query, idf)
        vecs = [_tfidf(b, idf) for b in bags]
        to_topic = _relative([cosine(query_vec, v) for v in vecs])
        to_page = _relative(cohesion(vecs))
        return [(TOPIC_WEIGHT * t + (1 - TOPIC_WEIGHT) * c)
                * (1.0 if subjects_agree(topic.subjects, item.subjects) else DISJOINT_SUBJECT)
                for t, c, item in zip(to_topic, to_page, items)]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
