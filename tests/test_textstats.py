"""Text statistics: the building blocks, and one example page as a guard.

The example pages (``eval_pages.py``) are small and written for this purpose,
so the numbers below are a regression guard, not evidence of quality on real
pages: if a change makes them worse, look again; that they pass says little more.
"""

import asyncio

import pytest

from themenseiten_check.collect import Item, TopicPage
from themenseiten_check.scoring import rerank
from eval_pages import PAGES
from themenseiten_check.textstats import (TextStatsReranker, cosine, ngrams, normalise,
                                          subjects_agree)

P = ("Physik",)


def item(i, title, desc="", kw=(), subjects=P):
    return Item(i, title, desc, tuple(kw), subjects, "", "")


def test_ngrams_bridge_compounds_and_inflection():
    a, b = ngrams("Linsen"), ngrams("Sammellinse")
    assert {"lins", "inse", "linse"} <= set(a) & set(b)
    assert "licht" in ngrams("Lichtbrechung")


def test_ngrams_skip_stopwords_and_short_words():
    assert not ngrams("und die der ab")


def test_normalise_composes_umlauts():
    assert normalise("Lu\u0308cke") == normalise("Lücke") == "lücke"


def test_subjects_agree():
    assert subjects_agree(P, ("physik", "Chemie")) == 1.0
    assert subjects_agree(P, ("Mathematik",)) == 0.0
    assert subjects_agree(P, ()) == 1.0 and subjects_agree((), ("Mathematik",)) == 1.0


def test_cosine_of_unit_vectors():
    assert cosine({"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0)
    assert cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_scores_are_relative_to_the_typical_material_and_capped():
    topic = TopicPage("t", "Optik", "", ("Licht",), "", (), subjects=P)
    items = [item("1", "Licht und Schatten"), item("2", "Licht im Alltag"),
             item("3", "Licht und Linse"), item("4", "Bruchrechnen")]
    scores = asyncio.run(TextStatsReranker().score_page(topic, items))
    assert max(scores) == 1.0 and all(0.0 <= s <= 1.0 for s in scores)
    assert scores[3] < 0.5


def test_disjoint_subject_caps_the_score():
    topic = TopicPage("t", "Optik", "", ("Licht",), "", (), subjects=P)
    items = [item("1", "Licht und Schatten"), item("2", "Licht und Farben", subjects=("Kunst",))]
    scores = asyncio.run(TextStatsReranker().score_page(topic, items))
    assert scores[1] <= 0.4 < scores[0]


def _measure(topic, fits, source):
    scored = asyncio.run(rerank(topic, TextStatsReranker(), source=source))
    pos = [s.score for s in scored if fits[s.item.id]]
    neg = [s.score for s in scored if not fits[s.item.id]]
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    return scored, auc


@pytest.mark.parametrize("name", list(PAGES))
def test_example_pages_with_description(name):
    """Default setting: measured AUC 0.99-1.00, no fitting material dropped."""
    topic, fits = PAGES[name]
    scored, auc = _measure(topic, fits, "description")
    assert auc >= 0.95
    assert all(s.keep for s in scored if fits[s.item.id])
    # Misfits of another subject are always dropped -- the subject cap alone does that.
    assert all(not s.keep for s in scored
               if not fits[s.item.id] and set(s.item.subjects) != set(topic.subjects))


def test_example_pages_find_most_misfits():
    # Measured at the default threshold: 17 of 18 misfits across the three
    # pages; the one kept is "Elektrischer Stromkreis" on Optik (0.61 vs 0.60).
    found = total = 0
    for topic, fits in PAGES.values():
        scored, _ = _measure(topic, fits, "description")
        found += sum(not s.keep for s in scored if not fits[s.item.id])
        total += sum(not f for f in fits.values())
    assert found >= total - 2


@pytest.mark.parametrize("name", list(PAGES))
def test_example_pages_with_compendium(name):
    # The long compendium text is the weaker yardstick (measured mean AUC 0.965).
    topic, fits = PAGES[name]
    _, auc = _measure(topic, fits, "compendium")
    assert auc >= 0.9


def test_cohesion_singles_out_the_odd_one():
    from themenseiten_check.textstats import cohesion
    vecs = [{"a": 1.0}, {"a": 0.8, "b": 0.6}, {"a": 1.0}, {"z": 1.0}]
    c = cohesion(vecs)
    assert c[3] == 0.0 and min(c[:3]) > 0.5


def test_source_changes_what_is_measured():
    topic, _ = PAGES["Optik"]
    texts = {src: asyncio.run(TextStatsReranker().score_page(topic, list(topic.items), src))
             for src in ("description", "compendium")}
    assert texts["description"] != texts["compendium"]
