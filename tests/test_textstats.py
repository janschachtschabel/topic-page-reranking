"""Text statistics: the building blocks, and one example page as a guard.

The example set is small and was written for this purpose, so the numbers
below are a regression guard, not evidence of quality on real pages: if a
change makes them worse, look again; that they pass says little more.
"""

import asyncio

import pytest

from themenseiten_check.collect import Item, TopicPage
from themenseiten_check.scoring import rerank
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


EXAMPLE = [
    (item("1", "Sammellinse und Zerstreuungslinse", "Wie Linsen Lichtstrahlen bündeln oder streuen", ["Linsen"]), True),
    (item("2", "Lichtbrechung am Wasserglas", "Ein einfacher Versuch zur Brechung", ["Brechung"]), True),
    (item("3", "Regenbogen erklärt", "Dispersion: weißes Licht wird in Spektralfarben zerlegt"), True),
    (item("4", "Totalreflexion im Lichtleiter", "Glasfaser und Grenzwinkel"), True),
    (item("5", "Das menschliche Auge", "Aufbau, Netzhaut und Akkommodation", ["Auge"]), True),
    (item("6", "Additive Farbmischung", "Rot, Grün und Blau ergeben Weiß", ["Farben"]), True),
    (item("7", "Lochkamera selbst bauen", "Bastelanleitung, Bildentstehung"), True),
    (item("8", "Spiegelbilder am ebenen Spiegel", "Reflexionsgesetz: Einfallswinkel gleich Ausfallswinkel", ["Reflexion"]), True),
    (item("9", "Schatten und Halbschatten", "Kernschatten bei zwei Lichtquellen"), True),
    (item("10", "Beugung am Doppelspalt", "Interferenzmuster von Laserlicht", ["Wellenoptik"]), True),
    (item("11", "Linsengleichung üben", "Brennweite, Gegenstandsweite und Bildweite berechnen"), True),
    (item("12", "Brüche kürzen und erweitern", "Arbeitsblatt Klasse 6", ["Bruchrechnung"], ("Mathematik",)), False),
    (item("13", "Der Dreißigjährige Krieg", "Ursachen und Folgen", [], ("Geschichte",)), False),
    (item("14", "Photosynthese", "Pflanzen nutzen Licht, um Zucker herzustellen", ["Licht"], ("Biologie",)), False),
    (item("15", "Elektrischer Stromkreis", "Reihen- und Parallelschaltung", ["Strom"]), False),
    (item("16", "Newtonsche Gesetze", "Kraft, Masse und Beschleunigung", ["Mechanik"]), False),
    (item("17", "Bruchrechnen mit Pizza", "Anteile anschaulich", [], ("Mathematik",)), False),
    (item("18", "Schall und Lautstärke", "Schallwellen und Frequenz", ["Akustik"]), False),
]
OPTIK = TopicPage(
    "t", "Optik",
    "Hier findest du zahlreiches kostenloses Material für Physik! Neben sorgfältig "
    "ausgewählten Inhalten für jede Art von Unterricht findest du auch kurzweilige Inhalte.",
    ("Licht", "Linse"), "", tuple(i for i, _ in EXAMPLE),
    compendium="Die Optik ist das Teilgebiet der Physik, das sich mit dem Licht beschäftigt: "
               "seiner Ausbreitung, Reflexion und Brechung, mit Linsen, Spiegeln und optischen "
               "Geräten wie dem Auge, mit Farben sowie mit Beugung und Interferenz in der Wellenoptik.",
    subjects=P)


def test_example_page_guard():
    scored = asyncio.run(rerank(OPTIK, TextStatsReranker()))
    fits = {i.id: f for i, f in EXAMPLE}
    pos = [s.score for s in scored if fits[s.item.id]]
    neg = [s.score for s in scored if not fits[s.item.id]]
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert auc >= 0.95                                       # measured 0.97
    assert all(s.keep for s in scored if fits[s.item.id])   # no fitting material dropped
    # Every other-subject misfit is dropped; same-subject ones (mechanics,
    # electricity, acoustics on an optics page) are what text statistics
    # cannot reliably tell apart -- measured: they stay.
    other = {"12", "13", "14", "17"}
    assert all(not s.keep for s in scored if s.item.id in other)
