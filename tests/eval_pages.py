"""Hand-written example pages for tuning and guarding the text statistics.

Three subjects, each with fitting materials, misfits of other subjects and the
hard case: misfits of the SAME subject (mechanics on an optics page). Written
for this purpose -- good for comparing variants of the method against each
other, not evidence of how it does on real pages. Real pages are judged in
the app (mark materials, read the measured threshold).
"""

from themenseiten_check.collect import Item, TopicPage


def _page(pid, title, description, keywords, compendium, subjects, rows):
    items, fits = [], {}
    for n, (t, d, kw, subj, fit) in enumerate(rows, start=1):
        iid = f"{pid}-{n}"
        items.append(Item(iid, t, d, tuple(kw), tuple(subj or subjects), "", ""))
        fits[iid] = fit
    topic = TopicPage(pid, title, description, tuple(keywords), "", tuple(items),
                      compendium=compendium, subjects=tuple(subjects))
    return topic, fits


P, M, I = ("Physik",), ("Mathematik",), ("Informatik",)

OPTIK = _page("optik", "Optik",
    "Licht breitet sich aus, wird reflektiert und gebrochen. Hier geht es um Spiegel, Linsen, "
    "Farben und optische Geräte wie Auge, Lupe und Kamera.",
    ["Licht", "Linse"],
    "Die Optik ist das Teilgebiet der Physik, das sich mit dem Licht beschäftigt: seiner "
    "Ausbreitung, Reflexion und Brechung, mit Linsen, Spiegeln und optischen Geräten wie dem "
    "Auge, mit Farben sowie mit Beugung und Interferenz in der Wellenoptik.", P, [
    ("Sammellinse und Zerstreuungslinse", "Wie Linsen Lichtstrahlen bündeln oder streuen", ["Linsen"], None, True),
    ("Lichtbrechung am Wasserglas", "Ein einfacher Versuch zur Brechung", ["Brechung"], None, True),
    ("Regenbogen erklärt", "Dispersion: weißes Licht wird in Spektralfarben zerlegt", [], None, True),
    ("Totalreflexion im Lichtleiter", "Glasfaser und Grenzwinkel", [], None, True),
    ("Das menschliche Auge", "Aufbau, Netzhaut und Akkommodation", ["Auge"], None, True),
    ("Additive Farbmischung", "Rot, Grün und Blau ergeben Weiß", ["Farben"], None, True),
    ("Lochkamera selbst bauen", "Bastelanleitung, Bildentstehung", [], None, True),
    ("Spiegelbilder am ebenen Spiegel", "Reflexionsgesetz: Einfallswinkel gleich Ausfallswinkel", ["Reflexion"], None, True),
    ("Schatten und Halbschatten", "Kernschatten bei zwei Lichtquellen", [], None, True),
    ("Beugung am Doppelspalt", "Interferenzmuster von Laserlicht", ["Wellenoptik"], None, True),
    ("Linsengleichung üben", "Brennweite, Gegenstandsweite und Bildweite berechnen", [], None, True),
    ("Wie funktioniert eine Lupe?", "Vergrößerung mit einer Sammellinse", ["Lupe"], None, True),
    ("Brüche kürzen und erweitern", "Arbeitsblatt Klasse 6", ["Bruchrechnung"], M, False),
    ("Der Dreißigjährige Krieg", "Ursachen und Folgen", [], ("Geschichte",), False),
    ("Photosynthese", "Pflanzen nutzen Licht, um Zucker herzustellen", ["Licht"], ("Biologie",), False),
    ("Elektrischer Stromkreis", "Reihen- und Parallelschaltung", ["Strom"], None, False),
    ("Newtonsche Gesetze", "Kraft, Masse und Beschleunigung", ["Mechanik"], None, False),
    ("Schall und Lautstärke", "Schallwellen und Frequenz", ["Akustik"], None, False),
])

GEOMETRIE = _page("geo", "Räumliche und ebene Geometrie",
    "Dreiecke, Vierecke und Kreise, Körper wie Würfel, Quader, Zylinder und Kugel: Flächen, "
    "Umfang, Volumen und Oberfläche berechnen und konstruieren.",
    ["Geometrie", "Flächen", "Körper"],
    "Die Geometrie untersucht ebene Figuren und räumliche Körper, ihre Eigenschaften, Lage "
    "und Maße. Dazu gehören Konstruktionen mit Zirkel und Lineal, Kongruenz und Ähnlichkeit, "
    "der Satz des Pythagoras, Flächen- und Volumenberechnung sowie Netze und Schrägbilder.", M, [
    ("Flächeninhalt von Dreiecken", "Grundseite mal Höhe durch zwei", ["Dreieck"], None, True),
    ("Satz des Pythagoras", "Rechtwinklige Dreiecke und Hypotenuse", ["Pythagoras"], None, True),
    ("Kreisumfang und Kreisfläche", "Die Zahl Pi entdecken", ["Kreis"], None, True),
    ("Volumen von Quader und Würfel", "Kantenlängen multiplizieren", ["Volumen"], None, True),
    ("Netze von Körpern", "Würfelnetze falten und erkennen", ["Körper"], None, True),
    ("Oberfläche eines Zylinders", "Mantelfläche und Grundflächen", [], None, True),
    ("Kongruente Dreiecke", "Kongruenzsätze SSS, SWS, WSW", [], None, True),
    ("Konstruieren mit Zirkel und Lineal", "Mittelsenkrechte und Winkelhalbierende", [], None, True),
    ("Vierecke im Haus der Vierecke", "Quadrat, Raute, Trapez, Parallelogramm", ["Vierecke"], None, True),
    ("Schrägbilder zeichnen", "Räumliche Darstellung von Körpern", [], None, True),
    ("Kugelvolumen", "Formel und Beispiele", ["Kugel"], None, True),
    ("Lineare Gleichungen lösen", "Äquivalenzumformungen üben", ["Gleichungen"], None, False),
    ("Prozentrechnung im Alltag", "Rabatte und Zinsen", ["Prozent"], None, False),
    ("Wahrscheinlichkeit beim Würfeln", "Laplace-Experimente", ["Stochastik"], None, False),
    ("Zellteilung", "Mitose und Meiose", [], ("Biologie",), False),
    ("Gedichtanalyse", "Reim, Metrum und Strophe", [], ("Deutsch",), False),
    ("Kreislauf des Wassers", "Verdunstung und Niederschlag", [], ("Geografie",), False),
])

TECHINF = _page("techinf", "Technische Informatik",
    "Wie ein Computer im Inneren arbeitet: Bits und Binärzahlen, logische Gatter, "
    "Schaltnetze, Prozessor, Speicher und Rechnerarchitektur.",
    ["Rechnerarchitektur", "Logik", "Hardware"],
    "Die Technische Informatik befasst sich mit dem Aufbau und der Funktionsweise von "
    "Rechnern: Zahlendarstellung im Binärsystem, boolesche Algebra und Logikgatter, "
    "Schaltnetze und Schaltwerke, Von-Neumann-Architektur, Prozessor, Speicherhierarchie "
    "und Mikrocontroller.", I, [
    ("Binärzahlen umrechnen", "Dezimal in Dual und zurück", ["Binärsystem"], None, True),
    ("Logikgatter AND, OR, NOT", "Wahrheitstabellen erstellen", ["Gatter"], None, True),
    ("Halbaddierer und Volladdierer", "Schaltnetze zum Addieren", [], None, True),
    ("Von-Neumann-Architektur", "Steuerwerk, Rechenwerk, Speicher, Bus", [], None, True),
    ("Wie arbeitet ein Prozessor?", "Befehlszyklus Fetch, Decode, Execute", ["CPU"], None, True),
    ("Speicherhierarchie", "Register, Cache, Arbeitsspeicher, Festplatte", ["Speicher"], None, True),
    ("Flipflops und Schaltwerke", "Speichern eines Bits", [], None, True),
    ("Mikrocontroller programmieren", "Erste Schritte mit dem Arduino", ["Mikrocontroller"], None, True),
    ("Boolesche Algebra", "Rechenregeln und Vereinfachung", [], None, True),
    ("Hexadezimalsystem", "Zahlen mit Basis 16", [], None, True),
    ("Datenschutz im Netz", "Persönliche Daten schützen", ["Datenschutz"], None, False),
    ("Python-Schleifen", "for und while üben", ["Programmierung"], None, False),
    ("SQL-Abfragen", "SELECT, WHERE und JOIN", ["Datenbanken"], None, False),
    ("Sortieralgorithmen", "Bubblesort und Quicksort", ["Algorithmen"], None, False),
    ("Die Französische Revolution", "1789 und die Folgen", [], ("Geschichte",), False),
    ("Bruchrechnen mit Pizza", "Anteile anschaulich", [], M, False),
])

PAGES = {"Optik": OPTIK, "Geometrie": GEOMETRIE, "Technische Informatik": TECHINF}
