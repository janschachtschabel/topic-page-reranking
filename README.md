# themenseiten-check

Vergleicht eine WLO-Themenseite, wie sie heute aussieht, mit einer nachbewerteten
Fassung: Jeder Inhalt wird per Reranking gegen das Thema der Seite gestellt –
Titel, **Beschreibungstext** und Schlagworte der Sammlung –, unpassende Inhalte
fallen heraus. Ergebnis ist eine HTML-Datei mit beiden Seiten nebeneinander.

Zwei Wege: eine **Web-App für Vercel**, in die man die URL einer Themenseite
eingibt, und eine **Kommandozeile**, die einen HTML-Bericht schreibt.

Gebaut auf [edu-sharing-python-client](https://github.com/janschachtschabel/edu-sharing-python-client),
voreingestellt auf WLO Staging (`https://repository.staging.openeduhub.net`, `mds_oeh`).

## Web-App (Vercel)

Zwei Wege zur Themenseite: **suchen** (Thema, optional Fach) über die
Repository-API – jeder Treffer zeigt, ob die Sammlung einen kompendialen Text
und ein Seitenlayout trägt, ein Klick prüft sie – oder **direkt eingeben**:
die Portal-URL der Themenseite, z. B.
`https://wordpress.staging.openeduhub.net/portal/physik-optik/`, eine
edu-sharing-URL oder die Sammlungs-ID.

Eine Portal-URL enthält keine Knoten-ID. Der Server ruft die Seite deshalb ab,
sammelt alle UUIDs aus ihrem HTML, fragt das Repositorium, welche davon
Sammlungen sind, und nimmt die mit dem Titel der `<h1>`. Wo das nicht eindeutig
ist, sucht er Sammlungen mit diesem Titel; bleibt es mehrdeutig, bietet die App
die Kandidaten zur Auswahl an, statt zu raten. Abgerufen werden nur Seiten von
`PORTAL_HOSTS` (Standard: `wordpress.staging.openeduhub.net`,
`wirlernenonline.de`, `www.wirlernenonline.de`), nur über https, Weiterleitungen
werden bei jedem Schritt erneut geprüft. Edu-sharing-URLs und IDs werden nicht
abgerufen, nur gelesen.

Vorausgefüllt ist die Optik-Seite; drei Beispiel-Knöpfe (Physik: Optik,
Mathematik: Räumliche und ebene Geometrie, Informatik: Technische Informatik)
prüfen mit einem Klick.

**Woran gemessen wird, ist wählbar:** an der **Beschreibung der Themenseite**
(Standard, kürzer), am **kompendialen Text** oder an **beidem** – jeweils
zusammen mit Titel und Schlagworten. Fehlt der gewählte Text, springt der andere
ein, und die Seite sagt das. Im Kopf stehen beide Texte; der genutzte ist
markiert und aufgeklappt. Kommandozeile: `--topic-text description|compendium|both`.

Anzeige: Kopf mit Titel, Beschreibung, kompendialem Text und
Schlagworten, darunter jede
Schwimmlinie der Seite mit zwei Spalten – **Ist-Stand** in Seitenreihenfolge
(unpassende rot durchgestrichen) und **Nach Reranking** nach Score, mit
Platzänderung und eingeklappter Liste „Aussortiert“. Danach die direkt
zugeordneten Sammlungsinhalte und je Unterthema eine eigene Schwimmlinie
(Standard an, wie auf der Portalseite). Die Zahl der direkt zugeordneten Inhalte
lässt sich mit der Angabe der Portalseite vergleichen („163 redaktionell
geprüfte Bildungsinhalte“). Die Schwelle
ist ein Schieberegler und wirkt sofort, ohne neu zu bewerten. Die zuletzt
geprüften Themenseiten stehen als Knöpfe unter dem Eingabefeld.

Woran gemessen wird: Titel, Beschreibung, Schlagworte und der kompendiale Text
der Sammlung (`ccm:oeh_collection_compendium_text`, über das Metadatenprofil
des Clients). Die Beschreibung ist auf Portalseiten oft allgemein („Hier
findest du zahlreiches kostenloses Material für Physik!“); der kompendiale
Text sagt, was das Thema ist.

Bewertung – kostenfrei, ohne Schlüssel und ohne Modell, per **Textstatistik**:

* **Zeichen-n-Gramme (4 und 5 Zeichen) mit TF-IDF**: Thema (Titel ×3,
  Schlagworte ×2, gewählter Text ×1,5) und Material
  (Titel ×2, Schlagworte ×1,5, Beschreibung ×1) werden als Vektoren verglichen,
  die Wortgewichte aus allen Inhalten der Seite. n-Gramme statt ganzer Wörter
  wegen deutscher Komposita und Beugung: „Linsen“ und „Sammellinse“ teilen kein
  Wort, aber „lins“, „inse“, „linse“.
* **Fachabgleich**: Ein Material, dessen Fächer keines mit der Themenseite
  teilen (Mathematik auf der Optik-Seite), bekommt höchstens 0,4.
* **Skala**: 1,0 heißt „so nah am Thema wie ein typischer Inhalt dieser Seite“
  (Median), Standardschwelle 0,5. Die Annahme dahinter: Der Großteil einer Seite
  passt. Ohne Modell gibt es keinen absoluten Maßstab.

Aus dem Reranker von wlo-mcp-sc übernommen: Felder unterschiedlich gewichten,
Stoppwörter ignorieren. Nicht übernommen: den Bonus für Metadatenqualität
(Lizenz, Vorschaubild, Herausgeber) – er bewertet, ob ein Datensatz gut gepflegt
ist, nicht, ob er zum Thema gehört.

Wie gut das ist: Auf einer selbst gebauten Beispielseite (Optik, 11 passende,
7 unpassende Materialien, `tests/test_textstats.py`) liegt die Reihung gegen den
kompendialen Text bei AUC 0,97 (alle passenden bleiben, alle fachfremden fallen
heraus), gegen eine allgemeine Portal-Beschreibung („Hier findest du …
Material für Physik“) bei 0,92 – dann tragen Titel und Schlagworte fast allein. Fachgleiche
Fehlzuordnungen (Mechanik, Akustik auf der Optik-Seite) bleiben dagegen über der
Schwelle – dafür reicht Wortstatistik nicht. Die Beispielseite ist klein und für
diesen Zweck geschrieben; die Zahl ist ein Wächter gegen Verschlechterung, kein
Qualitätsnachweis. Ausprobiert und verworfen: 3-Gramme, Endungen abschneiden,
Unterthemen-Titel als Themenvokabular – alle drei verschlechterten die Reihung.

Der Cross-Encoder der Kommandozeile (lokal, ebenfalls kostenfrei) läuft auf
Vercel nicht: torch sprengt die Funktionsgröße.

### Deployen

```bash
npm i -g vercel
vercel                 # im Projektordner; Vercel erkennt die FastAPI-App selbst
```

Umgebungsvariablen sind keine nötig. Optional (siehe `.env.example`):
`DEMO_PASSWORD` – jede Prüfung schickt einige hundert Anfragen an das
Repositorium; mit Passwort erscheint das Feld beim Laden –, `EDU_SHARING_URL`
(Standard: Staging), `EDU_SHARING_USER`/`EDU_SHARING_PASSWORD`, `PORTAL_HOSTS`.

Lokal: `vercel dev`. Die Funktion darf 300 s laufen (`vercel.json`, auf dem
Hobby-Plan das Maximum mit Fluid Compute). Gelesen werden bis zu 400 Inhalte;
die Zeit geht fast ganz auf das Lesen, die Bewertung selbst braucht
Millisekunden.
`.vercelignore` blendet `pyproject.toml` aus, damit Vercel aus
`requirements.txt` baut (Client auf einen Commit festgelegt).

## Kommandozeile

### Installation

```bash
pip install -e ".[dev]"         # Textstatistik
pip install -e ".[local,dev]"   # zusätzlich lokaler Cross-Encoder (zieht torch)
```

Python 3.11+. Umgebungsvariablen siehe `.env.example`; ohne Angaben läuft alles
anonym gegen Staging.

### Benutzung

```bash
themenseiten-check --search Optik --subject Physik # IDs von Themenseiten finden
themenseiten-check <sammlungs-id>                  # -> reports/<id>.html
themenseiten-check <sammlungs-id> --subcollections --threshold 0.4
themenseiten-check <sammlungs-id> --backend local  # Cross-Encoder
```

| Option | Bedeutung |
|---|---|
| `--backend textstats` | Textstatistik wie in der Web-App (Schwelle 0.50). Standard. |
| `--backend local` | Cross-Encoder `BAAI/bge-reranker-v2-m3` (mehrsprachig, lokal, kostenfrei, Score 0–1, Schwelle 0.30). |
| `--searches` | gespeicherte Suchen der Widgets angenähert ausführen |
| `--model` | anderes Modell für `--backend local` |
| `--threshold` | Mindestscore für „passend“ |
| `--subcollections` | Inhalte direkter Untersammlungen mitbewerten |
| `--no-widgets` | Seitenlayout (Schwimmlinien) ignorieren |
| `--max-items` | Obergrenze gelesener Inhalte (Standard 200) |

## Was gelesen wird – und was nicht

* die Materialien der Sammlung (Referenzen; bewertet wird der Datensatz dahinter),
* optional die direkten Untersammlungen,
* Widgets im Seitenlayout mit fester Liste (`sortedNodeIds`),
* Widgets mit gespeicherter Suche – nur auf Wunsch (Web-App: Standard an,
  CLI: `--searches`) und **angenähert**: Der Client führt sie bewusst nicht aus,
  weil ihre Filter `virtual:`-Felder enthalten, die das Metadatenset nicht kennt.
  Hier werden diese Felder weggelassen, die übrigen Filter mit ihren gespeicherten
  Werten gesendet und die ersten 24 Treffer gezeigt. Was weggelassen oder vom
  Repositorium nicht angewandt wurde, steht an der Schwimmlinie. Das Format von
  `propertyFilters` ist nicht dokumentiert; verwendet wird nur
  `{Eigenschaft: Wert | [Werte]}`.

Widgets ohne Knotenbezug (z. B. Redaktionsmitglieder) erscheinen als
„kein auswertbarer Inhalt“. Kürzungen und nicht mehr abrufbare Knoten stehen
unter den Hinweisen.

## Aufbau

| Datei | Aufgabe |
|---|---|
| `collect.py` | Themenseite in Schwimmlinien lesen: Seitenlayout, Sammlung, Untersammlungen |
| `saved_search.py` | gespeicherte Widget-Suchen angenähert ausführen |
| `urls.py` | Knoten-ID aus einer edu-sharing-URL oder ID lesen |
| `portal.py` | Portal-URL (WordPress) auf die Sammlung dahinter auflösen |
| `serialize.py` | Ergebnis als JSON für die Web-App |
| `search.py` | Themenseiten über die Sammlungssuche finden |
| `scoring.py` | Abfrage- und Dokumenttext bauen, sortieren, Schwelle anwenden |
| `textstats.py` | Textstatistik: n-Gramme, TF-IDF, Fachabgleich |
| `backends.py` | lokaler Cross-Encoder |
| `report.py` | HTML-Bericht der Kommandozeile |
| `cli.py` | Kommandozeile |
| `api/index.py` | Vercel-Funktion: `POST /api/analyze`, `POST /api/search` |
| `public/index.html` | Oberfläche der Web-App |

## Tests

```bash
pytest
```

Offline: `test_check.py` mit einem Fake-Repositorium, `test_integration.py` geht
durch die echte Bibliothek mit gemocktem HTTP, `test_portal.py` die Auflösung von Portal-URLs samt Host-Sperre, `test_textstats.py` die
Textstatistik samt Beispielseite, `test_web.py` URL-Erkennung und Endpunkte. Für die Web-Tests zusätzlich `pip install fastapi`.

## Welche Teile des Clients benutzt werden

| Frage | Aufruf |
|---|---|
| Themenseiten finden | `repo.flows.find_collections(text, subject=…, properties=…)` (WLO-Themenseiten sind Sammlungen; `find_pages` findet nur die mit Seitenlayout) |
| Texte der Themenseite | `repo.flows.describe(id)` – Beschreibung, Schlagworte, kompendialer Text über `repo.metadata_profile.compendium_property` |
| Schwimmlinien | `repo.flows.page(id, resolve_widgets=True)` |
| Inhalte | `repo.flows.collection_contents(id, …)`, `repo.flows.describe_many(ids)`, gespeicherte Suchen über `repo.flows.search(…, raw_filters=…)` |
| Bewertung | Stoppwörter aus `edusharing.language.GERMAN` |

## Grenzen

Die Schwimmlinien der WordPress-Seite selbst (etwa „Die neusten geprüften Inhalte
im gesamten Themenbereich“) sind Blöcke des Portals, keine Daten in edu-sharing,
und werden nicht nachgebildet. Gezeigt werden die Schwimmlinien des
edu-sharing-Seitenlayouts, falls die Sammlung eines hat, sowie Sammlung und
Unterthemen.

Die Schwellen sind Startwerte, nicht kalibriert. Vor einer Entscheidung über
Inhalte ein paar Seiten durchsehen und `--threshold` je Fach anpassen.
