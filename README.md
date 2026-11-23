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

Anzeige: Kopf mit Titel, Beschreibung, **kompendialem Text** und
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

Bewertung, beide über die B-API:

* **LLM-Bewertung** (Standard): ein Chatmodell vergibt je Material 0–10 für
  „gehört zum Thema“, 15 Materialien je Anfrage, bis zu vier parallel.
  Repository-Texte gehen als markierte Fremddaten in den Prompt.
* **Embeddings**: Kosinus-Ähnlichkeit, schneller, gröber.

Der Cross-Encoder der Kommandozeile läuft auf Vercel nicht (torch sprengt die
Funktionsgröße).

### Deployen

```bash
npm i -g vercel
vercel                 # im Projektordner; Framework-Preset "Other"
```

Umgebungsvariablen im Vercel-Projekt setzen (siehe `.env.example`):
`B_API_BASE_URL`, `B_API_KEY` (Pflicht), `DEMO_PASSWORD` (dringend empfohlen –
sonst kann jeder mit dem Link deinen B-API-Schlüssel verbrauchen), optional
`EDU_SHARING_URL` (Standard: Staging), `EDU_SHARING_USER`/`EDU_SHARING_PASSWORD`,
`LLM_MODEL`, `EMBEDDING_MODEL`, `PORTAL_HOSTS`. Ist `DEMO_PASSWORD` gesetzt, erscheint das
Passwortfeld nach dem ersten abgewiesenen Versuch.

Lokal: `vercel dev`. Die Funktion darf 300 s laufen (`vercel.json`, auf dem
Hobby-Plan das Maximum mit Fluid Compute). Gelesen werden bis zu 400 Inhalte;
mit LLM-Bewertung dauert eine Seite wie Optik (rund 360 Inhalte) ein bis zwei
Minuten, mit Embeddings Sekunden.
`.vercelignore` blendet `pyproject.toml` aus, damit Vercel aus
`requirements.txt` baut (Client auf einen Commit festgelegt).

## Kommandozeile

### Installation

```bash
pip install -e ".[local,dev]"   # mit lokalem Cross-Encoder (zieht torch)
pip install -e ".[dev]"         # nur B-API-Embeddings
```

Python 3.11+. Umgebungsvariablen siehe `.env.example`; ohne Angaben läuft alles
anonym gegen Staging.

### Benutzung

```bash
themenseiten-check --search Optik --subject Physik # IDs von Themenseiten finden
themenseiten-check <sammlungs-id>                  # -> reports/<id>.html
themenseiten-check <sammlungs-id> --subcollections --threshold 0.4
themenseiten-check <sammlungs-id> --backend bapi   # braucht B_API_BASE_URL, B_API_KEY
```

| Option | Bedeutung |
|---|---|
| `--backend local` | Cross-Encoder `BAAI/bge-reranker-v2-m3` (mehrsprachig, lokal, Score 0–1, Schwelle 0.30). Standard. |
| `--backend bapi` | Kosinus-Ähnlichkeit von Embeddings über die B-API, Provider `openai` (Schwelle 0.35). |
| `--backend llm` | LLM-Bewertung über die B-API wie in der Web-App (Schwelle 0.50). |
| `--searches` | gespeicherte Suchen der Widgets angenähert ausführen |
| `--model` | anderes Modell für das gewählte Backend |
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
| `backends.py` | Cross-Encoder, B-API-Embeddings, LLM-Bewertung |
| `report.py` | HTML-Bericht der Kommandozeile |
| `cli.py` | Kommandozeile |
| `api/index.py` | Vercel-Funktion: `POST /api/analyze`, `POST /api/search` |
| `index.html` | Oberfläche der Web-App |

## Tests

```bash
pytest
```

Offline: `test_check.py` mit einem Fake-Repositorium, `test_integration.py` geht
durch die echte Bibliothek mit gemocktem HTTP, `test_portal.py` die Auflösung von Portal-URLs samt Host-Sperre, `test_web.py` prüft URL-Erkennung,
LLM-Bewertung und den Endpunkt. Für die Web-Tests zusätzlich `pip install fastapi`.

## Welche Teile des Clients benutzt werden

| Frage | Aufruf |
|---|---|
| Themenseiten finden | `repo.flows.find_collections(text, subject=…, properties=…)` (WLO-Themenseiten sind Sammlungen; `find_pages` findet nur die mit Seitenlayout) |
| Texte der Themenseite | `repo.flows.describe(id)` – Beschreibung, Schlagworte, kompendialer Text über `repo.metadata_profile.compendium_property` |
| Schwimmlinien | `repo.flows.page(id, resolve_widgets=True)` |
| Inhalte | `repo.flows.collection_contents(id, …)`, `repo.flows.describe_many(ids)`, gespeicherte Suchen über `repo.flows.search(…, raw_filters=…)` |
| Bewertung | `BildungsAPI.chat`, `BildungsAPI.embeddings`, `edusharing.agent.as_untrusted` |

## Grenzen

Die Schwimmlinien der WordPress-Seite selbst (etwa „Die neusten geprüften Inhalte
im gesamten Themenbereich“) sind Blöcke des Portals, keine Daten in edu-sharing,
und werden nicht nachgebildet. Gezeigt werden die Schwimmlinien des
edu-sharing-Seitenlayouts, falls die Sammlung eines hat, sowie Sammlung und
Unterthemen.

Die Schwellen sind Startwerte, nicht kalibriert. Vor einer Entscheidung über
Inhalte ein paar Seiten durchsehen und `--threshold` je Fach anpassen.
