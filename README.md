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

**Wo die Texte herkommen:** Die Beschreibung liest die App wie der WLO-MCP-Server
(`getTopicPageContent`): zuerst `cclom:general_description` der Sammlung, dann
`collection.description` aus dem `collection`-Objekt der Knoten-Metadaten – das,
was die Themenseite als Intro über den Schwimmlinien zeigt –, zuletzt
`cm:description`. Die Quelle steht im Kopf der Seite. Der kompendiale Text
(`ccm:oeh_collection_compendium_text`) ist Markdown; Inhaltsverzeichnis,
Überschriftenzeichen und Links werden vor Anzeige und Abgleich entfernt.

**Woran gemessen wird, ist wählbar:** an der **Beschreibung der Themenseite**
(Standard, kürzer), an den **Überschriften des kompendialen Texts**, am
**kompendialen Text** oder an **Beschreibung und kompendialem Text** – jeweils
zusammen mit Titel und Schlagworten. Fehlt der gewählte Text, springt der andere
ein, und die Seite sagt das. Im Kopf stehen beide Texte; der genutzte ist
markiert und aufgeklappt. Kommandozeile: `--topic-text description|headings|compendium|both`.

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
* **Seitenzusammenhalt**: Jedes Material wird außerdem mit allen *anderen*
  Materialien der Seite zusammen verglichen. Fehlzuordnungen sind Ausreißer
  unter ihren Nachbarn, auch wenn sie mit dem Thementext Wörter teilen. Der Score
  ist je zur Hälfte Themennähe und Zusammenhalt, beide relativ zum Median der
  Seite (1,0 = so nah wie ein typischer Inhalt dieser Seite).
* **Fachabgleich**: Ein Material, dessen Fächer keines mit der Themenseite
  teilen (Mathematik auf der Optik-Seite), bekommt höchstens 0,4.
* **Schwelle 0,60** als Ausgangswert. Die Annahme dahinter: Der Großteil einer
  Seite passt. Ohne Modell gibt es keinen absoluten Maßstab.

Aus dem Reranker von wlo-mcp-sc übernommen: Felder unterschiedlich gewichten,
Stoppwörter ignorieren. Nicht übernommen: den Bonus für Metadatenqualität
(Lizenz, Vorschaubild, Herausgeber) – er bewertet, ob ein Datensatz gut gepflegt
ist, nicht, ob er zum Thema gehört.

**Abgestimmt** auf drei selbst geschriebene Beispielseiten (`tests/eval_pages.py`:
Optik, Räumliche und ebene Geometrie, Technische Informatik; je 10–12 passende,
6 unpassende Materialien, darunter fachgleiche wie Mechanik auf der Optik-Seite):

| Variante, gegen die Beschreibung | Reihung (AUC, Mittel) | bestes F1 für Unpassendes |
|---|---|---|
| nur Themennähe | 0,975 | 0,89 bei 0,40 |
| Themennähe und Zusammenhalt je zur Hälfte | **1,00** | **0,97 bei 0,60** |
| Zusammenhalt überwiegt (0,3 / 0,7) | 0,96 | 0,87 bei 0,75 |

Mit Schwelle 0,60 bleiben alle 33 passenden Materialien, 17 von 18 unpassenden
fallen heraus (übrig bleibt „Elektrischer Stromkreis“ auf Optik, 0,61). Gegen
den vollständigen kompendialen Text ist die Reihung schwächer (AUC 0,965).
Der Abstand an der Schwelle ist knapp, und die Seiten sind klein und für diesen
Zweck geschrieben: Die Zahlen vergleichen Varianten, sie belegen keine Qualität
auf echten Seiten. Ausprobiert und verworfen: 3-Gramme, Endungen abschneiden,
Unterthemen-Titel als Themenvokabular.

**Auf echten Seiten einstellen:** Jeder Inhalt hat die Knöpfe „passt“ und
„passt nicht“. Die Markierungen bleiben je Themenseite im Browser; die Leiste
oben zeigt dann die Reihungsgüte (AUC), wie viele unpassende die aktuelle
Schwelle erkennt und wie viele passende sie fälschlich aussortiert, schlägt die
Schwelle vor, die die Markierungen am besten trennt, und exportiert sie als
JSON – Grundlage, um Schwelle und Gewichte an echten Seiten nachzustellen.

### Alternative: kleiner Cross-Encoder (ONNX, int8)

Unter „Optionen → Verfahren“ lässt sich statt der Textstatistik ein
Cross-Encoder oder die Kombination aus beiden wählen. Der Cross-Encoder ist `cross-encoder/msmarco-MiniLM-L12-en-de-v1` (Deutsch/
Englisch), int8-quantisiert und als ONNX ausgeliefert, ohne torch. Er bewertet
jedes Paar (Thema, Material) zusammen und liest davon höchstens 256 Token; der
Thementext wird auf 96 Token gekürzt. Für ihn eignen sich die **Beschreibung**
oder die **Überschriften des kompendialen Texts** (h1–h3, ohne Inhaltsverzeichnis,
Dokumenttitel, „Teil 1 …“ und die nummerierten Schema-Abschnitte); vom
vollständigen kompendialen Text sähe er nur die ersten Zeilen, die Seite weist
darauf hin.

Seine Logits liegen bei „Thema gegen Material“ weit unter null und verschieben
sich von Seite zu Seite (msmarco ist auf Suchanfragen trainiert). Sie werden
deshalb auf die Skala der Seite gelegt: Median 0,5, zwei mittlere Abweichungen
darüber 1,0, darunter 0,0; Fachfremdes ×0,4.

**Gemessen mit dem ausgelieferten int8-Modell** an den Beispielseiten
(Beschreibung als Thementext; `tests/test_model_real.py` wacht darüber):

| Verfahren | Reihung (AUC) Optik / Geometrie / Techn. Informatik | Standardschwelle: passende behalten, unpassende raus |
|---|---|---|
| Textstatistik | 1,00 / 1,00 / 1,00 | 0,60: 33/33, 17/18 |
| Cross-Encoder | 0,93 / 0,91 / 0,82 | 0,25: 29/33, 13/18 |
| Kombination (je zur Hälfte) | 1,00 / 1,00 / 0,92 | 0,40: 32/33, 16/18 |

Mit dem vollständigen kompendialen Text als Thema fiel der Cross-Encoder auf
0,75 / 0,83 / 0,57 – er sieht nur dessen Anfang. Rechenzeit rund 25 ms je
Material, also etwa 10 s für 400; das Laden des Modells beim ersten Aufruf
etwa 1 s. Die Beispielseiten sind beim Abstimmen der Textstatistik entstanden
und begünstigen sie; welches Verfahren auf echten Seiten trägt, zeigen die
Markierungen „passt / passt nicht“ in der App.

**Modell im Repository** (`model/`, 121,5 MiB, liegt bei) – kein Git LFS nötig:
GitHub lehnt Dateien über 100 MiB ab; die größte Datei, das mehrsprachige
Vokabular, hat 91,6 MiB. `scripts/prepare_model.py` lädt es von Hugging Face,
quantisiert es dynamisch auf int8 und schreibt die Gewichte als ONNX-„external
data“ in Teildateien `weights-NN.bin` unter 95 MiB (ASCII-Namen, damit ein
Windows-Checkout sie anlegen kann); `manifest.json` hält Quelle, Revision,
Prüfsummen und einen fp32-gegen-int8-Vergleich fest.

Am einfachsten über GitHub: **Actions → „Modell vorbereiten“ → Run workflow**.
Die Action baut `model/` und committet es; Vercel deployt danach von selbst.
Lokal geht es mit `pip install -r scripts/requirements-model.txt` und
`python scripts/prepare_model.py`. Ohne `model/` ist der Cross-Encoder in der
App ausgegraut und nennt den Grund.

Größe auf Vercel: Modell ~118 MB plus onnxruntime, tokenizers, numpy (~160 MB)
und die übrigen Pakete – unter den 500 MB, die Vercel Python-Funktionen
erlaubt. Der erste Aufruf einer Instanz lädt das Modell (einige Sekunden),
danach bleibt es im Speicher. Lokal ohne ONNX: `--backend local` nutzt
sentence-transformers mit `BAAI/bge-reranker-v2-m3`.

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
| `--backend textstats` | Textstatistik wie in der Web-App (Schwelle 0.60). Standard. |
| `--backend onnx` | mitgelieferter int8-Cross-Encoder aus `model/` (Schwelle 0.25). |
| `--backend hybrid` | Textstatistik und Cross-Encoder je zur Hälfte (Schwelle 0.40). |
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
| `topic_text.py` | Beschreibung wie der MCP-Server lesen, Markdown zu Klartext |
| `urls.py` | Knoten-ID aus einer edu-sharing-URL oder ID lesen |
| `portal.py` | Portal-URL (WordPress) auf die Sammlung dahinter auflösen |
| `serialize.py` | Ergebnis als JSON für die Web-App |
| `search.py` | Themenseiten über die Sammlungssuche finden |
| `scoring.py` | Abfrage- und Dokumenttext bauen, sortieren, Schwelle anwenden |
| `textstats.py` | Textstatistik: n-Gramme, TF-IDF, Fachabgleich |
| `backends.py` | lokaler Cross-Encoder (sentence-transformers) |
| `onnx_reranker.py` | int8-Cross-Encoder aus `model/` (ONNX Runtime) |
| `scripts/prepare_model.py` | Modell laden, quantisieren, in Teile unter 100 MiB schreiben |
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
