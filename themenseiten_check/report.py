"""A self-contained HTML page: the topic page as it is, beside the reranked one.

Left: every material in page order, misfits struck through -- that column is
the point of the report, it shows what a visitor sees today. Right: what stays,
best first, with how far each one moved; the removed ones folded below.

All repository text is untrusted and escaped; only http(s) links are emitted.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from .collect import TopicPage
from .scoring import Scored

_CSS = """
:root{--ink:#1b2f5b;--paper:#fbfcfe;--line:#d9dfea;--muted:#5d6678;
--keep:#2e7d5b;--drop:#b4432f;--bar:#c9d6ee;--card:#fff;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--ink:#dfe7f7;--paper:#111a2c;--line:#2c3a55;
--muted:#9aa6bd;--keep:#6cc59b;--drop:#ef8a75;--bar:#2f4670;--card:#17233a;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}
header{padding:2.5rem 1.5rem 1.5rem;max-width:72rem;margin:auto}
h1{font:600 2rem/1.2 Georgia,"Times New Roman",serif;margin:0 0 .75rem}
h1 a{color:inherit;text-decoration-thickness:1px;text-underline-offset:.2em}
.yardstick{max-width:46rem;border-left:3px solid var(--ink);padding:.25rem 0 .25rem 1rem;
margin:0 0 1rem;color:var(--muted);white-space:pre-line}
.facts{color:var(--muted);margin:0}
.facts strong{color:var(--ink)}
main{display:grid;grid-template-columns:1fr 1fr;gap:2rem;max-width:72rem;margin:auto;padding:0 1.5rem 3rem}
@media (max-width:52rem){main{grid-template-columns:1fr}}
h2{font:600 1.15rem/1.3 Georgia,serif;margin:1rem 0 .75rem;padding-bottom:.4rem;border-bottom:1px solid var(--line)}
ol{list-style:none;margin:0;padding:0}
li{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.6rem .8rem;margin-bottom:.5rem}
li.drop{border-left:4px solid var(--drop)}
li.drop .title{text-decoration:line-through;text-decoration-color:var(--drop)}
li.keep{border-left:4px solid var(--keep)}
.row{display:flex;gap:.6rem;align-items:baseline}
.rank{color:var(--muted);min-width:2ch;text-align:right;font-variant-numeric:tabular-nums}
.title{font-weight:600;flex:1}
.title a{color:inherit}
.score{font-variant-numeric:tabular-nums;color:var(--muted);font-size:.9rem}
.bar{position:relative;height:4px;background:var(--line);border-radius:2px;margin:.4rem 0 0 2.6ch}
.bar i{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--muted)}
.bar span{display:block;height:100%;background:var(--bar);border-radius:2px}
li.keep .bar span{background:var(--keep)}li.drop .bar span{background:var(--drop)}
.meta{color:var(--muted);font-size:.85rem;margin:.3rem 0 0 2.6ch}
details{margin:.3rem 0 0 2.6ch;font-size:.9rem}
summary{cursor:pointer;color:var(--muted)}
summary:focus-visible,a:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.move{font-size:.85rem;font-variant-numeric:tabular-nums}
.up{color:var(--keep)}.down{color:var(--drop)}
.notes{max-width:72rem;margin:0 auto 1rem;padding:0 1.5rem;color:var(--muted);font-size:.9rem}
.empty{color:var(--muted);font-style:italic}
"""


def _link(title: str, url: str) -> str:
    if url.startswith(("https://", "http://")):
        return f'<a href="{escape(url)}" target="_blank" rel="noopener">{escape(title)}</a>'
    return escape(title)


def _entry(s: Scored, rank: int, threshold: float, *, show_move: bool) -> str:
    item = s.item
    state = "keep" if s.keep else "drop"
    move = ""
    if show_move:
        delta = s.original_rank - rank
        if delta:
            arrow, cls = ("▲", "up") if delta > 0 else ("▼", "down")
            move = f'<span class="move {cls}" title="vorher Platz {s.original_rank}">{arrow}{abs(delta)}</span>'
    details = ""
    if item.description or item.keywords:
        body = escape(item.description) if item.description else ""
        if item.keywords:
            body += f"<br><em>Schlagworte:</em> {escape(', '.join(item.keywords))}"
        details = f"<details><summary>Beschreibung</summary>{body}</details>"
    subjects = f", Fach: {escape(', '.join(item.subjects))}" if item.subjects else ""
    width = max(0.0, min(1.0, s.score)) * 100
    cut = max(0.0, min(1.0, threshold)) * 100
    return (
        f'<li class="{state}"><div class="row"><span class="rank">{rank}</span>'
        f'<span class="title">{_link(item.title, item.url)}</span>{move}'
        f'<span class="score">{s.score:.2f}</span></div>'
        f'<div class="bar"><span style="width:{width:.0f}%"></span>'
        f'<i style="left:{cut:.0f}%" title="Schwelle {threshold:.2f}"></i></div>'
        f'<div class="meta">{escape(item.source)}{subjects}</div>{details}</li>')


def _list(entries: list[str]) -> str:
    return f"<ol>{''.join(entries)}</ol>" if entries else '<p class="empty">Keine Inhalte.</p>'


def render(topic: TopicPage, scored: list[Scored], *, backend: str,
           threshold: float, repository: str) -> str:
    """The whole report as one HTML string."""
    original = sorted(scored, key=lambda s: s.original_rank)
    kept = [s for s in scored if s.keep]
    dropped = [s for s in scored if not s.keep]
    yardstick = escape(topic.description) or "<em>Diese Themenseite hat keinen Beschreibungstext – bewertet wird nur gegen den Titel.</em>"
    notes = "".join(f"<li>{escape(n)}</li>" for n in topic.notes)
    notes_html = f'<section class="notes"><strong>Hinweise zur Vollständigkeit</strong><ul>{notes}</ul></section>' if notes else ""
    title = escape(topic.title)

    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Themenseiten-Check: {title}</title><style>{_CSS}</style></head>
<body>
<header>
<h1>{_link(topic.title, topic.url)}</h1>
<p class="yardstick">{yardstick}</p>
<p class="facts"><strong>{len(dropped)} von {len(scored)}</strong> Inhalten passen laut Reranking nicht zum Thema.
Bewertet mit {escape(backend)} bei Schwelle {threshold:.2f}, Quelle {escape(repository)}, Stand {datetime.now():%d.%m.%Y %H:%M}.</p>
</header>
{notes_html}
<main>
<section><h2>Themenseite heute ({len(original)})</h2>
{_list([_entry(s, s.original_rank, threshold, show_move=False) for s in original])}</section>
<section><h2>Nach Reranking ({len(kept)})</h2>
{_list([_entry(s, i, threshold, show_move=True) for i, s in enumerate(kept, start=1)])}
<h2>Aussortiert ({len(dropped)})</h2>
{_list([_entry(s, s.original_rank, threshold, show_move=False) for s in dropped])}</section>
</main></body></html>
"""
