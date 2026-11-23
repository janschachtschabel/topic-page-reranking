"""Command line: one topic page in, one HTML comparison out.

    themenseiten-check <collection-id>                   # text statistics, no setup
    themenseiten-check <collection-id> --backend local   # cross-encoder (needs .[local])
    themenseiten-check --search Optik --subject Physik # find topic page ids
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from edusharing import AsyncRepository, EduSharingError

from .backends import DEFAULT_CROSS_ENCODER, CrossEncoderReranker
from .collect import load_topic_page
from .report import render
from .scoring import DEFAULT_TOPIC_TEXT, TOPIC_TEXTS, rerank
from .search import find_topic_pages
from .textstats import TextStatsReranker

STAGING = "https://repository.staging.openeduhub.net"


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="themenseiten-check", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("collection_id", nargs="?", help="ID der Sammlung hinter der Themenseite")
    p.add_argument("--search", metavar="BEGRIFF", help="Themenseiten zu einem Begriff auflisten und beenden")
    p.add_argument("--subject", default="", help="Fach für --search, z. B. Physik")
    p.add_argument("--backend", choices=("textstats", "local"), default="textstats")
    p.add_argument("--model", help=f"nur für --backend local, Vorgabe {DEFAULT_CROSS_ENCODER}")
    p.add_argument("--topic-text", choices=TOPIC_TEXTS, default=DEFAULT_TOPIC_TEXT,
                   help="Woran gemessen wird: Beschreibung, kompendialer Text oder beides")
    p.add_argument("--threshold", type=float, help="Mindestscore für 'passend' (Vorgabe je Backend)")
    p.add_argument("--max-items", type=int, default=200)
    p.add_argument("--subcollections", action="store_true", help="Inhalte direkter Untersammlungen einbeziehen")
    p.add_argument("--no-widgets", action="store_true", help="Seitenlayout (Schwimmlinien) ignorieren")
    p.add_argument("--searches", action="store_true",
                   help="Gespeicherte Suchen der Widgets angenähert ausführen")
    p.add_argument("--out", type=Path, help="Zieldatei (Vorgabe: reports/<id>.html)")
    args = p.parse_args(argv)
    if not args.collection_id and not args.search:
        p.error("collection_id oder --search angeben")
    if args.max_items < 1:
        p.error("--max-items muss mindestens 1 sein")
    return args


def _repository() -> AsyncRepository:
    url = os.environ.get("EDU_SHARING_URL", STAGING)
    user = os.environ.get("EDU_SHARING_USER", "")
    auth = (user, os.environ.get("EDU_SHARING_PASSWORD", "")) if user else None
    return AsyncRepository(url, auth=auth,
                           metadataset=os.environ.get("EDU_SHARING_METADATASET", "mds_oeh"))


async def _list_pages(repo: AsyncRepository, term: str, subject: str) -> int:
    found = await find_topic_pages(repo, term, subject=subject)
    marks = {True: "ja", False: "nein", None: "?"}
    for hit in found["hits"]:
        print(f"{hit['id']}  {hit['title']}  (Kompendientext: {marks[hit['has_compendium']]}, "
              f"Seitenlayout: {marks[hit['has_layout']]})")
    if not found["hits"]:
        print("Keine Sammlung gefunden.", file=sys.stderr)
    for u in found["unresolved"]:
        print(f"Hinweis: Fach {u.get('value')!r} unbekannt, nicht angewandt "
              f"(Vorschläge: {', '.join(u.get('suggestions') or []) or 'keine'})", file=sys.stderr)
    return 0


async def _run(args: argparse.Namespace) -> int:
    async with AsyncExitStack() as stack:
        repo = await stack.enter_async_context(_repository())
        if args.search:
            return await _list_pages(repo, args.search, args.subject)

        reranker = (CrossEncoderReranker(args.model or DEFAULT_CROSS_ENCODER)
                    if args.backend == "local" else TextStatsReranker())

        topic = await load_topic_page(repo, args.collection_id, max_items=args.max_items,
                                      include_subcollections=args.subcollections,
                                      include_widgets=not args.no_widgets,
                                      run_searches=args.searches)
        print(f"{topic.title}: {len(topic.items)} Inhalte geladen", file=sys.stderr)
        threshold = reranker.default_threshold if args.threshold is None else args.threshold
        scored = await rerank(topic, reranker, threshold, source=args.topic_text)

        out = args.out or Path("reports") / f"{topic.id}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(topic, scored, backend=reranker.name, threshold=threshold,
                              repository=repo.url), encoding="utf-8")
        dropped = sum(not s.keep for s in scored)
        print(f"{dropped} von {len(scored)} unpassend -> {out}", file=sys.stderr)
        for note in topic.notes:
            print(f"  Hinweis: {note}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        return asyncio.run(_run(args))
    except EduSharingError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
