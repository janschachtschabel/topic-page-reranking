"""Vercel entrypoint: one FastAPI app (Vercel's FastAPI preset finds it here and
routes every path to it; files in public/ are served first).
POST /api/analyze scores one topic page, POST /api/search finds topic pages.

No rewrite in vercel.json: a rewrite to /api/index replaces the path the app
sees, and every route then answers 404 (measured on the first deployment).

The repository comes from the environment (see .env.example); a request can
only choose *which* topic page is read. Scoring is text statistics: free, no
key, no model. With DEMO_PASSWORD set, every request must carry it -- the app
costs nothing, but each check sends a few hundred requests to the repository.
"""

import hmac
import logging
import os
import sys
from pathlib import Path
from typing import Literal

# The package lives next to api/, not on the function's import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Header, Request as HTTPRequest  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from edusharing import AsyncRepository, EduSharingError, NotFoundError  # noqa: E402
from edusharing.agent.safety import UnsafeUrlError  # noqa: E402
from themenseiten_check.collect import load_topic_page  # noqa: E402
from themenseiten_check.scoring import rerank  # noqa: E402
from themenseiten_check.search import find_topic_pages  # noqa: E402
from themenseiten_check.serialize import page_as_json  # noqa: E402
from themenseiten_check.onnx_reranker import HybridReranker, OnnxCrossEncoder, unavailable  # noqa: E402
from themenseiten_check.textstats import TextStatsReranker  # noqa: E402
from themenseiten_check.portal import DEFAULT_HOSTS, NeedsChoice, resolve  # noqa: E402

STAGING = "https://repository.staging.openeduhub.net"
#: Vercel serves public/ itself -- unless it detects this FastAPI app and routes
#: every path here, which is what it does today. Then this route answers "/".
INDEX = Path(__file__).resolve().parent.parent / "public" / "index.html"
log = logging.getLogger("themenseiten")
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class Request(BaseModel):
    url: str = Field(max_length=2000)
    topic_text: Literal["description", "headings", "compendium", "both"] = "description"
    backend: Literal["textstats", "crossencoder", "hybrid"] = "textstats"
    subcollections: bool = True
    run_searches: bool = True
    max_items: int = Field(default=400, ge=1, le=600)


class SearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    subject: str = Field(default="", max_length=100)


def _authorized(given: str) -> bool:
    expected = os.environ.get("DEMO_PASSWORD", "")
    return not expected or hmac.compare_digest(given, expected)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.exception_handler(404)
async def not_found(request: HTTPRequest, exc: Exception) -> JSONResponse:
    # Name the path the app actually received: when routing goes wrong this is
    # the one fact that shows where.
    return _error(404, f"Keine Route für {request.method} {request.url.path}.")


def _hosts() -> tuple[str, ...]:
    """Portal hosts the server may fetch; PORTAL_HOSTS overrides, comma-separated."""
    raw = os.environ.get("PORTAL_HOSTS", "")
    return tuple(h.strip().lower() for h in raw.split(",") if h.strip()) or DEFAULT_HOSTS


def _repository() -> AsyncRepository:
    user = os.environ.get("EDU_SHARING_USER", "")
    auth = (user, os.environ.get("EDU_SHARING_PASSWORD", "")) if user else None
    return AsyncRepository(os.environ.get("EDU_SHARING_URL", STAGING), auth=auth,
                           metadataset=os.environ.get("EDU_SHARING_METADATASET", "mds_oeh"))


@app.get("/", include_in_schema=False)
async def index():
    if not INDEX.is_file():
        return _error(500, "public/index.html fehlt im Deployment.")
    return FileResponse(INDEX, media_type="text/html; charset=utf-8")


@app.get("/api/config")
async def config():
    """What the page needs to know before the first request."""
    reason = unavailable()
    return {"password": bool(os.environ.get("DEMO_PASSWORD")),
            "crossencoder": {"available": not reason, "reason": reason}}


@app.post("/api/search")
async def search(body: SearchRequest, x_demo_password: str = Header(default="")):
    if not _authorized(x_demo_password):
        return _error(401, "Passwort fehlt oder ist falsch.")
    try:
        async with _repository() as repo:
            return await find_topic_pages(repo, body.text.strip(), subject=body.subject.strip())
    except EduSharingError as exc:
        log.warning("edu-sharing error: %s", exc)
        return _error(502, f"{type(exc).__name__}: {exc}")


@app.post("/api/analyze")
async def analyze(body: Request, x_demo_password: str = Header(default="")):
    if not _authorized(x_demo_password):
        return _error(401, "Passwort fehlt oder ist falsch.")
    collection_id = ""
    try:
        async with _repository() as repo:
            try:
                collection_id = await resolve(repo, body.url, hosts=_hosts())
            except NeedsChoice as exc:
                return JSONResponse({"error": str(exc), "choices": exc.choices}, status_code=409)
            except (ValueError, UnsafeUrlError) as exc:
                return _error(400, str(exc))
            topic = await load_topic_page(repo, collection_id, max_items=body.max_items,
                                          include_subcollections=body.subcollections,
                                          run_searches=body.run_searches)
            if body.backend in ("crossencoder", "hybrid"):
                reason = unavailable()
                if reason:
                    return _error(503, f"Cross-Encoder nicht verfügbar: {reason}")
                reranker = OnnxCrossEncoder() if body.backend == "crossencoder" else HybridReranker()
            else:
                reranker = TextStatsReranker()
            scored = await rerank(topic, reranker, source=body.topic_text)
            return page_as_json(topic, scored, backend=reranker.name,
                                threshold=reranker.default_threshold, repository=repo.url,
                                topic_text=body.topic_text)
    except NotFoundError:
        return _error(404, f"Keine Sammlung mit der ID {collection_id or '?'} gefunden.")
    except EduSharingError as exc:
        # Library messages carry no stack traces and no secrets; the type helps the reader.
        log.warning("edu-sharing error: %s", exc)
        return _error(502, f"{type(exc).__name__}: {exc}")
