"""Vercel function behind /api/* (one ASGI app, routed by vercel.json):
POST /api/analyze scores one topic page, POST /api/search finds topic pages.

Repository and B-API come from the environment (see .env.example); a request
can only choose *which* topic page and how it is scored. With DEMO_PASSWORD
set, every request must carry it -- otherwise anyone with the link spends the
B-API budget.
"""

import hmac
import logging
import os
import sys
from pathlib import Path
from typing import Literal

# The package lives next to api/, not on the function's import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Header  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from edusharing import AsyncRepository, EduSharingError, NotFoundError  # noqa: E402
from edusharing.agent.safety import UnsafeUrlError  # noqa: E402
from edusharing.bapi import BildungsAPI  # noqa: E402
from themenseiten_check.backends import EmbeddingReranker, LLMReranker  # noqa: E402
from themenseiten_check.collect import load_topic_page  # noqa: E402
from themenseiten_check.scoring import rerank  # noqa: E402
from themenseiten_check.search import find_topic_pages  # noqa: E402
from themenseiten_check.serialize import page_as_json  # noqa: E402
from themenseiten_check.portal import DEFAULT_HOSTS, NeedsChoice, resolve  # noqa: E402

STAGING = "https://repository.staging.openeduhub.net"
#: Vercel serves public/ itself -- unless it detects this FastAPI app and routes
#: every path here, which is what it does today. Then this route answers "/".
INDEX = Path(__file__).resolve().parent.parent / "public" / "index.html"
log = logging.getLogger("themenseiten")
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class Request(BaseModel):
    url: str = Field(max_length=2000)
    backend: Literal["llm", "embeddings"] = "llm"
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
            async with BildungsAPI.from_env() as api:
                reranker = (LLMReranker(api, os.environ.get("LLM_MODEL") or None)
                            if body.backend == "llm"
                            else EmbeddingReranker(api, os.environ.get("EMBEDDING_MODEL")
                                                   or "text-embedding-3-small"))
                topic = await load_topic_page(repo, collection_id, max_items=body.max_items,
                                              include_subcollections=body.subcollections,
                                              run_searches=body.run_searches)
                scored = await rerank(topic, reranker)
                return page_as_json(topic, scored, backend=reranker.name,
                                    threshold=reranker.default_threshold, repository=repo.url)
    except NotFoundError:
        return _error(404, f"Keine Sammlung mit der ID {collection_id or '?'} gefunden.")
    except EduSharingError as exc:
        # Library messages carry no stack traces and no secrets; the type helps the reader.
        log.warning("edu-sharing/B-API error: %s", exc)
        return _error(502, f"{type(exc).__name__}: {exc}")
    except RuntimeError as exc:
        log.warning("scoring failed: %s", exc)
        return _error(502, f"Bewertung fehlgeschlagen: {exc}")
