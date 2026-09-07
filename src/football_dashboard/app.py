"""Read-only Premier League prediction dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from football_dashboard.queries import (
    about_payload,
    overview_payload,
    performance_payload,
    results_payload,
    upcoming_payload,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
SPA_PAGES = {"upcoming", "results", "performance", "about"}

app = FastAPI(title="Premier League predictions", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/")
def index() -> FileResponse:
    return _page()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/overview")
def overview() -> dict:
    return overview_payload()


@app.get("/api/upcoming")
def upcoming(
    team: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=16, ge=10, le=20),
) -> dict:
    return upcoming_payload(
        team=team or None,
        date_from=date_from or None,
        date_to=date_to or None,
        page=page,
        page_size=page_size,
    )


@app.get("/api/results")
def results(
    team: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=16, ge=10, le=20),
) -> dict:
    return results_payload(
        team=team or None,
        date_from=date_from or None,
        date_to=date_to or None,
        page=page,
        page_size=page_size,
    )


@app.get("/api/performance")
def performance() -> dict:
    return performance_payload()


@app.get("/api/about")
def about() -> dict:
    return about_payload()


@app.get("/{page}")
def spa(page: str) -> FileResponse:
    if page not in SPA_PAGES:
        raise HTTPException(status_code=404, detail="Not found")
    return _page()
