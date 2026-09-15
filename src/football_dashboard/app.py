"""Read-only Top 5 European leagues prediction dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from football_dashboard.queries import (
    about_payload,
    competitions_payload,
    forecast_payload,
    match_detail_payload,
    overview_payload,
    performance_payload,
    results_payload,
    upcoming_payload,
)
from football_pipeline.competitions import CompetitionError, parse_competition
from football_pipeline.season_sim import ForecastUnavailable

STATIC_DIR = Path(__file__).resolve().parent / "static"
SPA_PAGES = {"upcoming", "results", "performance", "forecast", "about"}

app = FastAPI(title="MatchLab predictions", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _league(league: str | None) -> str:
    try:
        return parse_competition(league)
    except CompetitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return _page()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/competitions")
def competitions() -> dict:
    return competitions_payload()


@app.get("/api/overview")
def overview(league: str | None = Query(default=None)) -> dict:
    return overview_payload(competition=_league(league))


@app.get("/api/upcoming")
def upcoming(
    league: str | None = Query(default=None),
    team: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=16, ge=10, le=20),
) -> dict:
    return upcoming_payload(
        competition=_league(league),
        team=team or None,
        date_from=date_from or None,
        date_to=date_to or None,
        page=page,
        page_size=page_size,
    )


@app.get("/api/results")
def results(
    league: str | None = Query(default=None),
    team: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=16, ge=10, le=20),
) -> dict:
    return results_payload(
        competition=_league(league),
        team=team or None,
        date_from=date_from or None,
        date_to=date_to or None,
        page=page,
        page_size=page_size,
    )


@app.get("/api/performance")
def performance(league: str | None = Query(default=None)) -> dict:
    return performance_payload(competition=_league(league))


@app.get("/api/about")
def about(league: str | None = Query(default=None)) -> dict:
    return about_payload(competition=_league(league))


@app.get("/api/forecast")
def forecast(league: str | None = Query(default=None)) -> dict:
    try:
        return forecast_payload(competition=_league(league))
    except ForecastUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/matches/{match_id}")
def match_detail(match_id: int) -> dict:
    payload = match_detail_payload(match_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Match not found.")
    return payload


@app.get("/upcoming/{match_id}")
def upcoming_match(match_id: int) -> FileResponse:
    return _page()


@app.get("/results/{match_id}")
def result_match(match_id: int) -> FileResponse:
    return _page()


@app.get("/{page}")
def spa(page: str) -> FileResponse:
    if page not in SPA_PAGES:
        raise HTTPException(status_code=404, detail="Not found")
    return _page()
