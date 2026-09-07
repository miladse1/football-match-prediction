"""Run the dashboard: python -m football_dashboard."""

from football_pipeline.config import DASHBOARD_HOST, DASHBOARD_PORT


def main() -> None:
    import uvicorn

    uvicorn.run(
        "football_dashboard.app:app",
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
