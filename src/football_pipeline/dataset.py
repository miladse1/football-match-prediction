"""Assemble training_rows from matches + match_features with a time split."""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import date
from pathlib import Path

from football_pipeline.config import ROOT, VALID_END, TRAIN_END, TEST_END, MIN_PRIOR_N
from football_pipeline.competitions import DEFAULT_COMPETITION
from football_pipeline.db import connect
from football_pipeline.splits import ChronoSplit, assert_splits_are_chronological

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = (
    "home_win_rate_l5",
    "away_win_rate_l5",
    "home_goals_scored_avg_l5",
    "away_goals_scored_avg_l5",
    "home_goals_conceded_avg_l5",
    "away_goals_conceded_avg_l5",
    "home_points_l5",
    "away_points_l5",
    "home_gd_l5",
    "away_gd_l5",
    "home_home_win_rate_l5",
    "away_away_win_rate_l5",
    "h2h_home_win_rate_n",
    "home_elo",
    "away_elo",
    "home_draw_rate_l5",
    "away_draw_rate_l5",
    "home_home_draw_rate_l5",
    "away_away_draw_rate_l5",
    "h2h_draw_rate_n",
    "elo_abs_diff",
    "ppg_diff_l5",
    "gf_diff_l5",
    "ga_diff_l5",
    "recent_total_goals_avg_l5",
)

CSV_PATH = ROOT / "data" / "processed" / "training_rows.csv"


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def assemble_training_rows(
    *,
    train_end: date,
    valid_end: date,
    min_prior_n: int,
    test_end: date | None = None,
    competition: str = DEFAULT_COMPETITION,
) -> dict[str, int]:
    if min_prior_n < 1:
        raise ValueError("min_prior_n must be >= 1 (opening matches have no rolling form)")
    if test_end is None:
        test_end = parse_iso_date(TEST_END)

    split = ChronoSplit(train_end=train_end, valid_end=valid_end, test_end=test_end)
    feature_sql = ", ".join(f"f.{name}" for name in FEATURE_COLUMNS)
    insert_cols = ", ".join(FEATURE_COLUMNS)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM training_rows AS tr
                USING matches AS m, competitions AS c
                WHERE tr.match_id = m.id
                  AND m.competition_id = c.id
                  AND c.code = %s
                """,
                (competition,),
            )
            cur.execute(
                f"""
                INSERT INTO training_rows (
                    match_id, match_date, season_id, competition_id, result, result_code, split,
                    home_prior_n, away_prior_n, {insert_cols}, feature_version
                )
                SELECT
                    m.id, m.match_date, m.season_id, m.competition_id, m.result, m.result_code,
                    CASE
                        WHEN m.match_date <= %s THEN 'train'
                        WHEN m.match_date <= %s THEN 'valid'
                        WHEN m.match_date <= %s THEN 'test'
                    END,
                    f.home_prior_n, f.away_prior_n,
                    {feature_sql}, f.feature_version
                FROM matches AS m
                JOIN match_features AS f ON f.match_id = m.id
                JOIN competitions AS c ON c.id = m.competition_id
                WHERE c.code = %s
                  AND m.is_played
                  AND m.result_code IS NOT NULL
                  AND f.home_prior_n >= %s
                  AND f.away_prior_n >= %s
                  AND m.match_date <= %s
                """,
                (
                    split.train_end,
                    split.valid_end,
                    split.test_end,
                    competition,
                    min_prior_n,
                    min_prior_n,
                    split.test_end,
                ),
            )
            cur.execute(
                """
                SELECT tr.match_date, tr.split
                FROM training_rows AS tr
                JOIN matches AS m ON m.id = tr.match_id
                JOIN competitions AS c ON c.id = m.competition_id
                WHERE c.code = %s
                ORDER BY tr.match_date, tr.match_id
                """,
                (competition,),
            )
            labelled = list(cur.fetchall())
            assert_splits_are_chronological(labelled)
            cur.execute(
                """
                SELECT tr.split, tr.result_code, count(*)
                FROM training_rows AS tr
                JOIN matches AS m ON m.id = tr.match_id
                JOIN competitions AS c ON c.id = m.competition_id
                WHERE c.code = %s
                GROUP BY tr.split, tr.result_code
                """,
                (competition,),
            )
            classes: dict[str, dict[str, int]] = {}
            for split_name, result_code, n in cur.fetchall():
                label = {0: "away", 1: "draw", 2: "home"}.get(int(result_code), str(result_code))
                classes.setdefault(split_name, {})[label] = int(n)
            cur.execute(
                """
                SELECT tr.split, count(*)
                FROM training_rows AS tr
                JOIN matches AS m ON m.id = tr.match_id
                JOIN competitions AS c ON c.id = m.competition_id
                WHERE c.code = %s
                GROUP BY tr.split
                """,
                (competition,),
            )
            counts = {name: n for name, n in cur.fetchall()}
        conn.commit()

    _export_csv(competition=competition)
    summary = {
        "competition": competition,
        "rows": len(labelled),
        "train": counts.get("train", 0),
        "valid": counts.get("valid", 0),
        "test": counts.get("test", 0),
        "classes_by_split": classes,
        "train_end": train_end.isoformat(),
        "valid_end": valid_end.isoformat(),
        "test_end": split.test_end.isoformat() if split.test_end else None,
        "min_prior_n": min_prior_n,
    }
    logger.info(
        "training_rows: %s rows (train=%s valid=%s test=%s) cutoffs train_end=%s valid_end=%s min_prior_n=%s classes=%s",
        summary["rows"],
        summary["train"],
        summary["valid"],
        summary["test"],
        train_end.isoformat(),
        valid_end.isoformat(),
        min_prior_n,
        classes,
    )
    return summary


def _export_csv(*, competition: str = DEFAULT_COMPETITION) -> None:
    from football_pipeline.competitions import processed_dir

    dest = processed_dir(competition, root=ROOT) / "training_rows.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tr.match_id, tr.match_date, tr.split, tr.result_code, tr.home_prior_n, tr.away_prior_n,
                       """
                + ", ".join(f"tr.{name}" for name in FEATURE_COLUMNS)
                + """,
                       tr.feature_version
                FROM training_rows AS tr
                JOIN matches AS m ON m.id = tr.match_id
                JOIN competitions AS c ON c.id = m.competition_id
                WHERE c.code = %s
                ORDER BY tr.match_date, tr.match_id
                """,
                (competition,),
            )
            columns = [col.name for col in cur.description]
            rows = cur.fetchall()
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    logger.info("Wrote %s", dest)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Build the chronological training table.")
    parser.add_argument("--train-end", default=TRAIN_END, help="Inclusive last train date (YYYY-MM-DD)")
    parser.add_argument("--valid-end", default=VALID_END, help="Inclusive last valid date (YYYY-MM-DD)")
    parser.add_argument("--test-end", default=TEST_END, help="Inclusive last test date (YYYY-MM-DD). Later played matches are live-only.")
    parser.add_argument(
        "--min-prior-n",
        type=int,
        default=MIN_PRIOR_N,
        help="Drop matches where either club has fewer than this many prior games",
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    args = parser.parse_args()
    assemble_training_rows(
        train_end=parse_iso_date(args.train_end),
        valid_end=parse_iso_date(args.valid_end),
        test_end=parse_iso_date(args.test_end),
        min_prior_n=args.min_prior_n,
        competition=args.competition,
    )


if __name__ == "__main__":
    main()
