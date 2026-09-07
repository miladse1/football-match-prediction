-- Model-ready snapshot: features + target + chrono split label.
-- Rebuilt by python -m football_pipeline.dataset (derived table, not source of truth).

CREATE TABLE training_rows (
    match_id                   INTEGER PRIMARY KEY REFERENCES matches (id) ON DELETE CASCADE,
    match_date                 DATE NOT NULL,
    season_id                  INTEGER NOT NULL REFERENCES seasons (id),
    result                     CHAR(1) NOT NULL,
    result_code                SMALLINT NOT NULL,
    split                      TEXT NOT NULL,
    home_prior_n               INTEGER NOT NULL,
    away_prior_n               INTEGER NOT NULL,
    home_win_rate_l5           DOUBLE PRECISION,
    away_win_rate_l5           DOUBLE PRECISION,
    home_goals_scored_avg_l5   DOUBLE PRECISION,
    away_goals_scored_avg_l5   DOUBLE PRECISION,
    home_goals_conceded_avg_l5 DOUBLE PRECISION,
    away_goals_conceded_avg_l5 DOUBLE PRECISION,
    home_points_l5             DOUBLE PRECISION,
    away_points_l5             DOUBLE PRECISION,
    home_gd_l5                 DOUBLE PRECISION,
    away_gd_l5                 DOUBLE PRECISION,
    home_home_win_rate_l5      DOUBLE PRECISION,
    away_away_win_rate_l5      DOUBLE PRECISION,
    h2h_home_win_rate_n        DOUBLE PRECISION,
    home_elo                   DOUBLE PRECISION NOT NULL,
    away_elo                   DOUBLE PRECISION NOT NULL,
    feature_version            TEXT NOT NULL,
    built_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT training_rows_result_check CHECK (result IN ('H', 'D', 'A')),
    CONSTRAINT training_rows_result_code_check CHECK (result_code IN (0, 1, 2)),
    CONSTRAINT training_rows_split_check CHECK (split IN ('train', 'valid', 'test'))
);

CREATE INDEX training_rows_split_date_idx ON training_rows (split, match_date);

COMMENT ON TABLE training_rows IS
    'Derived ML table. split is assigned by match_date cutoffs, never by random shuffle.';
