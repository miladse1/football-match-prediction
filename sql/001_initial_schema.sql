-- Core schema: dimensions, matches, raw payloads, features, and ML tables.
-- Apply with: python -m football_pipeline.migrate
--
-- raw_match_payloads is the landing zone for CSV/JSON rows.

CREATE TABLE competitions (
    id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code        VARCHAR(8) NOT NULL,
    name        TEXT NOT NULL,
    country     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT competitions_code_unique UNIQUE (code)
);

CREATE TABLE seasons (
    id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    competition_id  INTEGER NOT NULL REFERENCES competitions (id),
    start_year      INTEGER NOT NULL,
    name            TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT seasons_competition_year_unique UNIQUE (competition_id, start_year)
);

CREATE TABLE teams (
    id               INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_name      TEXT NOT NULL,
    canonical_name   TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT teams_source_name_unique UNIQUE (source_name)
);

CREATE TABLE matches (
    id               INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    competition_id   INTEGER NOT NULL REFERENCES competitions (id),
    season_id        INTEGER NOT NULL REFERENCES seasons (id),
    match_date       DATE NOT NULL,
    kickoff_time     TIME,
    home_team_id     INTEGER NOT NULL REFERENCES teams (id),
    away_team_id     INTEGER NOT NULL REFERENCES teams (id),
    home_goals       INTEGER,
    away_goals       INTEGER,
    result           CHAR(1),
    result_code      SMALLINT,
    is_played        BOOLEAN NOT NULL DEFAULT FALSE,
    source_file      TEXT,
    source_row_hash  TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT matches_result_check CHECK (result IS NULL OR result IN ('H', 'D', 'A')),
    CONSTRAINT matches_result_code_check CHECK (result_code IS NULL OR result_code IN (0, 1, 2)),
    CONSTRAINT matches_goals_check CHECK (
        (home_goals IS NULL AND away_goals IS NULL)
        OR (home_goals >= 0 AND away_goals >= 0)
    ),
    CONSTRAINT matches_unique_fixture UNIQUE (competition_id, match_date, home_team_id, away_team_id)
);

CREATE INDEX matches_match_date_idx ON matches (match_date);
CREATE INDEX matches_home_team_date_idx ON matches (home_team_id, match_date);
CREATE INDEX matches_away_team_date_idx ON matches (away_team_id, match_date);
CREATE INDEX matches_season_id_idx ON matches (season_id);

-- One JSON object per CSV row. Unique (source_file, row_number) makes reruns idempotent.
CREATE TABLE raw_match_payloads (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    season_id    INTEGER NOT NULL REFERENCES seasons (id) ON DELETE CASCADE,
    source_url   TEXT NOT NULL,
    source_file  TEXT NOT NULL,
    row_number   INTEGER NOT NULL,
    payload      JSONB NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT raw_match_payloads_file_row_unique UNIQUE (source_file, row_number)
);

CREATE INDEX raw_match_payloads_season_id_idx ON raw_match_payloads (season_id);

CREATE TABLE match_features (
    match_id                 INTEGER PRIMARY KEY REFERENCES matches (id) ON DELETE CASCADE,
    home_win_rate_l5         DOUBLE PRECISION,
    away_win_rate_l5         DOUBLE PRECISION,
    home_goals_scored_avg_l5 DOUBLE PRECISION,
    away_goals_scored_avg_l5 DOUBLE PRECISION,
    home_goals_conceded_avg_l5 DOUBLE PRECISION,
    away_goals_conceded_avg_l5 DOUBLE PRECISION,
    home_points_l5           DOUBLE PRECISION,
    away_points_l5           DOUBLE PRECISION,
    home_gd_l5               DOUBLE PRECISION,
    away_gd_l5               DOUBLE PRECISION,
    home_home_win_rate_l5    DOUBLE PRECISION,
    away_away_win_rate_l5    DOUBLE PRECISION,
    h2h_home_win_rate_n      DOUBLE PRECISION,
    home_elo                 DOUBLE PRECISION,
    away_elo                 DOUBLE PRECISION,
    feature_version          TEXT NOT NULL,
    computed_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE model_runs (
    id               INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trained_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    train_start      DATE,
    train_end        DATE,
    valid_start      DATE,
    valid_end        DATE,
    test_start       DATE,
    test_end         DATE,
    feature_version  TEXT NOT NULL,
    algorithm        TEXT NOT NULL,
    metrics          JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_path    TEXT
);

CREATE TABLE predictions (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    match_id         INTEGER NOT NULL REFERENCES matches (id),
    model_run_id     INTEGER NOT NULL REFERENCES model_runs (id),
    p_away           DOUBLE PRECISION NOT NULL,
    p_draw           DOUBLE PRECISION NOT NULL,
    p_home           DOUBLE PRECISION NOT NULL,
    predicted_class  SMALLINT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT predictions_class_check CHECK (predicted_class IN (0, 1, 2)),
    CONSTRAINT predictions_probs_sum_check CHECK (
        abs((p_away + p_draw + p_home) - 1.0) < 0.01
    ),
    CONSTRAINT predictions_match_run_unique UNIQUE (match_id, model_run_id)
);

COMMENT ON TABLE raw_match_payloads IS
    'Immutable landing zone: original football-data.co.uk CSV rows as JSON.';
COMMENT ON TABLE matches IS
    'Curated fixtures. result_code: 0 away, 1 draw, 2 home.';
COMMENT ON TABLE match_features IS
    'Pre-kickoff features only. No full-time stats from the current match.';
