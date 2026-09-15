-- Scope teams and model runs to a competition so Top 5 leagues can share one schema
-- without colliding club identities or silently mixing training runs.

ALTER TABLE teams
    ADD COLUMN competition_id INTEGER REFERENCES competitions (id);

UPDATE teams AS t
SET competition_id = src.competition_id
FROM (
    SELECT DISTINCT ON (m.home_team_id)
        m.home_team_id AS team_id,
        m.competition_id
    FROM matches AS m
    ORDER BY m.home_team_id, m.id
) AS src
WHERE t.id = src.team_id
  AND t.competition_id IS NULL;

UPDATE teams AS t
SET competition_id = src.competition_id
FROM (
    SELECT DISTINCT ON (m.away_team_id)
        m.away_team_id AS team_id,
        m.competition_id
    FROM matches AS m
    ORDER BY m.away_team_id, m.id
) AS src
WHERE t.id = src.team_id
  AND t.competition_id IS NULL;

UPDATE teams
SET competition_id = (SELECT id FROM competitions WHERE code = 'E0')
WHERE competition_id IS NULL
  AND EXISTS (SELECT 1 FROM competitions WHERE code = 'E0');

ALTER TABLE teams
    ALTER COLUMN competition_id SET NOT NULL;

ALTER TABLE teams
    DROP CONSTRAINT teams_canonical_name_unique;

ALTER TABLE teams
    DROP CONSTRAINT teams_source_name_unique;

ALTER TABLE teams
    ADD CONSTRAINT teams_competition_canonical_unique UNIQUE (competition_id, canonical_name);

ALTER TABLE teams
    ADD CONSTRAINT teams_competition_source_unique UNIQUE (competition_id, source_name);

CREATE INDEX teams_competition_id_idx ON teams (competition_id);

ALTER TABLE model_runs
    ADD COLUMN competition_id INTEGER REFERENCES competitions (id);

UPDATE model_runs
SET competition_id = (SELECT id FROM competitions WHERE code = 'E0')
WHERE competition_id IS NULL
  AND EXISTS (SELECT 1 FROM competitions WHERE code = 'E0');

ALTER TABLE model_runs
    ALTER COLUMN competition_id SET NOT NULL;

CREATE INDEX model_runs_competition_id_idx ON model_runs (competition_id);

ALTER TABLE training_rows
    ADD COLUMN competition_id INTEGER REFERENCES competitions (id);

UPDATE training_rows AS tr
SET competition_id = m.competition_id
FROM matches AS m
WHERE m.id = tr.match_id
  AND tr.competition_id IS NULL;

CREATE INDEX training_rows_competition_id_idx ON training_rows (competition_id);

CREATE INDEX matches_competition_id_idx ON matches (competition_id);
