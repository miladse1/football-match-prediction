-- Teams are identified by canonical_name so aliases (Man City / Manchester City)
-- collapse to one row. matches cannot have the same club on both sides.

ALTER TABLE teams
    ADD CONSTRAINT teams_canonical_name_unique UNIQUE (canonical_name);

ALTER TABLE matches
    ADD CONSTRAINT matches_distinct_teams CHECK (home_team_id <> away_team_id);
