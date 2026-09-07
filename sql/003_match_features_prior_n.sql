-- How many played matches sit in the last-5 window (0–5).
-- Training can later drop rows with home_prior_n < 5.

ALTER TABLE match_features
    ADD COLUMN home_prior_n INTEGER,
    ADD COLUMN away_prior_n INTEGER;
