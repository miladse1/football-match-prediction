"""Pre-match rolling features in Spark, plus sequential Elo.

Window bounds are rowsBetween(-5, -1): the current match is never in its own form.
Elo is a sequential scan (each rating depends on the previous result), so it runs
on the driver after Spark windows. That is a real property of Elo, not a shortcut.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DateType, IntegerType, LongType, StringType, StructField, StructType
from pyspark.sql.window import Window

from football_pipeline.constants import FEATURE_VERSION

MATCH_ROW_SCHEMA = StructType(
    [
        StructField("match_id", LongType(), False),
        StructField("match_date", DateType(), False),
        StructField("kickoff_time", StringType(), True),
        StructField("home_team_id", LongType(), False),
        StructField("away_team_id", LongType(), False),
        StructField("home_goals", IntegerType(), True),
        StructField("away_goals", IntegerType(), True),
        StructField("result", StringType(), True),
        StructField("is_played", BooleanType(), False),
    ]
)

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0


def spark_session(app_name: str = "football-match-features") -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def _team_match_grain(played: DataFrame) -> DataFrame:
    home = played.select(
        "match_id",
        "match_date",
        "kickoff_time",
        F.col("home_team_id").alias("team_id"),
        F.lit(True).alias("is_home"),
        F.col("home_goals").alias("goals_for"),
        F.col("away_goals").alias("goals_against"),
    )
    away = played.select(
        "match_id",
        "match_date",
        "kickoff_time",
        F.col("away_team_id").alias("team_id"),
        F.lit(False).alias("is_home"),
        F.col("away_goals").alias("goals_for"),
        F.col("home_goals").alias("goals_against"),
    )
    grain = home.unionByName(away)
    return (
        grain.withColumn("win", F.when(F.col("goals_for") > F.col("goals_against"), 1.0).otherwise(0.0))
        .withColumn("draw", F.when(F.col("goals_for") == F.col("goals_against"), 1.0).otherwise(0.0))
        .withColumn(
            "points",
            F.when(F.col("goals_for") > F.col("goals_against"), 3.0)
            .when(F.col("goals_for") == F.col("goals_against"), 1.0)
            .otherwise(0.0),
        )
        .withColumn("goal_diff", F.col("goals_for") - F.col("goals_against"))
        .withColumn("goals_total", F.col("goals_for") + F.col("goals_against"))
    )


def add_rolling_form(played: DataFrame) -> DataFrame:
    """Last-5 (or fewer) played matches for each club, excluding the current row."""
    grain = _team_match_grain(played)
    by_team = (
        Window.partitionBy("team_id")
        .orderBy("match_date", "kickoff_time", "match_id")
        .rowsBetween(-5, -1)
    )
    rolling = (
        grain.withColumn("win_rate_l5", F.avg("win").over(by_team))
        .withColumn("draw_rate_l5", F.avg("draw").over(by_team))
        .withColumn("goals_scored_avg_l5", F.avg("goals_for").over(by_team))
        .withColumn("goals_conceded_avg_l5", F.avg("goals_against").over(by_team))
        .withColumn("goals_total_avg_l5", F.avg("goals_total").over(by_team))
        .withColumn("points_l5", F.sum("points").over(by_team))
        .withColumn("gd_l5", F.sum("goal_diff").over(by_team))
        .withColumn("prior_n", F.count(F.lit(1)).over(by_team))
    )

    venue = (
        Window.partitionBy("team_id", "is_home")
        .orderBy("match_date", "kickoff_time", "match_id")
        .rowsBetween(-5, -1)
    )
    venue_form = grain.withColumn("venue_win_rate_l5", F.avg("win").over(venue)).withColumn(
        "venue_draw_rate_l5", F.avg("draw").over(venue)
    ).select(
        "match_id", "team_id", "is_home", "venue_win_rate_l5", "venue_draw_rate_l5"
    )
    rolling = rolling.join(venue_form, on=["match_id", "team_id", "is_home"], how="left")

    home_cols = rolling.filter(F.col("is_home")).select(
        F.col("match_id"),
        F.col("win_rate_l5").alias("home_win_rate_l5"),
        F.col("draw_rate_l5").alias("home_draw_rate_l5"),
        F.col("goals_scored_avg_l5").alias("home_goals_scored_avg_l5"),
        F.col("goals_conceded_avg_l5").alias("home_goals_conceded_avg_l5"),
        F.col("goals_total_avg_l5").alias("home_goals_total_avg_l5"),
        F.col("points_l5").alias("home_points_l5"),
        F.col("gd_l5").alias("home_gd_l5"),
        F.col("venue_win_rate_l5").alias("home_home_win_rate_l5"),
        F.col("venue_draw_rate_l5").alias("home_home_draw_rate_l5"),
        F.col("prior_n").alias("home_prior_n"),
    )
    away_cols = rolling.filter(~F.col("is_home")).select(
        F.col("match_id"),
        F.col("win_rate_l5").alias("away_win_rate_l5"),
        F.col("draw_rate_l5").alias("away_draw_rate_l5"),
        F.col("goals_scored_avg_l5").alias("away_goals_scored_avg_l5"),
        F.col("goals_conceded_avg_l5").alias("away_goals_conceded_avg_l5"),
        F.col("goals_total_avg_l5").alias("away_goals_total_avg_l5"),
        F.col("points_l5").alias("away_points_l5"),
        F.col("gd_l5").alias("away_gd_l5"),
        F.col("venue_win_rate_l5").alias("away_away_win_rate_l5"),
        F.col("venue_draw_rate_l5").alias("away_away_draw_rate_l5"),
        F.col("prior_n").alias("away_prior_n"),
    )
    return home_cols.join(away_cols, on="match_id", how="inner")


def _targets_long(matches: DataFrame) -> DataFrame:
    home = matches.select(
        "match_id",
        "match_date",
        "kickoff_time",
        F.col("home_team_id").alias("team_id"),
        F.lit(True).alias("is_home"),
    )
    away = matches.select(
        "match_id",
        "match_date",
        "kickoff_time",
        F.col("away_team_id").alias("team_id"),
        F.lit(False).alias("is_home"),
    )
    return home.unionByName(away)


def _earlier(prefix_target: str, prefix_hist: str):
    return (F.col(f"{prefix_hist}.match_date") < F.col(f"{prefix_target}.match_date")) | (
        (F.col(f"{prefix_hist}.match_date") == F.col(f"{prefix_target}.match_date"))
        & (F.col(f"{prefix_hist}.match_id") < F.col(f"{prefix_target}.match_id"))
    )


def add_upcoming_form(upcoming: DataFrame, played: DataFrame) -> DataFrame:
    """Pre-kickoff last-5 form for unplayed matches, using played history only."""
    history = _team_match_grain(played).alias("h")
    targets = _targets_long(upcoming).alias("t")
    recency = Window.partitionBy("t.match_id", "t.team_id", "t.is_home").orderBy(
        F.col("h.match_date").desc(),
        F.col("h.kickoff_time").desc(),
        F.col("h.match_id").desc(),
    )

    def last5(extra_join):
        inner = targets.join(
            history,
            (F.col("t.team_id") == F.col("h.team_id")) & extra_join & _earlier("t", "h"),
            how="inner",
        )
        return inner.withColumn("rn", F.row_number().over(recency)).filter(F.col("rn") <= 5)

    overall = (
        last5(F.lit(True))
        .groupBy(
            F.col("t.match_id").alias("match_id"),
            F.col("t.team_id").alias("team_id"),
            F.col("t.is_home").alias("is_home"),
        )
        .agg(
            F.avg("h.win").alias("win_rate_l5"),
            F.avg("h.draw").alias("draw_rate_l5"),
            F.avg("h.goals_for").alias("goals_scored_avg_l5"),
            F.avg("h.goals_against").alias("goals_conceded_avg_l5"),
            F.avg("h.goals_total").alias("goals_total_avg_l5"),
            F.sum("h.points").alias("points_l5"),
            F.sum("h.goal_diff").alias("gd_l5"),
            F.count(F.lit(1)).alias("prior_n"),
        )
    )
    venue = (
        last5(F.col("t.is_home") == F.col("h.is_home"))
        .groupBy(
            F.col("t.match_id").alias("match_id"),
            F.col("t.team_id").alias("team_id"),
            F.col("t.is_home").alias("is_home"),
        )
        .agg(
            F.avg("h.win").alias("venue_win_rate_l5"),
            F.avg("h.draw").alias("venue_draw_rate_l5"),
        )
    )
    rolling = (
        _targets_long(upcoming)
        .select(
            F.col("match_id"),
            F.col("team_id"),
            F.col("is_home"),
        )
        .join(overall, on=["match_id", "team_id", "is_home"], how="left")
        .join(venue, on=["match_id", "team_id", "is_home"], how="left")
        .fillna({"prior_n": 0})
    )

    home_cols = rolling.filter(F.col("is_home")).select(
        F.col("match_id"),
        F.col("win_rate_l5").alias("home_win_rate_l5"),
        F.col("draw_rate_l5").alias("home_draw_rate_l5"),
        F.col("goals_scored_avg_l5").alias("home_goals_scored_avg_l5"),
        F.col("goals_conceded_avg_l5").alias("home_goals_conceded_avg_l5"),
        F.col("goals_total_avg_l5").alias("home_goals_total_avg_l5"),
        F.col("points_l5").alias("home_points_l5"),
        F.col("gd_l5").alias("home_gd_l5"),
        F.col("venue_win_rate_l5").alias("home_home_win_rate_l5"),
        F.col("venue_draw_rate_l5").alias("home_home_draw_rate_l5"),
        F.col("prior_n").alias("home_prior_n"),
    )
    away_cols = rolling.filter(~F.col("is_home")).select(
        F.col("match_id"),
        F.col("win_rate_l5").alias("away_win_rate_l5"),
        F.col("draw_rate_l5").alias("away_draw_rate_l5"),
        F.col("goals_scored_avg_l5").alias("away_goals_scored_avg_l5"),
        F.col("goals_conceded_avg_l5").alias("away_goals_conceded_avg_l5"),
        F.col("goals_total_avg_l5").alias("away_goals_total_avg_l5"),
        F.col("points_l5").alias("away_points_l5"),
        F.col("gd_l5").alias("away_gd_l5"),
        F.col("venue_win_rate_l5").alias("away_away_win_rate_l5"),
        F.col("venue_draw_rate_l5").alias("away_away_draw_rate_l5"),
        F.col("prior_n").alias("away_prior_n"),
    )
    return home_cols.join(away_cols, on="match_id", how="inner")


def add_head_to_head(current: DataFrame, history: DataFrame | None = None) -> DataFrame:
    """Share of previous *played* meetings won by the current home club."""
    if history is None:
        history = current
    current = current.alias("c")
    history = history.alias("h")
    same_pair = (
        F.least(F.col("c.home_team_id"), F.col("c.away_team_id"))
        == F.least(F.col("h.home_team_id"), F.col("h.away_team_id"))
    ) & (
        F.greatest(F.col("c.home_team_id"), F.col("c.away_team_id"))
        == F.greatest(F.col("h.home_team_id"), F.col("h.away_team_id"))
    )
    strictly_earlier = (F.col("h.match_date") < F.col("c.match_date")) | (
        (F.col("h.match_date") == F.col("c.match_date"))
        & (F.col("h.match_id") < F.col("c.match_id"))
    )
    current_home_won = (
        F.when(F.col("h.match_id").isNull(), None)
        .when(
            (F.col("h.home_team_id") == F.col("c.home_team_id")) & (F.col("h.result") == F.lit("H")),
            1.0,
        )
        .when(
            (F.col("h.away_team_id") == F.col("c.home_team_id")) & (F.col("h.result") == F.lit("A")),
            1.0,
        )
        .otherwise(0.0)
    )
    current_was_draw = F.when(F.col("h.match_id").isNull(), None).when(
        F.col("h.result") == F.lit("D"), 1.0
    ).otherwise(0.0)
    return (
        current.join(history, on=same_pair & strictly_earlier, how="left")
        .groupBy(F.col("c.match_id").alias("match_id"))
        .agg(
            F.avg(current_home_won).alias("h2h_home_win_rate_n"),
            F.avg(current_was_draw).alias("h2h_draw_rate_n"),
        )
    )


def elo_before_matches(rows: Sequence[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    """Pre-kickoff Elo. Ratings update only *after* each played match is stored."""
    ratings: dict[int, float] = {}
    before: dict[int, tuple[float, float]] = {}
    ordered = sorted(
        rows,
        key=lambda row: (row["match_date"], str(row["kickoff_time"] or ""), row["match_id"]),
    )
    for row in ordered:
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        home_elo = ratings.get(home_id, ELO_START)
        away_elo = ratings.get(away_id, ELO_START)
        before[int(row["match_id"])] = (home_elo, away_elo)
        if not row.get("is_played"):
            continue
        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away_elo - home_elo - ELO_HOME_ADVANTAGE) / 400.0)
        )
        if row["home_goals"] > row["away_goals"]:
            actual_home = 1.0
        elif row["home_goals"] < row["away_goals"]:
            actual_home = 0.0
        else:
            actual_home = 0.5
        ratings[home_id] = home_elo + ELO_K * (actual_home - expected_home)
        ratings[away_id] = away_elo + ELO_K * ((1.0 - actual_home) - (1.0 - expected_home))
    return before


def build_feature_frame(spark: SparkSession, rows: Sequence[dict[str, Any]]) -> DataFrame:
    played_rows = [row for row in rows if row.get("is_played")]
    upcoming_rows = [row for row in rows if not row.get("is_played")]
    if not played_rows:
        raise ValueError("No played matches to build features from")

    played = spark.createDataFrame(played_rows, schema=MATCH_ROW_SCHEMA)
    rolling = add_rolling_form(played)
    if upcoming_rows:
        upcoming = spark.createDataFrame(upcoming_rows, schema=MATCH_ROW_SCHEMA)
        rolling = rolling.unionByName(add_upcoming_form(upcoming, played))
        current = played.unionByName(upcoming, allowMissingColumns=True)
        h2h = add_head_to_head(current, history=played)
    else:
        h2h = add_head_to_head(played)

    features = rolling.join(h2h, on="match_id", how="left")
    elo = elo_before_matches(list(rows))
    elo_rows = [
        {"match_id": match_id, "home_elo": home_elo, "away_elo": away_elo}
        for match_id, (home_elo, away_elo) in elo.items()
    ]
    elo_df = spark.createDataFrame(elo_rows)
    home_prior = F.greatest(F.col("home_prior_n"), F.lit(1))
    away_prior = F.greatest(F.col("away_prior_n"), F.lit(1))
    return (
        features.join(elo_df, on="match_id", how="left")
        .withColumn("elo_abs_diff", F.abs(F.col("home_elo") - F.col("away_elo")))
        .withColumn("ppg_diff_l5", (F.col("home_points_l5") / home_prior) - (F.col("away_points_l5") / away_prior))
        .withColumn("gf_diff_l5", F.col("home_goals_scored_avg_l5") - F.col("away_goals_scored_avg_l5"))
        .withColumn("ga_diff_l5", F.col("home_goals_conceded_avg_l5") - F.col("away_goals_conceded_avg_l5"))
        .withColumn(
            "recent_total_goals_avg_l5",
            (F.col("home_goals_total_avg_l5") + F.col("away_goals_total_avg_l5")) / 2.0,
        )
        .withColumn("feature_version", F.lit(FEATURE_VERSION))
    )
