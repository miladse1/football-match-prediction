import os

# Host Python 3.14 + Java 25 cannot start Spark. Those tests run in docker compose.
if os.getenv("RUN_SPARK_TESTS") != "1":
    collect_ignore_glob = ["test_spark_features.py"]
