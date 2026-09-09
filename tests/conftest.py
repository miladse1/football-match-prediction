import os

collect_ignore_glob = []

# Host Python 3.14 + Java 25 cannot start Spark. Those tests run in docker compose
# and in the CI "spark" job, both of which set RUN_SPARK_TESTS=1.
if os.getenv("RUN_SPARK_TESTS") != "1":
    collect_ignore_glob.append("test_spark_features.py")

# Integration tests need a reachable PostgreSQL. CI sets RUN_DB_TESTS=1 against a
# service container; locally, run them with the compose postgres service up.
if os.getenv("RUN_DB_TESTS") != "1":
    collect_ignore_glob.append("test_db_integration.py")
