FROM python:3.11-slim-bookworm

# Spark 3.5 needs a Java 17 runtime. The host has Java 25, which Spark cannot start.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /opt/java

ENV JAVA_HOME=/opt/java
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONPATH=/opt/project/src
ENV PYSPARK_PYTHON=python
ENV SPARK_LOCAL_IP=127.0.0.1

WORKDIR /opt/project

RUN pip install --no-cache-dir \
    "pyspark==3.5.5" \
    "psycopg[binary]>=3.2" \
    "python-dotenv>=1.0" \
    "pytest>=8.0"
