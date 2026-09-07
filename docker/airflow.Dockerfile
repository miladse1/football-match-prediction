FROM apache/airflow:2.10.4-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 procps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /opt/java

ENV JAVA_HOME=/opt/java
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONPATH=/opt/project/src
ENV SPARK_LOCAL_IP=127.0.0.1
ENV PYSPARK_PYTHON=python

USER airflow
RUN pip install --no-cache-dir \
    "psycopg[binary]>=3.2" \
    "python-dotenv>=1.0" \
    "pandas>=2.2" \
    "numpy>=1.26" \
    "scikit-learn>=1.5" \
    "xgboost>=2.0" \
    "joblib>=1.4" \
    "pyspark==3.5.5"
