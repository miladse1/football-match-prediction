FROM python:3.11-slim-bookworm

WORKDIR /opt/project
ENV PYTHONPATH=/opt/project/src
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /opt/project/
COPY src /opt/project/src

RUN pip install --no-cache-dir \
    "psycopg[binary]>=3.2" \
    "python-dotenv>=1.0" \
    "numpy>=1.26" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32"

EXPOSE 8500
CMD ["python", "-m", "football_dashboard"]
