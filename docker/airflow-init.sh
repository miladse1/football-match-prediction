#!/usr/bin/env bash
set -euo pipefail

: "${AIRFLOW_ADMIN_USER:?Set AIRFLOW_ADMIN_USER in .env}"
: "${AIRFLOW_ADMIN_PASSWORD:?Set AIRFLOW_ADMIN_PASSWORD in .env}"

if [[ "${AIRFLOW_ADMIN_USER}" == "admin" && "${AIRFLOW_ADMIN_PASSWORD}" == "admin" ]]; then
  echo "Refusing default admin/admin credentials. Set a local user in .env." >&2
  exit 1
fi

python /opt/project/docker/ensure_airflow_db.py
airflow db migrate

if ! airflow users list 2>/dev/null | grep -qw "${AIRFLOW_ADMIN_USER}"; then
  airflow users create \
    --username "${AIRFLOW_ADMIN_USER}" \
    --firstname Dev \
    --lastname Admin \
    --role Admin \
    --email "${AIRFLOW_ADMIN_USER}@localhost" \
    --password "${AIRFLOW_ADMIN_PASSWORD}"
fi

airflow users reset-password --username "${AIRFLOW_ADMIN_USER}" --password "${AIRFLOW_ADMIN_PASSWORD}"

# Drop the old default account if this environment previously used admin/admin.
if [[ "${AIRFLOW_ADMIN_USER}" != "admin" ]]; then
  airflow users delete --username admin >/dev/null 2>&1 || true
fi
