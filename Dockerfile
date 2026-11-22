FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS base

ARG SQLGLOT_COMMIT=eb11aec4be8cb2122afe713a4816662c3f0fb2fa

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SKIP_INTEGRATION=0 \
    SQLGLOT_ROOT=/opt/sqlglot \
    MAPS_DIR=/app/maps

WORKDIR /app

COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

RUN python - <<'PY'
import io
import os
import shutil
import tarfile
import urllib.request

commit = os.environ.get("SQLGLOT_COMMIT", "eb11aec4be8cb2122afe713a4816662c3f0fb2fa")
url = f"https://github.com/tobymao/sqlglot/archive/{commit}.tar.gz"
data = urllib.request.urlopen(url, timeout=120).read()
os.makedirs("/opt", exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
    tf.extractall("/opt", filter="data")
shutil.move(f"/opt/sqlglot-{commit}", "/opt/sqlglot")
PY
RUN python -m compileall -q /usr/local/lib/python3.13 /opt/sqlglot/sqlglot /opt/sqlglot/tests \
    && chmod -R a+rX /usr/local/lib/python3.13 /opt/sqlglot /app

FROM base AS runtime

ENV MAPS_CACHE=/app/maps_cache.marshal \
    REPO_WORD_INDEX=/app/repo_word_index.marshal

COPY maps/ /app/maps/
COPY src/ /app/src/
RUN python -m compileall -q /app/src \
    && chmod -R a+rX /app/maps /app/src \
    && PYTHONPATH=/app/src python -c "from pathlib import Path; import diff_analyzer, selector; selector.write_maps_cache(Path('/app/maps'), Path('/app/maps_cache.marshal')); diff_analyzer.write_repo_word_index(Path('/opt/sqlglot'), Path('/app/repo_word_index.marshal'))" \
    && chmod a+r /app/maps_cache.marshal /app/repo_word_index.marshal

CMD ["python", "/app/src/main.py"]
