FROM python:3.10-slim

WORKDIR /app

COPY ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion/ /app/ingestion/

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "print('ok')" || exit 1

CMD ["python", "-m", "ingestion.pipeline"]
