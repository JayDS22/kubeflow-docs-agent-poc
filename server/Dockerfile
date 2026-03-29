FROM python:3.10-slim

WORKDIR /app

# install from server requirements (includes agent deps)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy agent and server packages
COPY agent/ /app/agent/
COPY server/ /app/server/

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health',timeout=3); assert r.status_code in (200,503)" || exit 1

# start both FastAPI and MCP server
CMD ["sh", "-c", "python -m server.mcp_server & uvicorn server.app:app --host 0.0.0.0 --port 8000"]
