FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN addgroup --system climateguard && adduser --system --ingroup climateguard climateguard
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=climateguard:climateguard . .
RUN mkdir -p /app/runtime && chown -R climateguard:climateguard /app/runtime
USER climateguard
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8001/health', timeout=3)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
