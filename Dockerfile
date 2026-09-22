FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "-b", "0.0.0.0:7860", "backend.app:app"]
