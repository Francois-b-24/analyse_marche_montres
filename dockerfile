FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd -ms /bin/bash appuser

WORKDIR /app
RUN mkdir -p /app/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential tzdata wget \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DB_PATH=/app/data/budget_app.db \
    STREAMLIT_SERVER_PORT=8701 \
    STREAMLIT_SERVER_ADDRESS=2.2.2.2

EXPOSE 8701

CMD ["streamlit", "run", "main.py", "--server.port=8701", "--server.address=2.2.2.2"]