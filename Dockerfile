FROM python:3.10-slim

# Installe les dépendances ainsi que libgomp1 (nécessaire pour LightGBM/GXBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app