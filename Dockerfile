# ============================
# Base image
# ============================
FROM python:3.11-slim

# Evita output bufferizzato
ENV PYTHONUNBUFFERED=1

# ============================
# Dipendenze di sistema
# ============================
# openseespy richiede librerie matematiche
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================
# Working directory
# ============================
WORKDIR /app

# ============================
# Python dependencies
# ============================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ============================
# Copia codice
# ============================
COPY main.py .

# ============================
# Avvio FastAPI
# ============================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
