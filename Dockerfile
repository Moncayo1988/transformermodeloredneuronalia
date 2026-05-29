FROM python:3.11-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860

# Dependencias del sistema (Tesseract + libGL para OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. Instalar PyTorch CPU-only primero (más liviano: ~700MB vs 2GB) ─────────
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── 2. Instalar el resto de dependencias ──────────────────────────────────────
COPY requirements_api.txt .

# Reemplazamos torch del requirements (ya instalado arriba) para evitar
# que pip lo sobreescriba con la versión CUDA
RUN grep -v "^torch" requirements_api.txt > requirements_cpu.txt && \
    pip install --no-cache-dir -r requirements_cpu.txt

# ── 3. Copiar el código del proyecto ──────────────────────────────────────────
COPY app.py .
COPY Modern/ ./Modern/

# ── 4. Crear carpeta de modelos (se descarga en runtime desde HuggingFace) ─────
RUN mkdir -p /app/modelos

# ── 5. Usuario no-root (requerido por HuggingFace Spaces) ─────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ── 6. Puerto 7860 (estándar de HuggingFace Spaces) ───────────────────────────
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
