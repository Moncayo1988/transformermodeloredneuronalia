"""
modal_app.py — Deployment en Modal.com
Detección de Placas Vehiculares (Popayán)
YOLO11 + EasyOCR + Transformer (98.22%)

ESTRUCTURA REQUERIDA en tu carpeta local:
    tu_proyecto/
    ├── modal_app.py          ← este archivo
    ├── app.py
    └── Modern/
        ├── __init__.py
        ├── modulo0_config.py
        ├── modulo1_deteccion_yolo.py
        ├── modulo2_ocr.py
        ├── modulo3_dataset.py
        └── modulo4_transformer.py

COMANDOS:
    pip install modal
    modal setup
    modal serve modal_app.py     # desarrollo
    modal deploy modal_app.py    # producción
"""

import modal
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

# ── IMAGEN ────────────────────────────────────────────────────────────────────
# REGLA MODAL: add_local_* SIEMPRE AL FINAL, después de apt y pip.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "tesseract-ocr",
        "libtesseract-dev",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgl1-mesa-glx",
    )
    .pip_install(
        "fastapi==0.111.0",
        "uvicorn[standard]==0.29.0",
        "python-multipart==0.0.9",
        "ultralytics>=8.0.0",
        "easyocr>=1.7.0",
        "huggingface_hub>=0.20.0",
        "opencv-python-headless>=4.8.0",
        "Pillow>=9.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "pytesseract>=0.3.10",
        "scikit-learn>=1.0.0",
        "seaborn>=0.12.0",
    )
    # add_local_* AL FINAL — esto es obligatorio en Modal 1.x
    # copy=True: los archivos se graban dentro de la imagen (necesario porque
    # hay pasos de build previos como pip_install).
    .add_local_file(str(PROJECT_DIR / "app.py"), "/proyecto/app.py", copy=True)
    .add_local_dir(str(PROJECT_DIR / "Modern"), "/proyecto/Modern", copy=True)
)

# ── APP Y VOLUMEN ─────────────────────────────────────────────────────────────
app = modal.App("placas-api", image=image)

volumen_modelos = modal.Volume.from_name("placas-modelos", create_if_missing=True)
VOL_PATH = "/vol/modelos"


# ── ENDPOINT ──────────────────────────────────────────────────────────────────
@app.function(
    gpu="T4",
    memory=4096,
    max_containers=2,
    scaledown_window=300,
    timeout=120,
    volumes={VOL_PATH: volumen_modelos},
    # secrets=[modal.Secret.from_name("mi-hf-token")],  # descomenta si HF es privado
)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    # Agrega /proyecto al path → 'from Modern.moduloX import' funciona igual que local
    if "/proyecto" not in sys.path:
        sys.path.insert(0, "/proyecto")

    # Redirige la descarga del transformer al volumen persistente
    os.makedirs(VOL_PATH, exist_ok=True)

    import app as app_module
    app_module.RUTA_MODELO = os.path.join(VOL_PATH, "transformer_pico_placa.pt")

    return app_module.app