"""
app.py — Endpoint FastAPI para Detección de Placas Vehiculares (Popayán)
=========================================================================
Corregido para usar las firmas reales de los módulos del proyecto:
  - modulo1: detectar_y_recortar_placa(ruta_imagen) → (recorte_rgb, img_marcada, metodo)
  - modulo2: elegir_mejor_candidato(candidatos)      → (placa, formato_ok)
             ocr_easyocr_multi / preprocesar_por_tipo / detectar_tipo_placa
  - modulo4: cargar_modelo()                         → TransformerPlacas
             predecir_pico_placa(placa, model)       → dict
"""

import os
import sys
import tempfile
import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("placas-api")

sys.path.insert(0, os.path.dirname(__file__))

# ── Descarga del modelo Transformer desde HuggingFace ─────────────────────────
from huggingface_hub import hf_hub_download

RUTA_MODELO = "modelos/transformer_pico_placa.pt"
os.makedirs("modelos", exist_ok=True)

if not os.path.exists(RUTA_MODELO):
    log.info("Descargando transformer_pico_placa.pt desde HuggingFace...")
    hf_hub_download(
        repo_id="Huntercito/Deteccion_Pico_y_Placa",
        filename="transformer_pico_placa.pt",
        local_dir="modelos",
        token=os.environ.get("HF_TOKEN"),
    )
    log.info("Modelo descargado.")

# ── Imports CORRECTOS según las firmas reales de los módulos ──────────────────
from Modern.modulo1_deteccion_yolo import detectar_y_recortar_placa
from Modern.modulo2_ocr import (
    ocr_easyocr_multi,
    preprocesar_por_tipo,
    detectar_tipo_placa,
    elegir_mejor_candidato,
)
from Modern.modulo4_transformer import cargar_modelo, predecir_pico_placa

# ── Cargar Transformer una sola vez al arrancar ───────────────────────────────
log.info("Cargando modelo Transformer...")
_transformer = cargar_modelo(ruta=RUTA_MODELO)
log.info("Transformer listo.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Detección de Placas — Popayán",
    description="YOLO11 + EasyOCR + Transformer (98.22%). Detecta placa y predice Pico y Placa.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "modelo": "YOLO11 + EasyOCR + Transformer", "precision_test": "98.22%"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Recibe imagen JPG/PNG → retorna placa, restricción y confianza."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG.")

    img_bytes = await file.read()

    # modulo1 recibe ruta de archivo, no array → guardamos en temporal
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(img_bytes)
        ruta_tmp = tmp.name

    try:
        # MÓDULO 1 — detectar y recortar placa
        recorte_rgb, _, metodo = detectar_y_recortar_placa(ruta_tmp)

        if recorte_rgb is None:
            return {
                "placa": None, "formato_valido": False,
                "restriccion": None, "confianza_pct": 0.0,
                "metodo_deteccion": metodo,
                "mensaje": "No se detectó ninguna placa en la imagen.",
            }

        # MÓDULO 2 — OCR sobre el recorte
        tipo_placa, _ = detectar_tipo_placa(recorte_rgb)
        recorte_pre   = preprocesar_por_tipo(recorte_rgb, tipo_placa)
        candidatos    = ocr_easyocr_multi(recorte_pre)
        placa, formato_ok = elegir_mejor_candidato(candidatos)

        if not placa or placa == "???":
            return {
                "placa": None, "formato_valido": False,
                "restriccion": None, "confianza_pct": 0.0,
                "metodo_deteccion": metodo,
                "mensaje": "Placa detectada pero el OCR no pudo leer el texto.",
            }

        # MÓDULO 4 — predicción Pico y Placa
        resultado = predecir_pico_placa(placa, model=_transformer, verbose=False)

        return {
            "placa"           : resultado["placa"],
            "formato_valido"  : formato_ok,
            "restriccion"     : resultado["restriccion"],
            "confianza_pct"   : resultado["confianza_pct"],
            "probabilidades"  : resultado["probabilidades"],
            "metodo_deteccion": metodo,
        }

    finally:
        os.unlink(ruta_tmp)


@app.post("/detect/texto")
def detect_texto(placa: str):
    """Prueba rápida: recibe texto de placa, retorna predicción sin imagen.
    Ejemplo: POST /detect/texto?placa=SKY424"""
    if not placa:
        raise HTTPException(status_code=400, detail="El parámetro 'placa' es requerido.")
    return predecir_pico_placa(placa, model=_transformer, verbose=False)