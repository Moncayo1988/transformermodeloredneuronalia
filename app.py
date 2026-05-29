"""
app.py — Endpoint FastAPI para Detección de Placas Vehiculares (Popayán)
YOLO11 + EasyOCR + Transformer (98.22%)
Adaptado para HuggingFace Spaces (puerto 7860, lazy loading)
"""

import os
import sys
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("placas-api")

# ── Paths ─────────────────────────────────────────────────────────────────────
_root   = os.path.dirname(os.path.abspath(__file__))
_modern = os.path.join(_root, "Modern")
for _p in [_root, _modern]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Detección de Placas — Popayán",
    description="YOLO11 + EasyOCR + Transformer (98.22%). Detecta placa y predice Pico y Placa.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Estado global (lazy: se cargan en el primer request) ─────────────────────
_transformer  = None
_modelos_listos = False
RUTA_MODELO   = os.path.join(_root, "modelos", "transformer_pico_placa.pt")


def _inicializar_modelos():
    """
    Descarga y carga todos los modelos la primera vez que se recibe un request.
    Esto evita que Render / HuggingFace Spaces maten el proceso por OOM
    durante el arranque antes de que el health-check responda.
    """
    global _transformer, _modelos_listos

    if _modelos_listos:
        return

    # Descarga transformer desde HuggingFace (si no existe en disco)
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.join(_root, "modelos"), exist_ok=True)

    if not os.path.exists(RUTA_MODELO):
        log.info("Descargando transformer_pico_placa.pt desde HuggingFace...")
        hf_hub_download(
            repo_id   = "Huntercito/Deteccion_Pico_y_Placa",
            filename  = "transformer_pico_placa.pt",
            local_dir = os.path.join(_root, "modelos"),
            token     = os.environ.get("HF_TOKEN"),
        )
        log.info("Modelo descargado.")

    from Modern.modulo4_transformer import cargar_modelo
    log.info("Cargando modelo Transformer...")
    _transformer = cargar_modelo(ruta=RUTA_MODELO)
    log.info("Transformer listo.")

    _modelos_listos = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Verifica que el servidor esté activo (no carga modelos)."""
    return {
        "status"         : "ok",
        "modelo"         : "YOLO11 + EasyOCR + Transformer",
        "precision_test" : "98.22%",
        "modelos_cargados": _modelos_listos,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Recibe imagen JPG/PNG → retorna placa, restricción Pico y Placa y confianza.
    Los modelos se cargan automáticamente en el primer llamado.
    """
    # Lazy loading: carga modelos solo la primera vez
    _inicializar_modelos()

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG.")

    img_bytes = await file.read()

    from Modern.modulo1_deteccion_yolo import detectar_y_recortar_placa
    from Modern.modulo2_ocr import (
        ocr_easyocr_multi, preprocesar_por_tipo,
        detectar_tipo_placa, elegir_mejor_candidato,
    )
    from Modern.modulo4_transformer import predecir_pico_placa

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(img_bytes)
        ruta_tmp = tmp.name

    try:
        recorte_rgb, _, metodo = detectar_y_recortar_placa(ruta_tmp)

        if recorte_rgb is None:
            return {
                "placa": None, "formato_valido": False,
                "restriccion": None, "confianza_pct": 0.0,
                "metodo_deteccion": metodo,
                "mensaje": "No se detectó ninguna placa en la imagen.",
            }

        tipo_placa, _ = detectar_tipo_placa(recorte_rgb)
        recorte_pre   = preprocesar_por_tipo(recorte_rgb, tipo_placa)
        candidatos    = ocr_easyocr_multi(recorte_pre)
        placa, fmt_ok = elegir_mejor_candidato(candidatos)

        if not placa or placa == "???":
            return {
                "placa": None, "formato_valido": False,
                "restriccion": None, "confianza_pct": 0.0,
                "metodo_deteccion": metodo,
                "mensaje": "Placa detectada pero OCR no pudo leer el texto.",
            }

        resultado = predecir_pico_placa(placa, model=_transformer, verbose=False)

        return {
            "placa"           : resultado["placa"],
            "formato_valido"  : fmt_ok,
            "restriccion"     : resultado["restriccion"],
            "confianza_pct"   : resultado["confianza_pct"],
            "probabilidades"  : resultado["probabilidades"],
            "metodo_deteccion": metodo,
        }

    finally:
        os.unlink(ruta_tmp)


@app.post("/detect/texto")
def detect_texto(placa: str):
    """
    Prueba rápida sin imagen: recibe texto de placa, retorna predicción.
    Ejemplo: POST /detect/texto?placa=SKY424
    """
    _inicializar_modelos()

    if not placa:
        raise HTTPException(status_code=400, detail="El parámetro 'placa' es requerido.")

    from Modern.modulo4_transformer import predecir_pico_placa
    return predecir_pico_placa(placa, model=_transformer, verbose=False)