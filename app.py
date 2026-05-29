from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, os, sys

sys.path.insert(0, os.path.dirname(__file__))

# ── Descarga del modelo desde HuggingFace ─────────────
from huggingface_hub import hf_hub_download

os.makedirs("modelos", exist_ok=True)
if not os.path.exists("modelos/transformer_pico_placa.pt"):
    hf_hub_download(
        repo_id="Huntercito/Deteccion_Pico_y_Placa",
        filename="transformer_pico_placa.pt",
        local_dir="modelos",
        token=os.environ.get("HF_TOKEN")
    )

# ── Imports del proyecto ───────────────────────────────
from Modern.modulo0_config import PATRON_ANTIGUA, PATRON_NUEVA
from Modern.modulo1_deteccion_yolo import detectar_placa
from Modern.modulo2_ocr import leer_placa
from Modern.modulo4_transformer import cargar_modelo, predecir

app = FastAPI(title="Detección de Placas - Popayán")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "modelo": "YOLO11 + Transformer 98.22%"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    placa, formato_ok = leer_placa(frame)
    return {"placa": placa, "formato_valido": formato_ok}