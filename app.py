from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, io, cv2, numpy as np
from PIL import Image
from Modern.main import procesar_imagen   # importa tu pipeline
from huggingface_hub import hf_hub_download
import os

# --- DESCARGA DEL MODELO DESDE HUGGING FACE ---
# Se ejecuta una sola vez al arrancar el servidor en Render
MODEL_PATH = hf_hub_download(
    repo_id="Huntercito/Deteccion_Pico_y_Placa",  # Tu repositorio real
    filename="transformer_pico_placa.pt",
    token=os.environ.get("HF_TOKEN")  # Se leerá de las variables de entorno en Render
)
# -----------------------------------------------

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
    resultado = procesar_imagen(frame)
    return resultado
