from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, io, cv2, numpy as np
from PIL import Image
from Modern.main import procesar_imagen   # importa tu pipeline

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