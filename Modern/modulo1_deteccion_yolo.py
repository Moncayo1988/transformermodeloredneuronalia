# ==============================================================================
# MÓDULO 1 — DETECCIÓN DE PLACA VEHICULAR CON YOLO
# Autor: Michael Giraldo Buitrón.
# Entrada : ruta de imagen (JPG, PNG, WEBP, AVIF)
# Salida  : recorte RGB de la placa, imagen marcada, método usado
# ==============================================================================

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

_modelo_yolo: YOLO = None

def obtener_modelo() -> YOLO:
    global _modelo_yolo
    if _modelo_yolo is None:
        try:
            print("Descargando modelo YOLO11 especializado...")
            path = hf_hub_download(
                repo_id="morsetechlab/yolov11-license-plate-detection",
                filename="license-plate-finetune-v1x.pt"
            )
            _modelo_yolo = YOLO(path)
            print("[OK] YOLO11 especializado cargado.")
        except Exception as e:
            print(f"[AVISO] Modelo especializado falló ({e}). Usando yolo11n...")
            _modelo_yolo = YOLO('yolo11n.pt')
    return _modelo_yolo

def _leer_imagen(ruta: str) -> np.ndarray | None:
    img = cv2.imread(ruta)
    if img is not None: return img
    try:
        return cv2.cvtColor(np.array(Image.open(ruta).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  [ERROR] No se pudo leer '{ruta}': {e}"); return None

def _mejorar_imagen(img: np.ndarray) -> np.ndarray:
    dn  = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
    l, a, b = cv2.split(cv2.cvtColor(dn, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    m = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    return cv2.filter2D(m, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))

def _fallback_hsv(img: np.ndarray) -> list[tuple]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    rangos = [
        (np.array([12,60,80]),  np.array([38,255,255])),
        (np.array([0,0,185]),   np.array([180,55,255])),
        (np.array([36,40,40]),  np.array([90,255,200])),
        (np.array([5,80,80]),   np.array([18,255,255])),
    ]
    mascara = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for bajo, alto in rangos:
        mascara = cv2.bitwise_or(mascara, cv2.inRange(hsv, bajo, alto))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (20,6))
    mascara = cv2.morphologyEx(cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    return [
        cv2.boundingRect(c) for c in cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if (lambda x,y,cw,ch: cw*ch >= h*w*0.003 and 1.5 <= cw/max(ch,1) <= 7.0 and ch>=12 and cw>=40)(*cv2.boundingRect(c))
    ]

def detectar_y_recortar_placa(ruta: str) -> tuple:
    model = obtener_modelo()
    img   = _leer_imagen(ruta)
    if img is None: return None, None, "error_lectura"
    h, w = img.shape[:2]

    def _yolo(imagen, umbral, etiqueta):
        boxes = model(imagen, conf=umbral, iou=0.45, verbose=False)[0].boxes
        if not len(boxes): return None, None, None
        best = boxes[boxes.conf.argmax()]
        return tuple(map(int, best.xyxy[0].tolist())), best.conf.item(), etiqueta

    bbox, conf_val, metodo = _yolo(img, 0.35, "YOLO (conf=0.35)")
    if bbox is None:
        img_m = _mejorar_imagen(img)
        bbox, conf_val, metodo = _yolo(img_m, 0.25, "YOLO + mejora")
        if bbox: img = img_m
    for umbral, etiq in [(0.15,"YOLO (conf=0.15)"),(0.05,"YOLO (conf=0.05)")]:
        if bbox is None:
            bbox, conf_val, metodo = _yolo(img, umbral, etiq)

    if bbox is None:
        print("  [INFO] Fallback por color HSV...")
        cands = _fallback_hsv(img)
        if not cands: return None, None, "sin_deteccion"
        x, y, cw, ch = max(cands, key=lambda b: b[2]*b[3])
        bbox, conf_val, metodo = (x, y, x+cw, y+ch), 0.0, "HSV fallback"

    xmin, ymin, xmax, ymax = bbox
    mg_h = max(8, int((ymax-ymin)*0.08)); mg_w = max(8, int((xmax-xmin)*0.05))
    xmin,ymin = max(0,xmin-mg_w), max(0,ymin-mg_h)
    xmax,ymax = min(w,xmax+mg_w), min(h,ymax+mg_h)

    recorte = img[ymin:ymax, xmin:xmax]
    if recorte.size == 0: return None, None, "recorte_vacio"

    nw, nh = int(recorte.shape[1]*4), int(recorte.shape[0]*4)
    recorte_rgb = cv2.cvtColor(cv2.resize(recorte,(nw,nh),interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)

    img_marcada = img.copy()
    color = (0,255,0) if conf_val > 0 else (0,165,255)
    cv2.rectangle(img_marcada, (xmin,ymin), (xmax,ymax), color, 3)
    cv2.putText(img_marcada, f"{conf_val*100:.1f}%" if conf_val>0 else "HSV",
                (xmin, max(ymin-10,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    print(f"  [OK] Placa detectada — {metodo}" + (f" | {conf_val*100:.1f}%" if conf_val>0 else ""))
    return recorte_rgb, cv2.cvtColor(img_marcada, cv2.COLOR_BGR2RGB), metodo

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python modulo1_deteccion_yolo.py <ruta_imagen>")
    else:
        r, m, met = detectar_y_recortar_placa(sys.argv[1])
        if r is not None:
            import matplotlib.pyplot as plt
            fig, (a1,a2) = plt.subplots(1,2,figsize=(12,4))
            a1.imshow(m); a1.set_title(f"Detección ({met})"); a1.axis('off')
            a2.imshow(r); a2.set_title("Recorte SR ×4"); a2.axis('off')
            plt.tight_layout(); plt.show()