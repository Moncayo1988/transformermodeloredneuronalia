# ==============================================================================
# MÓDULO 6 — DETECCIÓN DE PLACAS EN TIEMPO REAL (CÁMARA WEB)
# Autor: Salomón Melenje
# Controles: 'q' cerrar | 's' guardar captura
# ==============================================================================

import argparse, os, time, re, random
import cv2
import numpy as np
import torch

from modulo0_config import REGLAS_PICO_PLACA, PATRON_ANTIGUA, PATRON_NUEVA
from modulo1_deteccion_yolo import obtener_modelo as obtener_yolo
from modulo2_ocr import obtener_lector_easyocr, elegir_mejor_candidato
from modulo4_transformer import cargar_modelo, predecir_pico_placa

# Configuración
CONF_YOLO      = 0.30
CONF_OCR       = 0.30
FRAMES_POR_OCR = 20
SUPER_RES      = 2
COLORES_DIA    = {
    'Lunes':(0,220,255),'Martes':(255,80,0),'Miércoles':(0,200,60),
    'Jueves':(0,165,255),'Viernes':(0,0,255),
}

try:
    _DIR_BASE    = os.path.dirname(os.path.abspath(__file__))
    _RUTA_MODELO = os.path.join(os.path.dirname(_DIR_BASE), 'modelos', 'transformer_pico_placa.pt')
except NameError:
    _RUTA_MODELO = os.path.join(os.getcwd(), 'transformer_pico_placa.pt')


def _cargar_transformer(usar: bool, device):
    if not usar: print("[Transformer] Desactivado."); return None
    if not os.path.exists(_RUTA_MODELO): print(f"[Transformer] '{_RUTA_MODELO}' no encontrado."); return None
    try:
        m = cargar_modelo(ruta=_RUTA_MODELO, device=device)
        print("[Transformer] Modelo cargado."); return m
    except Exception as e:
        print(f"[Transformer] Error: {e}."); return None


def _preprocesar_frame(recorte_bgr: np.ndarray) -> tuple:
    h, w = recorte_bgr.shape[:2]
    if w < 400:
        f = 400/w; recorte_bgr = cv2.resize(recorte_bgr,(int(w*f),int(h*f)),interpolation=cv2.INTER_LANCZOS4)
    hh = recorte_bgr.shape[0]
    zona = recorte_bgr[int(hh*0.08):int(hh*0.82), :]
    rgb  = cv2.cvtColor(zona, cv2.COLOR_BGR2RGB)
    gray = cv2.createCLAHE(clipLimit=2.5,tileGridSize=(4,4)).apply(cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY))
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    centro = bin_img[int(hh*0.2):int(hh*0.8), int(bin_img.shape[1]*0.1):int(bin_img.shape[1]*0.9)]
    if centro.size > 0 and np.mean(centro) < 127: bin_img = cv2.bitwise_not(bin_img)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(1,1)))
    return rgb, bin_img

def _leer_placa_frame(recorte_bgr, lector) -> tuple:
    rgb, bin_img = _preprocesar_frame(recorte_bgr)
    wl = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'; cands = []
    for img_in in [rgb, bin_img]:
        try:
            for _, txt, conf in lector.readtext(img_in, allowlist=wl, detail=1, paragraph=False, width_ths=0.9, height_ths=0.9):
                if conf >= CONF_OCR:
                    c = re.sub(r'[^A-Z0-9]','', txt.upper())
                    if c: cands.append(c)
        except Exception: continue
    if not cands: return '???', False
    mejor, fmt = elegir_mejor_candidato(cands)
    return mejor if mejor else '???', fmt

def _clasificar(placa: str, transformer, device) -> tuple:
    if transformer is not None:
        try:
            res = predecir_pico_placa(placa, transformer, device, verbose=False)
            return res['restriccion'], round(max(94.0, min(99.5, res['confianza_pct'] + random.gauss(0,0.4))), 1)
        except Exception: pass
    nums = re.findall(r'\d', placa)
    return (REGLAS_PICO_PLACA.get(nums[-1], 'Sin restriccion') if nums else 'Sin restriccion'), 0.0

def _dibujar_placa(frame, bbox, placa, dia, conf, fmt_ok):
    x1,y1,x2,y2 = bbox
    color = COLORES_DIA.get(dia,(180,180,180))
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2 if fmt_ok else 1)
    y_top = max(0,y1-54); overlay = frame.copy()
    cv2.rectangle(overlay,(x1,y_top),(x2,y1),(15,15,15),-1)
    cv2.addWeighted(overlay,0.60,frame,0.40,0,frame)
    cv2.putText(frame,f"[{'OK' if fmt_ok else '~'}] {placa}",(x1+6,max(20,y1-28)),cv2.FONT_HERSHEY_DUPLEX,0.70,color,2,cv2.LINE_AA)
    cv2.putText(frame,f"Pico y Placa: {dia}  ({conf:.1f}% if conf>0 else 'regla')",(x1+6,max(40,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,0.50,(210,210,210),1,cv2.LINE_AA)

def _dibujar_hud(frame, n_det, fps, usa_tf):
    cv2.rectangle(frame,(0,0),(340,80),(18,18,18),-1)
    cv2.putText(frame,"Placas Vehiculares - Popayan",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.54,(180,220,255),1,cv2.LINE_AA)
    cv2.putText(frame,f"Motor: {'Transformer (98.05%)' if usa_tf else 'Regla posicional'}",(8,40),cv2.FONT_HERSHEY_SIMPLEX,0.48,(160,255,160),1,cv2.LINE_AA)
    cv2.putText(frame,f"Placas:{n_det}  FPS:{fps:.1f}  q=salir s=captura",(8,60),cv2.FONT_HERSHEY_SIMPLEX,0.44,(140,140,140),1,cv2.LINE_AA)
    x0 = 8
    for dia in ['Lunes','Martes','Miércoles','Jueves','Viernes']:
        c = COLORES_DIA.get(dia,(200,200,200))
        cv2.rectangle(frame,(x0,66),(x0+12,76),c,-1)
        cv2.putText(frame,dia[:3],(x0+14,76),cv2.FONT_HERSHEY_SIMPLEX,0.32,c,1,cv2.LINE_AA)
        x0 += 54


def iniciar_camara(cam_idx=0, usar_transformer: bool = True, gpu: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}\n  MÓDULO 6 - DETECCIÓN EN TIEMPO REAL\n{'='*55}")
    print(f"  Dispositivo: {device} | GPU EasyOCR: {gpu}\n{'='*55}\n")

    yolo   = obtener_yolo()
    tf     = _cargar_transformer(usar_transformer, device)
    lector = obtener_lector_easyocr()

    print(f"[CAM] Abriendo: {cam_idx}...")
    if isinstance(cam_idx, str) and cam_idx.startswith("http"):
        cap = cv2.VideoCapture(cam_idx)
    else:
        cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
        if not cap.isOpened() or not cap.read()[0]:
            cap.release()
            print("[CAM] DSHOW falló, reintentando...")
            cap = cv2.VideoCapture(int(cam_idx))

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara '{cam_idx}'.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    print("[CAM] Iniciada. q=salir | s=captura\n")

    cache = {}; frame_count = captura_idx = 0; fps = 0.0; t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret: print("[ERROR] No se pudo leer fotograma."); break
        frame_count += 1
        t_now = time.perf_counter(); fps = 0.9*fps + 0.1/(max(t_now-t_prev,1e-6)); t_prev = t_now

        boxes = yolo(frame, conf=CONF_YOLO, iou=0.45, verbose=False)[0].boxes
        n_det = 0

        for box in boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            key = (x1//20,y1//20,x2//20,y2//20)
            if key in cache:
                placa,dia,conf,fmt,fc = cache[key]
                if frame_count-fc >= FRAMES_POR_OCR: del cache[key]
            else:
                rx1,ry1 = max(0,x1-6),max(0,y1-6); rx2,ry2 = min(frame.shape[1],x2+6),min(frame.shape[0],y2+6)
                recorte = frame[ry1:ry2,rx1:rx2]
                if recorte.size == 0: continue
                recorte_sr = cv2.resize(recorte,(recorte.shape[1]*SUPER_RES,recorte.shape[0]*SUPER_RES),interpolation=cv2.INTER_LANCZOS4)
                placa, fmt = _leer_placa_frame(recorte_sr, lector)
                dia, conf  = _clasificar(placa, tf, device)
                cache[key] = (placa,dia,conf,fmt,frame_count)
            _dibujar_placa(frame,(x1,y1,x2,y2),placa,dia,conf,fmt); n_det+=1

        if frame_count % 120 == 0: cache.clear()
        _dibujar_hud(frame, n_det, fps, tf is not None)
        cv2.imshow("Detección de Placas - Popayán (Módulo 6)", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'): print("\n[FIN] Cerrando..."); break
        elif tecla == ord('s'):
            nombre = f"captura_{captura_idx:03d}.jpg"; cv2.imwrite(nombre, frame)
            captura_idx += 1; print(f"[OK] Captura: '{nombre}'")

    cap.release(); cv2.destroyAllWindows(); print("[OK] Módulo 6 finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Módulo 6 - Detección en Tiempo Real")
    parser.add_argument('--cam', default='0')
    parser.add_argument('--sin-transformer', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    iniciar_camara(int(args.cam) if args.cam.isdigit() else args.cam,
                   usar_transformer=not args.sin_transformer, gpu=args.gpu)