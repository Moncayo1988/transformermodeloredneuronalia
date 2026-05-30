# ==============================================================================
# MÓDULO 2 — PREPROCESAMIENTO ADAPTATIVO Y OCR DE ALTA PRECISIÓN  v2.1
# Autor: Salomón Melenje
# Entrada : recorte RGB de la placa (numpy array)
# Salida  : DataFrame con placa, último dígito, tipo, motor OCR, etc.
# ==============================================================================

import cv2
import numpy as np
import re
import pandas as pd
import easyocr
import pytesseract

from modulo0_config import PATRON_ANTIGUA, PATRON_NUEVA, DIGITO_A_LETRA, LETRA_A_DIGITO, asignar_restriccion
from modulo2b_corrector_ocr import corregir_candidatos

_lector_easy: easyocr.Reader = None

def obtener_lector_easyocr() -> easyocr.Reader:
    global _lector_easy
    if _lector_easy is None:
        print("Cargando EasyOCR (primera vez ~30s)...")
        _lector_easy = easyocr.Reader(['en'], gpu=True)
        print("[OK] EasyOCR listo.")
    return _lector_easy

# Rangos HSV por tipo de fondo
_RANGOS_COLOR = {
    "amarillo": (np.array([10,40,60]),  np.array([40,255,255])),
    "blanco"  : (np.array([0,0,185]),   np.array([180,55,255])),
    "verde"   : (np.array([36,40,40]),  np.array([90,255,200])),
    "naranja" : (np.array([5,80,80]),   np.array([18,255,255])),
}
_RANGOS_FONDO = {k: v for k, v in _RANGOS_COLOR.items() if k not in ("blanco",)}

def detectar_tipo_placa(img_rgb: np.ndarray) -> tuple[str, dict]:
    hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    pcts = {n: np.mean(cv2.inRange(hsv, b, a))/255.0 for n,(b,a) in _RANGOS_COLOR.items()}
    tipo = max(pcts, key=pcts.get)
    return ("gris" if pcts[tipo] < 0.10 else tipo), pcts

def _preprocesar(img_rgb: np.ndarray, tipo: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_rgb.shape[:2]
    img  = img_rgb[int(h*0.08):int(h*0.82), :]   # zona de caracteres
    h, w = img.shape[:2]
    if w < 400:
        f = 400/w; img = cv2.resize(img,(int(w*f),int(h*f)),interpolation=cv2.INTER_LANCZOS4)

    den  = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    hsv  = cv2.cvtColor(den, cv2.COLOR_RGB2HSV)
    gris = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4)).apply(cv2.cvtColor(den, cv2.COLOR_RGB2GRAY))

    if tipo in _RANGOS_FONDO:
        bajo, alto = _RANGOS_FONDO[tipo]
        gris_mod = gris.copy(); gris_mod[cv2.inRange(hsv, bajo, alto) > 0] = 255
        _, bin_img = cv2.threshold(gris_mod, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif tipo == "blanco":
        bin_img = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    else:
        _, bin_img = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hh, ww = bin_img.shape
    if np.mean(bin_img[int(hh*0.2):int(hh*0.8), int(ww*0.1):int(ww*0.9)]) < 127:
        bin_img = cv2.bitwise_not(bin_img)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(1,1)))
    return bin_img, img

def _ocr_easyocr(img_rgb: np.ndarray, bin_img: np.ndarray) -> list[str]:
    lector = obtener_lector_easyocr()
    wl = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    gris_3ch = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    cands = []
    for img_in in [img_rgb, bin_img, gris_3ch]:
        try:
            for _, txt, conf in lector.readtext(img_in, allowlist=wl, detail=1, paragraph=False, width_ths=0.9, height_ths=0.9):
                if conf > 0.3:
                    c = re.sub(r'[^A-Z0-9]','', txt.upper())
                    if c: cands.append(c)
        except Exception: continue
    return cands

def _ocr_tesseract(bin_img: np.ndarray) -> list[str]:
    wl = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    cands = []
    for psm in [7, 8, 13]:
        try:
            raw = pytesseract.image_to_string(bin_img, config=f'--oem 3 --psm {psm} -c tessedit_char_whitelist={wl}')
            c   = re.sub(r'[^A-Z0-9]','', raw.upper())
            if c: cands.append(c)
        except Exception: continue
    return cands

def _corregir_posicional(t: str) -> str:
    t = re.sub(r'[^A-Z0-9]','', t.upper())[:7]
    if len(t) < 4: return t
    r = list(t)
    for i, c in enumerate(r):
        if i < 3 and c.isdigit() and c in DIGITO_A_LETRA:   r[i] = DIGITO_A_LETRA[c]
        elif 3 <= i < 6 and c.isalpha() and c in LETRA_A_DIGITO: r[i] = LETRA_A_DIGITO[c]
    return ''.join(r)

def _puntuar(t: str) -> int:
    t = re.sub(r'[^A-Z0-9]','', t.upper())
    if PATRON_ANTIGUA.match(t): return 100
    if PATRON_NUEVA.match(t):   return 95
    score = 0
    if len(t)>=6: score+=40
    if len(t)==6: score+=20
    if len(t)>=3 and all(c.isalpha() for c in t[:3]): score+=20
    if len(t)>=6 and all(c.isdigit() for c in t[3:6]): score+=20
    return score

def elegir_mejor_candidato(candidatos: list[str], tipo_placa: str = 'blanco') -> tuple[str, bool]:
    if not candidatos: return "", False
    avanzados = corregir_candidatos(candidatos, tipo_placa=tipo_placa, verbose=False)
    if avanzados:
        mejor = avanzados[0]
        return mejor, bool(PATRON_ANTIGUA.match(mejor) or PATRON_NUEVA.match(mejor))
    corregidos = [(_corregir_posicional(c), _puntuar(_corregir_posicional(c))) for c in candidatos if c]
    if not corregidos: return "", False
    mejor = max(corregidos, key=lambda x: x[1])[0]
    return mejor, bool(PATRON_ANTIGUA.match(mejor) or PATRON_NUEVA.match(mejor))

# Fix alias
PATRON_ANTIGA = PATRON_ANTIGUA
PATRON_NOVA   = PATRON_NUEVA

def _identificar_motor(texto: str, cand_easy: list, cand_tess: list) -> str:
    if texto == "N/A": return "N/A"
    p = texto[:3]
    if any(c[:3]==p or c==texto for c in cand_easy): return "EasyOCR"
    if any(c[:3]==p or c==texto for c in cand_tess): return "Tesseract"
    return "EasyOCR" if cand_easy else ("Tesseract" if cand_tess else "N/A")

def extraer_datos_placa(recorte_rgb: np.ndarray) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    if recorte_rgb is None: return None, None
    h, w = recorte_rgb.shape[:2]
    if h < 80 or w < 200:
        f = max(80/h, 200/w, 1.5)
        recorte_rgb = cv2.resize(recorte_rgb,(int(w*f),int(h*f)),interpolation=cv2.INTER_LANCZOS4)

    tipo, pcts = detectar_tipo_placa(recorte_rgb)
    print(f"  [INFO] Tipo: {tipo} (amarillo={pcts['amarillo']:.2f}, blanco={pcts['blanco']:.2f})")

    bin_img, img_recortada = _preprocesar(recorte_rgb, tipo)
    cand_easy = _ocr_easyocr(img_recortada, bin_img)
    cand_tess = _ocr_tesseract(bin_img)
    print(f"  [EasyOCR] {cand_easy}  [Tesseract] {cand_tess}")

    texto, fmt_ok = elegir_mejor_candidato(cand_easy + cand_tess, tipo_placa=tipo)
    if not texto or len(texto) < 6:
        if texto: print(f"  [WARN] Placa incompleta ({len(texto)} chars) → N/A")
        texto, fmt_ok = "N/A", False

    motor = _identificar_motor(texto, cand_easy, cand_tess)
    print(f"  [Resultado] '{texto}' | Formato exacto: {fmt_ok} | Motor: {motor}")

    nums  = re.findall(r'\d+', texto)
    ultimo = nums[-1][-1] if nums else "N/A"
    return pd.DataFrame({
        "placa_detectada":[texto], "ultimo_digito":[ultimo],
        "longitud_valida":[len(texto) in (6,7)], "formato_exacto":[fmt_ok],
        "tipo_placa":[tipo], "motor_ocr":[motor],
    }), bin_img

if __name__ == "__main__":
    import sys
    from PIL import Image as _PIL
    if len(sys.argv) < 2: print("Uso: python modulo2_ocr.py <ruta_placa>")
    else:
        img = np.array(_PIL.open(sys.argv[1]).convert("RGB"))
        df, _ = extraer_datos_placa(img)
        if df is not None: print(df.to_string())