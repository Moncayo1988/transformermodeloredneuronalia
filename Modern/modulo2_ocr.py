# ==============================================================================
# MÓDULO 2 — PREPROCESAMIENTO ADAPTATIVO Y OCR DE ALTA PRECISIÓN
# ==============================================================================
# CAMBIOS v3:
#   - preprocesar_por_tipo() ya NO usa máscara HSV para amarillo/verde/naranja
#   - Nuevo flujo unificado: CLAHE + Otsu para TODOS los tipos excepto blanco
#   - detectar_tipo_placa() analiza solo la zona CENTRAL (evita carrocería)
#   - corregir_placa_colombiana() y elegir_mejor_candidato() sin cambios (v2)
# ==============================================================================

import cv2
import numpy as np
import re
import pandas as pd
import easyocr
import pytesseract

from modulo0_config import (
    PATRON_ANTIGUA, PATRON_NUEVA,
    DIGITO_A_LETRA, LETRA_A_DIGITO,
    CORRECCION_VISUAL,
    asignar_restriccion
)

# ------------------------------------------------------------------------------
# 1. EASYOCR SINGLETON
# ------------------------------------------------------------------------------
_lector_easy: easyocr.Reader = None

def obtener_lector_easyocr() -> easyocr.Reader:
    global _lector_easy
    if _lector_easy is None:
        print("Cargando EasyOCR (primera vez ~30s)...")
        _lector_easy = easyocr.Reader(['en'], gpu=True)
        print("[OK] EasyOCR listo.")
    return _lector_easy

# ------------------------------------------------------------------------------
# 2. DETECCIÓN DEL TIPO DE PLACA POR COLOR
# ------------------------------------------------------------------------------
_RANGOS_COLOR = {
    "amarillo": (np.array([10, 40, 60]),  np.array([40, 255, 255])),
    "blanco"  : (np.array([0, 0, 185]),   np.array([180, 55, 255])),
    "verde"   : (np.array([36, 40, 40]),  np.array([90, 255, 200])),
    "naranja" : (np.array([5, 80, 80]),   np.array([18, 255, 255])),
}

def detectar_tipo_placa(imagen_rgb: np.ndarray) -> tuple[str, dict]:
    """
    CAMBIO v3: analiza solo la zona central (30%-85% vertical, 10%-90% horizontal)
    para que la carrocería del vehículo no contamine la detección de color.
    """
    h, w = imagen_rgb.shape[:2]
    y1, y2 = int(h * 0.30), int(h * 0.85)
    x1, x2 = int(w * 0.10), int(w * 0.90)
    zona_central = imagen_rgb[y1:y2, x1:x2]

    hsv  = cv2.cvtColor(zona_central, cv2.COLOR_RGB2HSV)
    pcts = {
        nombre: np.mean(cv2.inRange(hsv, bajo, alto)) / 255.0
        for nombre, (bajo, alto) in _RANGOS_COLOR.items()
    }
    tipo = max(pcts, key=pcts.get)
    if pcts[tipo] < 0.15:
        tipo = "gris"
    return tipo, pcts

# ------------------------------------------------------------------------------
# 3. PREPROCESAMIENTO ADAPTATIVO
# ------------------------------------------------------------------------------
def recortar_zona_caracteres(imagen_rgb: np.ndarray) -> np.ndarray:
    h, w = imagen_rgb.shape[:2]
    return imagen_rgb[int(h * 0.05):int(h * 0.88), :]

def preprocesar_por_tipo(
    imagen_rgb: np.ndarray,
    tipo_placa: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    CAMBIO v3: eliminada la máscara HSV para amarillo/verde/naranja.
    Ahora todos los tipos usan CLAHE + Otsu, excepto blanco que usa
    umbral adaptativo gaussiano.

    Motivo: la máscara HSV borraba los caracteres cuando YOLO recortaba
    con carrocería incluida (color dominante = color del carro, no la placa).
    Otsu trabaja con contraste local y no tiene ese problema.
    """
    img = recortar_zona_caracteres(imagen_rgb)

    h, w = img.shape[:2]
    if w < 400:
        factor = 400 / w
        img = cv2.resize(img, (int(w * factor), int(h * factor)),
                         interpolation=cv2.INTER_LANCZOS4)

    # Denoising adaptado al tamaño
    if w < 300:
        img_den = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    else:
        img_den = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    gris  = cv2.cvtColor(img_den, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gris  = clahe.apply(gris)

    if tipo_placa == "blanco":
        bin_img = cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 3
        )
    else:
        # Otsu: robusto sin depender del color del fondo
        _, bin_img = cv2.threshold(
            gris, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Inversión dinámica
    hh, ww = bin_img.shape
    centro = bin_img[int(hh * 0.2):int(hh * 0.8), int(ww * 0.1):int(ww * 0.9)]
    if np.mean(centro) < 127:
        bin_img = cv2.bitwise_not(bin_img)

    # Morfología mínima
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,  k)

    return bin_img, img

# ------------------------------------------------------------------------------
# 4. MOTORES OCR
# ------------------------------------------------------------------------------
def ocr_easyocr_multi(imagen_rgb: np.ndarray, bin_img: np.ndarray) -> list[str]:
    lector    = obtener_lector_easyocr()
    whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    candidatos = []

    lab = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    img_contraste = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    for img_in in [imagen_rgb, img_contraste, bin_img]:
        try:
            res = lector.readtext(
                img_in, allowlist=whitelist,
                detail=1, paragraph=False,
                width_ths=0.9, height_ths=0.9
            )
            for _, texto, conf in res:
                if conf > 0.25:
                    limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
                    if limpio:
                        candidatos.append(limpio)
        except Exception:
            continue
    return candidatos

def ocr_tesseract_multi(bin_img: np.ndarray) -> list[str]:
    whitelist  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    candidatos = []
    for psm in [7, 8, 13]:
        cfg = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
        try:
            raw    = pytesseract.image_to_string(bin_img, config=cfg)
            limpio = re.sub(r'[^A-Z0-9]', '', raw.upper())
            if limpio:
                candidatos.append(limpio)
        except Exception:
            continue
    return candidatos

# ------------------------------------------------------------------------------
# 5. CORRECCIÓN Y PUNTUACIÓN  (sin cambios desde v2)
# ------------------------------------------------------------------------------
def _longitud_valida(texto: str) -> bool:
    return len(texto) in (6, 7)

def _extraer_candidato_principal(candidatos: list[str]) -> list[str]:
    validos = [c for c in candidatos if _longitud_valida(c)]
    if validos:
        validos.sort(key=lambda x: (abs(len(x) - 6), x))
        return validos
    return candidatos

def corregir_placa_colombiana(texto: str) -> str:
    texto = re.sub(r'[^A-Z0-9]', '', texto.upper())
    if len(texto) > 7:
        texto = texto[:7]
    if len(texto) < 4:
        return texto

    # Paso 1: corrección visual (ej. UJLY246 → JLY246)
    if len(texto) == 7 and texto[0] in CORRECCION_VISUAL:
        candidato_sin_prefijo = texto[1:]
        if (all(c.isalpha()  for c in candidato_sin_prefijo[:3]) and
                all(c.isdigit() for c in candidato_sin_prefijo[3:6])):
            texto = candidato_sin_prefijo

    # Paso 2: corrección posicional
    r = list(texto)
    for i, c in enumerate(r):
        if i < 3:
            if c.isdigit() and c in DIGITO_A_LETRA:
                r[i] = DIGITO_A_LETRA[c]
        elif i < 6:
            if c.isalpha() and c in LETRA_A_DIGITO:
                r[i] = LETRA_A_DIGITO[c]
    return ''.join(r)

def puntuar_candidato(texto: str) -> int:
    t = re.sub(r'[^A-Z0-9]', '', texto.upper())
    if PATRON_ANTIGUA.match(t): return 100
    if PATRON_NUEVA.match(t):   return 95
    score = 0
    if len(t) == 6:   score += 40
    elif len(t) == 7: score += 35
    elif len(t) >= 4: score += 10
    else:             score -= 20
    if len(t) >= 3 and all(c.isalpha()  for c in t[:3]): score += 20
    if len(t) >= 6 and all(c.isdigit()  for c in t[3:6]): score += 20
    return score

def elegir_mejor_candidato(candidatos: list[str]) -> tuple[str, bool]:
    if not candidatos:
        return "", False
    filtrados  = _extraer_candidato_principal(candidatos)
    corregidos = [corregir_placa_colombiana(c) for c in filtrados if c]
    puntuados  = [(c, puntuar_candidato(c)) for c in corregidos if c]
    if not puntuados:
        return "", False
    puntuados.sort(key=lambda x: x[1], reverse=True)
    mejor = puntuados[0][0]
    fmt   = bool(PATRON_ANTIGUA.match(mejor) or PATRON_NUEVA.match(mejor))
    return mejor, fmt

# ------------------------------------------------------------------------------
# 6. PIPELINE OCR COMPLETO
# ------------------------------------------------------------------------------
def extraer_datos_placa(recorte_rgb: np.ndarray) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    if recorte_rgb is None:
        return None, None

    h, w = recorte_rgb.shape[:2]
    if h < 80 or w < 200:
        factor      = max(80 / h, 200 / w, 1.5)
        recorte_rgb = cv2.resize(recorte_rgb,
                                  (int(w * factor), int(h * factor)),
                                  interpolation=cv2.INTER_LANCZOS4)

    tipo, pcts = detectar_tipo_placa(recorte_rgb)
    print(f"  [INFO] Tipo de placa: {tipo} "
          f"(amarillo={pcts['amarillo']:.2f}, blanco={pcts['blanco']:.2f})")

    bin_img, img_recortada = preprocesar_por_tipo(recorte_rgb, tipo)

    # --- DEBUG: guardar imágenes intermedias ---
    cv2.imwrite("debug_binaria.png", bin_img)
    cv2.imwrite("debug_recorte_rgb.png", cv2.cvtColor(img_recortada, cv2.COLOR_RGB2BGR))

    cand_easy = ocr_easyocr_multi(img_recortada, bin_img)
    cand_tess = ocr_tesseract_multi(bin_img)
    todos     = cand_easy + cand_tess

    print(f"  [EasyOCR]   {cand_easy}")
    print(f"  [Tesseract] {cand_tess}")

    texto_final, formato_exacto = elegir_mejor_candidato(todos)
    if not texto_final:
        texto_final = "N/A"

    motor = ("EasyOCR"
             if cand_easy and texto_final[:2] == cand_easy[0][:2]
             else "Tesseract")
    print(f"  [Resultado] '{texto_final}' | Formato exacto: {formato_exacto}")

    numeros       = re.findall(r'\d+', texto_final)
    ultimo_digito = numeros[-1][-1] if numeros else "N/A"

    df = pd.DataFrame({
        "placa_detectada" : [texto_final],
        "ultimo_digito"   : [ultimo_digito],
        "longitud_valida" : [len(texto_final) in (6, 7)],
        "formato_exacto"  : [formato_exacto],
        "tipo_placa"      : [tipo],
        "motor_ocr"       : [motor],
    })
    return df, bin_img

# ------------------------------------------------------------------------------
# 7. PRUEBA RÁPIDA
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from PIL import Image
    if len(sys.argv) < 2:
        print("Uso: python modulo2_ocr.py <ruta_recorte_placa>")
    else:
        img = np.array(Image.open(sys.argv[1]).convert("RGB"))
        df, binaria = extraer_datos_placa(img)
        if df is not None:
            print(df.to_string())