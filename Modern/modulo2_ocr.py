# ==============================================================================
# MÓDULO 2 — PREPROCESAMIENTO ADAPTATIVO Y OCR DE ALTA PRECISIÓN  v2.1
# ==============================================================================
# Responsabilidad:
#   - Detectar el tipo de placa por color (amarilla, blanca, verde, etc.)
#   - Preprocesar adaptativamente según el tipo detectado
#   - Ejecutar OCR triple: EasyOCR (color + binaria + escala de grises) + Tesseract
#   - Corregir y puntuar candidatos según la estructura colombiana
#   - Aplicar corrector OCR post-procesado (modulo2b_corrector_ocr)
#   - Retornar DataFrame con resultado y metadatos
#
# Cambios v2.1 respecto a v2.0:
#   [+] Tercera pasada EasyOCR en escala de grises (mejora binarización destruida)
#   [+] Integración de modulo2b_corrector_ocr en elegir_mejor_candidato()
#   [+] Descarte de placas con < 6 chars como N/A (placa fuera del frame)
#   [+] Campo 'placa_corregida' en el DataFrame para trazabilidad
#   [+] Mejor detección del motor OCR ganador
#
# Entrada : recorte RGB de la placa (numpy array)
# Salida  : DataFrame con placa, último dígito, tipo, motor OCR, etc.
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
    asignar_restriccion
)

# Corrector de errores OCR post-procesado
from modulo2b_corrector_ocr import corregir_placa, corregir_candidatos


# ------------------------------------------------------------------------------
# 1. INICIALIZACIÓN DE EASYOCR (singleton)
# ------------------------------------------------------------------------------

_lector_easy: easyocr.Reader = None


def obtener_lector_easyocr() -> easyocr.Reader:
    """Carga EasyOCR una sola vez (primera llamada tarda ~30s en GPU)."""
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
    Detecta el color dominante del fondo de la placa.
    Si ningún color supera el 10% de píxeles, retorna "gris".
    Retorna: (tipo_str, dict de porcentajes por color)
    """
    hsv  = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2HSV)
    pcts = {
        nombre: np.mean(cv2.inRange(hsv, bajo, alto)) / 255.0
        for nombre, (bajo, alto) in _RANGOS_COLOR.items()
    }
    tipo = max(pcts, key=pcts.get)
    if pcts[tipo] < 0.10:
        tipo = "gris"
    return tipo, pcts


# ------------------------------------------------------------------------------
# 3. PREPROCESAMIENTO ADAPTATIVO
# ------------------------------------------------------------------------------

def recortar_zona_caracteres(imagen_rgb: np.ndarray) -> np.ndarray:
    """
    Recorta el 74% central vertical del recorte de la placa.
    Elimina texto de ciudad inferior y marcos decorativos.
    """
    h, w = imagen_rgb.shape[:2]
    y1 = int(h * 0.08)
    y2 = int(h * 0.82)
    return imagen_rgb[y1:y2, :]


def preprocesar_por_tipo(
    imagen_rgb: np.ndarray,
    tipo_placa: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesamiento adaptado al color de fondo:
      - amarillo/verde/naranja : máscara HSV + Otsu
      - blanco                 : umbral adaptativo gaussiano (ventana 11)
      - gris/desconocido       : CLAHE + Otsu

    Pasos comunes:
      1. Recorte zona de caracteres
      2. Escala mínima 400 px de ancho
      3. Denoising bilateral suave
      4. CLAHE de contraste local
      5. Binarización según tipo
      6. Inversión dinámica fondo/texto
      7. Morfología mínima (kernel 1×1)

    Retorna: (imagen_binaria, recorte_RGB_limpio)
    """
    img = recortar_zona_caracteres(imagen_rgb)

    h, w = img.shape[:2]
    if w < 400:
        factor = 400 / w
        img = cv2.resize(img, (int(w * factor), int(h * factor)),
                         interpolation=cv2.INTER_LANCZOS4)

    img_den = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    hsv   = cv2.cvtColor(img_den, cv2.COLOR_RGB2HSV)
    gris  = cv2.cvtColor(img_den, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gris  = clahe.apply(gris)

    _RANGOS_FONDO = {
        "amarillo": (np.array([10, 40, 60]),  np.array([40, 255, 255])),
        "verde"   : (np.array([36, 40, 40]),  np.array([90, 255, 200])),
        "naranja" : (np.array([5, 80, 80]),   np.array([18, 255, 255])),
    }

    if tipo_placa in _RANGOS_FONDO:
        bajo, alto = _RANGOS_FONDO[tipo_placa]
        mask_fondo = cv2.inRange(hsv, bajo, alto)
        gris_mod   = gris.copy()
        gris_mod[mask_fondo > 0] = 255
        _, bin_img = cv2.threshold(gris_mod, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif tipo_placa == "blanco":
        bin_img = cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 3
        )
    else:
        _, bin_img = cv2.threshold(gris, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inversión dinámica: si el centro es oscuro, invertir
    hh, ww = bin_img.shape
    centro = bin_img[int(hh * 0.2):int(hh * 0.8), int(ww * 0.1):int(ww * 0.9)]
    if np.mean(centro) < 127:
        bin_img = cv2.bitwise_not(bin_img)

    # Morfología mínima — no destruye trazos finos
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)

    return bin_img, img


# ------------------------------------------------------------------------------
# 4. MOTORES OCR
# ------------------------------------------------------------------------------

def ocr_easyocr_multi(
    imagen_rgb: np.ndarray,
    bin_img: np.ndarray
) -> list[str]:
    """
    EasyOCR en TRES pasadas:
      1. Imagen a color (RGB)
      2. Imagen binarizada
      3. Imagen en escala de grises convertida a 3 canales
         → mejora casos donde la binarización destruye la imagen (bajo contraste,
           ángulo extremo, placa desgastada)

    Filtra candidatos con confianza > 0.3.
    Whitelist: solo A-Z y 0-9.

    Retorna: lista de strings candidatos limpios.
    """
    lector    = obtener_lector_easyocr()
    whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    candidatos = []

    # Tercera pasada: escala de grises como imagen RGB (3 canales)
    gris_3ch = cv2.cvtColor(
        cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY),
        cv2.COLOR_GRAY2RGB
    )

    for img_in in [imagen_rgb, bin_img, gris_3ch]:
        try:
            res = lector.readtext(
                img_in, allowlist=whitelist,
                detail=1, paragraph=False,
                width_ths=0.9, height_ths=0.9
            )
            for _, texto, conf in res:
                if conf > 0.3:
                    limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
                    if limpio:
                        candidatos.append(limpio)
        except Exception:
            continue
    return candidatos


def ocr_tesseract_multi(bin_img: np.ndarray) -> list[str]:
    """
    Tesseract con PSM 7, 8 y 13 para acumular candidatos.
    Whitelist: solo A-Z y 0-9.
    Retorna: lista de strings candidatos limpios.
    """
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
# 5. CORRECCIÓN Y PUNTUACIÓN DE CANDIDATOS
# ------------------------------------------------------------------------------

def corregir_placa_colombiana(texto: str) -> str:
    """
    Corrección posicional estricta según el formato colombiano:
      pos 0-2 → deben ser LETRAS  (dígito confundido → letra equivalente)
      pos 3-5 → deben ser DÍGITOS (letra confundida  → dígito equivalente)
      pos 6   → libre (placa nueva formato moto)
    """
    texto = re.sub(r'[^A-Z0-9]', '', texto.upper())
    if len(texto) > 7:
        texto = texto[:7]
    if len(texto) < 4:
        return texto

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
    """
    Puntúa qué tan probable es que un string sea una placa colombiana válida.
      100 pts → formato antiguo exacto (ABC123)
       95 pts → formato nuevo exacto   (ABC12D)
       < 60   → coincidencia parcial
    """
    t = re.sub(r'[^A-Z0-9]', '', texto.upper())
    if PATRON_ANTIGUA.match(t): return 100
    if PATRON_NUEVA.match(t):   return 95

    score = 0
    if len(t) >= 6: score += 40
    if len(t) == 6: score += 20
    if len(t) == 7: score += 15
    if len(t) >= 3 and all(c.isalpha()  for c in t[:3]): score += 20
    if len(t) >= 6 and all(c.isdigit()  for c in t[3:6]): score += 20
    return score


def elegir_mejor_candidato(
    candidatos: list[str],
    tipo_placa: str = 'blanco'
) -> tuple[str, bool]:
    """
    Pipeline de selección con dos etapas:

    Etapa A — Corrector OCR (modulo2b_corrector_ocr):
      Aplica correcciones posicionales, recuperación de longitud y validación.
      Si produce resultados válidos, retorna el más frecuente.

    Etapa B — Fallback posicional clásico:
      Aplica corregir_placa_colombiana() + puntuar_candidato() sobre todos
      los candidatos y retorna el de mayor puntaje.

    Retorna: (mejor_placa, formato_exacto_bool)
    """
    if not candidatos:
        return "", False

    # ── Etapa A: corrector OCR avanzado ───────────────────────────────────────
    corregidos_avanzados = corregir_candidatos(
        candidatos, tipo_placa=tipo_placa, verbose=False
    )
    if corregidos_avanzados:
        mejor = corregidos_avanzados[0]
        fmt   = bool(PATRON_ANTIGUA.match(mejor) or PATRON_NUEVA.match(mejor))
        return mejor, fmt

    # ── Etapa B: fallback posicional clásico ──────────────────────────────────
    corregidos = [corregir_placa_colombiana(c) for c in candidatos if c]
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

def extraer_datos_placa(
    recorte_rgb: np.ndarray
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """
    Pipeline OCR completo de alta precisión v2.1:
      1. Asegura resolución mínima (80×200 px)
      2. Detecta tipo de placa por color
      3. Recorte de zona de caracteres + preprocesamiento adaptativo
      4. OCR triple: EasyOCR (color + binaria + grises) + Tesseract (PSM 7, 8, 13)
      5. Corrección y puntuación de candidatos (corrector avanzado + fallback)
      6. Descarte de placas incompletas (< 6 chars → N/A)
      7. Construye DataFrame con resultado y metadatos

    Retorna: (DataFrame_resultado, imagen_binaria) o (None, None) si falla.
    """
    if recorte_rgb is None:
        return None, None

    # Resolución mínima garantizada
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

    cand_easy = ocr_easyocr_multi(img_recortada, bin_img)
    cand_tess = ocr_tesseract_multi(bin_img)
    todos     = cand_easy + cand_tess

    print(f"  [EasyOCR]   {cand_easy}")
    print(f"  [Tesseract] {cand_tess}")

    texto_final, formato_exacto = elegir_mejor_candidato(todos, tipo_placa=tipo)

    # ── Descarte de placa incompleta (recorte YOLO insuficiente) ──────────────
    if not texto_final or len(texto_final) < 6:
        if texto_final:
            print(f"  [WARN] Placa incompleta ({len(texto_final)} chars) "
                  f"— recorte YOLO insuficiente, se descarta como N/A")
        texto_final   = "N/A"
        formato_exacto = False

    # ── Identificar motor OCR ganador ─────────────────────────────────────────
    motor = _identificar_motor(texto_final, cand_easy, cand_tess)

    print(f"  [Resultado] '{texto_final}' | Formato exacto: {formato_exacto} | Motor: {motor}")

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


def _identificar_motor(
    texto_final: str,
    cand_easy  : list[str],
    cand_tess  : list[str]
) -> str:
    """
    Determina qué motor OCR produjo el candidato ganador.
    Compara el inicio de texto_final con los candidatos de cada motor.
    """
    if texto_final == "N/A":
        return "N/A"

    prefijo = texto_final[:3]

    for c in cand_easy:
        if c[:3] == prefijo or c == texto_final:
            return "EasyOCR"

    for c in cand_tess:
        if c[:3] == prefijo or c == texto_final:
            return "Tesseract"

    # Si el corrector modificó el candidato, buscar coincidencia parcial
    if cand_easy:
        return "EasyOCR"
    if cand_tess:
        return "Tesseract"

    return "N/A"


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