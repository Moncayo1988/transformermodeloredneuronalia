# ==============================================================================
# MÓDULO 1 — DETECCIÓN DE PLACA VEHICULAR CON YOLO
# ==============================================================================
# Responsabilidad:
#   - Cargar el modelo YOLO11 especializado en placas
#   - Detectar y recortar la región de la placa en una imagen
#   - Estrategia en cascada: YOLO (3 umbrales) → Fallback HSV
#   - Aplicar super-resolución ×4 (Lanczos4) al recorte final
#
# Entrada : ruta de imagen (JPG, PNG, WEBP, AVIF)
# Salida  : recorte RGB de la placa, imagen marcada, método usado
# ==============================================================================

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# ------------------------------------------------------------------------------
# 1. CARGA DEL MODELO YOLO
# ------------------------------------------------------------------------------

def cargar_modelo_yolo() -> YOLO:
    """
    Descarga y carga el modelo YOLO11 especializado en detección de placas.
    Si falla, usa el modelo genérico yolo11n como fallback.

    Retorna: instancia de YOLO lista para inferencia.
    """
    try:
        print("Descargando modelo YOLO11 especializado en placas...")
        model_path = hf_hub_download(
            repo_id="morsetechlab/yolov11-license-plate-detection",
            filename="license-plate-finetune-v1x.pt"
        )
        model = YOLO(model_path)
        print("[OK] YOLO11 especializado cargado.")
        return model
    except Exception as e:
        print(f"[AVISO] Modelo especializado falló ({e}). Usando yolo11n...")
        return YOLO('yolo11n.pt')


# Variable global del modelo (se inicializa una sola vez)
_modelo_yolo: YOLO = None


def obtener_modelo() -> YOLO:
    """Singleton: carga el modelo solo la primera vez que se llama."""
    global _modelo_yolo
    if _modelo_yolo is None:
        _modelo_yolo = cargar_modelo_yolo()
    return _modelo_yolo


# ------------------------------------------------------------------------------
# 2. UTILIDADES DE IMAGEN
# ------------------------------------------------------------------------------

def leer_imagen_universal(ruta: str) -> np.ndarray | None:
    """
    Lee una imagen en cualquier formato (JPG, PNG, WEBP, AVIF).
    Usa OpenCV primero; si falla, usa Pillow como fallback.

    Retorna: imagen en formato BGR (numpy) o None si no se puede leer.
    """
    img = cv2.imread(ruta)
    if img is not None:
        return img
    try:
        pil    = Image.open(ruta).convert("RGB")
        img_np = np.array(pil)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  [ERROR] No se pudo leer '{ruta}': {e}")
        return None


def mejorar_imagen_borrosa(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica denoising + CLAHE + sharpening para mejorar imágenes borrosas.
    Útil cuando YOLO no detecta la placa en la imagen original.
    """
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 6, 6, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    mejorada = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(mejorada, -1, kernel)


# ------------------------------------------------------------------------------
# 3. FALLBACK POR COLOR HSV
# ------------------------------------------------------------------------------

def fallback_color_hsv(img_bgr: np.ndarray) -> list[tuple]:
    """
    Detecta candidatos a placa por segmentación de color cuando YOLO falla.
    Busca regiones amarillas, blancas, verdes y naranjas con proporciones
    típicas de una placa (relación ancho/alto entre 1.5 y 7.0).

    Retorna: lista de (x, y, ancho, alto) de candidatos detectados.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]

    rangos = {
        "amarillo": (np.array([12, 60, 80]),  np.array([38, 255, 255])),
        "blanco"  : (np.array([0, 0, 185]),   np.array([180, 55, 255])),
        "verde"   : (np.array([36, 40, 40]),  np.array([90, 255, 200])),
        "naranja" : (np.array([5, 80, 80]),   np.array([18, 255, 255])),
    }

    mascara = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for _, (bajo, alto) in rangos.items():
        mascara = cv2.bitwise_or(mascara, cv2.inRange(hsv, bajo, alto))

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  k)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for cnt in contornos:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        prop = cw / float(ch) if ch > 0 else 0
        if area < h * w * 0.003 or prop < 1.5 or prop > 7.0 or ch < 12 or cw < 40:
            continue
        candidatos.append((x, y, cw, ch))
    return candidatos


# ------------------------------------------------------------------------------
# 4. DETECCIÓN PRINCIPAL EN CASCADA
# ------------------------------------------------------------------------------

def detectar_y_recortar_placa(
    ruta_imagen: str
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Detecta la placa vehicular en cascada:
      1. YOLO conf=0.35 (imagen original)
      2. YOLO conf=0.25 (imagen mejorada con denoising+CLAHE+sharpening)
      3. YOLO conf=0.15 (umbral mínimo)
      4. Fallback HSV   (segmentación por color)

    Aplica super-resolución ×4 (Lanczos4) al recorte final.

    Retorna:
      - recorte_rgb   : imagen RGB de la placa recortada y ampliada
      - img_marcada   : imagen original con bounding box dibujado
      - metodo        : string describiendo qué método encontró la placa
    """
    model = obtener_modelo()
    img   = leer_imagen_universal(ruta_imagen)
    if img is None:
        return None, None, "error_lectura"

    h, w = img.shape[:2]

    # -- Helper interno para intentar YOLO con un umbral dado --
    def intentar_yolo(imagen, umbral, etiqueta):
        res   = model(imagen, conf=umbral, iou=0.45, verbose=False)
        boxes = res[0].boxes
        if len(boxes) == 0:
            return None, None, None
        best   = boxes[boxes.conf.argmax()]
        coords = map(int, best.xyxy[0].tolist())
        return tuple(coords), best.conf.item(), etiqueta

    # Intento 1 — imagen original, conf alta
    bbox, conf_val, metodo = intentar_yolo(img, 0.35, "YOLO (conf=0.35)")

    # Intento 2 — imagen mejorada
    if bbox is None:
        print("  [INFO] Mejorando imagen borrosa...")
        img_m = mejorar_imagen_borrosa(img)
        bbox, conf_val, metodo = intentar_yolo(img_m, 0.25, "YOLO + mejora")
        if bbox:
            img = img_m

    # Intento 3 — umbral muy bajo
    if bbox is None:
        print("  [INFO] YOLO umbral muy bajo (0.15)...")
        bbox, conf_val, metodo = intentar_yolo(img, 0.15, "YOLO (conf=0.15)")

    # Intento 4 — fallback HSV
    if bbox is None:
        print("  [INFO] Fallback por color HSV...")
        candidatos = fallback_color_hsv(img)
        if not candidatos:
            print("  [ERROR] No se detectó ninguna placa.")
            return None, None, "sin_deteccion"
        candidatos.sort(key=lambda c: c[2] * c[3], reverse=True)
        x, y, cw, ch = candidatos[0]
        bbox      = (x, y, x + cw, y + ch)
        conf_val  = 0.0
        metodo    = "Fallback HSV"

    # -- Expansión de margen y recorte --
    xmin, ymin, xmax, ymax = bbox
    mg_h = max(8, int((ymax - ymin) * 0.08))
    mg_w = max(8, int((xmax - xmin) * 0.05))
    xmin = max(0, xmin - mg_w);  ymin = max(0, ymin - mg_h)
    xmax = min(w, xmax + mg_w);  ymax = min(h, ymax + mg_h)

    recorte = img[ymin:ymax, xmin:xmax]
    if recorte.size == 0:
        return None, None, "recorte_vacio"

    # -- Super-resolución ×4 con Lanczos4 --
    nw         = int(recorte.shape[1] * 4)
    nh         = int(recorte.shape[0] * 4)
    recorte_sr  = cv2.resize(recorte, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    recorte_rgb = cv2.cvtColor(recorte_sr, cv2.COLOR_BGR2RGB)

    # -- Imagen marcada para visualización --
    img_marcada = img.copy()
    color = (0, 255, 0) if conf_val > 0 else (0, 165, 255)
    cv2.rectangle(img_marcada, (xmin, ymin), (xmax, ymax), color, 3)
    label = f"{conf_val * 100:.1f}%" if conf_val > 0 else "HSV"
    cv2.putText(img_marcada, label, (xmin, max(ymin - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    print(f"  [OK] Placa detectada — {metodo}"
          + (f" | {conf_val * 100:.1f}%" if conf_val > 0 else ""))

    return recorte_rgb, cv2.cvtColor(img_marcada, cv2.COLOR_BGR2RGB), metodo


# ------------------------------------------------------------------------------
# 5. PRUEBA RÁPIDA
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python modulo1_deteccion_yolo.py <ruta_imagen>")
    else:
        recorte, marcada, metodo = detectar_y_recortar_placa(sys.argv[1])
        if recorte is not None:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.imshow(marcada); ax1.set_title(f"Detección ({metodo})"); ax1.axis('off')
            ax2.imshow(recorte); ax2.set_title("Recorte SR ×4");         ax2.axis('off')
            plt.tight_layout(); plt.show()