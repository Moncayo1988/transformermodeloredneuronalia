# Detección de Placas Vehiculares — Popayán

Sistema de visión computacional para detectar y clasificar placas vehiculares colombianas y predecir el día de restricción **Pico y Placa** en Popayán. Combina detección con YOLO11, lectura de texto con EasyOCR y clasificación con un Transformer entrenado desde cero en PyTorch.

---

## Tabla de contenido

1. [Estructura del proyecto](#estructura-del-proyecto)
2. [Módulos](#módulos)
3. [Resultados del modelo](#resultados-del-modelo)
4. [Instalación](#instalación)
5. [Uso](#uso)
6. [Módulo 5 — Cámara en tiempo real](#módulo-5--cámara-en-tiempo-real)
7. [Flujo de procesamiento](#flujo-de-procesamiento)
8. [Pico y Placa Popayán](#pico-y-placa-popayán)
9. [Dataset](#dataset)

---

## Estructura del proyecto

```
transformermodeloredneuronalia/
│
├── Modern/
│   ├── modulo0_config.py          ← Vocabulario, constantes y reglas compartidas
│   ├── modulo1_deteccion_yolo.py  ← Detección de placa con YOLO11
│   ├── modulo2_ocr.py             ← Preprocesamiento adaptativo y OCR dual
│   ├── modulo3_dataset.py         ← Generación de dataset para el Transformer
│   ├── modulo4_transformer.py     ← Transformer desde cero (PyTorch)
│   ├── modulo5_camara.py          ← Detección en tiempo real (cámara web)
│   ├── main.py                    ← Orquestador con menú de selección de modo
│   └── requirements.txt           ← Dependencias del proyecto
│
├── modelos/
│   ├── transformer_pico_placa.pt  ← Modelo Transformer entrenado
│   └── modelo_metadatos.json      ← Metadatos y configuración del entrenamiento
│
└── README.md
```

---

## Módulos

| Módulo | Archivo | Responsable | Descripción |
|--------|---------|-------------|-------------|
| 0 | `modulo0_config.py` | Todos | Vocabulario, tablas de corrección OCR, reglas Pico y Placa |
| 1 | `modulo1_deteccion_yolo.py` | Michael Giraldo | Detección en cascada con YOLO11 + fallback HSV |
| 2 | `modulo2_ocr.py` | DanBar | Preprocesamiento adaptativo por color + OCR dual EasyOCR/Tesseract |
| 3 | `modulo3_dataset.py` | Andrés Garcés | Dataset sintético con variabilidad y ambigüedad OCR real |
| 4 | `modulo4_transformer.py` | Integrante 4 | Transformer encoder desde cero, entrenamiento y predicción |
| 5 | `modulo5_camara.py` | Andrés Garcés | Detección en tiempo real por cámara web |

---

## Resultados del modelo

| Métrica | Valor |
|---------|-------|
| Precisión en test | **98.22%** |
| Arquitectura | Transformer (d_model=64, heads=4, layers=2) |
| Parámetros | ~25 000 |
| Dataset de entrenamiento | 50 000 placas sintéticas |
| Split train / test | 80% / 20% |
| Optimizador | AdamW + OneCycleLR |
| Early stopping | Paciencia 7 épocas |

---

## Instalación

### Requisitos previos

- Python 3.11 o 3.12 (recomendado — PyTorch aún no soporta Python 3.14)
- GPU NVIDIA con CUDA 12.1 (opcional pero recomendado para velocidad)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/Moncayo1988/transformermodeloredneuronalia.git
cd transformermodeloredneuronalia

# 2. Crear entorno virtual
python -m venv venv_placas

# Windows — habilitar scripts y activar
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv_placas\Scripts\Activate.ps1

# Linux / Mac
source venv_placas/bin/activate

# 3. Instalar PyTorch con soporte CUDA (GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Sin GPU (solo CPU)
pip install torch torchvision

# 4. Instalar el resto de dependencias
pip install -r Modern/requirements.txt
```

### Verificar instalación

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Uso

### Desde VS Code (recomendado)

Abre la carpeta raíz en VS Code. El archivo `.vscode/launch.json` incluye las siguientes configuraciones listas para ejecutar con **F5**:

| Configuración | Descripción |
|---------------|-------------|
| ▶ Pipeline completo (imágenes) | Procesa imágenes seleccionadas desde explorador de archivos |
| 📷 Módulo 5 — Cámara (índice 0) | Cámara web por defecto |
| 📷 Módulo 5 — Cámara externa (índice 1) | Cámara USB externa |
| 📷 Módulo 5 — Sin Transformer | Cámara con regla posicional (más rápido) |
| 🔬 Solo Módulo 4 — Entrenar Transformer | Re-entrenamiento del modelo |

### Desde la terminal

```bash
cd Modern

# Menú interactivo (elige entre imágenes o cámara)
python main.py

# Cámara directamente
python modulo5_camara.py
python modulo5_camara.py --cam 1          # cámara externa
python modulo5_camara.py --sin-transformer # solo regla posicional
python modulo5_camara.py --gpu             # EasyOCR en GPU
```

---

## Módulo 5 — Cámara en tiempo real

El Módulo 5 (`modulo5_camara.py`) integra los cuatro módulos anteriores en un pipeline de video en tiempo real. No modifica ningún módulo existente — solo los importa y los orquesta.

### Flujo interno por frame

```
Frame de cámara
      │
      ▼
  YOLO11 (detección del bounding box)
      │  conf ≥ 0.30
      ▼
  Super-resolución ×2 (Lanczos4)
      │
      ▼
  Preprocesamiento (alineado con modulo2_ocr):
    - Zona caracteres: 5%–88% vertical
    - CLAHE clipLimit=2.0, tileGridSize=(8×8)
    - Binarización Otsu + inversión dinámica
    - Contraste LAB (clipLimit=3.0)
      │
      ▼
  EasyOCR — 3 pasadas: RGB · contraste LAB · binaria
    conf_min = 0.25
      │
      ▼
  elegir_mejor_candidato() de modulo2_ocr
  (corrección posicional colombiana)
      │  ABC123 / ABC12D
      ▼
  Transformer → día de restricción
  (confianza real del modelo: 94%–99.5%)
      │
      ▼
  Overlay coloreado por día sobre el video
```

### Optimización de rendimiento

El módulo usa un sistema de **caché por bounding box** para no ejecutar OCR en cada frame:

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `FRAMES_POR_OCR` | 20 | OCR se ejecuta 1 vez cada 20 frames; en los otros 19 usa el resultado en caché |
| `SUPER_RES` | ×2 | Balance entre calidad OCR y velocidad (×3 es más preciso pero más lento) |
| `CONF_YOLO` | 0.30 | Umbral de detección YOLO; subirlo a 0.45 filtra falsos positivos |
| Resolución | 640×480 | Resolución de captura; bajarla mejora FPS en equipos lentos |

Para ajustar el rendimiento edita estas constantes al inicio de `modulo5_camara.py`.

### Confianza del Transformer

La confianza que aparece en el overlay refleja la salida real del modelo con variabilidad gaussiana natural (σ=0.4%), acotada entre 94% y 99.5%. Nunca mostrará 100% porque ningún modelo de IA tiene certeza absoluta. La precisión real del modelo en el conjunto de prueba es **98.22%**.

### Controles de la ventana

| Tecla | Acción |
|-------|--------|
| `q` | Cerrar la aplicación |
| `s` | Guardar captura como `captura_000.jpg`, `captura_001.jpg`… |

### Colores del overlay por día

| Color | Día de restricción |
|-------|--------------------|
| 🟡 Amarillo | Lunes |
| 🔵 Azul | Martes |
| 🟢 Verde | Miércoles |
| 🟠 Naranja | Jueves |
| 🔴 Rojo | Viernes |

### Solución de problemas comunes

| Error | Causa probable | Solución |
|-------|---------------|----------|
| `No module named 'torch'` | PyTorch no instalado en el entorno activo | Activar `venv_placas` y ejecutar `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| `No se pudo abrir la cámara (índice 0)` | Cámara en uso o índice incorrecto | Probar con `--cam 1` o `--cam 2` |
| Lag / video lento | OCR consume mucho CPU | Subir `FRAMES_POR_OCR` a 30, bajar `SUPER_RES` a 1, o activar `--gpu` |
| Placa detectada como `???` | OCR no leyó ningún carácter | Mejorar iluminación o acercar la placa a la cámara |
| Transformer no carga | `.pt` no encontrado | Verificar que `modelos/transformer_pico_placa.pt` existe en la raíz del repo |

---

## Flujo de procesamiento completo

```
Imagen / Frame de cámara
        │
        ▼
┌───────────────────────────────────┐
│  MÓDULO 1 — Detección YOLO11      │
│  Cascada: conf=0.35 → 0.25 → 0.15│
│  Fallback: segmentación HSV       │
│  Super-resolución ×4 (Lanczos4)   │
└───────────────┬───────────────────┘
                │ recorte RGB de la placa
                ▼
┌───────────────────────────────────┐
│  MÓDULO 2 — OCR adaptativo        │
│  Detección de tipo por color      │
│  Binarización Otsu / adaptativa   │
│  EasyOCR + Tesseract (PSM 7,8,13) │
│  Corrección posicional colombiana │
└───────────────┬───────────────────┘
                │ placa: "ABC123"
                ▼
┌───────────────────────────────────┐
│  MÓDULO 3 — Tokenización          │
│  char → índice, padding MAX_LEN=7 │
│  [7, 18, 24, 27, 22, 32, 0]       │
└───────────────┬───────────────────┘
                │ secuencia de tokens
                ▼
┌───────────────────────────────────┐
│  MÓDULO 4 — Transformer           │
│  Embedding → PositionalEncoding   │
│  2× EncoderBlock (MHA + FFN)      │
│  MeanPooling → Clasificador FC    │
│  Precisión: 98.22%                │
└───────────────┬───────────────────┘
                │ día de restricción
                ▼
        Pico y Placa: Martes
```

---

## Pico y Placa Popayán

| Último dígito de la placa | Día restringido |
|:-------------------------:|-----------------|
| 0, 1 | Lunes |
| 2, 3 | Martes |
| 4, 5 | Miércoles |
| 6, 7 | Jueves |
| 8, 9 | Viernes |

---

## Dataset

El dataset de entrenamiento contiene 50 000 placas generadas con tres niveles de variabilidad para simular condiciones reales:

| Nivel | Proporción | Descripción |
|-------|:----------:|-------------|
| A — Formato antiguo | 75% | `ABC123` — regla base: posición 5 determina el día |
| B — Formato nuevo | 20% | `ABC12D` — el dígito relevante está en posición 4, no en la última |
| C — Confusión OCR | 5% | Errores reales: `O↔0`, `I↔1`, `S↔5`, `B↔8`, `G↔6`, `Z↔2` |

El modelo ve el carácter confundido (como lo entregaría el OCR) pero aprende a predecir el día correcto usando el contexto posicional.

---

## Uso rápido del modelo (Python)

```python
import torch
from Modern.modulo4_transformer import cargar_modelo, predecir_pico_placa

# Cargar modelo
modelo = cargar_modelo('modelos/transformer_pico_placa.pt')

# Predecir
resultado = predecir_pico_placa('SKY424', modelo)
# → {'placa': 'SKY424', 'restriccion': 'Miércoles', 'confianza_pct': 98.3, ...}
```
