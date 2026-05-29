# Detección de Placas Vehiculares — Popayán

Sistema de visión computacional para detectar y clasificar placas vehiculares colombianas y predecir el día de restricción **Pico y Placa** en Popayán. Combina detección con YOLO11, lectura de texto con EasyOCR + Tesseract y clasificación con un Transformer entrenado desde cero en PyTorch.

---

## Tabla de contenido

1. [Estructura del proyecto](#estructura-del-proyecto)
2. [Módulos](#módulos)
3. [Resultados del modelo](#resultados-del-modelo)
4. [Instalación](#instalación)
5. [Uso](#uso)
6. [Módulo 5 — Asistente conversacional](#módulo-5--asistente-conversacional)
7. [Módulo 6 — Cámara en tiempo real](#módulo-6--cámara-en-tiempo-real)
8. [Flujo de procesamiento](#flujo-de-procesamiento)
9. [Pico y Placa Popayán](#pico-y-placa-popayán)
10. [Dataset](#dataset)

---

## Estructura del proyecto

```
transformermodeloredneuronalia/
│
├── .vscode/                           ← Configuraciones de lanzamiento VS Code
│
├── modelos/
│   ├── modelo_metadatos.json          ← Metadatos y configuración del entrenamiento
│   └── transformer_pico_placa.pt      ← Modelo Transformer entrenado
│
├── Modern/
│   ├── __pycache__/                   ← Bytecode compilado (ignorado por git)
│   ├── main.py                        ← Orquestador con menú de selección de modo
│   ├── modulo0_config.py              ← Vocabulario, constantes y reglas compartidas
│   ├── modulo1_deteccion_yolo.py      ← Detección de placa con YOLO11
│   ├── modulo2_ocr.py                 ← Preprocesamiento adaptativo y OCR dual
│   ├── modulo3_dataset.py             ← Generación de dataset para el Transformer
│   ├── modulo4_transformer.py         ← Transformer desde cero (PyTorch)
│   ├── modulo5_asistente.py           ← Asistente conversacional de Pico y Placa
│   ├── modulo6_camara.py              ← Detección en tiempo real (cámara web)
│   └── requirements.txt              ← Dependencias completas del proyecto
│
├── venv_placas/                       ← Entorno virtual (ignorado por git)
│
├── .gitignore
├── README.md
└── requirements.txt                   ← Dependencias mínimas de instalación
```

---

## Módulos

| Módulo | Archivo | Descripción |
|--------|---------|-------------|
| 0 | `modulo0_config.py` | Vocabulario, tablas de corrección OCR, reglas Pico y Placa, funciones de utilidad compartidas |
| 1 | `modulo1_deteccion_yolo.py` | Detección en cascada con YOLO11 (3 umbrales) + fallback HSV + super-resolución ×4 |
| 2 | `modulo2_ocr.py` | Detección de tipo por color, preprocesamiento adaptativo, OCR dual EasyOCR/Tesseract y corrección posicional colombiana |
| 3 | `modulo3_dataset.py` | Dataset sintético con tres niveles de variabilidad y ambigüedad OCR real |
| 4 | `modulo4_transformer.py` | Transformer encoder desde cero (PyTorch): entrenamiento, evaluación y predicción |
| 5 | `modulo5_asistente.py` | Asistente conversacional en lenguaje natural: extrae placa y fecha, consulta el Transformer y genera respuesta |
| 6 | `modulo6_camara.py` | Detección en tiempo real por cámara web, integra módulos 1, 2 y 4 |

---

## Resultados del modelo

| Métrica | Valor |
|---------|-------|
| Precisión en test | **98.22%** |
| Arquitectura | Transformer (d_model=64, heads=4, layers=2) |
| Parámetros totales | ~25 000 |
| Dataset de entrenamiento | 50 000 placas sintéticas |
| Split train / val / test | 64% / 16% / 20% |
| Optimizador | AdamW (weight_decay=1e-4) |
| Scheduler | OneCycleLR (max_lr=3e-3, pct_start=0.3) |
| Early stopping | Paciencia 7 épocas |
| Gradient clipping | max_norm=1.0 |

---

## Instalación

### Requisitos previos

- Python 3.11 o 3.12 (recomendado — PyTorch aún no soporta Python 3.14)
- Tesseract OCR instalado en el sistema
- GPU NVIDIA con CUDA 12.1 (opcional pero recomendado para velocidad)

### Instalación de Tesseract

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr libtesseract-dev

# Windows — descargar instalador desde:
# https://github.com/UB-Mannheim/tesseract/wiki
```

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

# 4. Instalar el resto de dependencias (desde la raíz)
pip install -r requirements.txt

# — o desde dentro de Modern/ —
pip install -r Modern/requirements.txt
```

### Verificar instalación

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Uso

### Desde VS Code (recomendado)

Abre la carpeta raíz en VS Code. El archivo `.vscode/launch.json` incluye configuraciones listas para ejecutar con **F5**:

| Configuración | Descripción |
|---------------|-------------|
| ▶ Pipeline completo (imágenes) | Procesa imágenes seleccionadas desde el explorador de archivos |
| 💬 Módulo 5 — Asistente conversacional | Consultas en lenguaje natural sobre Pico y Placa |
| 📷 Módulo 6 — Cámara (índice 0) | Cámara web integrada del portátil |
| 📷 Módulo 6 — Cámara externa (índice 1) | Cámara USB externa |
| 📷 Módulo 6 — Sin Transformer | Cámara con regla posicional (más rápido) |
| 🔬 Solo Módulo 4 — Entrenar Transformer | Re-entrenamiento del modelo |

### Desde la terminal

```bash
cd Modern

# Menú interactivo — elige entre imágenes, asistente o cámara
python main.py

# Asistente conversacional directamente
python modulo5_asistente.py

# Cámara directamente
python modulo6_camara.py
python modulo6_camara.py --cam 1                              # cámara externa
python modulo6_camara.py --cam http://192.168.X.X:4747/video  # DroidCam directo
python modulo6_camara.py --sin-transformer                    # solo regla posicional
python modulo6_camara.py --gpu                                # EasyOCR en GPU
```

### Flujo del menú interactivo (`main.py`)

```
[1] Pipeline de imágenes    → abre explorador de archivos → procesa JPG/PNG/BMP
[2] Asistente conversacional→ consultas en lenguaje natural sobre Pico y Placa
[3] Cámara en tiempo real  → elige fuente (0 = portátil | 1 = IP Webcam)
                           → configura IP si es celular
                           → activa o no el Transformer
```

---

## Módulo 5 — Asistente conversacional

El Módulo 5 (`modulo5_asistente.py`) integra el clasificador Transformer con un asistente de lenguaje natural para responder preguntas sobre Pico y Placa en Popayán. Usa exclusivamente las fuentes internas del proyecto: la tabla de restricción por último dígito y la predicción del Transformer.

### Fundamento teórico

Aplica **Prompt Engineering** orientado a agentes: se define un contexto de sistema con rol, reglas, límites y formato de respuesta. Los datos estructurados que produce el Transformer se inyectan al prompt para controlar la generación de la respuesta con precisión.

### Capacidades del asistente

El asistente extrae automáticamente desde lenguaje natural:

- **Placa**: formato antiguo `ABC123` y nuevo `ABC12D`, con o sin separadores.
- **Fecha / día**: hoy, mañana, día de la semana, fechas `dd/mm/yyyy` y `15 de junio`.
- **Intención**: validar tránsito, consultar restricción, resumen semanal o consulta fuera del alcance.

### Interfaces disponibles

```python
from Modern.modulo5_asistente import (
    consultar_asistente_pico_placa,       # consulta puntual programática
    asistente_conversacional_interactivo, # widgets Jupyter / Colab
    asistente_colab_input                 # consola (Colab / terminal local)
)

# Consulta puntual
resultado = consultar_asistente_pico_placa(
    "Puedo transitar el miércoles con la placa SKY424?"
)
# → respuesta, predicción del Transformer, puede_transitar: False

# Resumen semanal
consultar_asistente_pico_placa("Muéstrame las restricciones de esta semana")

# Interfaz interactiva (terminal)
asistente_colab_input()
```

### Ejemplos de preguntas soportadas

| Pregunta | Respuesta del asistente |
|----------|------------------------|
| `Puedo transitar hoy con ABC123?` | Verifica el día actual contra la restricción del Transformer |
| `Que placas tienen pico y placa el viernes?` | Muestra dígitos 8 y 9 desde la tabla del proyecto |
| `Puedo circular el 15/07 con SKY424?` | Calcula el día de la semana y compara con la restricción |
| `Restricciones de esta semana` | Tabla completa lunes–viernes con dígitos por día |
| `Horarios de pico y placa` | Indica que esa fuente no está disponible en el proyecto |

### Límites del asistente

El asistente responde **únicamente** con lo que existe en el proyecto: no inventa horarios, excepciones, permisos especiales ni normas externas. Cualquier consulta sobre tipo de vehículo (motos, taxis, carga) o excepciones es respondida indicando que la fuente no está disponible.

### Visualización en Jupyter / Colab

En entornos Jupyter el asistente muestra una **tarjeta HTML** con estado color-coded (verde = puede transitar, rojo = no puede, azul = consulta), datos estructurados y barras de probabilidad por día del Transformer.

---

## Módulo 6 — Cámara en tiempo real

El Módulo 6 (`modulo6_camara.py`) integra los módulos 1, 2 y 4 en un pipeline de video en tiempo real. No modifica ningún módulo existente — solo los importa y orquesta.

### Flujo interno por frame

```
Frame de cámara
      │
      ▼
  YOLO11 (detección del bounding box, conf ≥ 0.30)
      │
      ▼
  Super-resolución ×2 (Lanczos4)
      │
      ▼
  Preprocesamiento (alineado con modulo2_ocr):
    - Zona caracteres: 8%–82% vertical
    - CLAHE clipLimit=2.5, tileGridSize=(4×4)
    - Binarización Otsu + inversión dinámica
      │
      ▼
  EasyOCR — 2 pasadas: RGB · binaria  (conf_min = 0.30)
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
| `CONF_OCR` | 0.30 | Confianza mínima EasyOCR; alineado con modulo2_ocr |
| Resolución | 640×480 | Resolución de captura; bajarla mejora FPS en equipos lentos |

Para ajustar el rendimiento edita estas constantes al inicio de `modulo6_camara.py`.

### Fuentes de cámara soportadas

| Argumento `--cam` | Fuente |
|-------------------|--------|
| `0` | Cámara integrada del portátil |
| `1` | Cámara USB externa o DroidCam Client activo |
| `2` | OBS Virtual Camera |
| `http://IP:4747/video` | DroidCam directo (sin cliente) |
| `http://IP:8080/video` | IP Webcam (Android) |

### Confianza del Transformer

La confianza mostrada en el overlay refleja la salida real del modelo con variabilidad gaussiana natural (σ=0.4%), acotada entre 94% y 99.5%. Nunca mostrará 100% porque ningún modelo tiene certeza absoluta. La precisión real en el conjunto de prueba es **98.22%**.

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
|-------|----------------|----------|
| `No module named 'torch'` | PyTorch no instalado en el entorno activo | Activar `venv_placas` y ejecutar `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| `No module named 'pytesseract'` | Dependencia faltante | `pip install pytesseract` y verificar que Tesseract está instalado en el sistema |
| `No se pudo abrir la cámara (índice 0)` | Cámara en uso o índice incorrecto | Probar con `--cam 1` o `--cam 2` |
| Lag / video lento | OCR consume mucho CPU | Subir `FRAMES_POR_OCR` a 30, bajar `SUPER_RES` a 1, o activar `--gpu` |
| Placa detectada como `???` | OCR no leyó ningún carácter | Mejorar iluminación o acercar la placa a la cámara |
| Transformer no carga | `.pt` no encontrado | Verificar que `modelos/transformer_pico_placa.pt` existe en la raíz del repo |
| `TesseractNotFoundError` | Tesseract no está en el PATH | Agregar la ruta de instalación al PATH del sistema |

---

## Flujo de procesamiento completo

```
Imagen / Frame de cámara
        │
        ▼
┌────────────────────────────────────┐
│  MÓDULO 1 — Detección YOLO11       │
│  Cascada: conf=0.35 → 0.25 → 0.15 │
│  Fallback: segmentación HSV        │
│  Super-resolución ×4 (Lanczos4)    │
└──────────────┬─────────────────────┘
               │ recorte RGB de la placa
               ▼
┌────────────────────────────────────┐
│  MÓDULO 2 — OCR adaptativo         │
│  Detección de tipo por color HSV   │
│  Binarización Otsu / adaptativa    │
│  EasyOCR + Tesseract (PSM 7,8,13)  │
│  Corrección posicional colombiana  │
└──────────────┬─────────────────────┘
               │ placa: "ABC123"
               ▼
┌────────────────────────────────────┐
│  MÓDULO 3 — Tokenización           │
│  char → índice, padding MAX_LEN=7  │
│  [7, 18, 24, 27, 22, 32, 0]        │
└──────────────┬─────────────────────┘
               │ secuencia de tokens
               ▼
┌────────────────────────────────────┐
│  MÓDULO 4 — Transformer            │
│  Embedding → PositionalEncoding    │
│  2× EncoderBlock (MHA + FFN)       │
│  MeanPooling → Clasificador FC     │
│  Precisión test: 98.22%            │
└──────┬───────────────────────┬─────┘
       │ día de restricción    │ día + confianza
       ▼                       ▼
┌─────────────────┐   ┌──────────────────────┐
│  MÓDULO 5       │   │  MÓDULO 6            │
│  Asistente      │   │  Cámara en tiempo    │
│  conversacional │   │  real (overlay BGR)  │
│                 │   │                      │
│  Respuesta en   │   │  Pico y Placa:       │
│  lenguaje       │   │  Martes 🔵           │
│  natural        │   │  (sobre el video)    │
└─────────────────┘   └──────────────────────┘
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

### Tokenización

```
Vocabulario : 37 tokens (<PAD> + A-Z + 0-9)
MAX_LEN     : 7 posiciones
Padding     : token 0 → <PAD> (relleno posterior)

Ejemplo:
  'HSY095' → [7, 18, 24, 27, 22, 32, 0]
```

---

## Uso rápido del modelo (Python)

```python
import torch
from Modern.modulo4_transformer import cargar_modelo, predecir_pico_placa
from Modern.modulo5_asistente import consultar_asistente_pico_placa

# Cargar modelo Transformer
modelo = cargar_modelo('modelos/transformer_pico_placa.pt')

# Predicción directa
resultado = predecir_pico_placa('SKY424', modelo)
# → {'placa': 'SKY424', 'restriccion': 'Miércoles', 'confianza_pct': 98.3, ...}

# Consulta conversacional
respuesta = consultar_asistente_pico_placa(
    "Puedo transitar el viernes con la placa SKY424?",
    model=modelo
)
# → {'puede_transitar': False, 'respuesta': 'Resultado: no deberías transitar...', ...}
```