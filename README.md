# Detección de Placas Vehiculares — Pico y Placa Popayán

**YOLO11 + EasyOCR + Transformer desde cero — Precisión test: 98.22%**

Sistema de inteligencia artificial para detectar placas vehiculares y predecir la restricción de Pico y Placa en Popayán, Colombia. El pipeline integra detección de objetos (YOLO11), OCR de alta precisión (EasyOCR + Tesseract) y un clasificador Transformer entrenado completamente desde cero en PyTorch, desplegado vía Gradio + ngrok y publicado en HuggingFace Spaces.

---

## Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Estructura del Repositorio](#estructura-del-repositorio)
3. [Descripción de Módulos](#descripción-de-módulos)
4. [Flujo de Datos](#flujo-de-datos)
5. [Entornos de Ejecución](#entornos-de-ejecución)
6. [Instalación y Configuración](#instalación-y-configuración)
7. [Ejecución](#ejecución)
8. [Despliegue](#despliegue)
9. [Refactorización de Código](#refactorización-de-código)
10. [Relación entre Módulos y Notebook](#relación-entre-módulos-y-notebook)
11. [Reglas de Pico y Placa — Popayán](#reglas-de-pico-y-placa--popayán)
12. [Precisión del Modelo](#precisión-del-modelo)

---

## Arquitectura General

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ENTORNOS DE EJECUCIÓN                         │
│                                                                      │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │  Notebook Colab  │   │  Local (main.py) │   │  HuggingFace     │  │
│  │  V12.ipynb       │   │  + módulos .py   │   │  Spaces (app.py) │  │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘  │
│           │                      │                       │            │
│           └──────────────────────┴───────────────────────┘            │
│                                  │                                    │
│                     ┌────────────▼───────────┐                        │
│                     │   PIPELINE COMPARTIDO  │                        │
│                     └────────────┬───────────┘                        │
└──────────────────────────────────┼────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
     ┌─────▼──────┐         ┌──────▼──────┐         ┌─────▼──────┐
     │ Módulo 0   │         │ Módulo 1    │         │ Módulo 2   │
     │ Config     │◄────────│ YOLO11      │────────►│ EasyOCR +  │
     │ Vocabulario│         │ Detección   │         │ Tesseract  │
     └─────┬──────┘         └─────────────┘         └──────┬─────┘
           │                                               │
     ┌─────▼──────┐                                  ┌─────▼──────┐
     │ Módulo 3   │                                  │ Módulo 4   │
     │ Dataset    │─────────────────────────────────►│ Transformer│
     │ Sintético  │                                  │ PyTorch    │
     └────────────┘                                  └──────┬─────┘
                                                            │
                               ┌────────────────────────────┤
                               │                            │
                         ┌─────▼──────┐             ┌──────▼─────┐
                         │ Módulo 5   │             │ Módulo 6   │
                         │ Asistente  │             │ Cámara     │
                         │ Gradio     │             │ Tiempo Real│
                         └─────┬──────┘             └────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
   ┌──────────▼──────────┐           ┌──────────▼──────────┐
   │  Gradio + ngrok     │           │  HuggingFace Spaces  │
   │  (ejecución local)  │           │  (despliegue público)│
   │  http://127.0.0.1   │           │  huntercito-pico-    │
   │  + link ngrok       │           │  placa-popayan       │
   └─────────────────────┘           └─────────────────────┘
```

---

## Estructura del Repositorio

```
TransformerModeloRedNeuronalia/
│
├── Modern/                           # Carpeta de desarrollo local
│   ├── __init__.py                   # Agrega Modern/ al sys.path automáticamente
│   ├── main.py                       # Menú interactivo CLI (opciones 1–4)
│   ├── modulo0_config.py             # Vocabulario, constantes y reglas Pico y Placa
│   ├── modulo1_deteccion_yolo.py     # Detección de placa con YOLO11 (cascada 4 niveles)
│   ├── modulo2_ocr.py                # Preprocesamiento adaptativo + OCR dual
│   ├── modulo2b_corrector_ocr.py     # Corrector avanzado de errores OCR posicionales
│   ├── modulo3_dataset.py            # Generación de dataset sintético (~48k muestras)
│   ├── modulo4_transformer.py        # Arquitectura + entrenamiento del Transformer
│   ├── modulo5_asistente.py          # Asistente conversacional + Gradio integral
│   ├── modulo6_camara.py             # Detección en tiempo real por cámara web
│   ├── requirements.txt              # Dependencias completas para uso local
│   └── resultados_placas.csv         # Historial de detecciones (generado en ejecución)
│
├── pico-placa-popayan/               # Carpeta de producción (HuggingFace Spaces)
│   ├── modelos/
│   │   └── transformer_pico_placa.pt # Checkpoint del Transformer entrenado
│   ├── app.py                        # Punto de entrada para HuggingFace Spaces
│   ├── modulo0_config.py             # (mismo que Modern/ — sincronizado)
│   ├── modulo1_deteccion_yolo.py     # (mismo que Modern/ — sincronizado)
│   ├── modulo2_ocr.py                # (mismo que Modern/ — sincronizado)
│   ├── modulo2b_corrector_ocr.py     # (mismo que Modern/ — sincronizado)
│   ├── modulo3_dataset.py            # (mismo que Modern/ — sincronizado)
│   ├── modulo4_transformer.py        # (mismo que Modern/ — sincronizado)
│   ├── modulo5_asistente.py          # (mismo que Modern/ — sincronizado)
│   ├── modulo6_camara.py             # (mismo que Modern/ — sincronizado)
│   ├── registros_placas.db           # Base de datos SQLite de consultas
│   ├── packages.txt                  # Dependencias del sistema (Tesseract, etc.)
│   ├── requirements.txt              # Dependencias Python para HuggingFace
│   └── subir_modelo.py               # Utilidad para subir el .pt a HuggingFace Hub
│
├── modelos/                          # Checkpoint raíz (creado al entrenar localmente)
│   └── transformer_pico_placa.pt
│
├── Modelo_IA_Pico_Placa_V12.ipynb   # Notebook principal (Google Colab)
├── ngrok.exe                         # Túnel público para exposición local
├── app.py                            # app.py raíz (referencia, no en producción activa)
├── requirements.txt                  # Dependencias completas raíz
├── requirements_api.txt              # Dependencias mínimas para API
├── requirements_colab.txt            # Dependencias para Google Colab
├── render.yaml                       # Configuración intentada en Render.com (ver nota)
├── modal_app.py                      # Alternativa de despliegue explorada (Modal.com)
└── README.md                         # Este archivo
```

---

## Descripción de Módulos

### `modulo0_config.py` — Configuración Global

Fuente de verdad de todas las constantes del sistema. Ningún otro módulo duplica estos valores.

| Elemento | Descripción |
|---|---|
| `char2idx` / `idx2char` | Tokenización: `<PAD>` + A–Z + 0–9 → 37 tokens |
| `PATRON_ANTIGUA` | Regex `^[A-Z]{3}[0-9]{3}$` — placa colombiana clásica |
| `PATRON_NUEVA` | Regex `^[A-Z]{3}[0-9]{2}[A-Z]$` — placa colombiana nueva |
| `REGLAS_PICO_PLACA` | Mapeo dígito → día de restricción (Popayán) |
| `DIGITO_A_LETRA` / `LETRA_A_DIGITO` | Corrección posicional de errores OCR |
| `tokenizar_placa()` | Convierte string de placa en tensor de índices con padding |
| `asignar_restriccion()` | Retorna día de restricción dado el último dígito |

### `modulo1_deteccion_yolo.py` — Detección de Placa

Pipeline en cascada de 4 niveles para detectar la placa en cualquier condición de imagen:

1. YOLO11 `conf=0.35` (imagen original)
2. YOLO11 `conf=0.25` (imagen con denoising + CLAHE + sharpening)
3. YOLO11 `conf=0.15` (umbral mínimo)
4. Fallback por segmentación de color HSV (amarillo, blanco, verde, naranja)

Aplica super-resolución ×4 con interpolación Lanczos4 al recorte final. El modelo YOLO se descarga automáticamente desde HuggingFace en el primer uso mediante singleton.

**Función principal:** `detectar_y_recortar_placa(ruta)` → `(recorte_rgb, img_marcada, metodo)`

### `modulo2_ocr.py` — OCR de Alta Precisión

Preprocesamiento adaptativo según el color de fondo (amarillo, blanco, verde, naranja, gris), seguido de OCR dual:

- **EasyOCR:** 2 pasadas (imagen RGB + binarizada), confianza > 0.3
- **Tesseract:** PSM 7, 8 y 13 en paralelo

Los candidatos se fusionan y pasan por `modulo2b_corrector_ocr.py` para corrección posicional avanzada (O↔0, I↔1, S↔5, B↔8, etc.) según la estructura `[A-Z]{3}[0-9]{2,3}`.

**Función principal:** `extraer_datos_placa(recorte_rgb)` → `(DataFrame, imagen_binaria)`

### `modulo2b_corrector_ocr.py` — Corrector OCR Avanzado

Módulo especializado en recuperar placas con errores de lectura OCR:

- Corrección posicional por zona (letras pos 0–2, dígitos pos 3–5)
- Recuperación de placas de 5, 7 u 8 caracteres eliminando o insertando el carácter correcto
- Scoring de candidatos por similitud al formato oficial colombiano
- Validación final con regex `PATRON_ANTIGUA` y `PATRON_NUEVA`

**Función principal:** `corregir_placa(texto_raw, tipo_placa)` → `str | None`

### `modulo3_dataset.py` — Generación de Dataset

Genera el dataset de entrenamiento del Transformer con tres niveles de variabilidad controlada:

| Nivel | Proporción | Descripción |
|---|---|---|
| A | 75% | Placas antiguas limpias `ABC123` |
| B | 20% | Placas nuevas `ABC12D` |
| C | 5% | Placas con confusión OCR simulada (O↔0, I↔1…) |

Puede integrar placas reales del dataset Kaggle `andrewmvd/car-plate-detection` para enriquecer el entrenamiento.

**Función principal:** `preparar_datos_transformer(historico_df, df_reales, n_sintetico=48000)` → `list[(tokens, label)]`

### `modulo4_transformer.py` — Transformer desde Cero

Arquitectura completa implementada en PyTorch sin dependencias de HuggingFace:

```
tokens(B,7) → Embedding(B,7,64) → PositionalEncoding
            → 2×TransformerEncoderBlock(d_model=64, heads=4, d_ff=256)
            → MeanPooling → Dropout → Linear(64) → ReLU → Linear(5)
            → 5 clases (Lunes / Martes / Miércoles / Jueves / Viernes)
```

Entrenamiento con AdamW + OneCycleLR + early stopping (`paciencia=7`). El checkpoint se guarda en `modelos/transformer_pico_placa.pt`.

**Funciones principales:**
- `ejecutar_modulo4(datos_raw)` → modelo entrenado
- `predecir_pico_placa(placa, model, verbose)` → `{restriccion, confianza_pct, probabilidades}`
- `guardar_modelo(model, ruta)` / `cargar_modelo(ruta)` → checkpoint `.pt`

### `modulo5_asistente.py` — Asistente Conversacional + Gradio

Asistente en lenguaje natural para consultas de Pico y Placa. Extrae placa, fecha e intención desde texto libre y consulta el Transformer para la predicción. Usa únicamente fuentes internas del proyecto.

Capacidades de extracción:
- Placa desde texto libre con regex (formatos antiguo y nuevo)
- Fecha: "hoy", "mañana", nombre del día, dd/mm, "12 de mayo"
- Intención: validar tránsito, consultar día, consulta semanal, temas fuera de alcance

Interfaces:
- `consultar_asistente_pico_placa(pregunta)` — función base reutilizable
- `asistente_colab_input()` — consola con `input()` (usada por `main.py`)
- `desplegar_app_integral_gradio()` — app Gradio completa con YOLO + OCR + Transformer + chat

La app Gradio incluye corrección de espejo para cámara web (JavaScript), soporte de imagen por archivo o webcam, y panel conversacional en la misma interfaz.

### `modulo6_camara.py` — Detección en Tiempo Real

Captura video de cámara web, DroidCam o IP Webcam, detecta placas frame a frame con YOLO11, aplica OCR y predice el día de restricción con el Transformer. Dibuja overlay con color por día y confianza.

Modos de cámara:
- `cam_idx=0` → cámara integrada del portátil
- `cam_idx=1` → cámara USB / DroidCam Client
- `cam_idx="http://IP:PORT/video"` → stream de celular directo

Controles: `q` = salir | `s` = guardar captura

**Función principal:** `iniciar_camara(cam_idx, usar_transformer, gpu)`

---

## Flujo de Datos

```
Imagen / Video / Texto libre
        │
        ▼
┌───────────────┐
│  Módulo 1     │  detectar_y_recortar_placa()
│  YOLO11       │  → recorte_rgb (numpy, super-res ×4)
└───────┬───────┘
        │
        ▼
┌───────────────────────┐
│  Módulo 2             │  extraer_datos_placa()
│  EasyOCR + Tesseract  │  → DataFrame {placa, ultimo_digito, tipo_placa, motor_ocr}
│  + Corrector 2b       │
└───────┬───────────────┘
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
┌───────────────┐                           ┌──────────────────┐
│  Módulo 3     │  preparar_datos_          │  Módulo 4        │
│  Dataset      │  transformer()            │  predecir_       │
│  Sintético    │  → list[(tokens, label)]  │  pico_placa()    │
└───────┬───────┘                           └────────┬─────────┘
        │                                            │
        ▼                                            │
┌───────────────┐                                    │
│  Módulo 4     │  ejecutar_modulo4()                │
│  Entrena      │  → transformer_pico_placa.pt       │
└───────────────┘                                    │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ Módulo 5             │
                                          │ Asistente + Gradio   │
                                          └──────────┬───────────┘
                                                     │
                              ┌──────────────────────┤
                              │                      │
                    ┌─────────▼──────┐     ┌─────────▼──────────────┐
                    │  Gradio local  │     │  HuggingFace Spaces     │
                    │  + ngrok       │     │  (producción pública)   │
                    └────────────────┘     └────────────────────────┘
```

---

## Entornos de Ejecución

El proyecto soporta tres entornos completamente independientes que comparten la misma lógica de negocio:

| Entorno | Punto de entrada | Requirements | Despliegue |
|---|---|---|---|
| **Google Colab** | `Modelo_IA_Pico_Placa_V12.ipynb` | `requirements_colab.txt` | Gradio con `share=True` (link temporal 72h) |
| **Local (CLI)** | `Modern/main.py` | `requirements.txt` | Menú interactivo + Gradio + ngrok |
| **Producción** | `pico-placa-popayan/app.py` | `pico-placa-popayan/requirements.txt` | HuggingFace Spaces (permanente) |

### ¿Por qué tres entornos distintos?

- **Colab:** no tiene `__file__` en celdas, los modelos van en `/content/`, usa Gradio con `share=True` para enlace temporal.
- **Local:** acceso a cámara web (Módulo 6), `tkinter` para seleccionar archivos, Gradio local con ngrok para link público. El menú principal (`main.py`) ofrece las 4 opciones del pipeline.
- **HuggingFace Spaces:** `app.py` lanza directamente `desplegar_app_integral_gradio(share=False)` porque Spaces ya provee el link público permanente. El modelo se descarga desde `Huntercito/modelos-placas-popayan` vía `hf_hub_download`.

---

## Instalación y Configuración

### Uso Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/TransformerModeloRedNeuronalia.git
cd TransformerModeloRedNeuronalia

# 2. Crear entorno virtual
python -m venv venv_placas
source venv_placas/bin/activate        # Linux/Mac
venv_placas\Scripts\activate           # Windows

# 3. Instalar PyTorch
# GPU NVIDIA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Solo CPU:
pip install torch torchvision

# 4. Instalar Tesseract OCR (sistema)
# Ubuntu/Debian: sudo apt-get install tesseract-ocr libtesseract-dev
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 5. Instalar dependencias Python
pip install -r requirements.txt

# 6. Ejecutar
cd Modern
python main.py
```

### Uso en Google Colab

El notebook `Modelo_IA_Pico_Placa_V12.ipynb` es autónomo. Solo necesitas:

1. Abrir el notebook en Google Colab
2. Ejecutar las celdas en orden
3. El Módulo 5 lanza automáticamente la app Gradio con enlace público temporal

Para usar los módulos `.py` del repositorio dentro del notebook:

```python
!git clone https://github.com/TU_USUARIO/TransformerModeloRedNeuronalia.git /content/repo
import sys
sys.path.insert(0, '/content/repo/Modern')

from modulo4_transformer import cargar_modelo, predecir_pico_placa
```

### ngrok (Despliegue Local Público)

Para exponer la app local en internet coloca `ngrok.exe` en la carpeta `Modern/` y al elegir la opción `[4]` del menú se lanza automáticamente:

```
  ╔══════════════════════════════════════════════════════════════════╗
  ║  LINKS DE ACCESO:                                                ║
  ║  Local:       http://127.0.0.1:7860                              ║
  ║  Público:     https://xxxx-xxxx.ngrok-free.dev                   ║
  ║  HuggingFace: https://huntercito-pico-placa-popayan.hf.space     ║
  ╚══════════════════════════════════════════════════════════════════╝
```

Requiere cuenta gratuita en [ngrok.com](https://ngrok.com) y configurar el authtoken:
```bash
ngrok config add-authtoken TU_TOKEN
```

---

## Ejecución

### Menú Local (`Modern/main.py`)

```
============================================================
   DETECCIÓN DE PLACAS VEHICULARES — POPAYÁN
============================================================
  [1] Pipeline de imágenes (seleccionar archivos)
  [2] Asistente conversacional (Módulo 5)
  [3] Cámara en tiempo real (Módulo 6)
  [4] Interfaz Web de Producción (Dashboard de Despliegue)
  [0] Salir
============================================================
```

- **Opción 1:** abre diálogo para seleccionar imágenes, procesa con YOLO + OCR + Transformer y muestra resultado con matplotlib.
- **Opción 2:** inicia el asistente conversacional en consola. Soporta preguntas como *"¿Puedo transitar el miércoles con la placa ABC125?"*.
- **Opción 3:** abre la cámara (integrada o celular) con detección en tiempo real.
- **Opción 4:** lanza la app Gradio en `http://127.0.0.1:7860` y ngrok en segundo plano.

### Módulos Independientes

```bash
# Probar detección YOLO con una imagen
python Modern/modulo1_deteccion_yolo.py foto_vehiculo.jpg

# Probar OCR con un recorte de placa
python Modern/modulo2_ocr.py recorte_placa.png

# Generar dataset de prueba
python Modern/modulo3_dataset.py

# Entrenar Transformer desde cero
python Modern/modulo4_transformer.py

# Iniciar asistente conversacional
python Modern/modulo5_asistente.py

# Iniciar cámara en tiempo real
python Modern/modulo6_camara.py --cam 0
python Modern/modulo6_camara.py --cam http://192.168.1.X:8080/video
```

---

## Despliegue

### HuggingFace Spaces (Producción actual) ✅

**URL pública permanente:** [https://huntercito-pico-placa-popayan.hf.space](https://huntercito-pico-placa-popayan.hf.space)

La carpeta `pico-placa-popayan/` es el repositorio sincronizado con HuggingFace Spaces. El `app.py` lanza directamente `desplegar_app_integral_gradio(share=False)` porque Spaces ya expone el puerto públicamente.

El modelo Transformer se descarga automáticamente desde:
```
Huntercito/modelos-placas-popayan → transformer_pico_placa.pt
```

### Gradio + ngrok (Despliegue local)

Al elegir la opción `[4]` en `main.py` se inicia simultáneamente:
1. Gradio en `http://127.0.0.1:7860`
2. ngrok en segundo plano creando el túnel público

El recuadro con los tres links aparece automáticamente en consola cuando ngrok obtiene su URL.

### Render.com (Intentado — cuenta gratuita insuficiente)

Se configuró `render.yaml` y `app.py` para desplegar como servicio web en Render.com, pero la cuenta gratuita (512 MB de RAM) resultó insuficiente para cargar simultáneamente YOLO11, EasyOCR y el Transformer en producción. Por esta razón se migró a HuggingFace Spaces, que ofrece mayor RAM disponible para inferencia.

```yaml
# render.yaml (referencia histórica — no en uso activo)
buildCommand: pip install -r requirements_api.txt
startCommand: uvicorn app:app --host 0.0.0.0 --port 10000
healthCheckPath: /health
```

---

## Refactorización de Código

Durante el desarrollo se realizó una refactorización completa de todos los módulos para mejorar la mantenibilidad sin alterar la funcionalidad:

| Módulo | Líneas originales | Líneas refactorizadas | Reducción |
|---|---|---|---|
| `modulo0_config.py` | 128 | 75 | −41% |
| `modulo1_deteccion_yolo.py` | 246 | 124 | −50% |
| `modulo2b_corrector_ocr.py` | 424 | 122 | −71% |
| `modulo2_ocr.py` | 443 | 174 | −61% |
| `modulo3_dataset.py` | 348 | 127 | −63% |
| `modulo4_transformer.py` | 690 | 277 | −60% |
| `modulo5_asistente.py` | 1 008 | 372 | −63% |
| `modulo6_camara.py` | 457 | 177 | −61% |
| `main.py` | 333 | 181 | −46% |
| **TOTAL** | **4 077** | **1 629** | **−60%** |

Técnicas aplicadas: funciones helper privadas `_nombre()`, generación de diccionarios por comprensión, singletons de una línea, eliminación de constantes no usadas (`PROMPT_SISTEMA_ASISTENTE`), consolidación de interfaces redundantes (`desplegar_asistente_gradio` integrado en `desplegar_app_integral_gradio`), y eliminación del parámetro `mostrar_visual` al remover la visualización HTML de Colab/Jupyter no usada en producción.

---

## Relación entre Módulos y Notebook

### Principio de diseño

El notebook `Modelo_IA_Pico_Placa_V12.ipynb` y los módulos `.py` son **implementaciones paralelas** de la misma lógica, diseñadas para convivir sin interferir:

```
Notebook (Colab)              Módulos .py (local / HuggingFace)
─────────────────             ──────────────────────────────────
Funciones inline              Funciones importables
Estado global en celdas       Singletons controlados
Gradio con share=True         Gradio con share=False
torch.save() en /content/     torch.save() en modelos/
__file__ no disponible        __file__ con try/except
lector_easy global            obtener_lector_easyocr()
model_transformer global      cargar_modelo(ruta)
```

### Regla de oro

> **Las firmas públicas de los módulos son inmutables.** Nunca cambiar las firmas de `predecir_pico_placa()`, `consultar_asistente_pico_placa()`, `extraer_datos_placa()` o `asignar_restriccion()` sin verificar compatibilidad con el notebook y con `app.py`.

### Sincronización Modern/ → pico-placa-popayan/

Los módulos de ambas carpetas deben mantenerse idénticos. El flujo correcto de actualización es:

1. Desarrollar y probar en `Modern/`
2. Copiar los módulos actualizados a `pico-placa-popayan/`
3. Hacer `git push` al repositorio GitHub
4. Hacer `git push` al Space de HuggingFace (rama `main`)

---

## Reglas de Pico y Placa — Popayán

| Último dígito | Día de restricción |
|:---:|---|
| 0, 1 | Lunes |
| 2, 3 | Martes |
| 4, 5 | Miércoles |
| 6, 7 | Jueves |
| 8, 9 | Viernes |

Aplica de lunes a viernes en horario de restricción. El sistema **no inventa** horarios, excepciones, permisos, ni normas para motos, taxis o vehículos de carga — solo reporta lo codificado en las fuentes internas del proyecto.

---

## Precisión del Modelo

| Conjunto | Precisión |
|---|---|
| Entrenamiento | ~99.5% |
| Validación | ~98.8% |
| **Test (final)** | **98.22%** |

Entrenado con ~48 000 placas sintéticas con variabilidad controlada (formatos antiguo/nuevo + confusiones OCR simuladas) más placas reales opcionales del dataset Kaggle `andrewmvd/car-plate-detection`.

Hiperparámetros finales: `d_model=64`, `num_heads=4`, `d_ff=256`, `num_layers=2`, `dropout=0.1`, `epochs=30`, `batch_size=256`, optimizador AdamW + OneCycleLR.

---

*Proyecto académico — Popayán, Colombia.*