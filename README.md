# Detección de Placas Vehiculares — Pico y Placa Popayán

**YOLO11 + EasyOCR + Transformer desde cero — Precisión test: 98.22%**

Sistema de inteligencia artificial para detectar placas vehiculares y predecir la restricción de Pico y Placa aplicable en Popayán, Colombia. El pipeline integra detección de objetos, OCR de alta precisión y un clasificador Transformer entrenado desde cero en PyTorch.

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
9. [Relación entre Módulos y Notebook](#relación-entre-módulos-y-notebook)
10. [Reglas de Pico y Placa — Popayán](#reglas-de-pico-y-placa--popayán)

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ENTORNOS DE EJECUCIÓN                           │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Notebook Colab  │  │  Local (main.py) │  │  API (Render)    │  │
│  │  V12.ipynb       │  │  + módulos .py   │  │  app.py          │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                     │                      │            │
│           └─────────────────────┴──────────────────────┘            │
│                                 │                                   │
│                    ┌────────────▼───────────┐                       │
│                    │   PIPELINE COMPARTIDO  │                       │
│                    └────────────┬───────────┘                       │
└─────────────────────────────────┼───────────────────────────────────┘
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
                   ┌──────────▼──────────┐
                   │  FastAPI (app.py)   │
                   │  /detect (imagen)   │
                   │  /detect/texto      │
                   │  /health            │
                   └─────────────────────┘
```

---

## Estructura del Repositorio

```
TransformerModeloRedNeuronalia/
│
├── Modern/                          # Paquete principal del proyecto
│   ├── __init__.py                  # Agrega Modern/ al sys.path automáticamente
│   ├── modulo0_config.py            # Vocabulario, constantes y reglas Pico y Placa
│   ├── modulo1_deteccion_yolo.py    # Detección de placa con YOLO11
│   ├── modulo2_ocr.py               # Preprocesamiento adaptativo + OCR dual
│   ├── modulo3_dataset.py           # Generación de dataset sintético
│   ├── modulo4_transformer.py       # Arquitectura + entrenamiento del Transformer
│   ├── modulo5_asistente.py         # Asistente conversacional + Gradio
│   ├── modulo6_camara.py            # Detección en tiempo real por cámara
│   └── requirements.txt             # Dependencias del paquete Modern/
│
├── Modelo_IA_Pico_Placa_V12.ipynb  # Notebook principal (Colab) — fuente de verdad
│                                    # del despliegue Gradio. Contiene la misma
│                                    # lógica de los módulos de forma autónoma.
│
├── modelos/                         # Checkpoint del Transformer (creado al entrenar)
│   └── transformer_pico_placa.pt   # Generado por Módulo 4 / notebook
│
├── app.py                           # Servidor FastAPI (punto de entrada para Render)
├── render.yaml                      # Configuración de despliegue en Render.com
├── requirements.txt                 # Dependencias raíz para uso local completo
├── requirements_api.txt             # Dependencias mínimas para el servidor FastAPI
├── requirements_colab.txt           # Dependencias para Google Colab / Jupyter
├── main.py                          # Menú interactivo local (CLI)
├── resultados_placas.csv            # Historial de detecciones (generado en ejecución)
└── README.md                        # Este archivo
```

---

## Descripción de Módulos

### `modulo0_config.py` — Configuración Global

**Responsabilidad única:** define todas las constantes, vocabularios y reglas compartidas por el resto del sistema. Ningún otro módulo duplica estos valores.

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

Pipeline en cascada de 4 niveles para detectar la placa en cualquier imagen:

1. YOLO11 conf=0.35 (imagen original)
2. YOLO11 conf=0.25 (imagen con denoising + CLAHE + sharpening)
3. YOLO11 conf=0.15 (umbral mínimo)
4. Fallback por segmentación de color HSV

Aplica super-resolución ×4 con Lanczos4 al recorte final. El modelo YOLO se descarga automáticamente desde HuggingFace en el primer uso.

**Función principal:** `detectar_y_recortar_placa(ruta_imagen)` → `(recorte_rgb, img_marcada, metodo)`

### `modulo2_ocr.py` — OCR de Alta Precisión

Preprocesamiento adaptativo según el color de fondo de la placa (amarillo, blanco, verde, naranja, gris), seguido de OCR dual:

- **EasyOCR:** 2 pasadas (imagen RGB + binarizada), confianza > 0.3
- **Tesseract:** PSM 7, 8 y 13 en paralelo

Corrección posicional automática de errores OCR típicos (O↔0, I↔1, S↔5, B↔8, etc.) según la estructura `[A-Z]{3}[0-9]{2,3}`.

**Función principal:** `extraer_datos_placa(recorte_rgb)` → `(DataFrame, imagen_binaria)`

### `modulo3_dataset.py` — Generación de Dataset

Genera el dataset de entrenamiento del Transformer con tres niveles de variabilidad:

- **Nivel A (75%):** placas antiguas limpias `ABC123`
- **Nivel B (20%):** placas nuevas `ABC12D`
- **Nivel C (5%):** placas con confusión OCR simulada

Puede integrar placas reales procesadas por Módulos 1+2 (pipeline Kaggle) para enriquecer el entrenamiento.

**Función principal:** `preparar_datos_transformer(historico_df, df_reales, n_sintetico)` → `list[(tokens, label)]`

### `modulo4_transformer.py` — Transformer desde Cero

Arquitectura completa implementada en PyTorch sin dependencias de HuggingFace:

```
tokens(B,7) → Embedding(B,7,64) → PositionalEncoding
            → 2×TransformerEncoderBlock → MeanPooling
            → Clasificador FC → 5 clases (días de semana)
```

Hiperparámetros: `d_model=64`, `num_heads=4`, `d_ff=256`, `num_layers=2`, `dropout=0.1`.

Entrenamiento con AdamW + OneCycleLR + early stopping (paciencia=7). **Compatible con Colab:** la resolución de rutas usa `try/except` sobre `__file__` para funcionar tanto como `.py` como importado desde notebook.

**Funciones principales:**
- `ejecutar_modulo4(datos_raw)` → modelo entrenado
- `predecir_pico_placa(placa, model, verbose)` → dict con restricción y confianza
- `guardar_modelo(model, ruta)` / `cargar_modelo(ruta)` → checkpoint `.pt`

### `modulo5_asistente.py` — Asistente Conversacional

Asistente en lenguaje natural para consultas de Pico y Placa. Extrae placa, fecha e intención desde texto libre y consulta el Transformer para la predicción. Usa únicamente las fuentes internas del proyecto (no inventa normas externas).

Interfaces disponibles:
- `consultar_asistente_pico_placa(pregunta)` — función base
- `asistente_conversacional_interactivo()` — widgets Jupyter
- `asistente_colab_input()` — consola con `input()`

El despliegue Gradio (`desplegar_app_integral_gradio()` y `desplegar_asistente_gradio()`) **vive únicamente en el notebook** y no debe ser movido a este módulo.

### `modulo6_camara.py` — Detección en Tiempo Real

Captura video de cámara web o DroidCam, detecta placas frame a frame con YOLO, aplica OCR y predice el día de restricción con el Transformer. Dibuja overlay con color por día y confianza del modelo.

**Función principal:** `iniciar_camara(cam_idx, usar_transformer, gpu)`

Modos de cámara:
- `cam_idx=0` → cámara integrada del portátil
- `cam_idx=1` → cámara USB externa o DroidCam Client
- `cam_idx="http://IP:PORT/video"` → DroidCam sin cliente (stream directo)

---

## Flujo de Datos

```
Imagen / Video / Texto
        │
        ▼
┌───────────────┐
│  Módulo 1     │  detectar_y_recortar_placa()
│  YOLO11       │  → recorte_rgb (numpy)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Módulo 2     │  extraer_datos_placa()
│  OCR dual     │  → DataFrame {placa, ultimo_digito, tipo_placa...}
└───────┬───────┘
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
                                          │ consultar_asistente_ │
                                          │ pico_placa()         │
                                          └──────────┬───────────┘
                                                     │
                              ┌──────────────────────┤
                              │                      │
                    ┌─────────▼──────┐     ┌─────────▼──────────┐
                    │  Gradio (Web)  │     │  FastAPI (/detect)  │
                    │  [Notebook]    │     │  [Render / cloud]   │
                    └────────────────┘     └────────────────────┘
```

---

## Entornos de Ejecución

El proyecto soporta tres entornos completamente independientes que comparten la misma lógica de negocio:

| Entorno | Punto de entrada | Requirements | Despliegue |
|---|---|---|---|
| **Google Colab** | `Modelo_IA_Pico_Placa_V12.ipynb` | `requirements_colab.txt` | Gradio con `share=True` |
| **Local (CLI)** | `main.py` | `requirements.txt` | Menú interactivo |
| **API Cloud** | `app.py` | `requirements_api.txt` | Render.com / uvicorn |

### ¿Por qué tres entornos distintos?

- **Colab:** requiere `pillow-avif-plugin`, `gradio`, y monta Google Drive. No tiene `__file__` en las celdas. Los modelos se guardan en `/content/`.
- **Local:** puede usar la cámara web (Módulo 6), `tkinter` para seleccionar archivos, y tiene acceso al sistema de archivos completo. Los modelos van en `modelos/`.
- **API:** usa `opencv-python-headless` (sin GUI), carga los modelos de forma lazy, y descarga el checkpoint desde HuggingFace si no existe en disco.

---

## Instalación y Configuración

### Uso Local

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd TransformerModeloRedNeuronalia

# 2. Crear entorno virtual
python -m venv venv_placas
source venv_placas/bin/activate        # Linux/Mac
venv_placas\Scripts\activate           # Windows

# 3. Instalar PyTorch (elige según tu hardware)
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
python main.py
```

### Uso en Google Colab

El notebook `Modelo_IA_Pico_Placa_V12.ipynb` es autónomo. Las dependencias se instalan en las primeras celdas. Solo necesitas:

1. Abrir el notebook en Google Colab
2. Ejecutar las celdas en orden
3. En la celda del Módulo 5, se lanza automáticamente la app Gradio con enlace público

Para usar los módulos `.py` del repositorio dentro del notebook (opcional):

```python
# Clonar el repo y agregar Modern/ al path
!git clone <url-del-repo> /content/repo
import sys
sys.path.insert(0, '/content/repo/Modern')

# Ahora se pueden importar los módulos
from modulo4_transformer import cargar_modelo, predecir_pico_placa
```

### Variables de Entorno (API)

| Variable | Descripción | Requerida |
|---|---|---|
| `HF_TOKEN` | Token de HuggingFace para descargar el modelo privado | Sí (Render) |
| `RUTA_MODELO_TRANSFORMER` | Ruta personalizada al archivo `.pt` | No (usa default) |

---

## Ejecución

### Menú Local (`main.py`)

```
============================================================
   DETECCIÓN DE PLACAS VEHICULARES — POPAYÁN
============================================================
  [1] Pipeline de imágenes (seleccionar archivos)
  [2] Asistente conversacional (Módulo 5)
  [3] Cámara en tiempo real (Módulo 6)
  [0] Salir
```

### Módulos Independientes

Cada módulo puede ejecutarse de forma standalone para pruebas:

```bash
# Probar detección YOLO con una imagen
python Modern/modulo1_deteccion_yolo.py foto_vehiculo.jpg

# Probar OCR con un recorte de placa
python Modern/modulo2_ocr.py recorte_placa.png

# Generar dataset de prueba
python Modern/modulo3_dataset.py

# Entrenar Transformer desde cero (mini-dataset de prueba)
python Modern/modulo4_transformer.py

# Iniciar asistente conversacional
python Modern/modulo5_asistente.py

# Iniciar cámara en tiempo real
python Modern/modulo6_camara.py --cam 0
python Modern/modulo6_camara.py --cam http://192.168.1.X:8080/video
```

---

## Despliegue

### FastAPI en Render.com

El servidor se define en `app.py` y se configura en `render.yaml`.

**Endpoints disponibles:**

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servidor (no carga modelos) |
| `POST` | `/detect` | Recibe imagen JPG/PNG → retorna placa y restricción |
| `POST` | `/detect/texto` | Recibe texto de placa → retorna predicción |

**Flujo del primer request (`/detect`):**
1. Lazy loading: descarga `transformer_pico_placa.pt` desde HuggingFace si no existe
2. Carga YOLO11, EasyOCR y Transformer en memoria
3. Procesa la imagen con el pipeline completo
4. Retorna JSON con placa, restricción, confianza y probabilidades por día

**Configuración Render (`render.yaml`):**
```yaml
buildCommand: pip install -r requirements_api.txt
startCommand: uvicorn app:app --host 0.0.0.0 --port 10000
healthCheckPath: /health
```

Configurar en el dashboard de Render la variable de entorno `HF_TOKEN` con tu token de HuggingFace.

### App Gradio en Google Colab

El despliegue Gradio es autónomo dentro del notebook. Al ejecutar la última celda del Módulo 5:

```python
# Se lanza automáticamente si AUTO_DESPLEGAR_APP_INTEGRAL = True
app_integral_gradio = desplegar_app_integral_gradio(share=True)
```

Gradio genera un enlace público temporal (válido 72 horas). El `share=True` crea un túnel desde los servidores de Gradio hasta la sesión de Colab.

---

## Relación entre Módulos y Notebook

### Principio de diseño

El notebook `Modelo_IA_Pico_Placa_V12.ipynb` y los módulos `.py` son **implementaciones paralelas** de la misma lógica, diseñadas para convivir sin interferir:

```
Notebook (Colab)              Módulos .py (local/API)
─────────────────             ───────────────────────
Funciones inline              Funciones importables
Estado global en celdas       Singletons controlados
Gradio integrado              Exporta API limpia
torch.save() en /content/     torch.save() en modelos/
__file__ no disponible        __file__ con try/except
lector_easy global            obtener_lector_easyocr()
model_transformer global      cargar_modelo(ruta)
```

### Regla de oro de actualización

> **El despliegue Gradio del notebook es la fuente de verdad.** Cualquier actualización a los módulos `.py` debe ser compatible hacia atrás con las funciones que el notebook ya consume. Nunca cambiar las firmas de `predecir_pico_placa()`, `consultar_asistente_pico_placa()`, o `asignar_restriccion()` sin verificar primero el notebook.

### Sincronización manual (cuando el notebook necesita funcionalidades nuevas)

Si agregas lógica nueva a un módulo que quieres que el notebook también tenga, el proceso correcto es:

1. Implementar en el módulo `.py` con tests
2. Copiar la función al notebook dentro de la celda correspondiente
3. Verificar que el despliegue Gradio sigue funcionando sin cambios

---

## Reglas de Pico y Placa — Popayán

| Último dígito | Día de restricción |
|:---:|---|
| 0, 1 | Lunes |
| 2, 3 | Martes |
| 4, 5 | Miércoles |
| 6, 7 | Jueves |
| 8, 9 | Viernes |

Aplica de lunes a viernes en horario de restricción. El sistema **no inventa** horarios, excepciones, permisos, ni normas para motos, taxis o vehículos de carga — solo reporta lo que está codificado en las fuentes internas del proyecto.

---

## Precisión del Modelo

| Conjunto | Precisión |
|---|---|
| Entrenamiento | ~99.5% |
| Validación | ~98.8% |
| **Test (final)** | **98.22%** |

Entrenado con ~48.000 placas sintéticas con variabilidad controlada (formatos antiguo/nuevo + confusiones OCR simuladas) más placas reales opcionales del dataset Kaggle `andrewmvd/car-plate-detection`.

---

*Proyecto académico — Popayán, Colombia.*