# Proyecto Detección de Placas Vehiculares — Popayán

## Descripción
Sistema de visión computacional para detección y clasificación de placas vehiculares
con predicción de restricción de Pico y Placa en Popayán.

## Módulos
| Módulo | Responsable | Descripción |
|--------|-------------|-------------|
| 1 | Integrante 1 | Detección de placa con YOLO11 |
| 2 | Integrante 2 | Segmentación y OCR (EasyOCR + Tesseract) |
| 3 | Integrante 3 | Tokenización y dataset para Transformer |
| 4 | Integrante 4 | Transformer desde cero (PyTorch) |

## Resultados
- **Precisión del modelo:** 98.16%
- **Arquitectura:** Transformer (d_model=64, heads=4, layers=2)
- **Dataset:** 50.000 placas sintéticas con ambigüedad OCR real

## Archivos
- `modelos/transformer_pico_placa.pt` — modelo entrenado
- `modelos/modelo_metadatos.json` — metadatos y configuración
- `proyecto_placas_popayan.py` — código fuente completo

## Uso rápido
```python
import torch
checkpoint = torch.load('modelos/transformer_pico_placa.pt', map_location='cpu')
# Ver modelo_metadatos.json para la arquitectura completa
```

## Pico y Placa Popayán
| Último dígito | Día restringido |
|---------------|-----------------|
| 0, 1 | Lunes |
| 2, 3 | Martes |
| 4, 5 | Miércoles |
| 6, 7 | Jueves |
| 8, 9 | Viernes |
