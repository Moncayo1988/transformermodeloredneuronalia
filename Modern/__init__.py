"""
Modern/__init__.py
Agrega el directorio Modern/ al sys.path para que los módulos del proyecto
puedan importarse entre sí con rutas relativas (from modulo0_config import ...)
sin importar desde qué directorio se ejecute el servidor.
"""
import sys
import os

_modern_dir = os.path.dirname(os.path.abspath(__file__))
if _modern_dir not in sys.path:
    sys.path.insert(0, _modern_dir)