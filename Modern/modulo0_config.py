# ==============================================================================
# MÓDULO 0 — CONFIGURACIÓN GLOBAL, INSTALACIÓN Y VOCABULARIO
# ==============================================================================
# Responsabilidad:
#   - Instalar dependencias del proyecto
#   - Definir constantes compartidas por todos los módulos
#   - Vocabulario, tablas de corrección y patrones colombianos
#
# Todos los demás módulos importan desde aquí.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. INSTALACIÓN DE DEPENDENCIAS (ejecutar solo en Google Colab)
# ------------------------------------------------------------------------------
def instalar_dependencias():
    import subprocess
    subprocess.run(["pip", "install", "ultralytics", "huggingface_hub", "easyocr", "-q"])
    subprocess.run(["apt-get", "update", "-y", "-q"])
    subprocess.run(["apt-get", "install", "tesseract-ocr", "libtesseract-dev", "-y", "-q"])
    subprocess.run(["apt-get", "install", "libavif-dev", "-y", "-q"])
    subprocess.run(["pip", "install", "pytesseract", "pandas", "tensorflow",
                    "pillow-avif-plugin", "-q"])
    print("[OK] Dependencias instaladas.")


# ------------------------------------------------------------------------------
# 2. IMPORTS COMUNES
# ------------------------------------------------------------------------------
import re
import string

# ------------------------------------------------------------------------------
# 3. VOCABULARIO Y TOKENIZACIÓN
# ------------------------------------------------------------------------------
caracteres = list(string.ascii_uppercase + string.digits)
tokens     = ['<PAD>'] + caracteres
char2idx   = {char: idx for idx, char in enumerate(tokens)}
idx2char   = {idx: char for char, idx in char2idx.items()}

MAX_LEN    = 7
VOCAB_SIZE = len(char2idx)   # 37
EMBED_DIM  = 16

# ------------------------------------------------------------------------------
# 4. PATRONES DE PLACA COLOMBIANA
# ------------------------------------------------------------------------------
PATRON_ANTIGUA = re.compile(r'^[A-Z]{3}[0-9]{3}$')       # ABC123
PATRON_NUEVA   = re.compile(r'^[A-Z]{3}[0-9]{2}[A-Z]$')  # ABC12D

# ------------------------------------------------------------------------------
# 5. TABLAS DE CORRECCIÓN POSICIONAL (OCR → placa real)
# ------------------------------------------------------------------------------
# Dígito leído por OCR que debería ser una letra (posiciones 0-2)
DIGITO_A_LETRA = {
    '0': 'O', '1': 'I', '5': 'S', '8': 'B',
    '6': 'G', '4': 'A', '7': 'T', '2': 'Z',
    '3': 'J', 'W': 'H', '0': 'I',
}

# Letra leída por OCR que debería ser un dígito (posiciones 3-5)
LETRA_A_DIGITO = {
    'O': '0', 'I': '1', 'S': '5', 'B': '8',
    'D': '0', 'G': '6', 'Z': '2', 'T': '7',
    'Q': '0', 'U': '0', 'A': '4', 'L': '1',
    'J': '3', 'U': '0', 'M': 'N', 'I': '1',
    'O': '0',

}

# Confusiones visuales comunes en OCR (usadas en generación de dataset)
CONFUSIONES_OCR = {
    'O': '0', '0': 'O',
    'I': '1', '1': 'I',
    'S': '5', '5': 'S',
    'B': '8', '8': 'B',
    'G': '6', '6': 'G',
    'Z': '2', '2': 'Z',
}

# ------------------------------------------------------------------------------
# 6. REGLAS PICO Y PLACA — POPAYÁN
# ------------------------------------------------------------------------------
REGLAS_PICO_PLACA = {
    '0': 'Lunes',     '1': 'Lunes',
    '2': 'Martes',    '3': 'Martes',
    '4': 'Miércoles', '5': 'Miércoles',
    '6': 'Jueves',    '7': 'Jueves',
    '8': 'Viernes',   '9': 'Viernes',
}

DIAS_UNICOS = ['Jueves', 'Lunes', 'Miércoles', 'Martes', 'Viernes']
label2idx   = {dia: i for i, dia in enumerate(DIAS_UNICOS)}
idx2label   = {i: dia for dia, i in label2idx.items()}
NUM_CLASES  = len(DIAS_UNICOS)

# ------------------------------------------------------------------------------
# 7. FUNCIONES DE UTILIDAD COMPARTIDAS
# ------------------------------------------------------------------------------

def asignar_restriccion(ultimo_digito: str) -> str:
    """Retorna el día de restricción según el último dígito de la placa."""
    return REGLAS_PICO_PLACA.get(str(ultimo_digito), 'Sin restricción')


def validar_formato_colombiano(placa: str) -> bool:
    """Valida si la placa cumple el formato oficial colombiano (antiguo o nuevo)."""
    return bool(PATRON_ANTIGUA.match(placa) or PATRON_NUEVA.match(placa))


def tokenizar_placa(texto: str) -> list:
    """
    Convierte una placa en secuencia numérica con padding hasta MAX_LEN.
    Usa NumPy puro (sin TensorFlow) para compatibilidad con PyTorch.

    Ejemplo: 'HSY095' → [7, 18, 24, 27, 22, 32, 0]
    """
    import numpy as np
    texto     = re.sub(r'[^A-Z0-9]', '', str(texto).upper())
    secuencia = [char2idx[c] for c in texto if c in char2idx]
    # Padding manual hasta MAX_LEN
    secuencia = secuencia[:MAX_LEN]
    secuencia = secuencia + [0] * (MAX_LEN - len(secuencia))
    return np.array(secuencia)


if __name__ == "__main__":
    print(f"[OK] Config cargada — VOCAB_SIZE={VOCAB_SIZE} | MAX_LEN={MAX_LEN}")
    print(f"     Patrones: ANTIGUA={PATRON_ANTIGUA.pattern} | NUEVA={PATRON_NUEVA.pattern}")
    print(f"     Días Pico y Placa: {DIAS_UNICOS}")