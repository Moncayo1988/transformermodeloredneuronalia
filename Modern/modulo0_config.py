# ==============================================================================
# MÓDULO 0 — CONFIGURACIÓN GLOBAL, INSTALACIÓN Y VOCABULARIO
# ==============================================================================
# CAMBIOS v2:
#   - Eliminados duplicados silenciosos en DIGITO_A_LETRA y LETRA_A_DIGITO
#   - CONFUSIONES_OCR reestructuradas como lista de pares (sin pisarse)
#   - Nueva función: corregir_confusion_visual() para uso en Módulo 2
#   - Agregadas confusiones U↔J, W↔VV, I↔J que faltaban
# ==============================================================================

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
# REGLA: posiciones 0-2 deben ser LETRAS, posiciones 3-5 deben ser DÍGITOS.

# Dígito/símbolo leído por OCR que debería ser una LETRA (posiciones 0-2)
# SIN duplicados — cada clave aparece una sola vez
DIGITO_A_LETRA = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'J',   # 3 se parece a J visualmente
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'P',
}

# Letra leída por OCR que debería ser un DÍGITO (posiciones 3-5)
# SIN duplicados — cada clave aparece una sola vez
LETRA_A_DIGITO = {
    'O': '0',
    'D': '0',
    'Q': '0',
    'I': '1',
    'L': '1',
    'Z': '2',
    'J': '3',   # J se parece a 3 — era el bug de JLY246 → UJLY246
    'A': '4',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
    'P': '9',
    'U': '0',   # U confundida con O en zona de dígitos
}

# ------------------------------------------------------------------------------
# 6. CONFUSIONES VISUALES PARA GENERACIÓN DE DATASET
# ------------------------------------------------------------------------------
# Reestructuradas como lista de pares (A, B) donde A↔B se confunden.
# Esto evita los diccionarios con claves duplicadas que se pisaban en v1.

PARES_CONFUSION = [
    ('O', '0'),
    ('I', '1'),
    ('S', '5'),
    ('B', '8'),
    ('G', '6'),
    ('Z', '2'),
    ('J', '3'),   # ← causa del bug JLY → 3LY  /  3 → J
    ('U', 'J'),   # ← causa del bug UJLY246 (U confundida con J al inicio)
    ('U', '0'),   # ← U confundida con O/0 en zona numérica
    ('I', 'J'),
    ('W', 'H'),
    ('D', '0'),
    ('L', '1'),
    ('Q', '0'),
    ('A', '4'),
    ('T', '7'),
    ('P', '9'),
]

# Diccionario de confusiones para generación sintética (un sentido):
CONFUSIONES_OCR = {}
for a, b in PARES_CONFUSION:
    # Solo guardamos la dirección más probable (carácter real → confusión OCR)
    if a not in CONFUSIONES_OCR:
        CONFUSIONES_OCR[a] = b
    if b not in CONFUSIONES_OCR:
        CONFUSIONES_OCR[b] = a

# ------------------------------------------------------------------------------
# 7. TABLA DE CORRECCIÓN VISUAL (para Módulo 2, independiente de posición)
# ------------------------------------------------------------------------------
# Usada ANTES de la corrección posicional para limpiar errores visuales puros.
# Clave = lo que leyó el OCR, Valor = lo que probablemente es.

CORRECCION_VISUAL = {
    # Letras que el OCR confunde con otras letras
    'U': 'J',   # el caso más frecuente en tus pruebas (UJLY → JLY)
    # Nota: U→0 se maneja en LETRA_A_DIGITO (posición 3-5)
    # Los demás casos se cubren con las tablas posicionales
}

# ------------------------------------------------------------------------------
# 8. REGLAS PICO Y PLACA — POPAYÁN
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
# 9. FUNCIONES DE UTILIDAD COMPARTIDAS
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
    Ejemplo: 'HSY095' → [7, 18, 24, 27, 22, 32, 0]
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    texto     = re.sub(r'[^A-Z0-9]', '', str(texto).upper())
    secuencia = [char2idx[c] for c in texto if c in char2idx]
    return pad_sequences([secuencia], maxlen=MAX_LEN, padding='post', value=0)[0]


if __name__ == "__main__":
    print(f"[OK] Config cargada — VOCAB_SIZE={VOCAB_SIZE} | MAX_LEN={MAX_LEN}")
    print(f"     Patrones: ANTIGUA={PATRON_ANTIGUA.pattern} | NUEVA={PATRON_NUEVA.pattern}")
    print(f"     Días Pico y Placa: {DIAS_UNICOS}")
    print(f"\n     DIGITO_A_LETRA  : {DIGITO_A_LETRA}")
    print(f"     LETRA_A_DIGITO  : {LETRA_A_DIGITO}")
    print(f"     CORRECCION_VISUAL: {CORRECCION_VISUAL}")
    print(f"\n     Pares de confusión registrados: {len(PARES_CONFUSION)}")