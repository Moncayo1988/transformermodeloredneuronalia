# ==============================================================================
# MÓDULO 0 — CONFIGURACIÓN GLOBAL Y VOCABULARIO
# Autor: Salomón Melenje
# ==============================================================================

import re
import string

# ------------------------------------------------------------------------------
# VOCABULARIO Y TOKENIZACIÓN
# ------------------------------------------------------------------------------
caracteres = list(string.ascii_uppercase + string.digits)
tokens     = ['<PAD>'] + caracteres
char2idx   = {c: i for i, c in enumerate(tokens)}
idx2char   = {i: c for c, i in char2idx.items()}

MAX_LEN    = 7
VOCAB_SIZE = len(char2idx)   # 37
EMBED_DIM  = 16

# ------------------------------------------------------------------------------
# PATRONES DE PLACA COLOMBIANA
# ------------------------------------------------------------------------------
PATRON_ANTIGUA = re.compile(r'^[A-Z]{3}[0-9]{3}$')
PATRON_NUEVA   = re.compile(r'^[A-Z]{3}[0-9]{2}[A-Z]$')

# ------------------------------------------------------------------------------
# TABLAS DE CORRECCIÓN POSICIONAL (OCR → placa real)
# ------------------------------------------------------------------------------
DIGITO_A_LETRA = {'0':'O','1':'I','5':'S','8':'B','6':'G','4':'A','7':'T','2':'Z','3':'J'}
LETRA_A_DIGITO = {
    'O':'0','I':'1','S':'5','B':'8','D':'0','G':'6','Z':'2','T':'7',
    'Q':'0','U':'0','A':'4','L':'1','J':'3','M':'N',
}
CONFUSIONES_OCR = {'O':'0','0':'O','I':'1','1':'I','S':'5','5':'S','B':'8','8':'B','G':'6','6':'G','Z':'2','2':'Z'}

# ------------------------------------------------------------------------------
# REGLAS PICO Y PLACA — POPAYÁN
# ------------------------------------------------------------------------------
REGLAS_PICO_PLACA = {
    '0':'Lunes','1':'Lunes','2':'Martes','3':'Martes',
    '4':'Miércoles','5':'Miércoles','6':'Jueves','7':'Jueves',
    '8':'Viernes','9':'Viernes',
}
DIAS_UNICOS = ['Jueves','Lunes','Miércoles','Martes','Viernes']
label2idx   = {d: i for i, d in enumerate(DIAS_UNICOS)}
idx2label   = {i: d for d, i in label2idx.items()}
NUM_CLASES  = len(DIAS_UNICOS)

# ------------------------------------------------------------------------------
# FUNCIONES COMPARTIDAS
# ------------------------------------------------------------------------------

def asignar_restriccion(ultimo_digito: str) -> str:
    return REGLAS_PICO_PLACA.get(str(ultimo_digito), 'Sin restricción')

def validar_formato_colombiano(placa: str) -> bool:
    return bool(PATRON_ANTIGUA.match(placa) or PATRON_NUEVA.match(placa))

def tokenizar_placa(texto: str) -> 'np.ndarray':
    import numpy as np
    texto = re.sub(r'[^A-Z0-9]', '', str(texto).upper())
    seq   = [char2idx[c] for c in texto if c in char2idx][:MAX_LEN]
    seq  += [0] * (MAX_LEN - len(seq))
    return np.array(seq)

def instalar_dependencias():
    import subprocess
    subprocess.run(["pip","install","ultralytics","huggingface_hub","easyocr","-q"])
    subprocess.run(["apt-get","install","tesseract-ocr","libtesseract-dev","-y","-q"])
    subprocess.run(["pip","install","pytesseract","pandas","pillow-avif-plugin","-q"])
    print("[OK] Dependencias instaladas.")

if __name__ == "__main__":
    print(f"[OK] Config — VOCAB_SIZE={VOCAB_SIZE} | MAX_LEN={MAX_LEN} | Días: {DIAS_UNICOS}")