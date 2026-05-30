# ==============================================================================
# MÓDULO 2B — CORRECTOR DE ERRORES OCR PARA PLACAS COLOMBIANAS  v2.1
# ==============================================================================
#
# ERRORES QUE CORRIGE ESTE MÓDULO:
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Tipo              │ Ejemplo              │ Causa                             │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │ Dígito en letras  │ 0→O, 1→I, 8→B       │ Zona letras pos 0-2               │
# │ Letra en dígitos  │ O→0, I→1, S→5, A→4  │ Zona dígitos pos 3-5              │
# │ Fantasma inicio   │ JGLV753→GLV753       │ Artefacto borde bbox              │
# │ Intruso dígitos   │ ABC12I3→ABC123       │ Letra en zona de dígitos          │
# │ 7-8 chars         │ JGLV7531→GLV753      │ Múltiples artefactos              │
# │ Incompleta <4     │ FHJ4→None            │ Placa fuera del frame → N/A       │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │ NO corregibles (limitación del OCR — requieren mejor imagen):               │
# │  • Letras confundidas entre sí cuando el resultado ya es LLL+NNN válido:    │
# │    SXY leída SHY, WDE leída ROE, GHR leída GHP.                            │
# │  • Dígitos confundidos entre sí (9≠0): no hay regla segura sin romper       │
# │    placas legítimas que sí terminan en 9.                                  │
# │  • Y en zona letras: Y es letra válida en placas (HSY095); sin contexto    │
# │    externo no podemos saber si Y fue leída correctamente o debería ser V.  │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# USO:
#   from modulo2b_corrector_ocr import corregir_placa, corregir_candidatos
#   placa = corregir_placa("JGLV753")          # → "GLV753"
#   lista = corregir_candidatos(["1BC456", "CARTAGENA"])  # → ["IBC456"]
# ==============================================================================

import re
import unicodedata
from typing import Optional
from collections import Counter


# ------------------------------------------------------------------------------
# 1. TABLAS DE CONFUSIÓN OCR
# ------------------------------------------------------------------------------

# Zona de LETRAS (posiciones 0-2)
# Solo corregimos DÍGITOS que ocupan una posición de letra.
# NO incluimos Y→V: Y es letra válida en placas colombianas (HSY, UAY, etc.)
CONFUSIONES_EN_LETRAS = {
    '0': 'O',   # cero  → O
    '1': 'I',   # uno   → I
    '5': 'S',   # 5     → S
    '6': 'G',   # 6     → G
    '8': 'B',   # 8     → B
    '2': 'Z',   # 2     → Z
    '4': 'A',   # 4     → A  (raro)
    '3': 'E',   # 3     → E  (raro)
    '7': 'T',   # 7     → T  (raro)
    '9': 'P',   # 9     → P  (raro)
}

# Zona de DÍGITOS (posiciones 3-5)
# Solo corregimos LETRAS que ocupan una posición de dígito.
# NO incluimos correcciones dígito↔dígito (9→0, 6→0): son ambiguas y rompen
# placas legítimas que sí contienen esos dígitos.
CONFUSIONES_EN_DIGITOS = {
    'O': '0',   # O  → 0
    'Q': '0',   # Q  → 0
    'I': '1',   # I  → 1
    'L': '1',   # L  → 1
    'S': '5',   # S  → 5
    'G': '6',   # G  → 6
    'B': '8',   # B  → 8
    'Z': '2',   # Z  → 2
    'A': '4',   # A  → 4
    'D': '0',   # D  → 0
    'T': '7',   # T  → 7
    'U': '0',   # U  → 0
    'J': '1',   # J  → 1
    'P': '0',   # P  → 0
    'C': '0',   # C  → 0
    'E': '8',   # E  → 8   (raro)
    'H': '4',   # H  → 4
    'F': '7',   # F  → 7
    'N': '4',   # N  → 4
    'M': '0',   # M  → 0
    'V': '1',   # V  → 1   (confusión en algunos modelos)
    'X': '8',   # X  → 8
    'K': '4',   # K  → 4
    'R': '2',   # R  → 2
    'W': '0',   # W  → 0   (muy raro)
}

# Caracteres "fantasma" que EasyOCR agrega por artefactos de borde del bbox
_FANTASMAS = {'I', 'L', '1', 'J', '|'}


# ------------------------------------------------------------------------------
# 2. REGEXES DE VALIDACIÓN
# ------------------------------------------------------------------------------

_RE_ESTANDAR = re.compile(r'^[A-Z]{3}[0-9]{3}$')       # ABC123
_RE_MOTO     = re.compile(r'^[A-Z]{3}[0-9]{2}[A-Z]$')  # ABC12D
_RE_FLEXIBLE = re.compile(r'^[A-Z0-9]{5,7}$')          # 5-7 alfanum.


def _es_valido(texto: str, estricto: bool = True) -> bool:
    if estricto:
        return bool(_RE_ESTANDAR.match(texto) or _RE_MOTO.match(texto))
    return bool(_RE_FLEXIBLE.match(texto))


# ------------------------------------------------------------------------------
# 3. LIMPIEZA BÁSICA
# ------------------------------------------------------------------------------

def _limpiar(texto: str) -> str:
    if not texto:
        return ''
    texto = unicodedata.normalize('NFKD', texto)
    texto = texto.encode('ascii', 'ignore').decode('ascii')
    texto = texto.upper().strip()
    texto = re.sub(r'[-.\s·•*_]', '', texto)
    texto = re.sub(r'[^A-Z0-9]', '', texto)
    return texto


# ------------------------------------------------------------------------------
# 4. CORRECCIÓN POSICIONAL ESTÁNDAR
# ------------------------------------------------------------------------------

def _corregir_posicionalmente(texto: str) -> str:
    """
    Aplica tablas de confusión según posición.
    Solo opera con exactamente 6 caracteres.
    """
    if len(texto) != 6:
        return texto
    chars = list(texto)
    for i in range(6):
        c = chars[i]
        if i < 3:
            if c in CONFUSIONES_EN_LETRAS:
                chars[i] = CONFUSIONES_EN_LETRAS[c]
        else:
            if c in CONFUSIONES_EN_DIGITOS:
                chars[i] = CONFUSIONES_EN_DIGITOS[c]
    return ''.join(chars)


# ------------------------------------------------------------------------------
# 5. RECUPERACIÓN DE LONGITUD != 6
# ------------------------------------------------------------------------------

def _intentar_recuperar(texto: str) -> str:
    """
    Heurísticas para cadenas de 5, 7 u 8 caracteres.
    Para 8 chars: intenta eliminar 2 artefactos encadenando con la lógica de 7.
    """
    n = len(texto)

    # ── 7 CARACTERES ──────────────────────────────────────────────────────────
    if n == 7:
        # 1. Fantasma al inicio (J/I/L/1) → LLL-NNN
        if texto[0] in _FANTASMAS and texto[1:4].isalpha() and texto[4:].isdigit():
            return texto[1:]

        # 2. Fantasma al final
        if texto[-1] in _FANTASMAS and texto[:3].isalpha() and texto[3:6].isdigit():
            return texto[:6]

        # 3. LETRA intrusa en zona de dígitos (prioridad sobre eliminar dígitos)
        for idx in range(3, 7):
            if idx < len(texto) and texto[idx].isalpha():
                candidato = texto[:idx] + texto[idx+1:]
                if len(candidato) == 6:
                    corr = _corregir_posicionalmente(candidato)
                    if _es_valido(corr):
                        return candidato

        # 4. Guión o punto residual
        sin = re.sub(r'[-.]', '', texto)
        if len(sin) == 6:
            return sin

        # 5. Cualquier posición que produzca candidato válido
        for idx in range(7):
            candidato = texto[:idx] + texto[idx+1:]
            if len(candidato) == 6:
                corr = _corregir_posicionalmente(candidato)
                if _es_valido(corr):
                    return candidato

    # ── 8 CARACTERES ──────────────────────────────────────────────────────────
    if n == 8:
        # Primero: letras intrusas en zona de dígitos
        for idx in range(3, 8):
            if idx < len(texto) and texto[idx].isalpha():
                candidato = texto[:idx] + texto[idx+1:]
                if len(candidato) == 6:
                    corr = _corregir_posicionalmente(candidato)
                    if _es_valido(corr):
                        return candidato

        # Luego: eliminar 1 posición → 7 chars → encadenar con lógica de 7
        for idx in range(8):
            candidato7 = texto[:idx] + texto[idx+1:]   # 7 chars
            if len(candidato7) == 7:
                recuperado = _intentar_recuperar(candidato7)
                if len(recuperado) == 6:
                    corr = _corregir_posicionalmente(recuperado)
                    if _es_valido(corr):
                        return recuperado

    # ── 5 CARACTERES ──────────────────────────────────────────────────────────
    # No se puede completar sin saber qué faltó.

    return texto


# ------------------------------------------------------------------------------
# 6. CORRECCIÓN FLEXIBLE (último recurso)
# ------------------------------------------------------------------------------

def _correccion_flexible(texto: str) -> Optional[str]:
    """
    Corrige posiciones en el lado equivocado:
    letra en zona dígitos → convertir; dígito en zona letras → convertir.
    """
    chars = list(texto)
    cambios = False
    for i in range(6):
        c = chars[i]
        if i < 3 and c.isdigit() and c in CONFUSIONES_EN_LETRAS:
            chars[i] = CONFUSIONES_EN_LETRAS[c]
            cambios = True
        elif i >= 3 and c.isalpha() and c in CONFUSIONES_EN_DIGITOS:
            chars[i] = CONFUSIONES_EN_DIGITOS[c]
            cambios = True
    resultado = ''.join(chars)
    return resultado if cambios and resultado != texto else None


# ------------------------------------------------------------------------------
# 7. FUNCIÓN PRINCIPAL
# ------------------------------------------------------------------------------

def corregir_placa(
    texto_raw  : str,
    tipo_placa : str  = 'blanco',
    verbose    : bool = False,
) -> Optional[str]:
    """
    Cadena completa de corrección OCR para una placa colombiana.

    Pasos:
      1. Limpieza básica
      2. Descarte si < 4 chars
      3. Recuperación de longitud (7 y 8 chars encadenados)
      4. Corrección posicional (tablas dígito↔letra)
      5. Validación — retornar si pasa
      6. Corrección flexible (último recurso)
      7. None si todo falla

    Retorna str con placa válida o None.
    """
    if not texto_raw:
        return None

    log = [f'[CorrectorOCR] raw="{texto_raw}"']

    # 1. Limpieza
    texto = _limpiar(texto_raw)
    log.append(f'  [1] limpieza     : "{texto}"')

    # 2. Descarte rápido
    if len(texto) < 4:
        log.append(f'  [2] DESCARTADO   : {len(texto)} chars < 4')
        if verbose: print('\n'.join(log))
        return None

    # 3. Recuperación de longitud
    if len(texto) != 6:
        recuperado = _intentar_recuperar(texto)
        log.append(f'  [3] recuperación : "{texto}"({len(texto)}) → "{recuperado}"({len(recuperado)})')
        texto = recuperado
    else:
        log.append(f'  [3] longitud OK  : 6 chars')

    # 4. Corrección posicional
    if len(texto) == 6:
        corr = _corregir_posicionalmente(texto)
        log.append(f'  [4] posicional   : "{texto}" → "{corr}"')
        texto = corr
    else:
        log.append(f'  [4] posicional   : OMITIDA (longitud {len(texto)})')

    # 5. Validación
    if _es_valido(texto):
        log.append(f'  [5] VÁLIDO ✓     : "{texto}"')
        if verbose: print('\n'.join(log))
        return texto

    # 6. Corrección flexible
    if len(texto) == 6:
        flex = _correccion_flexible(texto)
        if flex and _es_valido(flex):
            log.append(f'  [6] flexible     : "{texto}" → "{flex}"')
            if verbose: print('\n'.join(log))
            return flex
        log.append(f'  [6] flexible     : sin resultado')

    # 7. Sin solución
    log.append(f'  [7] DESCARTADO   : "{texto}" no corregible')
    if verbose: print('\n'.join(log))
    return None


# ------------------------------------------------------------------------------
# 8. CORRECCIÓN DE LISTA DE CANDIDATOS
# ------------------------------------------------------------------------------

def corregir_candidatos(
    candidatos : list,
    tipo_placa : str  = 'blanco',
    verbose    : bool = False,
) -> list:
    """
    Aplica corregir_placa() a toda la lista de candidatos de EasyOCR/Tesseract.
    Retorna placas válidas sin duplicados, ordenadas por frecuencia.
    """
    corregidos = []
    for raw in candidatos:
        resultado = corregir_placa(raw, tipo_placa=tipo_placa, verbose=verbose)
        if resultado:
            corregidos.append(resultado)
    return [p for p, _ in Counter(corregidos).most_common()]


# ------------------------------------------------------------------------------
# 9. VALIDACIÓN CON DIAGNÓSTICO
# ------------------------------------------------------------------------------

def validar_formato_placa(texto: str) -> dict:
    """Validación completa con diagnóstico. Retorna dict con valido, tipo, mensaje."""
    r = {
        'valido': False, 'tipo': 'invalido',
        'longitud_ok': len(texto) == 6,
        'letras_ok': False, 'digitos_ok': False, 'mensaje': '',
    }
    if len(texto) != 6:
        r['mensaje'] = f'Longitud {len(texto)} (esperado 6)'
        return r
    r['letras_ok']  = texto[:3].isalpha()
    r['digitos_ok'] = texto[3:].isdigit()
    if _RE_ESTANDAR.match(texto):
        r.update({'valido': True, 'tipo': 'estandar'})
    elif _RE_MOTO.match(texto):
        r.update({'valido': True, 'tipo': 'moto'})
    elif _RE_FLEXIBLE.match(texto):
        r.update({'valido': True, 'tipo': 'flexible',
                  'mensaje': 'Posiblemente diplomática o antigua'})
    else:
        partes = []
        if not r['letras_ok']:  partes.append('letras incorrectas pos 0-2')
        if not r['digitos_ok']: partes.append('dígitos incorrectos pos 3-5')
        r['mensaje'] = ', '.join(partes)
    return r


# ==============================================================================
# PRUEBAS — python modulo2b_corrector_ocr.py
# ==============================================================================

if __name__ == '__main__':
    casos = [
        # ── Errores de producción (imágenes reales) ───────────────────────────
        ('SHY424',    'SHY424',  'img1: X→H — LLL+NNN válido, no corregible por reglas'),
        ('JGLV753',   'GLV753',  'img2: J fantasma al inicio ✓'),
        ('FHJ4',      None,      'img3: 4 chars — irrecuperable ✓'),
        ('GHP210',    'GHP210',  'img4: P→R — LLL+NNN válido, no corregible por reglas'),
        ('IIB722',    'IIB722',  'img5: binarización destruida — válido estructuralmente'),
        ('ROE576',    'ROE576',  'img6: R→W — LLL+NNN válido, no corregible por reglas'),
        ('HSY995',    'HSY995',  'img7: 9≠0 — dígito-vs-dígito, no corregible sin romper otros'),
        ('WGM000',    'WGM000',  'img8: placa correcta — sin cambios ✓'),
        ('FXT27',     None,      'img9: 5 chars — irrecuperable ✓'),
        # ── Correcciones funcionales ──────────────────────────────────────────
        ('1BC456',    'IBC456',  '1→I zona letras ✓'),
        ('ABC12I3',   'ABC123',  'I intrusa en zona dígitos ✓'),
        ('JGLV7531',  'GLV753',  '8 chars: J fantasma + 1 extra (encadenado) ✓'),
        ('AB0123',    'ABO123',  '0→O zona letras ✓'),
        ('8BC456',    'BBC456',  '8→B zona letras ✓'),
        ('ABC4S6',    'ABC456',  'S→5 zona dígitos ✓'),
        ('ABCO56',    'ABC056',  'O→0 zona dígitos ✓'),
        ('HDE576',    'HDE576',  'placa correcta — sin cambios ✓'),
        ('UAQ629',    'UAQ629',  'Q legítima — sin cambios ✓'),
        ('HSY095',    'HSY095',  'Y legítima — sin cambios ✓'),
        ('ABC123',    'ABC123',  'placa perfecta — sin cambios ✓'),
        ('  ABC-123 ','ABC123',  'limpieza espacios y guión ✓'),
        ('0BC456',    'OBC456',  '0→O zona letras ✓'),
        ('ABC4O6',    'ABC406',  'O→0 zona dígitos ✓'),
        ('XYZ123',    'XYZ123',  'X, Y, Z válidas — sin cambios ✓'),
        ('ABC1O3',    'ABC103',  'O→0 en pos 4 ✓'),
    ]

    print('=' * 74)
    print('  CORRECTOR OCR v2.1 — PRUEBAS CON CASOS REALES DE PRODUCCIÓN')
    print('=' * 74)
    print(f'{"Entrada":<14} {"Esperado":<12} {"Obtenido":<12} {"OK":<4} Descripción')
    print('-' * 74)

    ok = total = 0
    for raw, esperado, desc in casos:
        total += 1
        obtenido = corregir_placa(raw, verbose=False)
        paso = (obtenido == esperado)
        if paso: ok += 1
        marca = '✓' if paso else '✗'
        print(f'{raw!r:<14} {str(esperado):<12} {str(obtenido):<12} {marca}   {desc}')

    print('-' * 74)
    print(f'  Resultado: {ok}/{total} pruebas pasadas ({ok/total*100:.0f}%)')
    print()
    print('  img1/img4/img6: X→H, P→R, R→W — el OCR devuelve LLL+NNN válido.')
    print('  Sin contexto externo (base de datos de placas reales) no hay regla')
    print('  segura para distinguir SHY de SXY. Solución: mejor imagen.')
    print()
    print('  img7: HSY995 — 9≠0 es ambiguo; hay placas reales con 9.')
    print('  img5: DOU722 leída IIB722 — fallo de binarización upstream.')
    print('=' * 74)