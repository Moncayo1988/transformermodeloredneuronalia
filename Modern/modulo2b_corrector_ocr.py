# ==============================================================================
# MÓDULO 2B — CORRECTOR DE ERRORES OCR PARA PLACAS COLOMBIANAS  v2.1
# Autor: Salomón Melenje
# ==============================================================================

import re
import unicodedata
from typing import Optional
from collections import Counter

# Zona de LETRAS (pos 0-2): corregimos dígitos que ocupan posición de letra
CONFUSIONES_EN_LETRAS  = {'0':'O','1':'I','5':'S','6':'G','8':'B','2':'Z','4':'A','3':'E','7':'T','9':'P'}
# Zona de DÍGITOS (pos 3-5): corregimos letras que ocupan posición de dígito
CONFUSIONES_EN_DIGITOS = {
    'O':'0','Q':'0','I':'1','L':'1','S':'5','G':'6','B':'8','Z':'2','A':'4','D':'0',
    'T':'7','U':'0','J':'1','P':'0','C':'0','E':'8','H':'4','F':'7','N':'4','M':'0',
    'V':'1','X':'8','K':'4','R':'2','W':'0',
}
_FANTASMAS  = {'I','L','1','J','|'}
_RE_STD     = re.compile(r'^[A-Z]{3}[0-9]{3}$')
_RE_MOTO    = re.compile(r'^[A-Z]{3}[0-9]{2}[A-Z]$')
_RE_FLEX    = re.compile(r'^[A-Z0-9]{5,7}$')

def _es_valido(t: str, estricto: bool = True) -> bool:
    return bool((_RE_STD.match(t) or _RE_MOTO.match(t)) if estricto else _RE_FLEX.match(t))

def _limpiar(t: str) -> str:
    if not t: return ''
    t = unicodedata.normalize('NFKD', t).encode('ascii','ignore').decode('ascii').upper().strip()
    return re.sub(r'[^A-Z0-9]', '', re.sub(r'[-.\s·•*_]', '', t))

def _corregir_posicionalmente(t: str) -> str:
    if len(t) != 6: return t
    chars = list(t)
    for i, c in enumerate(chars):
        if i < 3 and c in CONFUSIONES_EN_LETRAS:   chars[i] = CONFUSIONES_EN_LETRAS[c]
        elif i >= 3 and c in CONFUSIONES_EN_DIGITOS: chars[i] = CONFUSIONES_EN_DIGITOS[c]
    return ''.join(chars)

def _intentar_recuperar(t: str) -> str:
    n = len(t)
    if n == 7:
        if t[0] in _FANTASMAS and t[1:4].isalpha() and t[4:].isdigit(): return t[1:]
        if t[-1] in _FANTASMAS and t[:3].isalpha() and t[3:6].isdigit(): return t[:6]
        for idx in range(3,7):
            if idx < len(t) and t[idx].isalpha():
                c = t[:idx]+t[idx+1:]
                if len(c)==6 and _es_valido(_corregir_posicionalmente(c)): return c
        sin = re.sub(r'[-.]','',t)
        if len(sin)==6: return sin
        for idx in range(7):
            c = t[:idx]+t[idx+1:]
            if len(c)==6 and _es_valido(_corregir_posicionalmente(c)): return c
    if n == 8:
        for idx in range(3,8):
            if idx < len(t) and t[idx].isalpha():
                c = t[:idx]+t[idx+1:]
                if len(c)==7: c = _intentar_recuperar(c)
                if len(c)==6 and _es_valido(_corregir_posicionalmente(c)): return c
        for i in range(8):
            for j in range(i+1,8):
                c = t[:i]+t[i+1:j]+t[j+1:]
                if len(c)==6 and _es_valido(_corregir_posicionalmente(c)): return c
    if n == 5:
        for pos in range(6):
            for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                c = t[:pos]+ch+t[pos:]
                if len(c)==6 and _es_valido(_corregir_posicionalmente(c)): return c
    return t

def _correccion_flexible(t: str) -> Optional[str]:
    if len(t) != 6: return None
    chars, cambios = list(t), False
    for i, c in enumerate(chars):
        if i < 3 and c.isdigit() and c in CONFUSIONES_EN_LETRAS:
            chars[i] = CONFUSIONES_EN_LETRAS[c]; cambios = True
        elif i >= 3 and c.isalpha() and c in CONFUSIONES_EN_DIGITOS:
            chars[i] = CONFUSIONES_EN_DIGITOS[c]; cambios = True
    r = ''.join(chars)
    return r if cambios and r != t else None

def corregir_placa(texto_raw: str, tipo_placa: str = 'blanco', verbose: bool = False) -> Optional[str]:
    if not texto_raw: return None
    t = _limpiar(texto_raw)
    if len(t) < 4: return None
    if len(t) != 6: t = _intentar_recuperar(t)
    if len(t) == 6:
        t = _corregir_posicionalmente(t)
        if _es_valido(t): return t
        flex = _correccion_flexible(t)
        if flex and _es_valido(flex): return flex
    return None

def corregir_candidatos(candidatos: list, tipo_placa: str = 'blanco', verbose: bool = False) -> list:
    return [p for p, _ in Counter(
        r for raw in candidatos for r in [corregir_placa(raw, tipo_placa, verbose)] if r
    ).most_common()]

def validar_formato_placa(texto: str) -> dict:
    r = {'valido':False,'tipo':'invalido','longitud_ok':len(texto)==6,'letras_ok':False,'digitos_ok':False,'mensaje':''}
    if len(texto) != 6:
        r['mensaje'] = f'Longitud {len(texto)} (esperado 6)'; return r
    r['letras_ok']  = texto[:3].isalpha()
    r['digitos_ok'] = texto[3:].isdigit()
    if _RE_STD.match(texto):   r.update({'valido':True,'tipo':'estandar'})
    elif _RE_MOTO.match(texto): r.update({'valido':True,'tipo':'moto'})
    elif _RE_FLEX.match(texto): r.update({'valido':True,'tipo':'flexible','mensaje':'Posiblemente diplomática'})
    else:
        partes = []
        if not r['letras_ok']:  partes.append('letras incorrectas pos 0-2')
        if not r['digitos_ok']: partes.append('dígitos incorrectos pos 3-5')
        r['mensaje'] = ', '.join(partes)
    return r

if __name__ == '__main__':
    casos = [('JGLV753','GLV753'),('1BC456','IBC456'),('ABC12I3','ABC123'),
             ('FHJ4',None),('ABCO56','ABC056'),('ABC123','ABC123')]
    print(f"{'Entrada':<14} {'Esperado':<12} {'Obtenido':<12} OK")
    for raw, esp in casos:
        obt  = corregir_placa(raw)
        marca = '✓' if obt == esp else '✗'
        print(f"{raw!r:<14} {str(esp):<12} {str(obt):<12} {marca}")