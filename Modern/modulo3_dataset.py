# ==============================================================================
# MÓDULO 3 — GENERACIÓN DE DATASET PARA EL TRANSFORMER
# Autor: Salomón Melenje
# Niveles: A(75%) limpia | B(20%) nueva | C(5%) con confusión OCR
# ==============================================================================

import re, string, random
import pandas as pd
import numpy as np

from modulo0_config import (
    char2idx, idx2char, MAX_LEN, VOCAB_SIZE, CONFUSIONES_OCR,
    DIAS_UNICOS, label2idx, idx2label, NUM_CLASES,
    asignar_restriccion, validar_formato_colombiano, tokenizar_placa
)

LETRAS  = list(string.ascii_uppercase)
DIGITOS = list(string.digits)


def generar_placa_con_variabilidad() -> tuple | None:
    r = random.random()
    if r < 0.75:
        placa = ''.join(random.choices(LETRAS,k=3)) + ''.join(random.choices(DIGITOS,k=3))
        ultimo_real = placa[5]
    elif r < 0.95:
        placa = ''.join(random.choices(LETRAS,k=3)) + ''.join(random.choices(DIGITOS,k=2)) + random.choice(LETRAS)
        ultimo_real = placa[4]
    else:
        nums = list(''.join(random.choices(DIGITOS,k=3)))
        pos  = random.randint(0,2)
        orig = nums[pos]
        if orig in CONFUSIONES_OCR: nums[pos] = CONFUSIONES_OCR[orig]
        placa = ''.join(random.choices(LETRAS,k=3)) + ''.join(nums)
        ultimo_real = orig

    label = label2idx.get(asignar_restriccion(ultimo_real))
    return (tokenizar_placa(placa).tolist(), label) if label is not None else None


def generar_dataset_sintetico(n: int = 50_000) -> list[tuple]:
    datos, intentos = [], 0
    while len(datos) < n and intentos < int(n*1.6):
        r = generar_placa_con_variabilidad(); intentos += 1
        if r: datos.append(r)
    print(f"[OK] Dataset sintético: {len(datos)} muestras (intentos: {intentos})")
    return datos


def convertir_placas_reales(df: pd.DataFrame) -> list[tuple]:
    if df is None or df.empty:
        print("[INFO] Sin placas reales."); return []
    datos = []
    for _, fila in df.iterrows():
        label = label2idx.get(fila.get('restriccion',''))
        if label is None: continue
        if 'tokens' in fila and isinstance(fila['tokens'], (list, np.ndarray)):
            tokens = list(fila['tokens'])
        else:
            p = str(fila.get('placa_detectada',''))
            if p in ('N/A','nan','') or len(p)<4: continue
            tokens = tokenizar_placa(p).tolist()
        datos.append((tokens, label))
    print(f"[OK] Placas reales convertidas: {len(datos)}")
    return datos


def construir_df_transformer(historico_df: pd.DataFrame) -> pd.DataFrame:
    if historico_df.empty:
        print("[AVISO] Historial vacío."); return pd.DataFrame()
    dataset = []
    for _, fila in historico_df.iterrows():
        p = str(fila['placa_detectada'])
        if p in ('N/A','nan',''): continue
        dataset.append({
            "placa": p, "tokens": tokenizar_placa(p).tolist(),
            "restriccion": asignar_restriccion(str(fila['ultimo_digito'])),
            "formato_valido": validar_formato_colombiano(p),
        })
    df = pd.DataFrame(dataset)
    if not df.empty:
        v = df["formato_valido"].sum()
        print(f"Total: {len(df)} | Válidos: {v} ({v/len(df)*100:.1f}%)")
    return df


def _distribucion(datos: list[tuple], titulo: str = "Distribución") -> None:
    conteo = {}
    for _, l in datos: conteo[idx2label[l]] = conteo.get(idx2label[l],0)+1
    total = len(datos)
    print(f"\n{titulo} ({total} muestras):")
    for d in sorted(conteo):
        print(f"  {d:<12}: {conteo[d]:>6} ({conteo[d]/total*100:.1f}%)  {'█'*(conteo[d]//(max(1,total//200)))}")


def preparar_datos_transformer(historico_df=None, df_reales=None,
                                n_sintetico: int = 48_000, semilla: int = 42) -> list[tuple]:
    random.seed(semilla); np.random.seed(semilla)
    print("="*60)
    print("MÓDULO 3 — PREPARANDO DATASET PARA EL TRANSFORMER")
    print("="*60)

    sinteticos = generar_dataset_sintetico(n_sintetico)
    reales = []
    if df_reales is not None and not df_reales.empty:
        reales = convertir_placas_reales(df_reales)
    elif historico_df is not None and not historico_df.empty:
        df_tmp = construir_df_transformer(historico_df)
        if not df_tmp.empty: reales = convertir_placas_reales(df_tmp)

    datos = sinteticos + reales
    random.shuffle(datos)
    _distribucion(datos, "Dataset combinado final")

    print(f"\nResumen: Sintéticos={len(sinteticos)} | Reales={len(reales)} | TOTAL={len(datos)}")
    print(f"\n  Vocabulario: {VOCAB_SIZE} tokens | MAX_LEN: {MAX_LEN}")
    ejemplo = 'HSY095'
    print(f"  Ejemplo: '{ejemplo}' → {tokenizar_placa(ejemplo).tolist()}")
    print("[OK] datos_raw listo para Módulo 4.")
    return datos


if __name__ == "__main__":
    datos = preparar_datos_transformer(n_sintetico=1_000)
    print(f"\nPrimeras 3 muestras:")
    for t, l in datos[:3]:
        print(f"  tokens={t}  label={l} ({idx2label[l]})")