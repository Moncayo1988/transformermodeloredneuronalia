# ==============================================================================
# MÓDULO 3 — GENERACIÓN DE DATASET PARA EL TRANSFORMER
# ==============================================================================
# Responsabilidad:
#   - Generar placas sintéticas con tres niveles de dificultad
#   - Integrar placas reales procesadas por Módulos 1 y 2
#   - Tokenizar placas y asignar etiquetas de Pico y Placa
#   - Exportar dataset final listo para entrenar el Transformer (Módulo 4)
#
# Niveles de variabilidad del dataset:
#   Nivel A (75%): placa limpia formato antiguo ABC123
#   Nivel B (20%): placa nueva formato ABC12D
#   Nivel C ( 5%): placa con confusión OCR en zona de dígitos
#
# Entradas : (opcional) DataFrame de placas reales del pipeline OCR
# Salidas  : datos_raw — lista de (tokens:list[int], label:int)
#            df_transformer — DataFrame con columnas enriquecidas
# ==============================================================================

import re
import string
import random
import pandas as pd
import numpy as np

from modulo0_config import (
    char2idx, idx2char,
    MAX_LEN, VOCAB_SIZE,
    CONFUSIONES_OCR,
    DIAS_UNICOS, label2idx, idx2label, NUM_CLASES,
    asignar_restriccion, validar_formato_colombiano, tokenizar_placa
)


# ------------------------------------------------------------------------------
# 1. CONSTANTES LOCALES
# ------------------------------------------------------------------------------
LETRAS  = list(string.ascii_uppercase)
DIGITOS = list(string.digits)


# ------------------------------------------------------------------------------
# 2. GENERADOR DE PLACAS SINTÉTICAS
# ------------------------------------------------------------------------------

def generar_placa_con_variabilidad() -> tuple[list, int] | None:
    """
    Genera una placa colombiana con tres niveles de dificultad:

    Nivel A (75%): placa limpia formato antiguo ABC123
      → el modelo aprende la regla base: pos[5] determina el día.

    Nivel B (20%): placa nueva formato ABC12D
      → el último dígito real está en posición 4, no en la posición final.
      → el modelo debe aprender a ignorar la letra final.

    Nivel C (5%): placa antigua con confusión OCR en un dígito
      → simula errores reales del OCR (O↔0, I↔1, S↔5, B↔8…)
      → el label siempre usa el dígito ORIGINAL (la verdad)
      → el modelo ve el carácter confundido y debe inferir el día correcto.

    Retorna: (tokens_list, label_int) o None si el caso es inválido.
    """
    rand = random.random()

    if rand < 0.75:
        # Formato antiguo limpio
        letras  = ''.join(random.choices(LETRAS,  k=3))
        numeros = ''.join(random.choices(DIGITOS, k=3))
        placa   = letras + numeros
        ultimo_real = placa[5]

    elif rand < 0.95:
        # Formato nuevo ABC12D
        letras      = ''.join(random.choices(LETRAS,  k=3))
        numeros     = ''.join(random.choices(DIGITOS, k=2))
        letra_final = random.choice(LETRAS)
        placa       = letras + numeros + letra_final
        ultimo_real = placa[4]   # el dígito real está en posición 4

    else:
        # Formato antiguo con confusión OCR en zona de dígitos
        letras   = ''.join(random.choices(LETRAS,  k=3))
        numeros  = list(''.join(random.choices(DIGITOS, k=3)))
        pos      = random.randint(0, 2)
        char_orig = numeros[pos]
        if char_orig in CONFUSIONES_OCR:
            numeros[pos] = CONFUSIONES_OCR[char_orig]
        placa       = letras + ''.join(numeros)
        ultimo_real = char_orig   # verdad: el dígito antes de confundirlo

    restriccion = asignar_restriccion(ultimo_real)
    label       = label2idx.get(restriccion)

    if label is None:
        return None   # caso edge: carácter sin mapeo válido

    tokens_ids = tokenizar_placa(placa)
    return tokens_ids.tolist(), label


def generar_dataset_sintetico(n: int = 50_000) -> list[tuple]:
    """
    Genera n muestras sintéticas con variabilidad real.
    Descarta casos inválidos y reintenta hasta alcanzar n muestras.

    Retorna: lista de (tokens_list, label_int)
    """
    datos    = []
    intentos = 0
    max_iter = int(n * 1.6)

    while len(datos) < n and intentos < max_iter:
        resultado = generar_placa_con_variabilidad()
        intentos += 1
        if resultado is not None:
            datos.append(resultado)

    print(f"[OK] Dataset sintético: {len(datos)} muestras "
          f"(intentos: {intentos})")
    return datos


# ------------------------------------------------------------------------------
# 3. CONVERSIÓN DE PLACAS REALES (OCR pipeline) AL FORMATO TRANSFORMER
# ------------------------------------------------------------------------------

def convertir_placas_reales(df_reales: pd.DataFrame) -> list[tuple]:
    """
    Convierte el DataFrame de placas reales (producido por el pipeline OCR)
    al formato (tokens_list, label_int) que usa el Transformer.

    Filtra filas con placa inválida o restricción no mapeada.

    Parámetro:
      df_reales — DataFrame con columnas: 'placa_detectada', 'restriccion', 'tokens'
                  (puede provenir del pipeline Kaggle o del pipeline principal)

    Retorna: lista de (tokens_list, label_int)
    """
    if df_reales is None or df_reales.empty:
        print("[INFO] Sin placas reales. Usando solo datos sintéticos.")
        return []

    datos = []
    for _, fila in df_reales.iterrows():
        restriccion = fila.get('restriccion', '')
        label       = label2idx.get(restriccion)
        if label is None:
            continue

        # Aceptar tokens pre-calculados o calcularlos desde la placa
        if 'tokens' in fila and isinstance(fila['tokens'], (list, np.ndarray)):
            tokens_ids = list(fila['tokens'])
        else:
            placa = str(fila.get('placa_detectada', ''))
            if placa in ('N/A', 'nan', '') or len(placa) < 4:
                continue
            tokens_ids = tokenizar_placa(placa).tolist()

        datos.append((tokens_ids, label))

    print(f"[OK] Placas reales convertidas: {len(datos)}")
    return datos


# ------------------------------------------------------------------------------
# 4. COMBINACIÓN Y ESTADÍSTICAS
# ------------------------------------------------------------------------------

def combinar_y_mezclar(
    datos_sinteticos: list[tuple],
    datos_reales: list[tuple]
) -> list[tuple]:
    """
    Combina datos sintéticos y reales, los mezcla aleatoriamente.
    Retorna la lista combinada.
    """
    combinados = datos_sinteticos + datos_reales
    random.shuffle(combinados)
    return combinados


def mostrar_distribucion(datos: list[tuple], titulo: str = "Distribución") -> None:
    """Imprime la distribución de clases del dataset."""
    conteo = {}
    for _, lbl in datos:
        dia = idx2label[lbl]
        conteo[dia] = conteo.get(dia, 0) + 1

    total = len(datos)
    print(f"\n{titulo} ({total} muestras):")
    for dia in sorted(conteo):
        barra = "█" * (conteo[dia] // max(1, total // 200))
        print(f"  {dia:<12}: {conteo[dia]:>6} ({conteo[dia]/total*100:.1f}%)  {barra}")


# ------------------------------------------------------------------------------
# 5. CONSTRUCCIÓN DEL DATAFRAME ENRIQUECIDO (para visualización/análisis)
# ------------------------------------------------------------------------------

def construir_df_transformer(
    historico_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye el DataFrame enriquecido del Módulo 3 a partir del historial
    de placas detectadas (producido por el pipeline principal).

    Columnas del resultado:
      placa | tokens | restriccion | formato_valido

    Parámetro:
      historico_df — DataFrame con columna 'placa_detectada' y 'ultimo_digito'
    """
    dataset = []

    if historico_df.empty:
        print("[AVISO] Historial vacío. No se puede construir el DataFrame.")
        return pd.DataFrame()

    for _, fila in historico_df.iterrows():
        placa = str(fila['placa_detectada'])
        if placa in ('N/A', 'nan', ''):
            continue

        ultimo      = str(fila['ultimo_digito'])
        restriccion = asignar_restriccion(ultimo)
        tokens_seq  = tokenizar_placa(placa)

        dataset.append({
            "placa"         : placa,
            "tokens"        : tokens_seq.tolist(),
            "restriccion"   : restriccion,
            "formato_valido": validar_formato_colombiano(placa)
        })

    df = pd.DataFrame(dataset)

    if not df.empty:
        total   = len(df)
        validos = df["formato_valido"].sum()
        print(f"\nTotal de registros : {total}")
        print(f"Formato válido     : {validos} ({validos/total*100:.1f}%)")
        print(f"Formato inválido   : {total - validos} (posibles errores OCR residuales)")

    return df


# ------------------------------------------------------------------------------
# 6. FUNCIÓN PRINCIPAL DEL MÓDULO
# ------------------------------------------------------------------------------

def preparar_datos_transformer(
    historico_df: pd.DataFrame = None,
    df_reales: pd.DataFrame = None,
    n_sintetico: int = 48_000,
    semilla: int = 42
) -> list[tuple]:
    """
    Función de entrada única del Módulo 3.
    Orquesta la generación sintética + conversión real + combinación.

    Parámetros:
      historico_df : DataFrame del pipeline OCR principal (puede ser None)
      df_reales    : DataFrame de placas Kaggle (puede ser None)
      n_sintetico  : número de muestras sintéticas a generar
      semilla      : semilla aleatoria para reproducibilidad

    Retorna: datos_raw — lista de (tokens_list, label_int) lista para Módulo 4
    """
    random.seed(semilla)
    np.random.seed(semilla)

    print("=" * 60)
    print("MÓDULO 3 — PREPARANDO DATASET PARA EL TRANSFORMER")
    print("=" * 60)

    # Datos sintéticos
    datos_sinteticos = generar_dataset_sintetico(n_sintetico)

    # Datos reales (si están disponibles)
    datos_reales = []
    if df_reales is not None and not df_reales.empty:
        datos_reales = convertir_placas_reales(df_reales)
    elif historico_df is not None and not historico_df.empty:
        # Intentar desde historial OCR si no hay Kaggle
        df_temp = construir_df_transformer(historico_df)
        if not df_temp.empty:
            datos_reales = convertir_placas_reales(df_temp)

    datos_raw = combinar_y_mezclar(datos_sinteticos, datos_reales)
    mostrar_distribucion(datos_raw, "Dataset combinado final")

    print(f"\nResumen:")
    print(f"  Sintéticos : {len(datos_sinteticos)}")
    print(f"  Reales     : {len(datos_reales)}")
    print(f"  TOTAL      : {len(datos_raw)}")
    print(f"\n[OK] datos_raw listo para Módulo 4.")

    # ── Resumen teórico visible en la sustentación ────────────────────────────
    print("\n" + "─"*60)
    print("FUNDAMENTO TEÓRICO — MÓDULO 3")
    print("─"*60)
    print(f"  Vocabulario    : {VOCAB_SIZE} tokens (<PAD> + A-Z + 0-9)")
    print(f"  MAX_LEN        : {MAX_LEN} posiciones (longitud máx. de placa)")
    print( "  Padding token  : 0 → <PAD> (relleno posterior)")
    print( "\n  Ejemplo de tokenización:")
    ejemplo = 'HSY095'
    toks    = tokenizar_placa(ejemplo).tolist()
    print(f"    '{ejemplo}' → {toks}")
    print( "\n  Positional Encoding:")
    print( "    Inyecta la posición (0-6) en cada embedding.")
    print( "    Crucial para que el Transformer entienda que")
    print( "    pos 0-2 = letras  |  pos 3-5 = dígitos")
    print(f"    'HSY095' ≠ '095HSY'  (mismo contenido, día distinto)")
    print("─"*60)

    return datos_raw


# ------------------------------------------------------------------------------
# 7. FUNDAMENTO TEÓRICO (impresión informativa)
# ------------------------------------------------------------------------------

def imprimir_fundamento_teorico() -> None:
    print("=" * 60)
    print("MÓDULO 3 — EMBEDDINGS Y POSITIONAL ENCODING")
    print("=" * 60)
    print(f"Tamaño del vocabulario : {VOCAB_SIZE} tokens")
    print(f"Dimensión del embedding: {16}")
    print("""
Fundamento teórico:
  Cada carácter se convierte en un vector denso de EMBED_DIM valores.
  El Positional Encoding inyecta la posición (0-6) para que la red
  entienda que pos 0-2 son letras y pos 3-5 son dígitos.

  'HSY095' ≠ '095HSY'  (mismo contenido, restricción distinta)
""")


# ------------------------------------------------------------------------------
# 8. PRUEBA RÁPIDA
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    imprimir_fundamento_teorico()
    datos = preparar_datos_transformer(n_sintetico=1_000)
    print(f"\nPrimeras 3 muestras:")
    for tokens, label in datos[:3]:
        print(f"  tokens={tokens}  label={label} ({idx2label[label]})")