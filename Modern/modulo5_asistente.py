# ==============================================================================
# MÓDULO 5 — ASISTENTE CONVERSACIONAL DE PICO Y PLACA
# ==============================================================================
# Responsabilidad:
#   - Integrar el clasificador Transformer con un asistente en lenguaje natural
#   - Responder preguntas conversacionales sobre Pico y Placa en Popayán
#   - Usar exclusivamente las fuentes internas del proyecto:
#       1. Tabla de restricción por último dígito
#       2. Predicción del Transformer (Módulo 4)
#
# Fundamento teórico (Prompt Engineering aplicado a agentes):
#   Se define un contexto de sistema con rol, reglas, límites y formato de
#   respuesta. Los datos estructurados producidos por el Transformer se
#   inyectan al prompt para controlar la respuesta del agente.
#
# Autor: Salomón Melenje
#
# Interfaces disponibles:
#   - consultar_asistente_pico_placa(pregunta, placa_contexto=None)
#   - asistente_conversacional_interactivo()   ← widgets Jupyter/Colab
#   - asistente_colab_input()                  ← consola (Colab/local)
#
# Uso directo:
#   python modulo5_asistente.py
# ==============================================================================

import re
import unicodedata
import html as html_lib
from datetime import datetime, timedelta

from modulo0_config import (
    REGLAS_PICO_PLACA, DIAS_UNICOS,
    label2idx, idx2label,
    asignar_restriccion
)
from modulo4_transformer import cargar_modelo, predecir_pico_placa


# ==============================================================================
# 1. PROMPT DEL AGENTE Y FUENTES
# ==============================================================================

PROMPT_SISTEMA_ASISTENTE = """
Eres un asistente de movilidad para Pico y Placa en Popayan.
Usa exclusivamente la prediccion del clasificador Transformer y la tabla de
restriccion del proyecto. No inventes horarios, excepciones, permisos ni normas
externas.

Reglas del agente:
1. Extrae placa, dia/fecha e intencion desde lenguaje natural.
2. Si hay placa, consulta el Transformer para obtener el dia de restriccion.
3. Si el usuario pregunta por un dia sin placa, responde con la tabla del proyecto.
4. Si el usuario pregunta por horarios, motos, taxis, carga, excepciones o permisos,
   indica claramente que esa fuente no esta disponible en el notebook.
5. Muestra el motivo de la decision, la regla aplicada y la confianza del modelo.
6. Si falta un dato necesario, pregunta por ese dato de forma breve.
"""

FUENTES_ASISTENTE = [
    "Transformer de clasificacion de Pico y Placa entrenado en el Modulo 4.",
    "Tabla del proyecto para Popayan: 0-1 Lunes, 2-3 Martes, 4-5 Miercoles, 6-7 Jueves, 8-9 Viernes.",
]

DIGITOS_POR_DIA = {
    "lunes"    : ["0", "1"],
    "martes"   : ["2", "3"],
    "miercoles": ["4", "5"],
    "jueves"   : ["6", "7"],
    "viernes"  : ["8", "9"],
}

DIA_POR_DIGITO = {
    "0": "Lunes",     "1": "Lunes",
    "2": "Martes",    "3": "Martes",
    "4": "Miercoles", "5": "Miercoles",
    "6": "Jueves",    "7": "Jueves",
    "8": "Viernes",   "9": "Viernes",
}

MESES_AGENTE = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

HORARIO_NO_DISPONIBLE      = "No disponible en las fuentes del notebook."
INFO_TIPO_NO_DISPONIBLE    = "El notebook clasifica por placa, no por tipo de vehiculo."


# ==============================================================================
# 2. EXTRACCIÓN DE INFORMACIÓN DESDE LENGUAJE NATURAL
# ==============================================================================

def normalizar_texto_agente(texto: str) -> str:
    """Normaliza texto para comparar días sin depender de tildes o mayúsculas."""
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def limpiar_placa_consulta(texto: str) -> str:
    """Deja una placa en formato compacto: solo letras y números en mayúscula."""
    return re.sub(r"[^A-Z0-9]", "", str(texto).upper())


def extraer_placa_desde_pregunta(pregunta: str) -> str | None:
    """Extrae placas colombianas antiguas (ABC123) y nuevas (ABC12D)."""
    texto = str(pregunta).upper()
    patrones = [
        r"\b([A-Z]{3})[\s\-]?([0-9]{3})\b",
        r"\b([A-Z]{3})[\s\-]?([0-9]{2})[\s\-]?([A-Z])\b",
    ]
    for patron in patrones:
        match = re.search(patron, texto)
        if match:
            return limpiar_placa_consulta("".join(match.groups()))
    return None


def obtener_placa_contexto(placa_contexto: str | None = None,
                            historico_df=None) -> str | None:
    """
    Usa la placa escrita por el usuario o, si no existe, intenta tomar la
    última placa válida detectada por el pipeline YOLO + OCR.
    """
    if placa_contexto:
        placa = limpiar_placa_consulta(placa_contexto)
        if placa and placa not in ("NA", "NAN"):
            return placa

    if historico_df is not None and not historico_df.empty:
        try:
            placas = historico_df.get("placa_detectada", [])
            for placa in reversed([str(p) for p in placas]):
                placa = limpiar_placa_consulta(placa)
                if placa and placa not in ("NA", "NAN") and len(placa) >= 5:
                    return placa
        except Exception:
            pass
    return None


def _dia_desde_clave(clave: str) -> str:
    """Devuelve el nombre del día usando las etiquetas reales del Transformer."""
    for dia in DIAS_UNICOS:
        if normalizar_texto_agente(dia) == clave:
            return dia
    fallback = {
        "lunes": "Lunes", "martes": "Martes", "miercoles": "Miercoles",
        "jueves": "Jueves", "viernes": "Viernes",
        "sabado": "Sabado", "domingo": "Domingo",
    }
    return fallback.get(clave, clave.capitalize())


def _clave_dia_desde_fecha(fecha: datetime) -> str:
    return ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"][fecha.weekday()]


def _formatear_fecha_consulta(fecha: datetime) -> str:
    clave = _clave_dia_desde_fecha(fecha)
    return f"{fecha.strftime('%Y-%m-%d')} ({_dia_desde_clave(clave)})"


def extraer_fecha_desde_pregunta(pregunta: str) -> dict:
    """Entiende días, hoy, mañana, dd/mm/yyyy, dd-mm-yyyy y '15 de junio'."""
    texto = normalizar_texto_agente(pregunta)
    ahora = datetime.now()

    if re.search(r"\besta semana\b", texto):
        return {"tipo": "semana", "texto": "Semana consultada", "dia": None, "clave": None}

    if re.search(r"\bhoy\b", texto):
        clave = _clave_dia_desde_fecha(ahora)
        return {"tipo": "fecha", "texto": _formatear_fecha_consulta(ahora),
                "dia": _dia_desde_clave(clave), "clave": clave}

    if re.search(r"\bmanana\b", texto):
        fecha = ahora + timedelta(days=1)
        clave = _clave_dia_desde_fecha(fecha)
        return {"tipo": "fecha", "texto": _formatear_fecha_consulta(fecha),
                "dia": _dia_desde_clave(clave), "clave": clave}

    match_num = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", texto)
    if match_num:
        dia_num = int(match_num.group(1))
        mes_num = int(match_num.group(2))
        anio = int(match_num.group(3)) if match_num.group(3) else ahora.year
        if anio < 100:
            anio += 2000
        try:
            fecha = datetime(anio, mes_num, dia_num)
            clave = _clave_dia_desde_fecha(fecha)
            return {"tipo": "fecha", "texto": _formatear_fecha_consulta(fecha),
                    "dia": _dia_desde_clave(clave), "clave": clave}
        except ValueError:
            return {"tipo": "invalida", "texto": "Fecha invalida", "dia": None, "clave": None}

    match_texto = re.search(r"\b(\d{1,2})\s+de\s+([a-z]+)(?:\s+de\s+(\d{4}))?\b", texto)
    if match_texto:
        dia_num = int(match_texto.group(1))
        mes_num = MESES_AGENTE.get(match_texto.group(2))
        anio = int(match_texto.group(3)) if match_texto.group(3) else ahora.year
        if mes_num:
            try:
                fecha = datetime(anio, mes_num, dia_num)
                clave = _clave_dia_desde_fecha(fecha)
                return {"tipo": "fecha", "texto": _formatear_fecha_consulta(fecha),
                        "dia": _dia_desde_clave(clave), "clave": clave}
            except ValueError:
                return {"tipo": "invalida", "texto": "Fecha invalida", "dia": None, "clave": None}

    for clave in ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"]:
        if re.search(rf"\b{clave}\b", texto):
            return {"tipo": "dia",
                    "texto": f"{_dia_desde_clave(clave)} (fecha exacta no especificada)",
                    "dia": _dia_desde_clave(clave), "clave": clave}

    return {"tipo": "no_especificada", "texto": "No especificada", "dia": None, "clave": None}


def extraer_dia_desde_pregunta(pregunta: str) -> str | None:
    """Compatibilidad con versiones anteriores."""
    return extraer_fecha_desde_pregunta(pregunta).get("dia")


def clasificar_intencion_consulta(pregunta: str) -> str:
    """Clasifica de forma simple la intención conversacional del usuario."""
    texto = normalizar_texto_agente(pregunta)
    if any(p in texto for p in ["horario","hora","excepcion","permiso",
                                  "moto","taxi","carga","camion"]):
        return "consulta_fuera_de_fuente"
    if "esta semana" in texto:
        return "consulta_semana"
    if any(p in texto for p in ["puedo","transitar","circular","pasar"]):
        return "validar_transito"
    if any(p in texto for p in ["que dia","cual dia","que placas","restriccion","pico"]):
        return "consultar_restriccion"
    return "consulta_general"


def extraer_ultimo_digito_relevante(placa: str) -> str | None:
    """Obtiene el último dígito numérico visible de una placa."""
    numeros = re.findall(r"\d", limpiar_placa_consulta(placa))
    return numeros[-1] if numeros else None


def interpretar_confianza(confianza: float) -> str:
    if confianza >= 90: return "alta"
    if confianza >= 70: return "media"
    return "baja"


def regla_por_dia(dia_clave: str) -> str:
    digitos = DIGITOS_POR_DIA.get(dia_clave)
    if not digitos:
        return "La tabla del proyecto no define restriccion para ese dia."
    return (f"Placas terminadas en {digitos[0]} y {digitos[1]} "
            f"tienen restriccion el {_dia_desde_clave(dia_clave)}.")


# ==============================================================================
# 3. CONSTRUCCIÓN DE PROMPT Y GENERACIÓN DE RESPUESTA
# ==============================================================================

def construir_prompt_agente(pregunta: str, placa: str | None,
                             fecha_ctx: dict, prediccion: dict | None,
                             intencion: str) -> str:
    """Construye el prompt de agente con contexto estructurado."""
    return f"""{PROMPT_SISTEMA_ASISTENTE}

Fuentes disponibles:
- {FUENTES_ASISTENTE[0]}
- {FUENTES_ASISTENTE[1]}

Entrada del usuario:
{pregunta}

Contexto estructurado:
- intencion: {intencion}
- placa: {placa or 'No especificada'}
- fecha_consultada: {fecha_ctx.get('texto')}
- dia_consultado: {fecha_ctx.get('dia') or 'No especificado'}
- dia_restriccion_transformer: {prediccion.get('restriccion') if prediccion else 'No consultado'}
- confianza_transformer: {prediccion.get('confianza_pct') if prediccion else 'No consultado'}
- probabilidades: {prediccion.get('probabilidades') if prediccion else 'No consultado'}

Tarea:
Redacta una respuesta clara, con resultado, motivo, regla aplicada y limites de la fuente.
"""


def generar_respuesta_semana() -> tuple[str, None]:
    filas = []
    for clave in ["lunes","martes","miercoles","jueves","viernes"]:
        filas.append(
            f"- {_dia_desde_clave(clave)}: placas terminadas en "
            f"{DIGITOS_POR_DIA[clave][0]} y {DIGITOS_POR_DIA[clave][1]}"
        )
    return (
        "Resumen semanal disponible para Popayan:\n"
        + "\n".join(filas)
        + "\n\nHorario: no disponible en las fuentes del notebook.\n"
        + "Excepciones/permisos: no disponibles en las fuentes del notebook."
    ), None


def generar_respuesta_sin_placa(fecha_ctx: dict, intencion: str) -> tuple[str, bool | None]:
    dia_clave = fecha_ctx.get("clave")

    if intencion == "consulta_semana":
        return generar_respuesta_semana()

    if dia_clave in DIGITOS_POR_DIA:
        digitos = DIGITOS_POR_DIA[dia_clave]
        respuesta = (
            f"Para {fecha_ctx['texto']}, la tabla del proyecto indica Pico y Placa para "
            f"placas terminadas en {digitos[0]} y {digitos[1]}.\n\n"
            f"Regla aplicada: {regla_por_dia(dia_clave)}\n"
            "Horario: no disponible en las fuentes del notebook.\n"
            "Excepciones/permisos: no disponibles en las fuentes del notebook.\n"
            "Si quieres validar un vehiculo especifico, escribe la placa."
        )
        return respuesta, None

    if dia_clave in ("sabado", "domingo"):
        return (
            f"Para {fecha_ctx['texto']}, la tabla del proyecto no define restriccion de Pico y Placa.\n"
            "Observacion: no puedo afirmar normas externas; solo reporto lo que existe en el notebook."
        ), True

    return (
        "Necesito conocer al menos el dia o la placa para ayudarte.\n"
        "Ejemplos: 'Puedo transitar el miercoles con placa ABC123' "
        "o 'Que placas tienen pico y placa el viernes?'."
    ), None


def generar_respuesta_agente(pregunta: str, placa: str | None, fecha_ctx: dict,
                              prediccion: dict | None,
                              intencion: str) -> tuple[str, bool | None]:
    """Genera una respuesta natural a partir de la salida del Transformer."""
    texto_norm = normalizar_texto_agente(pregunta)

    if intencion == "consulta_fuera_de_fuente":
        temas = []
        if any(t in texto_norm for t in ["horario","hora"]):
            temas.append("horarios")
        if any(t in texto_norm for t in ["moto","taxi","carga","camion"]):
            temas.append("tipo de vehiculo")
        if any(t in texto_norm for t in ["excepcion","permiso"]):
            temas.append("excepciones o permisos")
        tema_txt = ", ".join(temas) if temas else "esa informacion"
        return (
            f"No tengo informacion suficiente sobre {tema_txt} en las fuentes del notebook.\n\n"
            "Lo que si puedo consultar es Pico y Placa por placa y dia para Popayan "
            "usando el Transformer y la tabla del proyecto."
        ), None

    if not placa:
        return generar_respuesta_sin_placa(fecha_ctx, intencion)

    restriccion     = prediccion["restriccion"]
    confianza       = prediccion["confianza_pct"]
    confianza_txt   = interpretar_confianza(confianza)
    ultimo_digito   = extraer_ultimo_digito_relevante(placa)
    dia_consulta    = fecha_ctx.get("dia")
    dia_clave       = fecha_ctx.get("clave")
    restriccion_norm = normalizar_texto_agente(restriccion)
    puede_transitar  = None

    detalle_base = (
        f"Placa consultada: {placa}\n"
        f"Ultimo digito relevante: {ultimo_digito or 'No identificado'}\n"
        f"Restriccion estimada por el Transformer: {restriccion}\n"
        f"Confianza del modelo: {confianza:.2f}% ({confianza_txt})\n"
        f"Fuente: {FUENTES_ASISTENTE[1]}"
    )

    if dia_consulta is None:
        respuesta = (
            f"Para la placa {placa}, el Transformer predice restriccion el dia {restriccion}.\n\n"
            f"{detalle_base}\n\n"
            "Para decirte si puedes transitar, dime el dia o la fecha de la consulta."
        )

    elif dia_clave in ("sabado", "domingo"):
        puede_transitar = True
        respuesta = (
            f"Resultado: segun la tabla del proyecto, si puedes transitar el {dia_consulta} "
            f"con la placa {placa}.\n\n"
            f"Motivo: la tabla disponible solo define restricciones de lunes a viernes.\n"
            f"{detalle_base}\n"
            "Observacion: no se consultan normas externas al notebook."
        )

    elif dia_clave == restriccion_norm:
        puede_transitar = False
        respuesta = (
            f"Resultado: no deberias transitar el {dia_consulta} con la placa {placa}.\n\n"
            f"Motivo: el dia consultado coincide con la restriccion estimada por el Transformer.\n"
            f"Regla aplicada: {regla_por_dia(dia_clave)}\n"
            f"{detalle_base}"
        )
    else:
        puede_transitar = True
        respuesta = (
            f"Resultado: si puedes transitar el {dia_consulta} con la placa {placa}.\n\n"
            f"Motivo: el dia consultado no coincide con la restriccion estimada para la placa.\n"
            f"Regla del dia consultado: {regla_por_dia(dia_clave)}\n"
            f"{detalle_base}"
        )

    if confianza < 70:
        respuesta += "\n\nAdvertencia: la confianza es baja; conviene verificar la lectura OCR de la placa."

    return respuesta, puede_transitar


# ==============================================================================
# 4. VISUALIZACIÓN (HTML — Colab / Jupyter)
# ==============================================================================

def mostrar_respuesta_visual(resultado: dict) -> None:
    """Muestra la respuesta en una tarjeta HTML dentro de Colab/Jupyter."""
    estado = "Consulta de restriccion"
    color  = "#2563eb"
    fondo  = "#eff6ff"

    if resultado.get("puede_transitar") is True:
        estado = "Puede transitar"
        color  = "#15803d"
        fondo  = "#f0fdf4"
    elif resultado.get("puede_transitar") is False:
        estado = "No puede transitar"
        color  = "#b91c1c"
        fondo  = "#fef2f2"

    pred  = resultado.get("prediccion", {}) or {}
    probs = pred.get("probabilidades", {}) or {}
    filas_prob = ""
    for dia, prob in sorted(probs.items(), key=lambda item: item[1], reverse=True):
        ancho = max(2, min(100, float(prob)))
        filas_prob += f"""
        <div style="margin:6px 0;">
          <div style="display:flex; justify-content:space-between; gap:12px;">
            <span>{html_lib.escape(str(dia))}</span>
            <strong>{float(prob):.2f}%</strong>
          </div>
          <div style="height:8px; background:#e5e7eb; border-radius:999px;">
            <div style="height:8px; width:{ancho:.2f}%; background:{color};
                        border-radius:999px;"></div>
          </div>
        </div>
        """

    respuesta_html = html_lib.escape(resultado["respuesta"]).replace("\n", "<br>")
    html = f"""
    <div style="
        font-family:Arial, sans-serif;
        border:1px solid #d1d5db;
        border-left:8px solid {color};
        border-radius:8px;
        padding:18px 20px;
        background:{fondo};
        color:#111827;
        max-width:820px;">
      <div style="font-size:13px; font-weight:700; color:{color};
                  text-transform:uppercase; letter-spacing:.04em;">
        {estado}
      </div>
      <h3 style="margin:6px 0 10px 0;">Asistente de Pico y Placa</h3>
      <div style="font-size:15px; line-height:1.5; margin:0 0 14px 0;">
        {respuesta_html}
      </div>
      <div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr));
                  gap:10px; margin:12px 0;">
        <div><strong>Placa</strong><br>{html_lib.escape(str(resultado.get("placa") or "N/A"))}</div>
        <div><strong>Fecha/Dia</strong><br>{html_lib.escape(str(resultado.get("fecha_texto") or "No especificado"))}</div>
        <div><strong>Restriccion</strong><br>{html_lib.escape(str(pred.get("restriccion") or "N/A"))}</div>
        <div><strong>Confianza</strong><br>{html_lib.escape(str(pred.get("confianza_pct", "N/A")))}</div>
      </div>
      {('<div style="margin-top:12px;"><strong>Distribucion del Transformer</strong>' + filas_prob + '</div>') if filas_prob else ''}
    </div>
    """
    try:
        from IPython.display import HTML, display
        display(HTML(html))
    except Exception:
        print(resultado["respuesta"])


# ==============================================================================
# 5. PUNTO PRINCIPAL DE CONSULTA
# ==============================================================================

def consultar_asistente_pico_placa(
    pregunta: str,
    placa_contexto: str | None = None,
    historico_df=None,
    model=None,
    mostrar_visual: bool = True
) -> dict:
    """
    Punto principal de consulta. Recibe lenguaje natural, llama al Transformer
    y devuelve una respuesta conversacional más una visualización opcional.

    Parámetros:
      pregunta       : pregunta en lenguaje natural
      placa_contexto : placa opcional si no se menciona en la pregunta
      historico_df   : DataFrame de placas detectadas por el pipeline OCR
      model          : instancia del Transformer (None = carga automática)
      mostrar_visual : muestra tarjeta HTML en Jupyter/Colab

    Retorna: dict con pregunta, placa, fecha, predicción, respuesta y más.
    """
    fecha_ctx  = extraer_fecha_desde_pregunta(pregunta)
    placa      = (extraer_placa_desde_pregunta(pregunta)
                  or obtener_placa_contexto(placa_contexto, historico_df))
    intencion  = clasificar_intencion_consulta(pregunta)

    prediccion = None
    if placa:
        prediccion = predecir_pico_placa(placa, model=model, verbose=False)

    prompt_agente = construir_prompt_agente(
        pregunta, placa, fecha_ctx, prediccion or {}, intencion
    )
    respuesta, puede_transitar = generar_respuesta_agente(
        pregunta, placa, fecha_ctx, prediccion, intencion
    )

    resultado = {
        "pregunta"      : pregunta,
        "placa"         : placa,
        "fecha_texto"   : fecha_ctx.get("texto"),
        "dia_consulta"  : fecha_ctx.get("dia"),
        "prediccion"    : prediccion or {},
        "puede_transitar": puede_transitar,
        "intencion"     : intencion,
        "prompt_agente" : prompt_agente,
        "respuesta"     : respuesta,
    }

    if mostrar_visual:
        mostrar_respuesta_visual(resultado)

    return resultado


# ==============================================================================
# 6. INTERFACES INTERACTIVAS
# ==============================================================================

def asistente_conversacional_interactivo(model=None, placa_contexto: str | None = None,
                                          usar_widgets: bool = True) -> dict | None:
    """
    Lanza una interfaz interactiva. En Colab/Jupyter usa widgets; si no están
    disponibles, usa un bucle de consola hasta que el usuario escriba 'salir'.
    """
    if usar_widgets:
        try:
            try:
                from google.colab import output
                output.enable_custom_widget_manager()
            except Exception:
                pass

            import ipywidgets as widgets
            from IPython.display import display, clear_output

            pregunta     = widgets.Textarea(
                value="Puedo transitar un miercoles con la placa SKY424?",
                placeholder="Escribe tu pregunta sobre Pico y Placa",
                description="Pregunta:",
                layout=widgets.Layout(width="100%", height="90px"),
            )
            placa_manual = widgets.Text(
                value=placa_contexto or "",
                placeholder="Opcional si la pregunta no incluye placa",
                description="Placa:",
                layout=widgets.Layout(width="50%"),
            )
            boton  = widgets.Button(
                description="Consultar",
                button_style="primary",
                icon="search",
            )
            salida = widgets.Output()

            def _on_click(_):
                with salida:
                    clear_output()
                    consultar_asistente_pico_placa(
                        pregunta.value,
                        placa_contexto=placa_manual.value or placa_contexto,
                        model=model,
                        mostrar_visual=True,
                    )

            boton.on_click(_on_click)
            display(widgets.VBox([pregunta, placa_manual, boton, salida]))
            return {"pregunta": pregunta, "placa": placa_manual,
                    "boton": boton, "salida": salida}

        except Exception as e:
            print(f"[AVISO] Widgets no disponibles ({e}). Usando consola.")

    # Fallback consola
    print("\nASISTENTE DE PICO Y PLACA")
    print("Escribe una pregunta o 'salir' para terminar.\n")

    while True:
        pregunta = input("Pregunta> ").strip()
        if normalizar_texto_agente(pregunta) in ("salir", "exit", "quit"):
            print("Asistente finalizado.")
            break
        resultado = consultar_asistente_pico_placa(
            pregunta,
            placa_contexto=placa_contexto,
            model=model,
            mostrar_visual=False,
        )
        print(resultado["respuesta"])

    return None


def asistente_colab_input(model=None, placa_contexto: str | None = None,
                           historico_df=None, repetir: bool = True) -> dict | None:
    """
    Interfaz simple y confiable para Google Colab o terminal local.
    Usa input(), que siempre muestra un cuadro de texto aunque los widgets
    no se rendericen.
    """
    placa_sugerida = obtener_placa_contexto(placa_contexto, historico_df)

    print("\nASISTENTE DE PICO Y PLACA - CONSULTA EN LENGUAJE NATURAL")
    print("Ejemplos:")
    print("  - Puedo transitar hoy?")
    print("  - Puedo transitar el miercoles con la placa ABC123?")
    print("  - Que placas tienen pico y placa el viernes?")
    print("  - Muestrame las restricciones de esta semana")
    if placa_sugerida:
        print(f"\nPlaca detectada/sugerida: {placa_sugerida}")
        print("Si no escribes placa en la pregunta, usare esa placa como contexto.")
    print("Escribe 'salir' para terminar.\n")

    ultimo_resultado = None
    while True:
        pregunta = input("Escribe tu pregunta aqui: ").strip()
        if normalizar_texto_agente(pregunta) in ("salir", "exit", "quit"):
            print("Asistente finalizado.")
            break
        if not pregunta:
            print("[AVISO] No escribiste ninguna pregunta.")
            if not repetir:
                break
            continue

        ultimo_resultado = consultar_asistente_pico_placa(
            pregunta,
            placa_contexto=placa_sugerida,
            historico_df=historico_df,
            model=model,
            mostrar_visual=True,
        )

        if not repetir:
            break

    return ultimo_resultado


# ==============================================================================
# 7. DESPLIEGUE GRADIO — LOCAL (VSCode) Y COLAB
# ==============================================================================
# Estas funciones son el puente entre el pipeline (Módulos 1-4) y la interfaz
# web. Funcionan idénticamente desde VSCode (local) y desde Google Colab.
#
# Funciones expuestas:
#   - respuesta_asistente_gradio(pregunta, placa_manual)  ← adaptador Gradio
#   - desplegar_asistente_gradio(share)                   ← solo asistente
#   - procesar_imagen_app_integral(ruta_imagen)            ← pipeline completo
#   - responder_app_integral(pregunta, placa, estado)      ← asistente + imagen
#   - desplegar_app_integral_gradio(share)                 ← app completa ★
# ==============================================================================

def _instalar_gradio() -> None:
    """Instala Gradio si no está disponible en el entorno actual."""
    import subprocess
    print("[INFO] Instalando Gradio...")
    subprocess.check_call([__import__('sys').executable, "-m", "pip", "install", "gradio", "-q"])
    print("[OK] Gradio instalado.")


def respuesta_asistente_gradio(pregunta: str, placa_manual: str = "") -> str:
    """
    Adaptador del asistente conversacional para la interfaz Gradio.
    Recibe texto plano y devuelve Markdown con el resultado.
    """
    pregunta     = str(pregunta     or "").strip()
    placa_manual = str(placa_manual or "").strip()

    if not pregunta:
        return "Escribe una pregunta para consultar el Pico y Placa."

    placa_contexto = placa_manual or obtener_placa_contexto()
    resultado = consultar_asistente_pico_placa(
        pregunta,
        placa_contexto=placa_contexto,
        mostrar_visual=False,
    )

    pred   = resultado.get("prediccion", {}) or {}
    estado = "Consulta realizada"
    if resultado.get("puede_transitar") is True:
        estado = "Puede transitar ✅"
    elif resultado.get("puede_transitar") is False:
        estado = "No puede transitar ❌"

    partes = [f"## {estado}", resultado["respuesta"]]

    if pred:
        partes += [
            "",
            "### Datos del modelo",
            f"- **Placa:** `{resultado.get('placa')}`",
            f"- **Día consultado:** `{resultado.get('dia_consulta') or 'No especificado'}`",
            f"- **Restricción estimada:** `{pred.get('restriccion')}`",
            f"- **Confianza:** `{pred.get('confianza_pct')}%`",
        ]

    partes += [
        "",
        "### Límites de la fuente",
        "- Solo se consulta Pico y Placa por placa y día para Popayán.",
        "- Horarios, permisos, excepciones y tipo de vehículo no están disponibles.",
    ]

    return "\n".join(partes)


def procesar_imagen_app_integral(ruta_imagen) -> tuple:
    """
    Pipeline completo desde una imagen subida en Gradio:
      Módulo 1 (YOLO) → Módulo 2 (OCR) → Módulo 4 (Transformer)

    Retorna la tupla de 6 elementos que espera la interfaz Gradio:
      (imagen_marcada, recorte, binaria, placa_texto, resumen_md, estado_dict)
    """
    from modulo1_deteccion_yolo import detectar_y_recortar_placa
    from modulo2_ocr import extraer_datos_placa

    vacio = {"placa": None, "restriccion": None, "confianza": None}

    if ruta_imagen is None:
        return None, None, None, "", "Sube una imagen para iniciar el análisis.", vacio

    try:
        recorte_rgb, img_marcada_rgb, metodo = detectar_y_recortar_placa(ruta_imagen)
        if recorte_rgb is None:
            return None, None, None, "", "No se detectó ninguna placa en la imagen.", vacio

        df_ocr, imagen_binaria = extraer_datos_placa(recorte_rgb)
        if df_ocr is None or df_ocr.empty:
            return img_marcada_rgb, recorte_rgb, None, "", \
                   "Se detectó la zona de placa, pero el OCR no pudo leerla.", vacio

        placa      = str(df_ocr["placa_detectada"].values[0])
        ultimo     = str(df_ocr["ultimo_digito"].values[0])
        tipo_placa = str(df_ocr["tipo_placa"].values[0])
        motor_ocr  = str(df_ocr["motor_ocr"].values[0])
        restriccion_regla = asignar_restriccion(ultimo)

        pred = None
        if placa not in ("N/A", "nan", ""):
            pred = predecir_pico_placa(placa, verbose=False)

        if pred:
            restriccion_modelo = pred["restriccion"]
            confianza          = pred["confianza_pct"]
            confianza_txt      = interpretar_confianza(confianza)
        else:
            restriccion_modelo = "No disponible"
            confianza          = None
            confianza_txt      = "No disponible"

        estado = {
            "placa"       : placa,
            "ultimo_digito": ultimo,
            "restriccion" : restriccion_modelo,
            "confianza"   : confianza,
        }

        resumen = "\n".join([
            "## Análisis de imagen completado",
            f"- **Método de detección:** `{metodo}`",
            f"- **Placa detectada:** `{placa}`",
            f"- **Último dígito OCR:** `{ultimo}`",
            f"- **Tipo de placa:** `{tipo_placa}`",
            f"- **Motor OCR:** `{motor_ocr}`",
            f"- **Restricción por regla base:** `{restriccion_regla}`",
            f"- **Restricción por Transformer:** `{restriccion_modelo}`",
            f"- **Confianza:** `{confianza if confianza is not None else 'N/A'}%` ({confianza_txt})",
            "",
            "Ahora puedes preguntar: `¿Puedo transitar hoy?` o `¿Puedo transitar el viernes?`",
        ])

        return img_marcada_rgb, recorte_rgb, imagen_binaria, placa, resumen, estado

    except Exception as e:
        return None, None, None, "", f"Error al procesar la imagen: {e}", vacio


def responder_app_integral(pregunta: str, placa_manual: str, estado_imagen: dict) -> str:
    """Responde combinando la placa detectada por imagen y la pregunta del usuario."""
    estado_imagen  = estado_imagen or {}
    placa_contexto = str(placa_manual or "").strip() or estado_imagen.get("placa") or ""
    return respuesta_asistente_gradio(pregunta, placa_contexto)


def desplegar_asistente_gradio(share: bool = False) -> object:
    """
    Despliega solo el asistente conversacional como app web con Gradio.
    Útil para una demo rápida sin necesidad de subir imágenes.

    Parámetros:
      share : False → solo accesible en http://localhost:7860 (recomendado para VSCode)
              True  → genera enlace público temporal (útil en Colab)
    """
    try:
        import gradio as gr
    except ImportError:
        _instalar_gradio()
        import gradio as gr

    placa_sugerida = obtener_placa_contexto() or ""

    with gr.Blocks(title="Asistente Pico y Placa — Popayán") as demo:
        gr.Markdown(
            "# Asistente Pico y Placa — Popayán\n"
            "Consulta si una placa puede transitar según la tabla del proyecto "
            "y la predicción del Transformer."
        )
        gr.Markdown(
            "**Fuente interna:** Transformer del Módulo 4 + tabla de Pico y Placa. "
            "No inventa horarios, excepciones ni permisos."
        )

        with gr.Row():
            pregunta = gr.Textbox(
                label="Pregunta",
                placeholder="Ej: ¿Puedo transitar hoy con la placa ABC123?",
                lines=3,
            )
            placa = gr.Textbox(
                label="Placa (opcional)",
                value=placa_sugerida,
                placeholder="Opcional si la pregunta ya incluye la placa",
            )

        with gr.Row():
            boton   = gr.Button("Consultar", variant="primary")
            limpiar = gr.Button("Limpiar")

        salida = gr.Markdown(label="Respuesta")

        gr.Examples(
            examples=[
                ["¿Puedo transitar hoy?",                             placa_sugerida],
                ["¿Puedo transitar el miércoles con la placa SKY424?", ""],
                ["¿Qué placas tienen pico y placa el viernes?",        ""],
                ["Muéstrame las restricciones de esta semana",          ""],
            ],
            inputs=[pregunta, placa],
            outputs=salida,
            fn=respuesta_asistente_gradio,
            cache_examples=False,
        )

        boton.click(fn=respuesta_asistente_gradio,  inputs=[pregunta, placa], outputs=salida)
        pregunta.submit(fn=respuesta_asistente_gradio, inputs=[pregunta, placa], outputs=salida)
        limpiar.click(fn=lambda: ("", placa_sugerida, ""),
                      inputs=None, outputs=[pregunta, placa, salida])

    print("[OK] Lanzando asistente Gradio...")
    demo.launch(share=share, debug=False, inbrowser=True)
    return demo


def desplegar_app_integral_gradio(share: bool = False) -> object:
    """
    Despliega la app completa con Gradio:
      · Módulo 1 — Subida y detección de placa (YOLO)
      · Módulo 2 — OCR adaptativo
      · Módulo 4 — Predicción del Transformer
      · Módulo 5 — Asistente conversacional

    Funciona igual desde VSCode (local) y desde Google Colab.

    Parámetros:
      share : False → http://localhost:7860 (VSCode / uso local) ← recomendado
              True  → enlace público temporal (Colab o demo remota)
    """
    try:
        import gradio as gr
    except ImportError:
        _instalar_gradio()
        import gradio as gr

    with gr.Blocks(title="App Pico y Placa IA — Popayán") as demo:
        estado_imagen = gr.State({"placa": None, "restriccion": None, "confianza": None})

        gr.Markdown(
            "# App Pico y Placa IA — Popayán\n"
            "Sube una imagen del vehículo, detecta la placa con el pipeline del proyecto "
            "y consulta el asistente conversacional."
        )
        gr.Markdown(
            "**Flujo:** Módulo 1 detección → Módulo 2 OCR → Módulo 4 Transformer → Módulo 5 asistente."
        )

        # ── Sección de imagen ─────────────────────────────────────────────────
        with gr.Row():
            imagen = gr.Image(label="Imagen del vehículo", type="filepath")
            with gr.Column():
                boton_procesar  = gr.Button("Analizar imagen", variant="primary")
                placa_detectada = gr.Textbox(label="Placa detectada / editable",
                                             placeholder="ABC123")
                resumen_imagen  = gr.Markdown(label="Resultado de imagen")

        with gr.Row():
            imagen_marcada = gr.Image(label="Detección de placa")
            recorte_placa  = gr.Image(label="Recorte de placa")
            binaria        = gr.Image(label="Binarización OCR")

        # ── Sección conversacional ────────────────────────────────────────────
        gr.Markdown("## Consulta conversacional")
        pregunta = gr.Textbox(
            label="Pregunta",
            placeholder="Ej: ¿Puedo transitar hoy?",
            lines=3,
        )
        with gr.Row():
            boton_preguntar = gr.Button("Preguntar", variant="primary")
            limpiar         = gr.Button("Limpiar pregunta")
        salida = gr.Markdown(label="Respuesta del asistente")

        gr.Examples(
            examples=[
                "¿Puedo transitar hoy?",
                "¿Puedo transitar el miércoles?",
                "¿Qué placas tienen pico y placa el viernes?",
                "Muéstrame las restricciones de esta semana",
            ],
            inputs=pregunta,
            cache_examples=False,
        )

        # ── Eventos ───────────────────────────────────────────────────────────
        boton_procesar.click(
            fn=procesar_imagen_app_integral,
            inputs=imagen,
            outputs=[imagen_marcada, recorte_placa, binaria,
                     placa_detectada, resumen_imagen, estado_imagen],
        )
        boton_preguntar.click(
            fn=responder_app_integral,
            inputs=[pregunta, placa_detectada, estado_imagen],
            outputs=salida,
        )
        pregunta.submit(
            fn=responder_app_integral,
            inputs=[pregunta, placa_detectada, estado_imagen],
            outputs=salida,
        )
        limpiar.click(fn=lambda: "", inputs=None, outputs=pregunta)

    print("[OK] Lanzando app integral con Gradio...")
    demo.launch(share=share, debug=False, inbrowser=True)
    return demo


# ==============================================================================
# 8. PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MÓDULO 5 — ASISTENTE CONVERSACIONAL DE PICO Y PLACA")
    print("=" * 70)
    print("[OK] Funciones disponibles:")
    print("     - consultar_asistente_pico_placa(pregunta, placa_contexto=None)")
    print("     - asistente_conversacional_interactivo()")
    print("     - asistente_colab_input()")
    print("     - desplegar_asistente_gradio(share=False)")
    print("     - desplegar_app_integral_gradio(share=False)")
    print()
    desplegar_app_integral_gradio(share=False)