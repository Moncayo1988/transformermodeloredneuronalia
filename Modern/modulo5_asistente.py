# ==============================================================================
# MÓDULO 5 — ASISTENTE CONVERSACIONAL DE PICO Y PLACA
# Autor: Salomón Melenje
# ==============================================================================

import re, os, unicodedata
from datetime import datetime, timedelta

from modulo0_config import DIAS_UNICOS, label2idx, idx2label, asignar_restriccion
from modulo4_transformer import predecir_pico_placa


# ==============================================================================
# CONSTANTES
# ==============================================================================

FUENTE_TABLA = "Tabla Popayan: 0-1 Lunes, 2-3 Martes, 4-5 Miercoles, 6-7 Jueves, 8-9 Viernes."

DIGITOS_POR_DIA = {
    "lunes":["0","1"],"martes":["2","3"],"miercoles":["4","5"],
    "jueves":["6","7"],"viernes":["8","9"],
}
MESES = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,
    "noviembre":11,"diciembre":12,
}
ORDEN_SEMANA = ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"]
TEMAS_FUERA  = {
    "horario":"horarios","hora":"horarios","moto":"tipo de vehiculo",
    "taxi":"tipo de vehiculo","carga":"tipo de vehiculo","camion":"tipo de vehiculo",
    "excepcion":"excepciones o permisos","permiso":"excepciones o permisos",
}
_INTENCIONES_GENERALES = ("consultar_restriccion","consulta_semana","consulta_fuera_de_fuente")


# ==============================================================================
# UTILIDADES
# ==============================================================================

def _norm(t: str) -> str:
    t = unicodedata.normalize("NFD", str(t).lower().strip())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")

def _limpiar_placa(t: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(t).upper())

def _dia_desde_clave(c: str) -> str:
    for d in DIAS_UNICOS:
        if _norm(d) == c: return d
    return {"lunes":"Lunes","martes":"Martes","miercoles":"Miercoles",
            "jueves":"Jueves","viernes":"Viernes","sabado":"Sabado","domingo":"Domingo"
            }.get(c, c.capitalize())

def _clave_desde_fecha(f: datetime) -> str: return ORDEN_SEMANA[f.weekday()]
def _fmt_fecha(f: datetime) -> str: return f"{f.strftime('%Y-%m-%d')} ({_dia_desde_clave(_clave_desde_fecha(f))})"

def interpretar_confianza(c: float) -> str:
    return "alta" if c >= 90 else "media" if c >= 70 else "baja"

def regla_por_dia(clave: str) -> str:
    d = DIGITOS_POR_DIA.get(clave)
    return (f"Placas terminadas en {d[0]} y {d[1]} tienen restriccion el {_dia_desde_clave(clave)}."
            if d else "Sin restriccion definida para ese dia.")

def extraer_placa_desde_pregunta(p: str) -> str | None:
    for pat in [r"\b([A-Z]{3})[\s\-]?([0-9]{3})\b", r"\b([A-Z]{3})[\s\-]?([0-9]{2})[\s\-]?([A-Z])\b"]:
        m = re.search(pat, str(p).upper())
        if m: return _limpiar_placa("".join(m.groups()))
    return None

def obtener_placa_contexto(placa_ctx=None, historico_df=None) -> str | None:
    if placa_ctx:
        p = _limpiar_placa(placa_ctx)
        if p and p not in ("NA","NAN"): return p
    if historico_df is not None and not historico_df.empty:
        for p in reversed([str(x) for x in historico_df.get("placa_detectada",[])]):
            p = _limpiar_placa(p)
            if p and p not in ("NA","NAN") and len(p) >= 5: return p
    return None


# ==============================================================================
# EXTRACCIÓN DE FECHA E INTENCIÓN
# ==============================================================================

def extraer_fecha_desde_pregunta(pregunta: str) -> dict:
    texto = _norm(pregunta); ahora = datetime.now()
    def _r(tipo, txt, dia, clave): return {"tipo":tipo,"texto":txt,"dia":dia,"clave":clave}

    if "esta semana" in texto: return _r("semana","Semana consultada",None,None)
    if re.search(r"\bhoy\b",   texto): c=_clave_desde_fecha(ahora); return _r("fecha",_fmt_fecha(ahora),_dia_desde_clave(c),c)
    if re.search(r"\bmanana\b",texto): f=ahora+timedelta(days=1); c=_clave_desde_fecha(f); return _r("fecha",_fmt_fecha(f),_dia_desde_clave(c),c)

    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", texto)
    if m:
        anio = int(m.group(3) or ahora.year); anio = anio+2000 if anio<100 else anio
        try: f=datetime(anio,int(m.group(2)),int(m.group(1))); c=_clave_desde_fecha(f); return _r("fecha",_fmt_fecha(f),_dia_desde_clave(c),c)
        except ValueError: return _r("invalida","Fecha invalida",None,None)

    m = re.search(r"\b(\d{1,2})\s+de\s+([a-z]+)(?:\s+de\s+(\d{4}))?\b", texto)
    if m:
        mes = MESES.get(m.group(2))
        if mes:
            try: f=datetime(int(m.group(3) or ahora.year),mes,int(m.group(1))); c=_clave_desde_fecha(f); return _r("fecha",_fmt_fecha(f),_dia_desde_clave(c),c)
            except ValueError: return _r("invalida","Fecha invalida",None,None)

    for clave in ORDEN_SEMANA:
        if re.search(rf"\b{clave}\b", texto):
            f = ahora + timedelta(days=(ORDEN_SEMANA.index(clave)-ahora.weekday())%7)
            return _r("dia",f"{_dia_desde_clave(clave)} {f.strftime('%d/%m/%Y')}",_dia_desde_clave(clave),clave)

    return _r("no_especificada","No especificada",None,None)


def clasificar_intencion(pregunta: str) -> str:
    t = _norm(pregunta)
    if any(p in t for p in TEMAS_FUERA):                                        return "consulta_fuera_de_fuente"
    if "esta semana" in t:                                                       return "consulta_semana"
    if any(p in t for p in ["puedo","transitar","circular","pasar"]):            return "validar_transito"
    if any(p in t for p in ["que dia","cual dia","que placas","restriccion","pico"]): return "consultar_restriccion"
    return "consulta_general"


# ==============================================================================
# GENERACIÓN DE RESPUESTA
# ==============================================================================

def _detalle(placa: str, pred: dict) -> str:
    nums = re.findall(r"\d", _limpiar_placa(placa)); c = pred["confianza_pct"]
    return (f"Placa: {placa} | Digito: {nums[-1] if nums else 'N/A'}\n"
            f"Transformer: {pred['restriccion']} | Confianza: {c:.2f}% ({interpretar_confianza(c)})\n"
            f"Fuente: {FUENTE_TABLA}")

def _respuesta_sin_placa(fecha_ctx: dict, intencion: str) -> tuple:
    clave = fecha_ctx.get("clave")
    if intencion == "consulta_semana":
        filas = "\n".join(f"- {_dia_desde_clave(c)}: dígitos {DIGITOS_POR_DIA[c][0]} y {DIGITOS_POR_DIA[c][1]}"
                          for c in ["lunes","martes","miercoles","jueves","viernes"])
        return f"Resumen semanal Popayan:\n{filas}\n\nHorarios/excepciones: no disponibles.", None
    if clave in DIGITOS_POR_DIA:
        d = DIGITOS_POR_DIA[clave]
        return (f"Para {fecha_ctx['texto']}: Pico y Placa para dígitos {d[0]} y {d[1]}.\n"
                f"Regla: {regla_por_dia(clave)}\nEscribe tu placa para validar."), None
    if clave in ("sabado","domingo"):
        return f"Para {fecha_ctx['texto']}: sin restriccion definida.\nNo se consultan normas externas.", True
    return "Necesito el dia o la placa. Ej: 'Puedo transitar el miercoles con placa ABC123?'", None

def generar_respuesta_agente(pregunta: str, placa, fecha_ctx: dict, prediccion, intencion: str) -> tuple:
    t = _norm(pregunta)
    if intencion == "consulta_fuera_de_fuente":
        temas = ", ".join({v for k,v in TEMAS_FUERA.items() if k in t}) or "esa informacion"
        return f"No tengo informacion sobre {temas} en las fuentes del notebook.", None
    if not placa:
        return _respuesta_sin_placa(fecha_ctx, intencion)

    r=prediccion["restriccion"]; c=prediccion["confianza_pct"]
    dia=fecha_ctx.get("dia"); clave=fecha_ctx.get("clave")
    det=_detalle(placa,prediccion); aviso="\n\nAdvertencia: confianza baja; verifica OCR." if c<70 else ""

    if dia is None:
        return f"Para {placa}, Transformer predice restriccion el {r}.\n{det}\nDime el dia para validar.", None
    if clave in ("sabado","domingo"):
        return f"Si puedes transitar el {dia} con {placa}.\nSin restriccion fines de semana.\n{det}", True
    if clave == _norm(r):
        return f"No deberias transitar el {dia} con {placa}.\nCoincide con restriccion Transformer.\nRegla: {regla_por_dia(clave)}\n{det}{aviso}", False
    return f"Si puedes transitar el {dia} con {placa}.\nNo coincide con restriccion estimada.\nRegla: {regla_por_dia(clave)}\n{det}{aviso}", True


# ==============================================================================
# CONSULTA PRINCIPAL
# ==============================================================================

def consultar_asistente_pico_placa(pregunta: str, placa_contexto=None,
                                    historico_df=None, model=None) -> dict:
    fecha_ctx  = extraer_fecha_desde_pregunta(pregunta)
    placa      = extraer_placa_desde_pregunta(pregunta) or obtener_placa_contexto(placa_contexto, historico_df)
    intencion  = clasificar_intencion(pregunta)
    prediccion = predecir_pico_placa(placa, model=model, verbose=False) if placa else None
    respuesta, puede_transitar = generar_respuesta_agente(pregunta, placa, fecha_ctx, prediccion, intencion)
    return {
        "pregunta":pregunta,"placa":placa,
        "fecha_texto":fecha_ctx.get("texto"),"dia_consulta":fecha_ctx.get("dia"),
        "prediccion":prediccion or {},"puede_transitar":puede_transitar,
        "intencion":intencion,"respuesta":respuesta,
    }


# ==============================================================================
# INTERFAZ CONSOLA
# ==============================================================================

def asistente_colab_input(model=None, placa_contexto=None, historico_df=None, repetir=True):
    placa = obtener_placa_contexto(placa_contexto, historico_df)
    print("\nASISTENTE DE PICO Y PLACA" + (f"\nPlaca sugerida: {placa}" if placa else ""))
    print("Escribe 'salir' para terminar.\n")
    ultimo = None
    while True:
        q = input("Pregunta: ").strip()
        if _norm(q) in ("salir","exit","quit"): print("Asistente finalizado."); break
        if not q:
            print("[AVISO] Pregunta vacía.")
            if not repetir: break
            continue
        ultimo = consultar_asistente_pico_placa(q, placa_contexto=placa, historico_df=historico_df, model=model)
        print(ultimo["respuesta"])
        if not repetir: break
    return ultimo


# ==============================================================================
# GRADIO — RESPUESTA Y PROCESAMIENTO DE IMAGEN
# ==============================================================================

def respuesta_asistente_gradio(pregunta: str, placa_manual: str = "") -> str:
    pregunta = str(pregunta or "").strip()
    if not pregunta: return "Escribe una pregunta para consultar el Pico y Placa."
    intencion = clasificar_intencion(pregunta)
    placa_ctx = (placa_manual or None) if intencion in _INTENCIONES_GENERALES else (placa_manual or obtener_placa_contexto())
    r    = consultar_asistente_pico_placa(pregunta, placa_contexto=placa_ctx)
    pred = r.get("prediccion") or {}
    estado = ("Puede transitar ✅" if r.get("puede_transitar") is True
              else "No puede transitar ❌" if r.get("puede_transitar") is False
              else "Consulta realizada")
    partes = [f"## {estado}", r["respuesta"]]
    if pred:
        partes += ["","### Datos del modelo",
                   f"- **Placa:** `{r.get('placa')}`",
                   f"- **Día:** `{r.get('dia_consulta') or 'No especificado'}`",
                   f"- **Restricción:** `{pred.get('restriccion')}`",
                   f"- **Confianza:** `{pred.get('confianza_pct')}%`"]
    partes += ["","### Límites","- Solo Pico y Placa por placa y día para Popayán.",
               "- Horarios, permisos y tipo de vehículo no disponibles."]
    return "\n".join(partes)

def responder_app_integral(pregunta: str, placa_manual: str, estado_imagen: dict) -> str:
    estado_imagen = estado_imagen or {}
    intencion     = clasificar_intencion(str(pregunta or ""))
    placa_ctx     = ("" if intencion in _INTENCIONES_GENERALES
                     else str(placa_manual or "").strip() or estado_imagen.get("placa") or "")
    return respuesta_asistente_gradio(pregunta, placa_ctx)

def procesar_imagen_app_integral(imagen_np, desde_webcam: bool = False) -> tuple:
    import numpy as np, tempfile, cv2
    from modulo1_deteccion_yolo import detectar_y_recortar_placa
    from modulo2_ocr import extraer_datos_placa

    vacio = {"placa":None,"restriccion":None,"confianza":None}
    if imagen_np is None: return None,None,None,"","Sube o captura una imagen para iniciar.",vacio
    try:
        if desde_webcam: imagen_np = np.ascontiguousarray(imagen_np[:,::-1])
        imagen_bgr = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, imagen_bgr); ruta=tmp.name; tmp.close()
        try:    recorte_rgb,img_marcada,metodo = detectar_y_recortar_placa(ruta)
        finally:
            try: os.remove(ruta)
            except Exception: pass

        if recorte_rgb is None: return None,None,None,"","No se detectó ninguna placa.",vacio
        df_ocr, binaria = extraer_datos_placa(recorte_rgb)
        if df_ocr is None or df_ocr.empty: return img_marcada,recorte_rgb,None,"","OCR no pudo leer la placa.",vacio

        placa = str(df_ocr["placa_detectada"].values[0]); ultimo = str(df_ocr["ultimo_digito"].values[0])
        pred  = predecir_pico_placa(placa,verbose=False) if placa not in ("N/A","nan","") else None
        r_mod = pred["restriccion"] if pred else "No disponible"; conf = pred["confianza_pct"] if pred else None

        estado  = {"placa":placa,"ultimo_digito":ultimo,"restriccion":r_mod,"confianza":conf}
        resumen = "\n".join([
            "## Análisis completado",
            f"- **Método:** `{metodo}` | **Placa:** `{placa}` | **Dígito:** `{ultimo}`",
            f"- **Tipo:** `{df_ocr['tipo_placa'].values[0]}` | **OCR:** `{df_ocr['motor_ocr'].values[0]}`",
            f"- **Restricción regla base:** `{asignar_restriccion(ultimo)}`",
            f"- **Restricción Transformer:** `{r_mod}` | **Confianza:** `{conf if conf is not None else 'N/A'}%`"
            + (f" ({interpretar_confianza(conf)})" if conf else ""),
            "","Ahora puedes preguntar: `¿Puedo transitar hoy?`",
        ])
        return img_marcada, recorte_rgb, binaria, placa, resumen, estado
    except Exception as e:
        return None,None,None,"",f"Error al procesar: {e}",vacio


# ==============================================================================
# CORRECCIÓN DE ESPEJO WEBCAM (JavaScript)
# ==============================================================================

_JS_ESPEJO = """
function corregirEspejoWebcam() {
    const s = document.createElement('style');
    s.textContent = '.gradio-container video { transform: scaleX(-1) !important; }';
    document.head.appendChild(s);
    const orig = CanvasRenderingContext2D.prototype.drawImage;
    CanvasRenderingContext2D.prototype.drawImage = function(src, ...a) {
        if (src instanceof HTMLVideoElement) {
            const w=this.canvas.width; this.save(); this.translate(w,0); this.scale(-1,1);
            if (a.length>=2 && typeof a[0]==='number') a[0]=w-a[0]-(a[2]||w);
            orig.call(this,src,...a); this.restore();
        } else { orig.call(this,src,...a); }
    };
}
"""


# ==============================================================================
# DESPLIEGUE GRADIO
# ==============================================================================

def _instalar_gradio():
    import subprocess, sys
    subprocess.check_call([sys.executable,"-m","pip","install","gradio","-q"])

def desplegar_app_integral_gradio(share: bool = False, server_port: int = 7860,
                                   server_name: str = "127.0.0.1") -> object:
    """App completa: YOLO → OCR → Transformer → Asistente conversacional."""
    try: import gradio as gr
    except ImportError: _instalar_gradio(); import gradio as gr

    with gr.Blocks(title="App Pico y Placa IA — Popayán", js=_JS_ESPEJO) as demo:
        estado_img = gr.State({"placa":None,"restriccion":None,"confianza":None})
        gr.Markdown("# App Pico y Placa IA — Popayán\n"
                    "**Flujo:** Sube una imagen → YOLO → OCR → Transformer → Asistente.")

        fuente = gr.State("upload")
        with gr.Tabs():
            with gr.Tab("📁 Subir archivo"):
                img_up  = gr.Image(label="Imagen (archivo)",  type="numpy", sources=["upload"])
            with gr.Tab("📷 Cámara web"):
                gr.HTML('<div style="background:rgba(251,191,36,.08);border-left:4px solid #f59e0b;'
                        'border-radius:8px;padding:10px 16px;margin-bottom:8px;font-size:13.5px;color:#d4a017">'
                        '📷 <b>Nota:</b> el preview puede verse en espejo — se corrige automáticamente.</div>')
                img_cam = gr.Image(label="Imagen (cámara)", type="numpy", sources=["webcam"])

        with gr.Row():
            with gr.Column():
                btn_proc   = gr.Button("Analizar imagen", variant="primary")
                placa_edit = gr.Textbox(label="Placa detectada / editable", placeholder="ABC123")
                resumen    = gr.Markdown(label="Resultado")

        img_up.change( fn=lambda _:"upload", inputs=img_up,  outputs=fuente)
        img_cam.change(fn=lambda _:"webcam", inputs=img_cam, outputs=fuente)

        with gr.Row():
            img_det = gr.Image(label="Detección")
            img_rec = gr.Image(label="Recorte")
            img_bin = gr.Image(label="Binarización OCR")

        gr.Markdown("## Consulta conversacional")
        pregunta = gr.Textbox(label="Pregunta", placeholder="¿Puedo transitar hoy?", lines=3)
        with gr.Row():
            btn_preg = gr.Button("Preguntar", variant="primary")
            btn_limp = gr.Button("Limpiar pregunta")
        salida = gr.Markdown(label="Respuesta")
        gr.Examples(examples=["¿Puedo transitar hoy?","¿Puedo transitar el miércoles?",
                               "¿Qué placas tienen pico y placa el viernes?","Restricciones de esta semana"],
                    inputs=pregunta, cache_examples=False)

        btn_proc.click(
            fn=lambda u,c,f: procesar_imagen_app_integral(c if f=="webcam" else u, desde_webcam=(f=="webcam")),
            inputs=[img_up,img_cam,fuente],
            outputs=[img_det,img_rec,img_bin,placa_edit,resumen,estado_img])
        btn_preg.click(fn=responder_app_integral, inputs=[pregunta,placa_edit,estado_img], outputs=salida)
        pregunta.submit(fn=responder_app_integral, inputs=[pregunta,placa_edit,estado_img], outputs=salida)
        btn_limp.click(fn=lambda:"", inputs=None, outputs=pregunta)

    print("[OK] Lanzando app integral con Gradio...")
    demo.launch(share=share, server_port=server_port, server_name=server_name, debug=False, inbrowser=True)
    return demo


if __name__ == "__main__":
    print("\n"+"="*70+"\nMÓDULO 5 — ASISTENTE CONVERSACIONAL DE PICO Y PLACA\n"+"="*70)
    desplegar_app_integral_gradio(share=True)