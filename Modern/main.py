# ==============================================================================
# MAIN — ORQUESTADOR DEL PIPELINE COMPLETO (Versión Local)
# Autor: Salomón Melenje
# ==============================================================================

import pandas as pd
import os, sys, time, threading, subprocess
from tkinter import Tk, filedialog

from modulo0_config          import asignar_restriccion
from modulo1_deteccion_yolo  import detectar_y_recortar_placa
from modulo2_ocr             import extraer_datos_placa
from modulo3_dataset         import preparar_datos_transformer
from modulo4_transformer     import ejecutar_modulo4, predecir_pico_placa
from modulo5_asistente       import asistente_colab_input, desplegar_app_integral_gradio
from modulo6_camara          import iniciar_camara


# ==============================================================================
# PIPELINE LOCAL
# ==============================================================================

def _fila_vacia() -> pd.DataFrame:
    return pd.DataFrame({"placa_detectada":["N/A"],"ultimo_digito":["N/A"],
                          "longitud_valida":[False],"formato_exacto":[False],
                          "tipo_placa":["N/A"],"motor_ocr":["N/A"]})

def _continuar_procesando() -> bool:
    while True:
        r = input("\n¿Procesar otra imagen? (s/n): ").strip().lower()
        if r in ('s','si','y','yes'): return True
        if r in ('n','no'):           return False
        print("   Por favor responde 's' o 'n'")

def _visualizar_resultado(img_marcada, imagen_binaria, recorte_rgb, df, metodo, nombre):
    import matplotlib.pyplot as plt
    placa = df['placa_detectada'].values[0]
    dia   = asignar_restriccion(df['ultimo_digito'].values[0])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(img_marcada);    axes[0].set_title(f"1. Localización YOLO\n({metodo})", fontsize=11); axes[0].axis('off')
    axes[1].imshow(imagen_binaria, cmap='gray'); axes[1].set_title(f"2. Binarización\nTipo: {df['tipo_placa'].values[0]}", fontsize=11); axes[1].axis('off')
    axes[2].imshow(recorte_rgb);    axes[2].set_title(f"Placa: {placa}  ({df['motor_ocr'].values[0]})\nPico y Placa: {dia}", fontsize=11); axes[2].axis('off')
    plt.suptitle(nombre, fontsize=10, color='gray'); plt.tight_layout(); plt.show()
    print(f"\n  {'─'*50}\n  PLACA DETECTADA : {placa}\n  RESTRICCIÓN     : {dia}\n  TIPO PLACA      : {df['tipo_placa'].values[0]}\n  {'─'*50}")

def pipeline_imagen_local() -> pd.DataFrame:
    historico = pd.DataFrame(columns=["placa_detectada","ultimo_digito","longitud_valida","formato_exacto","tipo_placa","motor_ocr"])
    print("🚀 Iniciando Pipeline de Detección de Placas (Modo Local)\n")
    while True:
        root = Tk(); root.withdraw()
        ruta = filedialog.askopenfilename(title="Seleccionar imagen",
                                          filetypes=[("Imágenes","*.jpg *.jpeg *.png *.JPG *.PNG *.bmp")])
        if not ruta: print("\n[FIN] No se seleccionaron más imágenes."); break
        print(f"\n>>> Procesando: {os.path.basename(ruta)}")
        recorte, marcada, metodo = detectar_y_recortar_placa(ruta)
        if recorte is None:
            print("  [ERROR] No se detectó ninguna placa.")
            historico = pd.concat([historico, _fila_vacia()], ignore_index=True)
            if not _continuar_procesando(): break
            continue
        df, binaria = extraer_datos_placa(recorte)
        if df is None:
            if not _continuar_procesando(): break
            continue
        _visualizar_resultado(marcada, binaria, recorte, df, metodo, os.path.basename(ruta))
        historico = pd.concat([historico, df], ignore_index=True)
        if not _continuar_procesando(): break
    return historico

def exportar_csv(historico: pd.DataFrame, ruta: str = 'resultados_placas.csv'):
    historico.to_csv(ruta, index=False)
    print(f"\n[OK] Resultados guardados en: {ruta}\n"); print(historico)

def ejecutar_pipeline_completo(usar_kaggle=False, n_sintetico=48000, epochs=30, ruta_modelo="transformer_pico_placa.pt"):
    print("="*70+"\n          PIPELINE COMPLETO - DETECCIÓN DE PLACAS\n"+"="*70)
    exportar_csv(pipeline_imagen_local())
    print("\nPipeline finalizado correctamente.")


# ==============================================================================
# NGROK — TÚNEL PÚBLICO AUTOMÁTICO
# ==============================================================================

def _lanzar_ngrok(puerto: int = 7860, ruta_ngrok: str = "ngrok.exe") -> None:
    import shutil, urllib.request, json
    directorio = os.path.dirname(os.path.abspath(__file__))
    exe = next((c for c in [
        os.path.join(directorio, ruta_ngrok),
        os.path.join(directorio, "..", ruta_ngrok),
        shutil.which("ngrok") or "",
    ] if c and os.path.isfile(c)), None)

    if not exe:
        print("  [NGROK] ngrok.exe no encontrado. Descárgalo en: https://ngrok.com/download"); return

    def _run():
        try:
            subprocess.Popen([exe,"http",str(puerto)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            url = None
            for _ in range(20):
                time.sleep(0.5)
                try:
                    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
                        for t in json.loads(r.read()).get("tunnels",[]):
                            if t.get("proto") == "https": url = t["public_url"]; break
                    if url: break
                except Exception: continue
            if url:
                hf = "https://huntercito-pico-placa-popayan.hf.space"
                print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
                print(f"  ║  LINKS DE ACCESO:                                                ║")
                print(f"  ║  Local:       http://127.0.0.1:7860                              ║")
                print(f"  ║  Público:     {url:<52}║")
                print(f"  ║  HuggingFace: {hf:<52}║")
                print(f"  ╚══════════════════════════════════════════════════════════════════╝")
                print("  Abre cualquier link en Chrome de tu celular o PC.\n")
            else:
                print("  [NGROK] No se obtuvo link público. Verifica: ngrok config add-authtoken TU_TOKEN")
        except Exception as e:
            print(f"  [NGROK] Error: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ==============================================================================
# MENÚ PRINCIPAL
# ==============================================================================

def _mostrar_menu():
    print("\n"+"="*60+"\n   DETECCIÓN DE PLACAS VEHICULARES — POPAYÁN\n"+"="*60)
    print("  [1] Pipeline de imágenes (seleccionar archivos)")
    print("  [2] Asistente conversacional (Módulo 5)")
    print("  [3] Cámara en tiempo real (Módulo 6)")
    print("  [4] Interfaz Web de Producción (Dashboard de Despliegue)")
    print("  [0] Salir\n"+"="*60)


if __name__ == "__main__":
    while True:
        _mostrar_menu()
        while True:
            opcion = input("\nElige una opción (1, 2, 3, 4 o 0 para salir): ").strip()
            if opcion in ('0','1','2','3','4'): break
            print("   Por favor elige 1, 2, 3, 4 o 0.")

        if opcion == '0':
            print("\n  ¡Hasta luego!\n"); break

        elif opcion == '1':
            ejecutar_pipeline_completo()
            input("\n  Presiona Enter para volver al menú...")

        elif opcion == '2':
            print("\n  Iniciando asistente... (escribe 'salir' para volver al menú)\n")
            asistente_colab_input(repetir=True)

        elif opcion == '3':
            print("\n  Fuente de video:\n    0 = cámara integrada\n    1 = celular (IP Webcam / DroidCam)")
            while True:
                cam_op = input("  Elige (0 o 1): ").strip()
                if cam_op in ('0','1'): break
                print("  Por favor elige 0 o 1.")
            if cam_op == '1':
                ip  = input("  IP del celular (ej: 192.168.80.21:8080): ").strip()
                cam = f"http://{ip}/video"
            else:
                cam = 0
            usar_tf = input("  ¿Usar Transformer? (s/n, Enter=s): ").strip().lower() not in ('n','no')
            iniciar_camara(cam_idx=cam, usar_transformer=usar_tf, gpu=False)

        elif opcion == '4':
            print("\n  Iniciando la app web con Gradio + ngrok...")
            print("  Local:       http://127.0.0.1:7860")
            print("  HuggingFace: https://huntercito-pico-placa-popayan.hf.space")
            print("  Público:     se generará automáticamente (espera unos segundos).")
            print("  Presiona Ctrl+C en esta terminal para detener y volver al menú.\n")
            _lanzar_ngrok(puerto=7860)
            try:
                desplegar_app_integral_gradio(share=False, server_port=7860, server_name="127.0.0.1")
            except KeyboardInterrupt:
                print("\n\n  [OK] App detenida. Volviendo al menú...\n")