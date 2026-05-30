# ==============================================================================
# MAIN — ORQUESTADOR DEL PIPELINE COMPLETO (Versión Local)
# ==============================================================================

import pandas as pd
import os
import sys
import time
import threading
import subprocess
from tkinter import Tk, filedialog

# Módulos del proyecto
from modulo0_config  import asignar_restriccion
from modulo1_deteccion_yolo import detectar_y_recortar_placa
from modulo2_ocr     import extraer_datos_placa
from modulo3_dataset import preparar_datos_transformer
from modulo4_transformer import ejecutar_modulo4, predecir_pico_placa

# Módulo 5 — Asistente conversacional + despliegue Gradio
from modulo5_asistente import asistente_colab_input, desplegar_app_integral_gradio

# Módulo 6 — Cámara en tiempo real
from modulo6_camara import iniciar_camara


# ==============================================================================
# 1. PIPELINE LOCAL - SELECCIÓN DE IMAGEN
# ==============================================================================

def pipeline_imagen_local() -> pd.DataFrame:
    """
    Pipeline para uso LOCAL:
    - Abre ventana para seleccionar imagen
    - Procesa una o varias imágenes
    """
    historico_df = pd.DataFrame(columns=[
        "placa_detectada", "ultimo_digito", "longitud_valida",
        "formato_exacto", "tipo_placa", "motor_ocr"
    ])

    print("🚀 Iniciando Pipeline de Detección de Placas (Modo Local)\n")

    while True:
        root = Tk()
        root.withdraw()

        ruta_imagen = filedialog.askopenfilename(
            title="Seleccionar imagen de vehículo",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.JPG *.PNG *.bmp")]
        )

        if not ruta_imagen:
            print("\n[FIN] No se seleccionaron más imágenes.")
            break

        print(f"\n>>> Procesando: {os.path.basename(ruta_imagen)}")

        # MÓDULO 1 — Detección YOLO
        recorte_rgb, img_marcada_rgb, metodo = detectar_y_recortar_placa(ruta_imagen)

        if recorte_rgb is None:
            print("  [ERROR] No se detectó ninguna placa.")
            historico_df = pd.concat([historico_df, _fila_vacia()], ignore_index=True)
            if not _continuar_procesando():
                break
            continue

        # MÓDULO 2 — OCR
        df_resultado, imagen_binaria = extraer_datos_placa(recorte_rgb)
        if df_resultado is None:
            if not _continuar_procesando():
                break
            continue

        # Visualización
        _visualizar_resultado(
            img_marcada_rgb, imagen_binaria, recorte_rgb,
            df_resultado, metodo, os.path.basename(ruta_imagen)
        )

        historico_df = pd.concat([historico_df, df_resultado], ignore_index=True)

        if not _continuar_procesando():
            break

    return historico_df


def _fila_vacia() -> pd.DataFrame:
    return pd.DataFrame({
        "placa_detectada" : ["N/A"], "ultimo_digito": ["N/A"],
        "longitud_valida" : [False],  "formato_exacto": [False],
        "tipo_placa"      : ["N/A"],  "motor_ocr": ["N/A"]
    })


def _continuar_procesando() -> bool:
    """Pregunta si quiere procesar otra imagen."""
    while True:
        resp = input("\n¿Procesar otra imagen? (s/n): ").strip().lower()
        if resp in ['s', 'si', 'y', 'yes']:
            return True
        if resp in ['n', 'no']:
            return False
        print("   Por favor responde 's' o 'n'")


def _visualizar_resultado(img_marcada, imagen_binaria, recorte_rgb, df_resultado, metodo, nombre_archivo):
    """Muestra los 3 paneles de diagnóstico (YOLO · binarización · recorte)."""
    placa = df_resultado['placa_detectada'].values[0]
    dia   = asignar_restriccion(df_resultado['ultimo_digito'].values[0])

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(img_marcada)
    axes[0].set_title(f"1. Localización YOLO\n({metodo})", fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(imagen_binaria, cmap='gray')
    axes[1].set_title(f"2. Binarización\nTipo: {df_resultado['tipo_placa'].values[0]}", fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(recorte_rgb)
    axes[2].set_title(
        f"Placa: {placa}  ({df_resultado['motor_ocr'].values[0]})\n"
        f"Pico y Placa: {dia}",
        fontsize=11
    )
    axes[2].axis('off')

    plt.suptitle(nombre_archivo, fontsize=10, color='gray')
    plt.tight_layout()
    plt.show()

    print(f"\n  {'─'*50}")
    print(f"  PLACA DETECTADA : {placa}")
    print(f"  RESTRICCIÓN     : {dia}")
    print(f"  TIPO PLACA      : {df_resultado['tipo_placa'].values[0]}")
    print(f"  {'─'*50}")


# ==============================================================================
# 2. EXPORTACIÓN CSV
# ==============================================================================

def exportar_csv(historico_df: pd.DataFrame, ruta: str = 'resultados_placas.csv'):
    historico_df.to_csv(ruta, index=False)
    print(f"\n[OK] Resultados guardados en: {ruta}")
    print(historico_df)


# ==============================================================================
# 3. EJECUCIÓN COMPLETA (Opción 1)
# ==============================================================================

def ejecutar_pipeline_completo(
    usar_kaggle: bool = False,
    n_sintetico: int = 48000,
    epochs: int = 30,
    ruta_modelo: str = "transformer_pico_placa.pt"
) -> None:

    print("="*70)
    print("          PIPELINE COMPLETO - DETECCIÓN DE PLACAS")
    print("="*70)

    historico_df = pipeline_imagen_local()
    exportar_csv(historico_df)

    print("\nPipeline finalizado correctamente.")



# ==============================================================================
# 5. NGROK — TÚNEL PÚBLICO AUTOMÁTICO
# ==============================================================================

def _lanzar_ngrok(puerto: int = 7860, ruta_ngrok: str = "ngrok.exe") -> None:
    """
    Lanza ngrok en un hilo separado y espera hasta obtener el link público.
    Busca ngrok.exe en la misma carpeta del proyecto y en el PATH del sistema.
    """
    import shutil, urllib.request, json

    # Buscar ngrok: carpeta del script → PATH del sistema
    directorio = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(directorio, ruta_ngrok),   # mismo directorio que main.py
        os.path.join(directorio, "..", ruta_ngrok),  # raíz del proyecto
        shutil.which("ngrok") or "",             # en el PATH
    ]
    exe = next((c for c in candidatos if c and os.path.isfile(c)), None)

    if not exe:
        print("  [NGROK] ngrok.exe no encontrado. Colócalo en la carpeta del proyecto.")
        print("  [NGROK] Descárgalo en: https://ngrok.com/download")
        return

    def _run():
        try:
            subprocess.Popen(
                [exe, "http", str(puerto)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Esperar hasta que la API local de ngrok esté lista (máx 10 s)
            url_publica = None
            for _ in range(20):
                time.sleep(0.5)
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:4040/api/tunnels", timeout=2
                    ) as resp:
                        data = json.loads(resp.read())
                        tunnels = data.get("tunnels", [])
                        for t in tunnels:
                            if t.get("proto") == "https":
                                url_publica = t["public_url"]
                                break
                        if url_publica:
                            break
                except Exception:
                    continue

            if url_publica:
                print(f"\n  ╔══════════════════════════════════════════════════════╗")
                print(f"  ║  LINK PÚBLICO (celular / internet):                  ║")
                print(f"  ║  {url_publica:<52}║")
                print(f"  ╚══════════════════════════════════════════════════════╝")
                print("  Abre ese link en Chrome de tu celular.\n")
            else:
                print("  [NGROK] No se pudo obtener el link público.")
                print("  [NGROK] Verifica que el authtoken esté configurado:")
                print("          ngrok config add-authtoken TU_TOKEN")
        except Exception as e:
            print(f"  [NGROK] Error al lanzar ngrok: {e}")

    hilo = threading.Thread(target=_run, daemon=True)
    hilo.start()

# ==============================================================================
# 4. MENÚ PRINCIPAL
# ==============================================================================

def _mostrar_menu() -> None:
    """Imprime el menú principal."""
    print("\n" + "="*60)
    print("   DETECCIÓN DE PLACAS VEHICULARES — POPAYÁN")
    print("="*60)
    print("  [1] Pipeline de imágenes (seleccionar archivos)")
    print("  [2] Asistente conversacional (Módulo 5)")
    print("  [3] Cámara en tiempo real (Módulo 6)")
    print("  [4] Interfaz Web de Producción (Dashboard de Despliegue)")
    print("  [0] Salir")
    print("="*60)


if __name__ == "__main__":

    while True:
        _mostrar_menu()

        while True:
            opcion = input("\nElige una opción (1, 2, 3, 4 o 0 para salir): ").strip()
            if opcion in ['0', '1', '2', '3', '4']:
                break
            print("   Por favor elige 1, 2, 3, 4 o 0.")

        # ── Salir ──────────────────────────────────────────────────────────────
        if opcion == '0':
            print("\n  ¡Hasta luego!\n")
            break

        # ── Opción 1 — Pipeline de imágenes ───────────────────────────────────
        elif opcion == '1':
            ejecutar_pipeline_completo(
                usar_kaggle=False,
                n_sintetico=48000,
                epochs=30,
                ruta_modelo="transformer_pico_placa.pt"
            )
            input("\n  Presiona Enter para volver al menú...")

        # ── Opción 2 — Asistente conversacional ───────────────────────────────
        elif opcion == '2':
            print("\n  Iniciando asistente de Pico y Placa...")
            print("  (escribe 'salir' dentro del asistente para volver al menú)\n")
            asistente_colab_input(repetir=True)

        # ── Opción 3 — Cámara en tiempo real ──────────────────────────────────
        elif opcion == '3':
            print("\n  Fuente de video:")
            print("    0 = cámara integrada del portátil")
            print("    1 = celular (IP Webcam / DroidCam)")

            while True:
                cam_opcion = input("  Elige (0 o 1): ").strip()
                if cam_opcion in ['0', '1']:
                    break
                print("  Por favor elige el 0 o el 1.")

            if cam_opcion == '1':
                ip = input("  IP del celular (ej: 192.168.80.21:8080): ").strip()
                cam = f"http://{ip}/video"
            else:
                cam = 0

            tf = input("  ¿Usar Transformer? (s/n, Enter=s): ").strip().lower()
            usar_tf = tf not in ['n', 'no']

            iniciar_camara(
                cam_idx          = cam,
                usar_transformer = usar_tf,
                gpu              = False
            )
            # La cámara vuelve al menú automáticamente al presionar 'q'

        # ── Opción 4 — Interfaz Web de Producción ─────────────────────────────
        elif opcion == '4':
            print("\n  Iniciando la app web con Gradio + ngrok...")
            print("  Local:   http://127.0.0.1:7860")
            print("  Público: se generará automáticamente (espera unos segundos).")
            print("  Presiona Ctrl+C en esta terminal para detener y volver al menú.\n")
            _lanzar_ngrok(puerto=7860)          # arranca ngrok en segundo plano
            try:
                desplegar_app_integral_gradio(share=False)   # Gradio solo local
            except KeyboardInterrupt:
                print("\n\n  [OK] App detenida. Volviendo al menú...\n")