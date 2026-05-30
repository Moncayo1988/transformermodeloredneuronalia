# ==============================================================================
# MAIN — ORQUESTADOR DEL PIPELINE COMPLETO (Versión Local)
# ==============================================================================

import pandas as pd
import os
import sys
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
            # Lanza Gradio localmente en http://127.0.0.1:7860
            # Se abre el navegador automáticamente (inbrowser=True en modulo5)
            # Presionar Ctrl+C en la terminal para detener y volver al menú
            print("\n  Iniciando la app web con Gradio...")
            print("  Se abrirá en http://127.0.0.1:7860 (local) y generará un link público para celular.")
            print("  Presiona Ctrl+C en esta terminal para detener y volver al menú.\n")
            try:
                desplegar_app_integral_gradio(share=True)
            except KeyboardInterrupt:
                print("\n\n  [OK] App detenida. Volviendo al menú...\n")