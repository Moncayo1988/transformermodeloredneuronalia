# ==============================================================================
# MAIN — ORQUESTADOR DEL PIPELINE COMPLETO (Versión Local)
# ==============================================================================

import pandas as pd
import os
from tkinter import Tk, filedialog

# Módulos del proyecto
from modulo0_config  import asignar_restriccion
from modulo1_deteccion_yolo import detectar_y_recortar_placa
from modulo2_ocr     import extraer_datos_placa
from modulo3_dataset import preparar_datos_transformer
from modulo4_transformer import ejecutar_modulo4, predecir_pico_placa

# Módulo 5 — Cámara en tiempo real
from modulo5_camara import iniciar_camara


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
        # Crear ventana oculta para seleccionar archivo
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
    """Pregunta si quiere procesar otra imagen"""
    while True:
        resp = input("\n¿Procesar otra imagen? (s/n): ").strip().lower()
        if resp in ['s', 'si', 'y', 'yes']:
            return True
        if resp in ['n', 'no']:
            return False
        print("   Por favor responde 's' o 'n'")


def _visualizar_resultado(img_marcada, imagen_binaria, recorte_rgb, df_resultado, metodo, nombre_archivo):
    """Muestra los 3 paneles (igual que antes)"""
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
# 3. EJECUCIÓN COMPLETA
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

    # FASE 1 — Procesamiento de imágenes locales
    historico_df = pipeline_imagen_local()
    exportar_csv(historico_df)

    print("\nPipeline finalizado correctamente.")


# ==============================================================================
# 4. PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    # Menú de selección de modo
    print("\n" + "="*60)
    print("   DETECCIÓN DE PLACAS VEHICULARES — POPAYÁN")
    print("="*60)
    print("  [1] Pipeline de imágenes (seleccionar archivos)")
    print("  [2] Cámara en tiempo real (Módulo 5)")
    print("="*60)

    while True:
        opcion = input("\nElige una opción (1 o 2): ").strip()
        if opcion in ['1', '2']:
            break
        print("   Por favor elige 1 o 2.")

    if opcion == '1':
        # Pipeline original — sin cambios
        ejecutar_pipeline_completo(
            usar_kaggle=False,
            n_sintetico=48000,
            epochs=30,
            ruta_modelo="transformer_pico_placa.pt"
        )

    elif opcion == '2':
        # Módulo 5 — Cámara en tiempo real
        print("\n  Índice de cámara:")
        print("    0 = cámara por defecto")
        print("    1 = cámara externa")
        cam = input("  Elige índice (Enter para 0): ").strip()
        cam = int(cam) if cam.isdigit() else 0

        tf = input("  ¿Usar Transformer? (s/n, Enter=s): ").strip().lower()
        usar_tf = tf not in ['n', 'no']

        iniciar_camara(
            cam_idx          = cam,
            usar_transformer = usar_tf,
            gpu              = False
        )
