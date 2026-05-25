# ==============================================================================
# MÓDULO 4 — TRANSFORMER DESDE CERO EN PyTorch
# ==============================================================================
# Responsabilidad:
#   - Definir la arquitectura completa del Transformer (desde cero, sin HuggingFace)
#   - Entrenar con early stopping y OneCycleLR scheduler
#   - Evaluar en conjunto de prueba y generar métricas (reporte + matriz confusión)
#   - Guardar/cargar checkpoints
#   - Exponer función de predicción para uso externo
#
# Arquitectura:
#   tokens(B,7) → Embedding(B,7,64) → PositionalEncoding → 2×EncoderBlock
#                → MeanPool → Clasificador(64→5 días)
#
# Entradas  : datos_raw — lista de (tokens:list, label:int) del Módulo 3
# Salidas   : modelo entrenado, métricas, checkpoint .pt
# ==============================================================================

import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from modulo0_config import (
    VOCAB_SIZE, MAX_LEN, NUM_CLASES,
    DIAS_UNICOS, label2idx, idx2label,
    char2idx, idx2char,
    tokenizar_placa
)


# ==============================================================================
# 1. VERIFICACIÓN DE GPU
# ==============================================================================

def verificar_dispositivo() -> torch.device:
    """Detecta GPU disponible e imprime información del hardware."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[GPU] Dispositivo activo: {device}")
    if device.type == "cuda":
        print(f"      Nombre GPU    : {torch.cuda.get_device_name(0)}")
        print(f"      Memoria total : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("      [AVISO] Sin GPU detectada. Entrenando en CPU.")
    return device


# ==============================================================================
# 2. DATASET Y DATALOADERS
# ==============================================================================

class PlacaDataset(Dataset):
    """Dataset PyTorch de secuencias tokenizadas de placas vehiculares."""

    def __init__(self, datos: list[tuple]):
        self.tokens = torch.tensor([d[0] for d in datos], dtype=torch.long)
        self.labels = torch.tensor([d[1] for d in datos], dtype=torch.long)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


def preparar_dataloaders(
    datos_raw: list[tuple],
    batch_size: int = 256,
    train_frac: float = 0.80,
    val_frac: float = 0.20,
    semilla: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Divide datos_raw en train/val/test y crea los DataLoaders.

    Proporciones por defecto:
      Train 80% (de los cuales 20% es validación) | Test 20%

    Retorna: (train_loader, val_loader, test_loader)
    """
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    pin = (device_type == "cuda")

    dataset = PlacaDataset(datos_raw)
    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_test  = n_total - n_train

    train_ds, test_ds = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(semilla)
    )

    n_val = int(n_train * val_frac)
    n_tr  = n_train - n_val
    train_sub, val_sub = random_split(
        train_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                               pin_memory=pin, num_workers=2)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False,
                               pin_memory=pin, num_workers=2)
    test_loader  = DataLoader(test_ds,   batch_size=batch_size, shuffle=False,
                               pin_memory=pin, num_workers=2)

    print(f"\nDataLoaders — Train: {n_tr} | Val: {n_val} | Test: {n_test} "
          f"| Batch: {batch_size}")
    return train_loader, val_loader, test_loader


# ==============================================================================
# 3. ARQUITECTURA DEL TRANSFORMER DESDE CERO
# ==============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding fijo con senos y cosenos (Vaswani et al. 2017):

      PE(pos, 2k)   = sin(pos / 10000^(2k/d_model))
      PE(pos, 2k+1) = cos(pos / 10000^(2k/d_model))

    Imprescindible porque el Transformer no tiene recurrencia y no sabe
    en qué posición se encuentra cada token. Para placas vehiculares,
    la posición importa: el dígito en pos 5 determina el día de restricción,
    no el de pos 3.
    """

    def __init__(self, d_model: int, max_len: int = MAX_LEN, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class MultiHeadAttention(nn.Module):
    """
    Atención multi-cabeza con producto punto escalado, desde cero:

      scores = softmax(Q @ K^T / sqrt(d_k)) @ V

    El escalado por sqrt(d_k) evita que el softmax se sature cuando
    d_k es grande, lo que haría desaparecer los gradientes.

    Cada cabeza aprende relaciones distintas entre posiciones:
      - Una cabeza puede detectar que pos 5 determina la clase
      - Otra puede validar que pos 0-2 son letras (contexto estructural)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.d_k)

    def _split_heads(self, x: torch.Tensor, B: int) -> torch.Tensor:
        return x.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B  = x.size(0)
        Q  = self._split_heads(self.W_Q(x), B)
        K  = self._split_heads(self.W_K(x), B)
        V  = self._split_heads(self.W_V(x), B)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx  = torch.matmul(attn, V)
        ctx  = ctx.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_O(ctx)


class TransformerEncoderBlock(nn.Module):
    """
    Bloque encoder estándar:
      1. MultiHeadAttention + Residual + LayerNorm
      2. FFN (Linear → ReLU → Dropout → Linear) + Residual + LayerNorm

    LayerNorm: normaliza por capa, más estable que BatchNorm para
    secuencias cortas donde el batch puede variar en tamaño.

    Conexiones residuales: permiten que el gradiente fluya directamente
    a capas anteriores sin atenuarse (solución al vanishing gradient).
    """

    def __init__(self, d_model: int, num_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TransformerPlacas(nn.Module):
    """
    Transformer completo para clasificación de Pico y Placa.

    Flujo:
      tokens(B,7) → Embedding(B,7,d_model) → PositionalEncoding
                 → N×EncoderBlock → MeanPooling → Clasificador FC

    Hiperparámetros por defecto (probados en dataset de 50k):
      d_model=64 | num_heads=4 (d_k=16) | d_ff=256 | num_layers=2
      dropout=0.1 | dropout_clasificador=0.3

    El MeanPooling con máscara de padding evita que los tokens <PAD>
    contribuyan al vector de representación final de la secuencia.
    """

    def __init__(
        self,
        vocab_size: int  = VOCAB_SIZE,
        d_model: int     = 64,
        num_heads: int   = 4,
        d_ff: int        = 256,
        num_layers: int  = 2,
        num_classes: int = NUM_CLASES,
        max_len: int     = MAX_LEN,
        dropout: float   = 0.1
    ):
        super().__init__()
        self.embedding      = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding   = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        emb = self.pos_encoding(self.embedding(x))
        out = emb
        for block in self.encoder_blocks:
            out = block(out, mask)

        # Mean pooling con máscara de padding
        pad_mask   = (x != 0).unsqueeze(-1).float()
        out_pooled = (out * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)
        return self.classifier(out_pooled)


# ==============================================================================
# 4. CICLO DE ENTRENAMIENTO
# ==============================================================================

def _entrenar_epoch(
    model: TransformerPlacas,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device
) -> tuple[float, float]:
    model.train()
    total_loss, correctos, total = 0.0, 0, 0
    for tok, lbl in loader:
        tok, lbl = tok.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(tok)
        loss   = criterion(logits, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * tok.size(0)
        correctos  += (logits.argmax(1) == lbl).sum().item()
        total      += tok.size(0)
    return total_loss / total, correctos / total


def _evaluar(
    model: TransformerPlacas,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss, correctos, total = 0.0, 0, 0
    with torch.no_grad():
        for tok, lbl in loader:
            tok, lbl = tok.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
            logits     = model(tok)
            loss       = criterion(logits, lbl)
            total_loss += loss.item() * tok.size(0)
            correctos  += (logits.argmax(1) == lbl).sum().item()
            total      += tok.size(0)
    return total_loss / total, correctos / total


def entrenar(
    model: TransformerPlacas,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int    = 30,
    lr: float      = 3e-4,
    paciencia: int = 7
) -> dict:
    """
    Entrena el Transformer con:
      - Optimizador: AdamW (weight_decay=1e-4)
      - Scheduler  : OneCycleLR (max_lr=3e-3, pct_start=0.3)
      - Early stopping con paciencia configurable
      - Gradient clipping (max_norm=1.0)

    Retorna: dict con historial de métricas y mejor val_acc.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3,
        steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.3
    )

    historial  = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    mejor_acc  = 0.0
    sin_mejora = 0
    mejor_estado = None

    print(f"\nEntrenando {epochs} épocas (early stop paciencia={paciencia})...\n")
    header = f"{'Época':>6} | {'Train Loss':>10} | {'Train Acc':>9} | " \
             f"{'Val Loss':>9} | {'Val Acc':>8} | {'LR':>10}"
    print(header)
    print("─" * 65)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _entrenar_epoch(model, train_loader, criterion,
                                           optimizer, scheduler, device)
        vl_loss, vl_acc = _evaluar(model, val_loader, criterion, device)

        historial['train_loss'].append(tr_loss)
        historial['train_acc'].append(tr_acc)
        historial['val_loss'].append(vl_loss)
        historial['val_acc'].append(vl_acc)

        lr_actual = scheduler.get_last_lr()[0]
        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc*100:>8.2f}% | "
              f"{vl_loss:>9.4f} | {vl_acc*100:>7.2f}% | {lr_actual:>10.2e}")

        if vl_acc > mejor_acc:
            mejor_acc    = vl_acc
            mejor_estado = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            sin_mejora   = 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                print(f"\n[Early Stop] Sin mejora por {paciencia} épocas.")
                break

    if mejor_estado:
        model.load_state_dict({k: v.to(device) for k, v in mejor_estado.items()})
        print(f"\n[OK] Mejor checkpoint restaurado (val_acc={mejor_acc*100:.2f}%)")

    historial['mejor_val_acc'] = mejor_acc
    return historial


# ==============================================================================
# 5. EVALUACIÓN Y MÉTRICAS
# ==============================================================================

def evaluar_test(
    model: TransformerPlacas,
    test_loader: DataLoader,
    device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evalúa el modelo en el conjunto de test.
    Retorna: (test_accuracy, preds_array, true_array)
    """
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = _evaluar(model, test_loader, criterion, device)

    model.eval()
    todos_preds, todos_true = [], []
    with torch.no_grad():
        for tok, lbl in test_loader:
            tok = tok.to(device, non_blocking=True)
            todos_preds.extend(model(tok).argmax(1).cpu().numpy())
            todos_true.extend(lbl.numpy())

    preds = np.array(todos_preds)
    true  = np.array(todos_true)

    n_errores = (preds != true).sum()
    print(f"\nPérdida test  : {test_loss:.4f}")
    print(f"Precisión test: {test_acc*100:.2f}%")
    print(f"Errores       : {n_errores} de {len(true)} "
          f"({n_errores/len(true)*100:.2f}%)")

    return test_acc, preds, true


def graficar_entrenamiento(historial: dict, test_acc: float = None) -> None:
    """Genera gráficas de accuracy y loss por época."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot([a * 100 for a in historial['train_acc']], label='Entrenamiento', lw=2)
    axes[0].plot([a * 100 for a in historial['val_acc']],   label='Validación',   lw=2, ls='--')
    if test_acc is not None:
        axes[0].axhline(y=test_acc * 100, color='red', ls=':', lw=1.5,
                        label=f'Test final: {test_acc*100:.2f}%')
    axes[0].set_title("Precisión del Transformer por Época")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim([85, 101]); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(historial['train_loss'], label='Entrenamiento', lw=2)
    axes[1].plot(historial['val_loss'],   label='Validación',   lw=2, ls='--')
    axes[1].set_title("Función de Pérdida (CrossEntropy)")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def graficar_confusion(preds: np.ndarray, true: np.ndarray, test_acc: float) -> None:
    """Genera y muestra la matriz de confusión."""
    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DIAS_UNICOS, yticklabels=DIAS_UNICOS, linewidths=0.5)
    plt.title(f"Matriz de Confusión — Precisión: {test_acc*100:.2f}%")
    plt.xlabel("Predicción"); plt.ylabel("Valor Real")
    plt.tight_layout(); plt.show()

    print("\nREPORTE DE CLASIFICACIÓN:\n")
    print(classification_report(true, preds, target_names=DIAS_UNICOS))


# ==============================================================================
# 6. GUARDAR Y CARGAR MODELO
# ==============================================================================

RUTA_MODELO_DEFAULT = "transformer_pico_placa.pt"


def guardar_modelo(
    model: TransformerPlacas,
    ruta: str = RUTA_MODELO_DEFAULT,
    hiperparametros: dict = None
) -> None:
    """Guarda el modelo y su configuración en un archivo .pt."""
    hp = hiperparametros or {
        'vocab_size': VOCAB_SIZE, 'd_model': 64, 'num_heads': 4,
        'd_ff': 256, 'num_layers': 2, 'num_classes': NUM_CLASES,
        'max_len': MAX_LEN
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx'        : char2idx,
        'idx2char'        : idx2char,
        'label2idx'       : label2idx,
        'idx2label'       : idx2label,
        'hyperparams'     : hp
    }, ruta)
    print(f"[OK] Modelo guardado: '{ruta}'")


def cargar_modelo(
    ruta: str = RUTA_MODELO_DEFAULT,
    device: torch.device = None
) -> TransformerPlacas:
    """Carga el modelo desde un checkpoint .pt."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ruta, map_location=device)
    hp         = checkpoint['hyperparams']
    modelo = TransformerPlacas(
        vocab_size=hp['vocab_size'], d_model=hp['d_model'],
        num_heads=hp['num_heads'],   d_ff=hp['d_ff'],
        num_layers=hp['num_layers'], num_classes=hp['num_classes'],
        max_len=hp['max_len']
    ).to(device)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo.eval()
    print(f"[OK] Modelo cargado desde '{ruta}'")
    return modelo


# ==============================================================================
# 7. FUNCIÓN DE PREDICCIÓN
# ==============================================================================

def predecir_pico_placa(
    placa: str,
    model: TransformerPlacas = None,
    device: torch.device = None,
    verbose: bool = True
) -> dict:
    """
    Predice el día de Pico y Placa para una placa dada.

    Parámetros:
      placa   : string de la placa (ej. 'SKY424', 'abc12D')
      model   : instancia del Transformer (si None, busca checkpoint en disco)
      device  : torch.device (si None, detecta automáticamente)
      verbose : imprime tabla con probabilidades por día

    Retorna: dict con claves placa, restriccion, confianza_pct, probabilidades
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = cargar_modelo(device=device)

    model.eval()
    placa_clean = re.sub(r'[^A-Z0-9]', '', placa.upper())
    entrada     = torch.tensor(
        [tokenizar_placa(placa_clean).tolist()], dtype=torch.long
    ).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(entrada), dim=1).squeeze()
        clase = probs.argmax().item()
        dia   = idx2label[clase]
        conf  = probs[clase].item() * 100

    prob_por_dia = {
        DIAS_UNICOS[i]: round(probs[i].item() * 100, 2)
        for i in range(NUM_CLASES)
    }

    if verbose:
        print("\n" + "=" * 50)
        print(f"  PLACA       : {placa_clean}")
        print(f"  RESTRICCIÓN : {dia}")
        print(f"  CONFIANZA   : {conf:.2f}%")
        print("  Distribución:")
        for d, p in sorted(prob_por_dia.items(), key=lambda x: -x[1]):
            barra = "█" * int(p / 5)
            print(f"    {d:<12}: {p:>6.2f}%  {barra}")
        print("=" * 50)

    return {
        'placa'         : placa_clean,
        'restriccion'   : dia,
        'confianza_pct' : round(conf, 2),
        'probabilidades': prob_por_dia
    }


# ==============================================================================
# 8. PIPELINE COMPLETO DEL MÓDULO 4
# ==============================================================================

def ejecutar_modulo4(
    datos_raw: list[tuple],
    epochs: int    = 30,
    batch_size: int = 256,
    paciencia: int  = 7,
    ruta_modelo: str = RUTA_MODELO_DEFAULT
) -> TransformerPlacas:
    """
    Ejecuta el pipeline completo: dataset → entrenamiento → evaluación → guardado.

    Parámetro:
      datos_raw — lista de (tokens, label) del Módulo 3

    Retorna: modelo entrenado listo para inferencia.
    """
    print("\n" + "=" * 70)
    print("MÓDULO 4 — TRANSFORMER INTELIGENTE (PyTorch)")
    print("=" * 70)

    device = verificar_dispositivo()

    train_loader, val_loader, test_loader = preparar_dataloaders(
        datos_raw, batch_size=batch_size
    )

    model = TransformerPlacas().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nArquitectura: d_model=64 | heads=4 | d_ff=256 | layers=2")
    print(f"Parámetros totales: {total_params:,}")

    historial = entrenar(model, train_loader, val_loader, device,
                         epochs=epochs, paciencia=paciencia)

    print("\n" + "=" * 70)
    print("EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA")
    print("=" * 70)
    test_acc, preds, true = evaluar_test(model, test_loader, device)

    graficar_entrenamiento(historial, test_acc)
    graficar_confusion(preds, true, test_acc)

    guardar_modelo(model, ruta_modelo)

    # Pruebas rápidas
    print("\n--- PRUEBAS DE PREDICCIÓN ---")
    for placa in ["SKY424", "ABC129", "MLT567", "HSY095", "ZZZ001", "PQR830"]:
        predecir_pico_placa(placa, model, device)

    print(f"\n[OK] Módulo 4 completado. Precisión test: {test_acc*100:.2f}%")
    return model


# ==============================================================================
# 9. PRUEBA RÁPIDA STANDALONE
# ==============================================================================
if __name__ == "__main__":
    # Genera un mini-dataset sintético para probar el módulo sin dependencias
    from modulo3_dataset import preparar_datos_transformer

    datos = preparar_datos_transformer(n_sintetico=5_000)
    modelo = ejecutar_modulo4(datos, epochs=5, paciencia=3)