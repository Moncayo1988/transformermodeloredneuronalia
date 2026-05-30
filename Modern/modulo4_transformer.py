# ==============================================================================
# MÓDULO 4 — TRANSFORMER DESDE CERO EN PyTorch
# Autor: Salomón Melenje
# Arquitectura: tokens(B,7) → Embedding → PositionalEncoding → 2×EncoderBlock → MeanPool → FC(5 días)
# ==============================================================================

import math, re, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from modulo0_config import (
    VOCAB_SIZE, MAX_LEN, NUM_CLASES, DIAS_UNICOS,
    label2idx, idx2label, char2idx, idx2char, tokenizar_placa
)

# Ruta del modelo
_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
_DIR_REPO = _DIR if os.path.isdir(os.path.join(_DIR,"modelos")) else os.path.dirname(_DIR)
RUTA_MODELO_DEFAULT = os.path.join(_DIR_REPO, "modelos", "transformer_pico_placa.pt")


def verificar_dispositivo() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GPU] Dispositivo: {device}")
    if device.type == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)} | "
              f"Mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    else: print("      [AVISO] Sin GPU — entrenando en CPU.")
    return device


# ==============================================================================
# DATASET
# ==============================================================================

class PlacaDataset(Dataset):
    def __init__(self, datos: list[tuple]):
        self.tokens = torch.tensor([d[0] for d in datos], dtype=torch.long)
        self.labels = torch.tensor([d[1] for d in datos], dtype=torch.long)
    def __len__(self): return len(self.tokens)
    def __getitem__(self, idx): return self.tokens[idx], self.labels[idx]


def preparar_dataloaders(datos_raw, batch_size=256, train_frac=0.80, val_frac=0.20, semilla=42):
    pin = torch.cuda.is_available()
    ds  = PlacaDataset(datos_raw); n = len(ds)
    n_train = int(n*train_frac); n_test = n-n_train
    train_ds, test_ds = random_split(ds,[n_train,n_test],generator=torch.Generator().manual_seed(semilla))
    n_val = int(n_train*val_frac); n_tr = n_train-n_val
    train_sub, val_sub = random_split(train_ds,[n_tr,n_val],generator=torch.Generator().manual_seed(0))
    kw = dict(batch_size=batch_size, pin_memory=pin, num_workers=2)
    print(f"DataLoaders — Train:{n_tr} | Val:{n_val} | Test:{n_test} | Batch:{batch_size}")
    return (DataLoader(train_sub, shuffle=True, **kw),
            DataLoader(val_sub,   shuffle=False, **kw),
            DataLoader(test_ds,   shuffle=False, **kw))


# ==============================================================================
# ARQUITECTURA
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LEN, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1), :])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads; self.d_k = d_model//num_heads
        self.W_Q = nn.Linear(d_model,d_model,bias=False)
        self.W_K = nn.Linear(d_model,d_model,bias=False)
        self.W_V = nn.Linear(d_model,d_model,bias=False)
        self.W_O = nn.Linear(d_model,d_model,bias=False)
        self.dropout = nn.Dropout(dropout); self.scale = math.sqrt(self.d_k)
    def _split(self, x, B): return x.view(B,-1,self.num_heads,self.d_k).transpose(1,2)
    def forward(self, x, mask=None):
        B  = x.size(0)
        Q,K,V = self._split(self.W_Q(x),B), self._split(self.W_K(x),B), self._split(self.W_V(x),B)
        s = torch.matmul(Q,K.transpose(-2,-1))/self.scale
        if mask is not None: s = s.masked_fill(mask==0,float('-inf'))
        ctx = torch.matmul(self.dropout(torch.softmax(s,dim=-1)), V)
        return self.W_O(ctx.transpose(1,2).contiguous().view(B,-1,self.num_heads*self.d_k))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.drop  = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.attention(x, mask)))
        return self.norm2(x + self.drop(self.ffn(x)))

class TransformerPlacas(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, num_heads=4, d_ff=256,
                 num_layers=2, num_classes=NUM_CLASES, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.embedding      = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding   = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_blocks = nn.ModuleList([TransformerEncoderBlock(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.classifier     = nn.Sequential(nn.Dropout(0.3),nn.Linear(d_model,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,num_classes))
    def forward(self, x, mask=None):
        out = self.pos_encoding(self.embedding(x))
        for blk in self.encoder_blocks: out = blk(out, mask)
        pad  = (x != 0).unsqueeze(-1).float()
        pooled = (out*pad).sum(dim=1) / pad.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


# ==============================================================================
# ENTRENAMIENTO
# ==============================================================================

def _epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train(); total_loss = correctos = total = 0
    for tok, lbl in loader:
        tok, lbl = tok.to(device,non_blocking=True), lbl.to(device,non_blocking=True)
        optimizer.zero_grad(); logits = model(tok); loss = criterion(logits, lbl)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item()*tok.size(0); correctos += (logits.argmax(1)==lbl).sum().item(); total += tok.size(0)
    return total_loss/total, correctos/total

def _evaluar(model, loader, criterion, device):
    model.eval(); total_loss = correctos = total = 0
    with torch.no_grad():
        for tok, lbl in loader:
            tok, lbl = tok.to(device,non_blocking=True), lbl.to(device,non_blocking=True)
            logits = model(tok); loss = criterion(logits, lbl)
            total_loss += loss.item()*tok.size(0); correctos += (logits.argmax(1)==lbl).sum().item(); total += tok.size(0)
    return total_loss/total, correctos/total

def entrenar(model, train_loader, val_loader, device, epochs=30, lr=3e-4, paciencia=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3,
                steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.3)
    historial = {'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
    mejor_acc = sin_mejora = 0; mejor_estado = None
    print(f"\nEntrenando {epochs} épocas (early stop={paciencia})...\n")
    print(f"{'Época':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'LR':>10}")
    print("─"*65)
    for epoch in range(1, epochs+1):
        tl, ta = _epoch(model, train_loader, criterion, optimizer, scheduler, device)
        vl, va = _evaluar(model, val_loader, criterion, device)
        for k, v in zip(['train_loss','train_acc','val_loss','val_acc'],[tl,ta,vl,va]): historial[k].append(v)
        print(f"{epoch:>6} | {tl:>10.4f} | {ta*100:>8.2f}% | {vl:>9.4f} | {va*100:>7.2f}% | {scheduler.get_last_lr()[0]:>10.2e}")
        if va > mejor_acc:
            mejor_acc = va; mejor_estado = {k:v.cpu().clone() for k,v in model.state_dict().items()}; sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia: print(f"\n[Early Stop] Sin mejora por {paciencia} épocas."); break
    if mejor_estado:
        model.load_state_dict({k:v.to(device) for k,v in mejor_estado.items()})
        print(f"\n[OK] Mejor checkpoint restaurado (val_acc={mejor_acc*100:.2f}%)")
    historial['mejor_val_acc'] = mejor_acc
    return historial


# ==============================================================================
# EVALUACIÓN Y MÉTRICAS
# ==============================================================================

def evaluar_test(model, test_loader, device):
    test_loss, test_acc = _evaluar(model, test_loader, nn.CrossEntropyLoss(), device)
    model.eval(); preds, true = [], []
    with torch.no_grad():
        for tok, lbl in test_loader:
            preds.extend(model(tok.to(device)).argmax(1).cpu().numpy()); true.extend(lbl.numpy())
    preds, true = np.array(preds), np.array(true)
    n_err = (preds!=true).sum()
    print(f"Pérdida: {test_loss:.4f} | Precisión: {test_acc*100:.2f}% | Errores: {n_err}/{len(true)}")
    return test_acc, preds, true

def graficar_entrenamiento(historial, test_acc=None):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].plot([a*100 for a in historial['train_acc']], label='Train', lw=2)
    axes[0].plot([a*100 for a in historial['val_acc']],   label='Val',   lw=2, ls='--')
    if test_acc: axes[0].axhline(y=test_acc*100, color='red', ls=':', label=f'Test:{test_acc*100:.2f}%')
    axes[0].set(title="Precisión", xlabel="Época", ylabel="Acc (%)", ylim=[85,101]); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(historial['train_loss'], label='Train', lw=2)
    axes[1].plot(historial['val_loss'],   label='Val',   lw=2, ls='--')
    axes[1].set(title="Pérdida (CrossEntropy)", xlabel="Época"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def graficar_confusion(preds, true, test_acc):
    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DIAS_UNICOS, yticklabels=DIAS_UNICOS, linewidths=0.5)
    plt.title(f"Matriz de Confusión — {test_acc*100:.2f}%"); plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.tight_layout(); plt.show()
    print("\nREPORTE:\n" + classification_report(true, preds, target_names=DIAS_UNICOS))


# ==============================================================================
# GUARDAR / CARGAR
# ==============================================================================

def guardar_modelo(model, ruta=RUTA_MODELO_DEFAULT, hiperparametros=None):
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
    hp = hiperparametros or {'vocab_size':VOCAB_SIZE,'d_model':64,'num_heads':4,'d_ff':256,'num_layers':2,'num_classes':NUM_CLASES,'max_len':MAX_LEN}
    torch.save({'model_state_dict':model.state_dict(),'char2idx':char2idx,'idx2char':idx2char,'label2idx':label2idx,'idx2label':idx2label,'hyperparams':hp}, ruta)
    print(f"[OK] Modelo guardado: '{ruta}'")

def cargar_modelo(ruta=RUTA_MODELO_DEFAULT, device=None):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ruta):
        cwd = os.path.join(os.getcwd(), os.path.basename(ruta))
        if os.path.exists(cwd): ruta = cwd
        else: raise FileNotFoundError(f"Modelo no encontrado: '{ruta}'")
    try:    ck = torch.load(ruta, map_location=device, weights_only=True)
    except TypeError: ck = torch.load(ruta, map_location=device)
    hp = ck['hyperparams']
    model = TransformerPlacas(**{k:hp[k] for k in ['vocab_size','d_model','num_heads','d_ff','num_layers','num_classes','max_len']}).to(device)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    print(f"[OK] Modelo cargado: '{ruta}'")
    return model


# ==============================================================================
# PREDICCIÓN
# ==============================================================================

def predecir_pico_placa(placa: str, model=None, device=None, verbose=True) -> dict:
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None: model = cargar_modelo(device=device)
    model.eval()
    entrada = torch.tensor([tokenizar_placa(re.sub(r'[^A-Z0-9]','',placa.upper())).tolist()], dtype=torch.long).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(entrada),dim=1).squeeze()
        clase = probs.argmax().item(); dia = idx2label[clase]; conf = probs[clase].item()*100
    prob_por_dia = {DIAS_UNICOS[i]: round(probs[i].item()*100,2) for i in range(NUM_CLASES)}
    if verbose:
        print(f"\n  PLACA: {placa.upper()} | RESTRICCIÓN: {dia} | CONFIANZA: {conf:.2f}%")
        for d, p in sorted(prob_por_dia.items(), key=lambda x: -x[1]):
            print(f"    {d:<12}: {p:>6.2f}%  {'█'*int(p/5)}")
    return {'placa':placa.upper(),'restriccion':dia,'confianza_pct':round(conf,2),'probabilidades':prob_por_dia}


# ==============================================================================
# PIPELINE COMPLETO
# ==============================================================================

def ejecutar_modulo4(datos_raw, epochs=30, batch_size=256, paciencia=7, ruta_modelo=RUTA_MODELO_DEFAULT):
    print("\n"+"="*70+"\nMÓDULO 4 — TRANSFORMER INTELIGENTE (PyTorch)\n"+"="*70)
    device = verificar_dispositivo()
    train_l, val_l, test_l = preparar_dataloaders(datos_raw, batch_size=batch_size)
    model = TransformerPlacas().to(device)
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    historial = entrenar(model, train_l, val_l, device, epochs=epochs, paciencia=paciencia)
    test_acc, preds, true = evaluar_test(model, test_l, device)
    graficar_entrenamiento(historial, test_acc); graficar_confusion(preds, true, test_acc)
    guardar_modelo(model, ruta_modelo)
    for p in ["SKY424","ABC129","MLT567","HSY095","ZZZ001","PQR830"]:
        predecir_pico_placa(p, model, device)
    print(f"\n[OK] Módulo 4 completado. Precisión test: {test_acc*100:.2f}%")
    return model

if __name__ == "__main__":
    from modulo3_dataset import preparar_datos_transformer
    ejecutar_modulo4(preparar_datos_transformer(n_sintetico=5_000), epochs=5, paciencia=3)