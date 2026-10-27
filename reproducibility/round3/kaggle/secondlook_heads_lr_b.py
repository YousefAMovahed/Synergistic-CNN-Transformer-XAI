# Matched short-budget comparison on frozen ImageNet backbones (ConvNeXt-Small, Swin-Small): the Phase-1
# cross-attention head against single-backbone heads and a late-fusion head, all trained with the same loss,
# optimizer, budget and checkpoint rule on the same cached backbone features; plus cross-entropy variants of
# the cross-attention and late-fusion heads, which separate the fusion from the loss.
# Datasets: AD5 (random 80/20 split per seed, as in the archived notebook) and the four MedMNIST tasks
# (official splits, 28x28 images resized to 224). Round 3: ten seeds (the first three reproduce the earlier run),
# the loss weight alpha of the Phase-1 objective (0.25, 0.5, 0.75, 1.0 = focal only) for the cross-attention head,
# and linear probes on the pooled features of each backbone and of both (learning rate chosen on validation).
import os, sys, json, time, random, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "medmnist"], check=False)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm, medmnist
from medmnist import INFO
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

OUT = "/kaggle/working"; DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = open(os.path.join(OUT, "log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ------------------------------------------------------------------ frozen backbones and feature cache
cnn = timm.create_model("convnext_small", pretrained=True, num_classes=0, global_pool="").to(DEV).eval()
swin = timm.create_model("swin_small_patch4_window7_224", pretrained=True, num_classes=0, global_pool="").to(DEV).eval()

@torch.no_grad()
def extract(ds):
    C, S, Y = [], [], []
    for x, y in DataLoader(ds, batch_size=64, shuffle=False, num_workers=2):
        x = x.to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            c = cnn.forward_features(x).flatten(2).permute(0, 2, 1)          # (B, 49, 768), as in Phase 1
            s = swin.forward_features(x).flatten(1, 2)                       # (B, 49, 768), as in Phase 1
        C.append(c.half().cpu()); S.append(s.half().cpu()); Y.append(torch.as_tensor(np.asarray(y)).reshape(-1))
    return torch.cat(C), torch.cat(S), torch.cat(Y).long()

class MedMNISTDataset(Dataset):
    def __init__(self, d, tf): self.d, self.tf = d, tf
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        img, lab = self.d[i]; return self.tf(img), int(np.asarray(lab).reshape(-1)[0])

# ------------------------------------------------------------------ heads (same width as the Phase-1 head)
class CrossAttentionFusionModule(nn.Module):
    def __init__(self, d=384, h=12, ff=384, p=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
        self.norm1, self.norm2, self.dropout = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(p)
    def forward(self, c, t):
        a, _ = self.mha(query=t, key=c, value=c)
        x = self.norm1(t + self.dropout(a))
        return self.norm2(x + self.dropout(self.ffn(x)))

class Head(nn.Module):
    def __init__(self, kind, K, d=384):
        super().__init__(); self.kind = kind
        if kind in ("xattn", "late", "cnn"): self.cnn_proj = nn.Linear(768, d)
        if kind in ("xattn", "late", "swin"): self.swin_proj = nn.Linear(768, d)
        if kind == "xattn": self.fusion = CrossAttentionFusionModule(d)
        w = 2 * d if kind == "late" else d
        self.norm = nn.LayerNorm(w)
        self.proj = nn.Sequential(nn.Linear(w, d), nn.ReLU(), nn.Linear(d, 128))
        self.cls = nn.Linear(w, K)
    def forward(self, c, s):
        if self.kind == "xattn":
            z = self.norm(self.fusion(self.cnn_proj(c), self.swin_proj(s))).mean(1)
        elif self.kind == "late":
            z = self.norm(torch.cat([self.cnn_proj(c).mean(1), self.swin_proj(s).mean(1)], 1))
        elif self.kind == "cnn":
            z = self.norm(self.cnn_proj(c)).mean(1)
        else:
            z = self.norm(self.swin_proj(s)).mean(1)
        return self.cls(z), self.proj(z)

def focal(logits, y, a=0.25, g=2.0, ls=0.1):
    ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=ls); pt = torch.exp(-ce)
    return (a * (1 - pt) ** g * ce).mean()

def supcon(f, y, t=0.07):
    f = F.normalize(f, dim=1); y = y.view(-1, 1)
    m = torch.eq(y, y.T).float(); sim = f @ f.T / t
    sim = sim - sim.max(1, keepdim=True)[0].detach()
    eye = torch.eye(f.shape[0], device=f.device, dtype=torch.bool); m.masked_fill_(eye, 0)
    ex = torch.exp(sim) * (~eye).float(); lp = sim - torch.log(ex.sum(1, keepdim=True) + 1e-9)
    return -((m * lp).sum(1) / (m.sum(1) + 1e-9)).mean()

def evaluate(head, C, S, Y):
    head.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(Y), 256):
            P.append(head(C[i:i + 256].to(DEV).float(), S[i:i + 256].to(DEV).float())[0].argmax(1).cpu())
    p = torch.cat(P).numpy(); y = Y.numpy()
    return dict(acc=float(accuracy_score(y, p)), ba=float(balanced_accuracy_score(y, p)),
                f1=float(f1_score(y, p, average="macro", zero_division=0)))

def train_head(kind, loss, K, tr, va, te, seed, epochs=5, alpha=0.5, lr=5e-5):
    set_seed(seed)
    head = Head(kind, K).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    Ctr, Str, Ytr = tr; g = torch.Generator(); g.manual_seed(seed)
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, epochs + 1):
        head.train()
        for idx in torch.randperm(len(Ytr), generator=g).split(16):
            c, s, y = Ctr[idx].to(DEV).float(), Str[idx].to(DEV).float(), Ytr[idx].to(DEV)
            logits, emb = head(c, s)
            if loss == "ce": l = F.cross_entropy(logits, y, label_smoothing=0.1)
            elif alpha >= 1.0: l = focal(logits, y)
            else: l = alpha * focal(logits, y) + (1 - alpha) * supcon(emb, y)
            opt.zero_grad(); l.backward(); opt.step()
        v = evaluate(head, *va)["ba"]
        if v > best: best, best_ep, best_state = v, ep, {k: t.detach().clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    r = evaluate(head, *te); r.update(best_epoch=best_ep, val_ba=best, params=int(sum(p.numel() for p in head.parameters())))
    return r

class Linear(nn.Module):
    """Linear probe on the mean-pooled tokens of one backbone or of both (concatenated)."""
    def __init__(self, which, K):
        super().__init__(); self.which = which
        self.fc = nn.Linear(1536 if which == "both" else 768, K)
    def forward(self, c, s):
        z = {"cnn": lambda: c.mean(1), "swin": lambda: s.mean(1), "both": lambda: torch.cat([c.mean(1), s.mean(1)], 1)}[self.which]()
        return self.fc(z), None

def train_linear(which, K, tr, va, te, seed, epochs=5):
    best_overall = None
    for lr in (1e-4, 1e-3, 1e-2):
        set_seed(seed); head = Linear(which, K).to(DEV)
        opt = torch.optim.AdamW(head.parameters(), lr=lr)
        Ctr, Str, Ytr = tr; g = torch.Generator(); g.manual_seed(seed)
        best, best_state, best_ep = -1.0, None, 0
        for ep in range(1, epochs + 1):
            head.train()
            for idx in torch.randperm(len(Ytr), generator=g).split(16):
                l = F.cross_entropy(head(Ctr[idx].to(DEV).float(), Str[idx].to(DEV).float())[0], Ytr[idx].to(DEV))
                opt.zero_grad(); l.backward(); opt.step()
            v = evaluate(head, *va)["ba"]
            if v > best: best, best_ep, best_state = v, ep, {k: t.detach().clone() for k, t in head.state_dict().items()}
        if best_overall is None or best > best_overall[0]: best_overall = (best, best_ep, lr, best_state)
    head = Linear(which, K).to(DEV); head.load_state_dict(best_overall[3])
    r = evaluate(head, *te); r.update(best_epoch=best_overall[1], val_ba=best_overall[0], lr=best_overall[2])
    return r

VARIANTS = [("xattn", "combined"), ("xattn", "ce"), ("late", "combined"), ("late", "ce"), ("cnn", "combined"), ("swin", "combined")]
ALPHAS = [0.25, 0.75, 1.0]
SEEDS = [42, 123, 2026, 0, 1, 2, 3, 4, 5, 6]
PART = ['pneumoniamnist', 'dermamnist']
RESULTS = {}

def run_dataset(name, K, splits_for_seed):
    RESULTS[name] = {}
    for seed in SEEDS:
        tr, va, te = splits_for_seed(seed)
        for kind, loss in VARIANTS:
            runs = [dict(train_head(kind, loss, K, tr, va, te, seed, lr=lr), lr=lr) for lr in (5e-5, 2e-4, 1e-3)]
            RESULTS[name].setdefault(f"{kind}|{loss}|tuned", {})[str(seed)] = max(runs, key=lambda r: r["val_ba"])
            RESULTS[name].setdefault(f"{kind}|{loss}|all", {})[str(seed)] = runs
        for which in ("cnn", "swin", "both"):
            RESULTS[name].setdefault(f"linear|{which}", {})[str(seed)] = train_linear(which, K, tr, va, te, seed)
        log(f"{name} seed {seed}: " + ", ".join(f"{k} {v[str(seed)]['ba']:.4f}" for k, v in RESULTS[name].items() if not k.endswith("|all")))
        json.dump(RESULTS, open(os.path.join(OUT, "results.json"), "w"))

# ------------------------------------------------------------------ AD5
def find_src():
    for d, sub, _ in os.walk("/kaggle/input"):
        if "train" in sub and "test" in sub and len(os.listdir(os.path.join(d, "train"))) == 5:
            return d
    raise RuntimeError("AD5 not found")
src = find_src(); tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])
full = ImageFolder(os.path.join(src, "train"), transform=tf); test = ImageFolder(os.path.join(src, "test"), transform=tf)
t0 = time.time(); Cf, Sf, Yf = extract(full); Ct, St, Yt = extract(test); log(f"AD5 features {len(Yf)}+{len(Yt)} in {time.time()-t0:.0f}s")
def ad5_splits(seed):
    set_seed(seed); n = len(full); ntr = int(0.8 * n)
    tr_ds, va_ds = random_split(range(n), [ntr, n - ntr])      # same permutation as random_split on the dataset
    ti, vi = torch.tensor(list(tr_ds.indices)), torch.tensor(list(va_ds.indices))
    return (Cf[ti], Sf[ti], Yf[ti]), (Cf[vi], Sf[vi], Yf[vi]), (Ct, St, Yt)
if "AD5" in PART: run_dataset("AD5", 5, ad5_splits)
del Cf, Sf

# ------------------------------------------------------------------ MedMNIST
os.makedirs("/tmp/medmnist", exist_ok=True)
for flag in [f for f in ("breastmnist", "retinamnist", "pneumoniamnist", "dermamnist") if f in PART]:
    info = INFO[flag]; DC = getattr(medmnist, info["python_class"])
    tl = [transforms.Resize((224, 224)), transforms.ToTensor(), NORM]
    if info["n_channels"] == 1: tl.insert(0, transforms.Grayscale(num_output_channels=3))
    tfm = transforms.Compose(tl); feats = {}
    t0 = time.time()
    for split in ("train", "val", "test"):
        feats[split] = extract(MedMNISTDataset(DC(split=split, download=True, root="/tmp/medmnist"), tfm))
    log(f"{flag} features in {time.time()-t0:.0f}s: " + ", ".join(f"{k} {len(v[2])}" for k, v in feats.items()))
    run_dataset(flag, len(info["label"]), lambda seed, f=feats: (f["train"], f["val"], f["test"]))
    del feats

log("RESULTSJSON " + json.dumps(RESULTS))
log("DONE")
