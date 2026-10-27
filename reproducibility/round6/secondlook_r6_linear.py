# Round-6: cross-attention, late fusion and a linear probe on both backbones, each with its learning rate chosen
# on validation from one grid, on OASIS-1 (participant split) and on RetinaMNIST-224 -> APTOS 2019.
# Round-3 experiments on data without the problems of AD5 and of 28x28 MedMNIST:
#   oasis_subject : OASIS-1 (Kaggle slice-level release), one MRI session per participant (mpr-1), three classes by
#                   Clinical Dementia Rating (0, 0.5, >=1), split 60/20/20 BY PARTICIPANT with a different draw per seed,
#                   so no participant contributes slices to more than one partition.
#   retina224, derma224 : RetinaMNIST and DermaMNIST from MedMNIST+ at 224x224 (official splits).
# Phase 1 is trained exactly as in the archived notebook (frozen ConvNeXt-Small and Swin-Small, cross-attention head,
# focal + SupCon, AdamW 5e-5, batch 16, 5 epochs, best validation balanced accuracy), here on cached backbone features,
# which gives the same model because the frozen backbones have no stochastic layers. Single-backbone and late-fusion
# heads are trained the same way. Probabilities and raw Grad-CAM maps of the hybrid and of the ConvNeXt-only model are
# saved for the offline review.
import os, sys, json, time, random, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "grad-cam", "medmnist"], check=False)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm, medmnist
from medmnist import INFO
from scipy.ndimage import binary_dilation
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

OUT = "/kaggle/working"; ARR = os.path.join(OUT, "arrays"); os.makedirs(ARR, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = open(os.path.join(OUT, "log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
SEEDS = [42, 123, 2026]
RESULTS = {}

# ------------------------------------------------------------------ frozen backbones
cnn = timm.create_model("convnext_small", pretrained=True, num_classes=0, global_pool="").to(DEV).eval()
swin = timm.create_model("swin_small_patch4_window7_224", pretrained=True, num_classes=0, global_pool="").to(DEV).eval()
for p in list(cnn.parameters()) + list(swin.parameters()): p.requires_grad_(False)

@torch.no_grad()
def extract(ds):
    C, S = [], []
    for x, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4):
        x = x.to(DEV)
        C.append(cnn.forward_features(x).flatten(2).permute(0, 2, 1).half().cpu())
        S.append(swin.forward_features(x).flatten(1, 2).half().cpu())
    return torch.cat(C), torch.cat(S)

# ------------------------------------------------------------------ heads (Phase-1 width and layer names)
class CrossAttentionFusionModule(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, ff_dim=384, dropout_rate=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(ff_dim, embed_dim))
        self.norm1, self.norm2, self.dropout = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.Dropout(dropout_rate)
    def forward(self, c, t):
        a, _ = self.mha(query=t, key=c, value=c)
        x = self.norm1(t + self.dropout(a))
        return self.norm2(x + self.dropout(self.ffn(x)))

class Head(nn.Module):
    def __init__(self, kind, K, d=384):
        super().__init__(); self.kind = kind
        if kind in ("xattn", "late", "cnn"): self.cnn_proj = nn.Linear(768, d)
        if kind in ("xattn", "late", "swin"): self.transformer_proj = nn.Linear(768, d)
        if kind == "xattn": self.fusion_module = CrossAttentionFusionModule(d)
        w = 2 * d if kind == "late" else d
        self.post_fusion_norm = nn.LayerNorm(w)
        self.projection_head = nn.Sequential(nn.Linear(w, d), nn.ReLU(), nn.Linear(d, 128))
        self.classifier = nn.Linear(w, K)
    def forward(self, c, s):
        if self.kind == "xattn":
            z = self.post_fusion_norm(self.fusion_module(self.cnn_proj(c), self.transformer_proj(s))).mean(1)
        elif self.kind == "late":
            z = self.post_fusion_norm(torch.cat([self.cnn_proj(c).mean(1), self.transformer_proj(s).mean(1)], 1))
        elif self.kind == "cnn":
            z = self.post_fusion_norm(self.cnn_proj(c)).mean(1)
        else:
            z = self.post_fusion_norm(self.transformer_proj(s)).mean(1)
        return self.classifier(z), self.projection_head(z)

class Full(nn.Module):
    """Backbones + trained head, returning logits (used for probabilities and Grad-CAM)."""
    def __init__(self, head):
        super().__init__(); self.cnn_backbone, self.transformer_backbone, self.head = cnn, swin, head
    def forward(self, x):
        c = self.cnn_backbone.forward_features(x).flatten(2).permute(0, 2, 1)
        s = self.transformer_backbone.forward_features(x).flatten(1, 2) if self.head.kind != "cnn" else None
        return self.head(c, s)[0]

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

@torch.no_grad()
def head_predict(head, C, S):
    head.eval(); out = []
    for i in range(0, len(C), 256):
        out.append(F.softmax(head(C[i:i + 256].to(DEV).float(), S[i:i + 256].to(DEV).float())[0], 1).cpu())
    return torch.cat(out).numpy()

def metrics(y, p):
    return dict(acc=float(accuracy_score(y, p)), ba=float(balanced_accuracy_score(y, p)),
                f1=float(f1_score(y, p, average="macro", zero_division=0)))

def train_head(kind, loss, K, Ctr, Str, Ytr, Cva, Sva, Yva, seed, epochs=5):
    set_seed(seed); head = Head(kind, K).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=5e-5); g = torch.Generator(); g.manual_seed(seed)
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, epochs + 1):
        head.train()
        for idx in torch.randperm(len(Ytr), generator=g).split(16):
            c, s, y = Ctr[idx].to(DEV).float(), Str[idx].to(DEV).float(), Ytr[idx].to(DEV)
            logits, emb = head(c, s)
            l = F.cross_entropy(logits, y, label_smoothing=0.1) if loss == "ce" else 0.5 * focal(logits, y) + 0.5 * supcon(emb, y)
            opt.zero_grad(); l.backward(); opt.step()
        v = balanced_accuracy_score(Yva.numpy(), head_predict(head, Cva, Sva).argmax(1))
        if v > best: best, best_ep, best_state = v, ep, {k: t.detach().clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    return head, best_ep, float(best)

class ListDS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]; return TF(Image.open(p).convert("RGB")), y

class ArrDS(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        img, lab = self.d[i]; return TF(img.convert("RGB")), int(np.asarray(lab).reshape(-1)[0])


# ------------------------------------------------------------------ one learning-rate grid for three heads
class LinearBoth(nn.Module):
    kind = "linboth"
    def __init__(self, K):
        super().__init__(); self.fc = nn.Linear(1536, K)
    def forward(self, c, s):
        return self.fc(torch.cat([c.mean(1), s.mean(1)], 1)), None
def fit(kind, K, Ctr, Str, Ytr, Cva, Sva, Yva, seed, lr):
    set_seed(seed)
    head = (LinearBoth(K) if kind == "linboth" else Head(kind, K)).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr); g = torch.Generator(); g.manual_seed(seed)
    best, state = -1.0, None
    for ep in range(5):
        head.train()
        for idx in torch.randperm(len(Ytr), generator=g).split(16):
            logits = head(Ctr[idx].to(DEV).float(), Str[idx].to(DEV).float())[0]
            l = F.cross_entropy(logits, Ytr[idx].to(DEV), label_smoothing=0.1)
            opt.zero_grad(); l.backward(); opt.step()
        v = balanced_accuracy_score(Yva.numpy(), head_predict(head, Cva, Sva).argmax(1))
        if v > best: best, state = v, {k: t.detach().clone() for k, t in head.state_dict().items()}
    head.load_state_dict(state); return head, best
GRID = (5e-5, 2e-4, 1e-3, 1e-2)
def tuned(kind, K, tr_, va_, seed):
    runs = [fit(kind, K, *tr_, *va_, seed, lr) for lr in GRID]
    return max(runs, key=lambda r: r[1])[0]
RESULTS = {}
def find_oasis():
    for d, sub, _ in os.walk("/kaggle/input"):
        low = [s.lower() for s in sub]
        if any("non" in s and "demented" in s for s in low) and any("very" in s for s in low):
            return d
    raise RuntimeError("OASIS folders not found")
root = find_oasis(); log("OASIS root", root, sorted(os.listdir(root)))
def cdr_class(folder):
    f = folder.lower()
    if "non" in f: return 0
    if "very" in f: return 1
    return 2                                   # Mild (CDR 1) and Moderate (CDR 2) dementia
per_subject = {}
for folder in sorted(os.listdir(root)):
    if not os.path.isdir(os.path.join(root, folder)): continue
    for fn in os.listdir(os.path.join(root, folder)):
        parts = fn.split("_")
        if len(parts) < 5 or not fn.lower().endswith((".jpg", ".png")): continue
        subj, mpr = parts[0] + "_" + parts[1], parts[3]
        per_subject.setdefault(subj, {}).setdefault((cdr_class(folder), folder, mpr), []).append(os.path.join(root, folder, fn))
items, groups, conflicts = [], [], 0
subjects = sorted(per_subject)
subj_class = {}
for si, subj in enumerate(subjects):
    keys = per_subject[subj]
    if len({k[0] for k in keys}) > 1: conflicts += 1
    key = sorted(keys, key=lambda k: (k[2], k[1]))[0]          # first acquisition (mpr-1 where present)
    subj_class[subj] = key[0]
    for p in sorted(keys[key]): items.append((p, key[0])); groups.append(si)
groups = np.array(groups); Y_oas = torch.tensor([y for _, y in items])
cls_subj = {c: [s for s in subjects if subj_class[s] == c] for c in range(3)}
log(f"OASIS: {len(subjects)} participants ({ {c: len(v) for c, v in cls_subj.items()} }), {len(items)} slices "
    f"({ {c: int((Y_oas == c).sum()) for c in range(3)} }), participants with conflicting labels: {conflicts}")
RESULTS["oasis_info"] = dict(participants={c: len(v) for c, v in cls_subj.items()}, slices={c: int((Y_oas == c).sum()) for c in range(3)},
                             conflicts=conflicts)
ds_oas = ListDS(items)
t0 = time.time(); C_oas, S_oas = extract(ds_oas); log(f"OASIS features in {time.time()-t0:.0f}s")
sidx = {s: i for i, s in enumerate(subjects)}
def oasis_splits(seed):
    rng = np.random.default_rng(seed); tr_s, va_s, te_s = [], [], []
    for c in range(3):
        s = list(rng.permutation(cls_subj[c])); n = len(s)
        n_te = max(1, int(round(0.2 * n))); n_va = max(1, int(round(0.2 * n)))
        te_s += s[:n_te]; va_s += s[n_te:n_te + n_va]; tr_s += s[n_te + n_va:]
    pick = lambda ss: np.where(np.isin(groups, [sidx[x] for x in ss]))[0]
    return pick(tr_s), pick(va_s), pick(te_s)

RESULTS["oasis_subject"] = {}
for seed in SEEDS:
    tr, va, te = oasis_splits(seed); r = {}
    for kind in ("xattn", "late", "linboth"):
        h = tuned(kind, 3, (C_oas[tr], S_oas[tr], Y_oas[tr]), (C_oas[va], S_oas[va], Y_oas[va]), seed)
        r[kind] = metrics(Y_oas[te].numpy(), head_predict(h, C_oas[te], S_oas[te]).argmax(1))
    RESULTS["oasis_subject"][str(seed)] = r; log("oasis", seed, {k: round(v["ba"], 4) for k, v in r.items()})
    json.dump(RESULTS, open(os.path.join(OUT, "results.json"), "w"))
del C_oas, S_oas
os.makedirs("/tmp/medmnist", exist_ok=True)
import glob
def find_aptos():
    for d, sub, files in os.walk("/kaggle/input"):
        low = [s.lower() for s in sub]
        if any("no_dr" in s for s in low) and any("prolif" in s for s in low):
            return d
    raise RuntimeError("APTOS folders not found")
root = find_aptos(); log("APTOS root", root, sorted(os.listdir(root)))
GRADE = {"no_dr": 0, "mild": 1, "moderate": 2, "severe": 3, "proliferate_dr": 4}
items = []
for folder in sorted(os.listdir(root)):
    g = GRADE.get(folder.lower())
    if g is None: continue
    for fn in sorted(os.listdir(os.path.join(root, folder))):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")): items.append((os.path.join(root, folder, fn), g))
log("APTOS images", len(items), {g: sum(1 for _, y in items if y == g) for g in range(5)})
os.makedirs("/tmp/medmnist", exist_ok=True)
info = INFO["retinamnist"]; DC = getattr(medmnist, info["python_class"])
parts = {sp: DC(split=sp, download=True, root="/tmp/medmnist", size=224) for sp in ("train", "val", "test")}
ds_ret = torch.utils.data.ConcatDataset([ArrDS(parts[sp]) for sp in ("train", "val", "test")])
n = {sp: len(parts[sp]) for sp in parts}
Y_ret = torch.tensor([int(np.asarray(parts[sp].labels[i]).reshape(-1)[0]) for sp in ("train", "val", "test") for i in range(n[sp])])
ds_apt = ListDS(items); Y_apt = torch.tensor([y for _, y in items])
C_r, S_r = extract(ds_ret); C_a, S_a = extract(ds_apt)

tr = np.arange(n["train"]); va = np.arange(n["train"], n["train"] + n["val"]); te = np.arange(n["train"] + n["val"], len(Y_ret))
RESULTS["aptos"] = {}
for seed in SEEDS:
    r = {}
    for kind in ("xattn", "late", "linboth"):
        h = tuned(kind, 5, (C_r[tr], S_r[tr], Y_ret[tr]), (C_r[va], S_r[va], Y_ret[va]), seed)
        r[kind] = dict(internal=metrics(Y_ret[te].numpy(), head_predict(h, C_r[te], S_r[te]).argmax(1)), aptos=metrics(Y_apt.numpy(), head_predict(h, C_a, S_a).argmax(1)))
    RESULTS["aptos"][str(seed)] = r; log("aptos", seed, {k: (round(v["internal"]["ba"], 4), round(v["aptos"]["ba"], 4)) for k, v in r.items()})
    json.dump(RESULTS, open(os.path.join(OUT, "results.json"), "w"))
log("RESULTSJSON " + json.dumps(RESULTS)); log("DONE")
