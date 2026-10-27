# Round-4 AD5 experiments, part A: Phase 1 with cross-entropy; frozen ResNet-50 and EfficientNetV2-S with the Phase-1 head.
import os, sys, json, time, random, shutil, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "grad-cam", "optuna"], check=False)

import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm, optuna
from scipy.ndimage import binary_dilation
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

OUT = "/kaggle/working"
LOG = open(os.path.join(OUT, "log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ------------------------------------------------------------------ data (original cell 9)
def find_src():
    for d, sub, _ in os.walk("/kaggle/input"):
        if "train" in sub and "test" in sub:
            if len(os.listdir(os.path.join(d, "train"))) == 5 and len(os.listdir(os.path.join(d, "test"))) == 5:
                return d
    raise RuntimeError("AD5 train/test folders not found")
base_input_dir = find_src()
base_work_dir = "/tmp/dataset_alzheimer1"
if not os.path.exists(base_work_dir):
    shutil.copytree(base_input_dir, base_work_dir)
train_dir = os.path.join(base_work_dir, "train")
rename_map = {"MildDemented": "Mild", "ModerateDemented": "Moderate", "NonDemented": "None",
              "SevereDemented": "Servere", "VeryMildDemented": "Very Mild"}
for folder in os.listdir(train_dir):
    if os.path.isdir(os.path.join(train_dir, folder)) and folder in rename_map:
        os.rename(os.path.join(train_dir, folder), os.path.join(train_dir, rename_map[folder]))

# ------------------------------------------------------------------ original configuration
class Config:
    DATASET_NAME = 'Alzheimer5Class1'
    DATA_DIR = base_work_dir
    SEED = 42
    IMG_SIZE = 224
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_SAVE_PATH = "/tmp/phase1.pth"
    CNN_MODEL_NAME = "convnext_small"
    SWIN_MODEL_NAME = "swin_small_patch4_window7_224"
    EMBED_DIM = 384
    NUM_HEADS = 12
    FF_DIM = 384
    BATCH_SIZE = 16
    HEAD_TRAIN_EPOCHS = 5
    INITIAL_LR = 5e-5
    LABEL_SMOOTHING = 0.1
    ALPHA = 0.5
    SCL_TEMPERATURE = 0.07
    NUM_WORKERS = 2
    N_TRIALS_OPTIMIZATION = 50

ACTIVATION_PERCENTILE, DILATION_SIZE, CONSENSUS_THRESHOLD = 95, 3, 0.90
TAU_ORIGINAL = 0.6585

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt)**self.gamma * ce).mean()

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        identity_mask = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
        mask.masked_fill_(identity_mask, 0)
        exp_logits = torch.exp(logits) * (~identity_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return -mean_log_prob_pos.mean()

class CrossAttentionFusionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(ff_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, cnn_features, transformer_features):
        attn_output, _ = self.mha(query=transformer_features, key=cnn_features, value=cnn_features)
        x = self.norm1(transformer_features + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))

class HybridVisionModel(nn.Module):
    def __init__(self, cnn_model_name, swin_model_name, num_classes, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.cnn_backbone = timm.create_model(cnn_model_name, pretrained=True, num_classes=0, global_pool='')
        self.transformer_backbone = timm.create_model(swin_model_name, pretrained=True, num_classes=0, global_pool='')
        self.cnn_proj = nn.Linear(self.cnn_backbone.num_features, embed_dim)
        self.transformer_proj = nn.Linear(self.transformer_backbone.num_features, embed_dim)
        self.fusion_module = CrossAttentionFusionModule(embed_dim, num_heads, ff_dim)
        self.post_fusion_norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.projection_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 128))
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        cnn_features_map = self.cnn_backbone.forward_features(x)
        cnn_sequence = cnn_features_map.flatten(2).permute(0, 2, 1)
        cnn_proj = self.cnn_proj(cnn_sequence)
        transformer_features_map = self.transformer_backbone.forward_features(x)
        transformer_sequence = transformer_features_map.flatten(1, 2)
        transformer_proj = self.transformer_proj(transformer_sequence)
        fused = self.fusion_module(cnn_proj, transformer_proj)
        norm_fused = self.post_fusion_norm(fused)
        pooled_fused = self.pool(norm_fused.permute(0, 2, 1))
        embedding = self.flatten(pooled_fused)
        projected_embedding = self.projection_head(embedding)
        logits = self.classifier(embedding)
        return logits, projected_embedding

def run_phase1_training(cfg, train_loader, val_loader, num_classes):
    model = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, num_classes, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
    criterion_cls = FocalLoss(label_smoothing=cfg.LABEL_SMOOTHING).to(cfg.DEVICE)
    criterion_scl = SupervisedContrastiveLoss(temperature=cfg.SCL_TEMPERATURE).to(cfg.DEVICE)
    for param in model.cnn_backbone.parameters(): param.requires_grad = False
    for param in model.transformer_backbone.parameters(): param.requires_grad = False
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.INITIAL_LR)
    best_metric, best_epoch, hist = 0.0, 0, []
    for epoch in range(cfg.HEAD_TRAIN_EPOCHS):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad()
            logits, embeddings = model(inputs)
            loss_cls = criterion_cls(logits, labels)
            loss_scl = criterion_scl(embeddings, labels)
            loss = cfg.ALPHA * loss_cls + (1 - cfg.ALPHA) * loss_scl
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                logits, _ = model(inputs)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        hist.append(dict(epoch=epoch + 1, train_loss=train_loss / len(train_loader.dataset), val_ba=float(val_balanced_acc)))
        log(f"  epoch {epoch+1} train_loss {train_loss/len(train_loader.dataset):.4f} val_BA {val_balanced_acc:.4f}")
        if val_balanced_acc > best_metric:
            best_metric, best_epoch = val_balanced_acc, epoch + 1
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
    return best_epoch, hist

# ------------------------------------------------------------------ Grad-CAM cache
class LogitsOnly(nn.Module):
    """Returns only the logits so that Grad-CAM can be computed for a batch; per-image maps are
    identical to the batch-size-1 computation of the original notebook (no cross-sample layers)."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m(x)[0]

def collect(model, loader, cfg):
    """Softmax probabilities, labels and Grad-CAM maps for the predicted class of every image."""
    wrapped = LogitsOnly(model).eval()
    P, Y = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            P.append(F.softmax(wrapped(inputs.to(cfg.DEVICE)), dim=1).float().cpu().numpy()); Y.append(labels.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    pred = P.argmax(1)
    cam = GradCAM(model=wrapped, target_layers=[model.cnn_backbone.stages[-1].blocks[-1].conv_dw])
    G, k = [], 0
    for inputs, _ in loader:
        g = cam(input_tensor=inputs.to(cfg.DEVICE), targets=[ClassifierOutputTarget(int(c)) for c in pred[k:k + len(inputs)]])
        G.append(g.astype(np.float32)); k += len(inputs)
    cam.activations_and_grads.release()
    return P, Y, np.concatenate(G)

def instance_mask(g, p_act):
    hotspot = g >= np.percentile(g, p_act)
    return binary_dilation(hotspot, structure=np.ones((DILATION_SIZE, DILATION_SIZE))).astype(np.uint8)

def consensus_masks(P, Y, G, p_act, t_cons, K):
    pred = P.argmax(1); out, info = {}, {}
    for c in range(K):
        ids = np.where((pred == c) & (Y == c))[0]
        if len(ids) < 5:
            info[c] = f"only {len(ids)} correct images"; continue
        freq = np.mean(np.stack([instance_mask(G[i], p_act) for i in ids]), axis=0)
        cm = freq >= t_cons
        info[c] = int(cm.sum())
        if cm.sum() < 100: continue
        out[c] = cm
    return out, info

def review(P, G, masks, tau, p_act, require_pred_mask=True):
    """Original rule. Returns y1, y2, trigger flags and the class proposed by the masks (-1 if none)."""
    top = np.argsort(-P, 1); y1, y2 = top[:, 0], top[:, 1]; p1 = P[np.arange(len(P)), y1]
    trig = (p1 < tau) & (np.isin(y1, list(masks.keys())) if require_pred_mask else True) & (len(masks) > 0)
    prop = np.full(len(P), -1)
    for i in np.where(trig)[0]:
        m = instance_mask(G[i], p_act); best = -1
        for c, C in masks.items():
            s = int(np.sum(m & C))
            if s > best: best, prop[i] = s, c
    return y1, y2, trig, prop

def counts(Y, base, new):
    b = int(((base != Y) & (new == Y)).sum()); c = int(((base == Y) & (new != Y)).sum())
    ch = int((base != new).sum())
    return dict(acc=float(accuracy_score(Y, new)), ba=float(balanced_accuracy_score(Y, new)),
                f1=float(f1_score(Y, new, average="macro", zero_division=0)), changed=ch, corrected=b,
                harmed=c, w2w=ch - b - c, net=b - c, mcnemar_p=float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0)

def ece_brier(P, Y, bins=15):
    conf, pred, ece = P.max(1), P.argmax(1), 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        s = (conf > lo) & (conf <= lo + 1 / bins)
        if s.any(): ece += s.mean() * abs((pred[s] == Y[s]).mean() - conf[s].mean())
    return float(ece), float(((P - np.eye(P.shape[1])[Y]) ** 2).sum(1).mean())

def evaluate_rule(name, Pt, Yt, Gt, masks, tau, p_act, rng):
    base = Pt.argmax(1)
    y1, y2, trig, prop = review(Pt, Gt, masks, tau, p_act)
    sl = base.copy(); acc_sl = trig & (prop == y2); sl[acc_sl] = y2[acc_sl]
    r = dict(tau=float(tau), triggered=int(trig.sum()), secondlook=counts(Yt, base, sl))
    r["proposed_by_mask"] = {int(c): int((prop[trig] == c).sum()) for c in np.unique(prop[trig])} if trig.any() else {}
    unf = base.copy(); unf[trig] = y2[trig]; r["unfiltered"] = counts(Yt, base, unf)
    g = trig & np.isin(y2, list(masks.keys())); cg = base.copy(); cg[g] = y2[g]; r["class_gate"] = counts(Yt, base, cg)
    # same rule without the predicted-class condition in the trigger (as described in the manuscript draft)
    y1b, y2b, trigb, propb = review(Pt, Gt, masks, tau, p_act, require_pred_mask=False)
    alt = base.copy(); a = trigb & (propb == y2b); alt[a] = y2b[a]
    r["trigger_without_pred_mask"] = dict(triggered=int(trigb.sum()), **counts(Yt, base, alt))
    n = int(acc_sl.sum()); ids = np.where(trig)[0]; sims = []
    for _ in range(2000):
        o = base.copy()
        if n:
            pick = rng.choice(ids, size=n, replace=False); o[pick] = y2[pick]
        sims.append([balanced_accuracy_score(Yt, o), int(((base == Yt) & (o != Yt)).sum()), int(((base != Yt) & (o == Yt)).sum())])
    sims = np.array(sims, dtype=float)
    r["random_filter"] = dict(accept_n=n, mean_corrected=float(sims[:, 2].mean()), mean_harmed=float(sims[:, 1].mean()),
                              mean_ba=float(sims[:, 0].mean()),
                              p_harmed_le=float((sims[:, 1] <= r["secondlook"]["harmed"]).mean()),
                              p_corrected_ge=float((sims[:, 2] >= r["secondlook"]["corrected"]).mean()))
    audit = [dict(idx=int(i), y1=int(y1[i]), p1=float(Pt[i, y1[i]]), y2=int(y2[i]), p2=float(Pt[i, y2[i]]),
                  mask=int(prop[i]), true=int(Yt[i]), final=int(sl[i])) for i in np.where(trig)[0]]
    return r, audit


# ------------------------------------------------------------------ round 4, part A (AD5, five seeds, archived splits)
#   hybrid_ce   : the Phase-1 hybrid trained with cross-entropy (label smoothing 0.1) instead of focal + SupCon
#   resnet_head : frozen ImageNet ResNet-50 with the Phase-1 head and recipe (projection, LayerNorm, mean pooling)
#   effnet_head : frozen ImageNet EfficientNetV2-S with the Phase-1 head and recipe
# Probabilities and raw Grad-CAM maps of the predicted class are saved for the offline review, as in round 3.
from torchvision import models as tvm

ARR = os.path.join(OUT, "arrays"); os.makedirs(ARR, exist_ok=True)
SEEDS = [42, 123, 2026, 0, 1]
MODE = "a"
RESULTS = {}

class RawGradCAM(GradCAM):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.raw = []
    def get_cam_image(self, *a, **k):
        cam = super().get_cam_image(*a, **k)
        self.raw.append(np.asarray(cam, dtype=np.float32)); return cam

def collect2(wrapped, layer, loader):
    wrapped.eval(); P, Y = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            P.append(F.softmax(wrapped(inputs.to(cfg.DEVICE)), dim=1).float().cpu().numpy()); Y.append(labels.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y); pred = P.argmax(1)
    cam = RawGradCAM(model=wrapped, target_layers=[layer]); k = 0
    for inputs, _ in loader:
        cam(input_tensor=inputs.to(cfg.DEVICE), targets=[ClassifierOutputTarget(int(c)) for c in pred[k:k + len(inputs)]]); k += len(inputs)
    raw = np.concatenate(cam.raw); cam.activations_and_grads.release()
    return P, Y, raw

class FrozenCNNHead(nn.Module):
    """A frozen torchvision CNN with the Phase-1 head (projection to 384, LayerNorm, mean over the 7x7 tokens)."""
    def __init__(self, arch, K, d=384):
        super().__init__(); self.arch = arch
        if arch == "resnet50":
            m = tvm.resnet50(weights="IMAGENET1K_V1"); self.backbone = nn.Sequential(*list(m.children())[:-2]); c = 2048
        else:
            m = tvm.efficientnet_v2_s(weights="IMAGENET1K_V1"); self.backbone = m.features; c = 1280
        self.proj = nn.Linear(c, d); self.norm = nn.LayerNorm(d)
        self.projection_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 128)); self.classifier = nn.Linear(d, K)
    def forward(self, x):
        z = self.norm(self.proj(self.backbone(x).flatten(2).permute(0, 2, 1))).mean(1)
        return self.classifier(z), self.projection_head(z)
    def cam_layer(self):
        return self.backbone[-1][-1] if self.arch == "resnet50" else self.backbone[-1]

def train_generic(model, frozen_prefixes, loss_kind, train_loader, val_loader, path):
    """Phase-1 loop: frozen backbone(s), AdamW 5e-5, batch 16, 5 epochs, checkpoint = best validation BA."""
    for n, p in model.named_parameters():
        if any(n.startswith(f) for f in frozen_prefixes): p.requires_grad = False
    focal_l = FocalLoss(label_smoothing=cfg.LABEL_SMOOTHING).to(cfg.DEVICE)
    scl_l = SupervisedContrastiveLoss(temperature=cfg.SCL_TEMPERATURE).to(cfg.DEVICE)
    ce_l = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.INITIAL_LR)
    best, best_epoch = -1.0, 0
    for epoch in range(cfg.HEAD_TRAIN_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            opt.zero_grad(); logits, emb = model(inputs)
            loss = ce_l(logits, labels) if loss_kind == "ce" else cfg.ALPHA * focal_l(logits, labels) + (1 - cfg.ALPHA) * scl_l(emb, labels)
            loss.backward(); opt.step()
        model.eval(); yl, yp = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                yp.extend(model(inputs.to(cfg.DEVICE))[0].argmax(1).cpu().numpy()); yl.extend(labels.numpy())
        v = balanced_accuracy_score(yl, yp); log(f"    epoch {epoch+1} val_BA {v:.4f}")
        if v > best: best, best_epoch = v, epoch + 1; torch.save(model.state_dict(), path)
    return best_epoch

data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
cfg = Config()

def summarize(name, Pt, Yt):
    p = Pt.argmax(1)
    return dict(acc=float(accuracy_score(Yt, p)), ba=float(balanced_accuracy_score(Yt, p)), f1=float(f1_score(Yt, p, average="macro", zero_division=0)))

if MODE == "a":
    for seed in SEEDS:
        t0 = time.time(); set_seed(seed)
        full_train_dataset = ImageFolder(root=os.path.join(cfg.DATA_DIR, 'train'), transform=data_transform)
        test_dataset = ImageFolder(root=os.path.join(cfg.DATA_DIR, 'test'), transform=data_transform)
        K = len(full_train_dataset.classes)
        tr_n = int(0.8 * len(full_train_dataset)); va_n = len(full_train_dataset) - tr_n
        train_dataset, val_dataset = random_split(full_train_dataset, [tr_n, va_n])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
        res = {}
        # ---- hybrid with cross-entropy
        set_seed(seed)
        m = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, K, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
        be = train_generic(m, ("cnn_backbone", "transformer_backbone"), "ce", train_loader, val_loader, "/tmp/hyb_ce.pth"); del m
        m = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, K, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
        m.load_state_dict(torch.load("/tmp/hyb_ce.pth", map_location=cfg.DEVICE)); m.eval()
        layer = m.cnn_backbone.stages[-1].blocks[-1].conv_dw
        Pv, Yv, Rv = collect2(LogitsOnly(m), layer, val_loader); Pt, Yt, Rt = collect2(LogitsOnly(m), layer, test_loader)
        np.savez_compressed(os.path.join(ARR, f"hybridce_seed{seed}.npz"), Pv=Pv, Yv=Yv, Pt=Pt, Yt=Yt, Rv=Rv, Rt=Rt)
        res["hybrid_ce"] = dict(best_epoch=be, **summarize("hybrid_ce", Pt, Yt)); del m; torch.cuda.empty_cache()
        # ---- frozen CNN backbones with the Phase-1 head and recipe
        for arch in ("resnet50", "effv2s"):
            set_seed(seed)
            m = FrozenCNNHead(arch, K).to(cfg.DEVICE)
            be = train_generic(m, ("backbone",), "combined", train_loader, val_loader, f"/tmp/{arch}_head.pth"); del m
            m = FrozenCNNHead(arch, K).to(cfg.DEVICE); m.load_state_dict(torch.load(f"/tmp/{arch}_head.pth", map_location=cfg.DEVICE)); m.eval()
            Pv, Yv, Rv = collect2(LogitsOnly(m), m.cam_layer(), val_loader); Pt, Yt, Rt = collect2(LogitsOnly(m), m.cam_layer(), test_loader)
            np.savez_compressed(os.path.join(ARR, f"{arch}head_seed{seed}.npz"), Pv=Pv, Yv=Yv, Pt=Pt, Yt=Yt, Rv=Rv, Rt=Rt)
            res[f"{arch}_head"] = dict(best_epoch=be, **summarize(arch, Pt, Yt)); del m; torch.cuda.empty_cache()
        res["minutes"] = (time.time() - t0) / 60; RESULTS[str(seed)] = res
        json.dump(RESULTS, open(os.path.join(OUT, "results.json"), "w"))
        log(f"seed {seed}: " + ", ".join(f"{k} acc {v['acc']:.4f} BA {v['ba']:.4f}" for k, v in res.items() if isinstance(v, dict)) + f" ({res['minutes']:.1f} min)")

# ------------------------------------------------------------------ round 4, part B: where does the AD5 gain come from?
# The training directory alone is split 60/20/20 (stratified, per seed): Phase 1 is trained on 60%, the masks and
# thresholds come from the 20% validation part, and the review is evaluated on (i) the held-out 20% of the training
# directory (same source, no folder boundary), (ii) the same images under synthetic intensity shifts, and (iii) the
# official test directory.
import torchvision.transforms.functional as TF
class Shifted(torch.utils.data.Dataset):
    def __init__(self, base, idx, kind, val):
        self.base, self.idx, self.kind, self.val = base, list(idx), kind, val
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        path, y = self.base.samples[self.idx[i]]
        img = self.base.loader(path)
        if self.kind == "gamma": img = TF.adjust_gamma(img, self.val)
        elif self.kind == "contrast": img = TF.adjust_contrast(img, self.val)
        elif self.kind == "blur": img = TF.gaussian_blur(img, kernel_size=9, sigma=self.val)
        return data_transform(img), y

SHIFTS = [("none", 0), ("gamma", 0.6), ("gamma", 1.6), ("contrast", 0.6), ("contrast", 1.5), ("blur", 2.0)]
if MODE == "b":
    for seed in SEEDS:
        t0 = time.time(); set_seed(seed)
        base = ImageFolder(root=os.path.join(cfg.DATA_DIR, 'train'))
        test_dataset = ImageFolder(root=os.path.join(cfg.DATA_DIR, 'test'), transform=data_transform)
        K = len(base.classes); y = np.array(base.targets); rng = np.random.default_rng(seed)
        tr, va, te = [], [], []
        for c in range(K):
            ix = rng.permutation(np.where(y == c)[0]); n = len(ix); a, b = int(0.6 * n), int(0.8 * n)
            tr += ix[:a].tolist(); va += ix[a:b].tolist(); te += ix[b:].tolist()
        train_loader = DataLoader(Shifted(base, tr, "none", 0), batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(Shifted(base, va, "none", 0), batch_size=16, shuffle=False, num_workers=2)
        set_seed(seed)
        m = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, K, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
        be = train_generic(m, ("cnn_backbone", "transformer_backbone"), "combined", train_loader, val_loader, "/tmp/hyb_in.pth"); del m
        m = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, K, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
        m.load_state_dict(torch.load("/tmp/hyb_in.pth", map_location=cfg.DEVICE)); m.eval()
        layer = m.cnn_backbone.stages[-1].blocks[-1].conv_dw
        out = {}
        Pv, Yv, Rv = collect2(LogitsOnly(m), layer, val_loader); out.update(Pv=Pv, Yv=Yv, Rv=Rv)
        res = dict(best_epoch=be, n=[len(tr), len(va), len(te)])
        for kind, val in SHIFTS:
            Pt, Yt, Rt = collect2(LogitsOnly(m), layer, DataLoader(Shifted(base, te, kind, val), batch_size=16, shuffle=False, num_workers=2))
            tag = f"{kind}{val}"; out.update({f"P_{tag}": Pt, f"Y_{tag}": Yt, f"R_{tag}": Rt}); res[tag] = summarize(tag, Pt, Yt)
        Pt, Yt, Rt = collect2(LogitsOnly(m), layer, DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2))
        out.update(P_testdir=Pt, Y_testdir=Yt, R_testdir=Rt); res["testdir"] = summarize("testdir", Pt, Yt)
        np.savez_compressed(os.path.join(ARR, f"inshift_seed{seed}.npz"), **out)
        del m; torch.cuda.empty_cache()
        res["minutes"] = (time.time() - t0) / 60; RESULTS[str(seed)] = res
        json.dump(RESULTS, open(os.path.join(OUT, "results.json"), "w"))
        log(f"seed {seed}: " + ", ".join(f"{k} acc {v['acc']:.4f} BA {v['ba']:.4f}" for k, v in res.items() if isinstance(v, dict)) + f" ({res['minutes']:.1f} min)")
log("RESULTSJSON " + json.dumps(RESULTS, default=float))
log("DONE")
