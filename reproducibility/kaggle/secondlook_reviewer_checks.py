# Reviewer checks on top of the archived protocol (normalized scores, Swin/fusion Grad-CAM, directional
# control, temperature-scaling and class-bias baselines, duplicate exclusion; AD5 and BreastMNIST).
# Shared code: the ORIGINAL AD5 SecondLook notebook (cells 9, 18 and 20 of the authors' Kaggle
# notebook that produced 84.33% -> 92.37%), extended with controls. Model, losses, Phase-1 training
# (frozen ConvNeXt/Swin backbones, 5 epochs, checkpoint = best validation balanced accuracy,
# torch random_split 80/20 after set_seed), consensus masks (p_act 95, 3x3 dilation, t_cons 0.90,
# >=5 images, >=100 pixels) and the review rule (p1 < tau AND predicted class has a valid mask;
# switch to y2 iff argmax_c |M_x & C_c| == y2) are copied from that notebook.
# Primary threshold: the original constant tau = 0.6585, applied unchanged to every seed.
# Secondary threshold: 90th percentile of the confidences of misclassified validation images.
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


cfg = Config()

# ------------------------------------------------------------------ reviewer checks
import math
DUP_IDX = np.array([64, 69, 70, 72, 74, 76, 77, 80, 83, 84, 86, 88, 90, 91, 93, 94, 96, 98, 100, 101, 103, 107, 109, 113, 115, 116, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197, 198, 199, 200, 201], dtype=int)   # AD5 test images with an exact pixel duplicate in the training directory

def tokens_to_grid(t):
    if t.dim() == 4:                       # timm Swin blocks: (B, H, W, C)
        return t.permute(0, 3, 1, 2)
    b, l, c = t.shape; h = int(round(math.sqrt(l)))
    return t.reshape(b, h, h, c).permute(0, 3, 1, 2)

def predict_probs(model, loader):
    wrapped = LogitsOnly(model).eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            P.append(F.softmax(wrapped(x.to(cfg.DEVICE)), 1).float().cpu().numpy()); Y.append(np.asarray(y).reshape(-1))
    return np.concatenate(P), np.concatenate(Y)

def cam_maps(model, loader, pred, layer, reshape=None):
    wrapped = LogitsOnly(model).eval()
    cam = GradCAM(model=wrapped, target_layers=[layer], reshape_transform=reshape)
    G, k = [], 0
    for x, _ in loader:
        g = cam(input_tensor=x.to(cfg.DEVICE), targets=[ClassifierOutputTarget(int(c)) for c in pred[k:k + len(x)]])
        G.append(g.astype(np.float32)); k += len(x)
    cam.activations_and_grads.release()
    return np.concatenate(G)

def score(kind, m, C):
    inter = float(np.sum(m & C))
    if kind == 'raw': return inter
    if kind == 'iou': return inter / max(1.0, float(np.sum(m | C)))
    if kind == 'dice': return 2 * inter / max(1.0, float(m.sum() + C.sum()))
    return inter / max(1.0, float(C.sum()))      # 'area': intersection divided by the template area

def review_k(P, G, masks, tau, p_act, kind):
    top = np.argsort(-P, 1); y1, y2 = top[:, 0], top[:, 1]; p1 = P[np.arange(len(P)), y1]
    trig = (p1 < tau) & np.isin(y1, list(masks.keys())) & (len(masks) > 0)
    prop = np.full(len(P), -1); MB = {c: C.astype(bool) for c, C in masks.items()}
    for i in np.where(trig)[0]:
        m = instance_mask(G[i], p_act).astype(bool); best = -1.0
        for c in sorted(MB):
            s = score(kind, m, MB[c])
            if s > best: best, prop[i] = s, c
    return y1, y2, trig, prop

def apply_rule(P, y1, y2, sw):
    out = P.argmax(1).copy(); out[sw] = y2[sw]; return out

def fit_temperature(P, Y):
    L = np.log(P + 1e-12); best = (1e9, 1.0)
    for T in np.arange(0.25, 5.001, 0.05):
        z = L / T; z = z - z.max(1, keepdims=True); q = np.exp(z); q /= q.sum(1, keepdims=True)
        best = min(best, (float(-np.log(q[np.arange(len(Y)), Y] + 1e-12).mean()), float(T)))
    return best[1]

def scale(P, T):
    z = np.log(P + 1e-12) / T; z = z - z.max(1, keepdims=True); q = np.exp(z); return q / q.sum(1, keepdims=True)

def ts_switch(Pv, Yv, Pt):
    """Temperature scaling, then switch to the second-ranked class when the calibrated top-1 probability is
    below a threshold tuned on the validation split (no switch if no threshold improves validation BA)."""
    T = fit_temperature(Pv, Yv); Qv, Qt = scale(Pv, T), scale(Pt, T)
    def sw(Q, tau):
        top = np.argsort(-Q, 1); out = top[:, 0].copy(); s = Q[np.arange(len(Q)), top[:, 0]] < tau; out[s] = top[s, 1]; return out
    best = (balanced_accuracy_score(Yv, Qv.argmax(1)), 0.0)
    for tau in np.arange(0.30, 0.951, 0.025):
        best = max(best, (balanced_accuracy_score(Yv, sw(Qv, tau)), float(tau)))
    return sw(Qt, best[1]), dict(T=T, tau=best[1])

def class_bias(Pv, Yv, Pt):
    """Class-specific bias on the log-probabilities, tuned on the validation split by coordinate ascent (BA)."""
    Lv, Lt = np.log(Pv + 1e-12), np.log(Pt + 1e-12); b = np.zeros(Pv.shape[1])
    cur = balanced_accuracy_score(Yv, (Lv + b).argmax(1))
    for _ in range(3):
        for k in range(len(b)):
            for v in np.arange(-3, 3.001, 0.25):
                bb = b.copy(); bb[k] = v; s = balanced_accuracy_score(Yv, (Lv + bb).argmax(1))
                if s > cur + 1e-12: cur, b = s, bb
    return (Lt + b).argmax(1), dict(bias=b.round(2).tolist(), val_ba=float(cur))

def evaluate_all(Pv, Yv, Gv_by_src, Pt, Yt, Gt_by_src, p_act, t_cons, tau, K, names, dup=None):
    base = Pt.argmax(1); R = {}
    for src in Gv_by_src:
        masks, info = consensus_masks(Pv, Yv, Gv_by_src[src], p_act, t_cons, K)
        r = dict(valid_masks=[names[c] for c in masks], mask_pixels={names[c]: int(m.sum()) for c, m in masks.items()})
        if not masks:
            R[src] = r; continue
        largest = max(masks, key=lambda c: masks[c].sum())
        for kind in ('raw', 'iou', 'dice', 'area'):
            y1, y2, trig, prop = review_k(Pt, Gt_by_src[src], masks, tau, p_act, kind)
            out = apply_rule(Pt, y1, y2, trig & (prop == y2))
            r[kind] = counts(Yt, base, out); r[kind]['triggered'] = int(trig.sum())
            r[kind]['prop_largest_share'] = float((prop[trig] == largest).mean()) if trig.any() else None
            if dup is not None:
                keep = np.ones(len(Yt), bool); keep[dup] = False
                r[kind]['nodup'] = counts(Yt[keep], base[keep], out[keep])
                r[kind]['changed_in_dup'] = int(((out != base) & ~keep).sum())
            if kind == 'raw':
                r['directional_largest'] = counts(Yt, base, apply_rule(Pt, y1, y2, trig & (y2 == largest)))
                r['unfiltered'] = counts(Yt, base, apply_rule(Pt, y1, y2, trig))
                r['class_gate'] = counts(Yt, base, apply_rule(Pt, y1, y2, trig & np.isin(y2, list(masks.keys()))))
        R[src] = r
    o, info = ts_switch(Pv, Yv, Pt); R['ts_switch'] = dict(**counts(Yt, base, o), **info)
    o, info = class_bias(Pv, Yv, Pt); R['class_bias'] = dict(**counts(Yt, base, o), **info)
    if dup is not None:
        keep = np.ones(len(Yt), bool); keep[dup] = False
        R['phase1_nodup'] = counts(Yt[keep], base[keep], base[keep])
    return R

def sources(model):
    return {'convnext': (model.cnn_backbone.stages[-1].blocks[-1].conv_dw, None),
            'swin': (model.transformer_backbone.layers[-1].blocks[-1].norm1, tokens_to_grid),
            'fusion': (model.fusion_module, tokens_to_grid)}

RESULTS = {"ad5": {}, "breast": {}}
SEEDS = [42, 123, 2026, 0, 1]

# ------------------------------------------------------------------ Part A: AD5, archived short-budget protocol
data_transform = transforms.Compose([transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
for seed in SEEDS:
    t0 = time.time(); set_seed(seed)
    full_train_dataset = ImageFolder(root=os.path.join(cfg.DATA_DIR, 'train'), transform=data_transform)
    test_dataset = ImageFolder(root=os.path.join(cfg.DATA_DIR, 'test'), transform=data_transform)
    K, names = len(full_train_dataset.classes), full_train_dataset.classes
    tr_n = int(0.8 * len(full_train_dataset)); va_n = len(full_train_dataset) - tr_n
    train_dataset, val_dataset = random_split(full_train_dataset, [tr_n, va_n])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    best_epoch, hist = run_phase1_training(cfg, train_loader, val_loader, K)
    model = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, K, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE)); model.eval()
    Pv, Yv = predict_probs(model, val_loader); Pt, Yt = predict_probs(model, test_loader)
    Gv, Gt, errs = {}, {}, {}
    for src, (layer, rs) in sources(model).items():
        try:
            Gv[src] = cam_maps(model, val_loader, Pv.argmax(1), layer, rs)
            Gt[src] = cam_maps(model, test_loader, Pt.argmax(1), layer, rs)
        except Exception as e:
            errs[src] = repr(e)[:300]
    res = dict(best_epoch=best_epoch, phase1=counts(Yt, Pt.argmax(1), Pt.argmax(1)), cam_errors=errs)
    res.update(evaluate_all(Pv, Yv, Gv, Pt, Yt, Gt, ACTIVATION_PERCENTILE, CONSENSUS_THRESHOLD, TAU_ORIGINAL, K, names, dup=DUP_IDX))
    res['minutes'] = (time.time() - t0) / 60
    RESULTS['ad5'][str(seed)] = res
    c = res.get('convnext', {}).get('raw', {})
    log(f"AD5 seed {seed}: P1 acc {res['phase1']['acc']:.4f} | SL raw +{c.get('corrected')}/-{c.get('harmed')} | "
        f"iou {res.get('convnext', {}).get('iou', {}).get('net')} | swin {res.get('swin', {}).get('raw', {}).get('net')} | "
        f"fusion {res.get('fusion', {}).get('raw', {}).get('net')} | dir {res.get('convnext', {}).get('directional_largest', {}).get('net')} | "
        f"TS {res['ts_switch']['net']} | bias {res['class_bias']['net']} | errs {errs}")
    json.dump(RESULTS, open(os.path.join(OUT, 'results.json'), 'w'), default=float)
    del model; torch.cuda.empty_cache()

# ------------------------------------------------------------------ Part B: BreastMNIST, archived MedMNIST protocol
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "medmnist"], check=False)
import medmnist
from medmnist import INFO
from torch.utils.data import Dataset

class MedMNISTDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset; self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform: image = self.transform(image)
        return image, torch.from_numpy(np.asarray(label)).squeeze().long()

info = INFO['breastmnist']; DataClass = getattr(medmnist, info['python_class']); names_b = [info['label'][str(i)] for i in range(len(info['label']))]
tf_b = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224)), transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
os.makedirs('/tmp/medmnist', exist_ok=True)
for seed in SEEDS:
    t0 = time.time(); set_seed(seed)
    tr_raw = DataClass(split='train', download=True, root='/tmp/medmnist'); va_raw = DataClass(split='val', download=True, root='/tmp/medmnist')
    te_raw = DataClass(split='test', download=True, root='/tmp/medmnist')
    train_loader = DataLoader(MedMNISTDataset(tr_raw, tf_b), batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(MedMNISTDataset(va_raw, tf_b), batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(MedMNISTDataset(te_raw, tf_b), batch_size=16, shuffle=False, num_workers=2)
    best_epoch, hist = run_phase1_training(cfg, train_loader, val_loader, 2)
    model = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, 2, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE)); model.eval()
    Pv, Yv = predict_probs(model, val_loader); Pt, Yt = predict_probs(model, test_loader)
    layer, _ = sources(model)['convnext']
    Gv = {'convnext': cam_maps(model, val_loader, Pv.argmax(1), layer)}; Gt = {'convnext': cam_maps(model, test_loader, Pt.argmax(1), layer)}
    wrong = Pv.max(1)[Pv.argmax(1) != Yv]; tau_err = float(wrong.mean()) if len(wrong) else 0.85
    masks, _ = consensus_masks(Pv, Yv, Gv['convnext'], ACTIVATION_PERCENTILE, CONSENSUS_THRESHOLD, 2)
    p_act, t_cons, tau, used = ACTIVATION_PERCENTILE, CONSENSUS_THRESHOLD, tau_err, False
    if len(masks) <= 1:
        used = True
        def objective(trial):
            tc = trial.suggest_float('consensus_threshold', 0.4, 0.95, step=0.05)
            pa = trial.suggest_int('activation_percentile', 85, 99)
            ct = trial.suggest_float('confidence_threshold', 0.5, 0.95, step=0.05)
            mk, _ = consensus_masks(Pv, Yv, Gv['convnext'], pa, tc, 2)
            if len(mk) < 2: return 0.0
            a, b2, trg, pr = review(Pv, Gv['convnext'], mk, ct, pa)
            o = Pv.argmax(1).copy(); s = trg & (pr == b2); o[s] = b2[s]
            return balanced_accuracy_score(Yv, o)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=50)
        bp = study.best_params; p_act, t_cons = bp['activation_percentile'], bp['consensus_threshold']
        tau = min(tau_err, bp['confidence_threshold'])
    res = dict(best_epoch=best_epoch, phase1=counts(Yt, Pt.argmax(1), Pt.argmax(1)), optuna_used=used, p_act=p_act,
               t_cons=t_cons, tau=tau, tau_err=tau_err)
    res.update(evaluate_all(Pv, Yv, Gv, Pt, Yt, Gt, p_act, t_cons, tau, 2, names_b))
    res['minutes'] = (time.time() - t0) / 60
    RESULTS['breast'][str(seed)] = res
    c = res.get('convnext', {})
    log(f"Breast seed {seed}: P1 acc {res['phase1']['acc']:.4f} tau {tau:.4f} masks {c.get('valid_masks')} | raw {c.get('raw', {}).get('net')} "
        f"({c.get('raw', {}).get('corrected')}/{c.get('raw', {}).get('harmed')}) | iou {c.get('iou', {}).get('net')} | dice {c.get('dice', {}).get('net')} "
        f"| area {c.get('area', {}).get('net')} | dir {c.get('directional_largest', {}).get('net')} | TS {res['ts_switch']['net']} | bias {res['class_bias']['net']}")
    json.dump(RESULTS, open(os.path.join(OUT, 'results.json'), 'w'), default=float)
    del model; torch.cuda.empty_cache()
log("RESULTSJSON " + json.dumps(RESULTS, default=float))
log("DONE")
