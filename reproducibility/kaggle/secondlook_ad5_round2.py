# Faithful re-run of the ORIGINAL AD5 SecondLook notebook (cells 9, 18 and 20 of the authors' Kaggle
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


# ------------------------------------------------------------------ round-2 checks on AD5: threshold sweep, target-half comparison
from scipy.optimize import minimize
cfg = Config()
DUP_CONFLICT = np.array([64, 69, 72, 116, 124], dtype=int)   # test images labeled Moderate whose pixel duplicate is labeled Severe in train
TAUS = [0.50, 0.55, 0.60, 0.6585, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SEEDS = [42, 123, 2026, 0, 1]
N_SPLITS = 200
RESULTS = {}

def ba(y, p): return balanced_accuracy_score(y, p)

def sl_predict(P, G, masks, tau, p_act):
    y1, y2, trig, prop = review(P, G, masks, tau, p_act)
    out = P.argmax(1).copy(); s = trig & (prop == y2); out[s] = y2[s]
    return out, trig, y2

def bias_fit(L, Y, metric):
    b = np.zeros(L.shape[1]); cur = metric(Y, (L + b).argmax(1))
    for _ in range(3):
        for k in range(len(b)):
            for v in np.arange(-3, 3.001, 0.25):
                bb = b.copy(); bb[k] = v; s = metric(Y, (L + bb).argmax(1))
                if s > cur + 1e-12: cur, b = s, bb
    return b

def temp_fit(L, Y):
    best = (1e9, 1.0)
    for T in np.arange(0.25, 5.001, 0.05):
        z = L / T; z = z - z.max(1, keepdims=True); q = np.exp(z); q /= q.sum(1, keepdims=True)
        best = min(best, (float(-np.log(q[np.arange(len(Y)), Y] + 1e-12).mean()), float(T)))
    return best[1]

def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def top2_switch(Q, tau):
    top = np.argsort(-Q, 1); out = top[:, 0].copy(); s = Q[np.arange(len(Q)), top[:, 0]] < tau; out[s] = top[s, 1]; return out

def bcts_fit(L, Y):
    K = L.shape[1]
    def nll(th):
        z = L / np.exp(th[0]) + th[1:]; z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(len(Y)), Y] - np.log(np.exp(z).sum(1))).mean())
    th = minimize(nll, np.zeros(K + 1), method='L-BFGS-B').x
    return np.exp(th[0]), th[1:]

def em(P, prior, iters=200):
    q = prior.copy()
    for _ in range(iters):
        W = P * (q / prior); W /= W.sum(1, keepdims=True); q = W.mean(0)
    return q

data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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
    best_epoch, hist = run_phase1_training(cfg, train_loader, val_loader, K)
    model = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, K, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE)); model.eval()
    Pv, Yv, Gv = collect(model, val_loader, cfg)
    Pt, Yt, Gt = collect(model, test_loader, cfg)
    base = Pt.argmax(1)
    masks, _ = consensus_masks(Pv, Yv, Gv, ACTIVATION_PERCENTILE, CONSENSUS_THRESHOLD, K)
    largest = max(masks, key=lambda c: masks[c].sum()) if masks else -1
    res = dict(phase1=counts(Yt, base, base), valid_masks=sorted(int(c) for c in masks), largest=int(largest))

    # ---- threshold sweep on the full test set
    SLP = {}
    res['sweep'] = {}
    for tau in TAUS:
        out, trig, y2 = sl_predict(Pt, Gt, masks, tau, ACTIVATION_PERCENTILE); SLP[tau] = out
        d = base.copy(); s = trig & (y2 == largest); d[s] = y2[s]
        res['sweep'][str(tau)] = dict(triggered=int(trig.sum()), secondlook=counts(Yt, base, out), directional=counts(Yt, base, d))
    log(f"seed {seed} sweep: " + ", ".join(f"{t}:{res['sweep'][str(t)]['secondlook']['net']:+d}" for t in TAUS))

    # ---- the five test images whose pixel duplicate is labeled Severe in the training directory
    res['conflict_images'] = [dict(idx=int(i), true=int(Yt[i]), phase1=int(base[i]), p1=float(Pt[i].max()),
                                   secondlook=int(SLP[0.6585][i])) for i in DUP_CONFLICT]

    # ---- target-half comparison: every method is tuned on half A of the test directory and evaluated on half B
    train_prior = np.bincount(np.array(full_train_dataset.targets)[train_dataset.indices], minlength=K) / tr_n
    Lt = np.log(Pt + 1e-12); Lv = np.log(Pv + 1e-12)
    Tb, bb = bcts_fit(Lv, Yv)                                   # label-free on the target: calibration on source validation
    Pb = softmax(Lt / Tb + bb); qb = em(Pb, train_prior)
    bcts_em = (Pb * qb / train_prior).argmax(1)
    q_em = em(Pt, train_prior); em_plain = (Pt * q_em / train_prior).argmax(1)
    rng = np.random.default_rng(seed)
    by_class = [np.where(Yt == c)[0] for c in range(K)]
    rec = {m: [] for m in ('sl_always', 'sl_enabled', 'sl_tau_tuned', 'bias_acc', 'bias_ba', 'reweight_halfA',
                           'ts_switch', 'bcts_em', 'em')}
    recba = {m: [] for m in rec}
    for r in range(N_SPLITS):
        A = np.concatenate([rng.permutation(ix)[:len(ix) // 2] for ix in by_class]); mA = np.zeros(len(Yt), bool); mA[A] = True; B = ~mA
        def score(pred):
            return int((pred[B] == Yt[B]).sum() - (base[B] == Yt[B]).sum()), float(ba(Yt[B], pred[B]) - ba(Yt[B], base[B]))
        preds = {}
        preds['sl_always'] = SLP[0.6585]
        netA = int((SLP[0.6585][mA] == Yt[mA]).sum() - (base[mA] == Yt[mA]).sum())
        dbaA = ba(Yt[mA], SLP[0.6585][mA]) - ba(Yt[mA], base[mA])
        preds['sl_enabled'] = SLP[0.6585] if (netA > 0 and dbaA > 0 and len(masks) >= 2) else base
        gains = {t: int((SLP[t][mA] == Yt[mA]).sum() - (base[mA] == Yt[mA]).sum()) for t in TAUS}
        tbest = max(TAUS, key=lambda t: (gains[t], -t))
        preds['sl_tau_tuned'] = SLP[tbest] if gains[tbest] > 0 else base
        preds['bias_acc'] = (Lt + bias_fit(Lt[mA], Yt[mA], accuracy_score)).argmax(1)
        preds['bias_ba'] = (Lt + bias_fit(Lt[mA], Yt[mA], ba)).argmax(1)
        qA = np.bincount(Yt[mA], minlength=K) / mA.sum()
        preds['reweight_halfA'] = (Pt * qA / train_prior).argmax(1)
        T = temp_fit(Lt[mA], Yt[mA]); Q = softmax(Lt / T)
        accs = [(accuracy_score(Yt[mA], top2_switch(Q[mA], t)), -t, t) for t in np.arange(0.30, 0.951, 0.025)]
        best = max(accs); base_acc = accuracy_score(Yt[mA], base[mA])
        preds['ts_switch'] = top2_switch(Q, best[2]) if best[0] > base_acc else base
        preds['bcts_em'] = bcts_em; preds['em'] = em_plain
        for m in rec:
            n, d = score(preds[m]); rec[m].append(n); recba[m].append(d)
    res['target_half'] = {m: dict(mean_net_B=float(np.mean(rec[m])), sd_net_B=float(np.std(rec[m])),
                                  frac_positive=float(np.mean(np.array(rec[m]) > 0)), mean_dBA_B=float(np.mean(recba[m])))
                          for m in rec}
    res['minutes'] = (time.time() - t0) / 60
    RESULTS[str(seed)] = res
    log(f"seed {seed} target-half (mean net on half B): " + ", ".join(f"{m} {res['target_half'][m]['mean_net_B']:+.1f}" for m in rec))
    json.dump(RESULTS, open(os.path.join(OUT, 'results.json'), 'w'), default=float)
    del model; torch.cuda.empty_cache()
log("RESULTSJSON " + json.dumps(RESULTS, default=float))
log("DONE")
