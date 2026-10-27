# Round-3 AD5 experiments for the SecondLook paper. The archived Phase-1 code (cells 9, 18 and 20 of the
# authors' Kaggle notebook) is copied unchanged; on top of it, the review layer is applied to three further
# classifiers trained on the same split (ConvNeXt-Small alone with the Phase-1 head and recipe, and ResNet-50 and
# EfficientNetV2-S with the archived baseline recipe). Probabilities and raw Grad-CAM maps are saved for offline
# analysis (mask controls, reversed class prior, bootstrap intervals, MC dropout, selective prediction).
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


# ------------------------------------------------------------------ round 3: SecondLook as a plug-in for four classifiers
# For every seed the same random 80/20 split of the training directory is used by all four models:
#   hybrid   : Phase 1 of the archived notebook (run_phase1_training above, unchanged)
#   convnext : ConvNeXt-Small alone, frozen, with the same head width, loss, optimizer, budget and checkpoint rule
#   resnet50 : torchvision ResNet-50 (ImageNet weights), fine-tuned in full with the archived baseline recipe
#   effv2s   : torchvision EfficientNetV2-S (ImageNet weights), same baseline recipe
# For each model we save softmax probabilities and the raw (pre-upsampling) Grad-CAM maps of the predicted class
# on the validation split and the test directory, so that the review layer and its controls can be recomputed
# offline. The default review (p_act 95, t_cons 0.90, tau = 0.6585 and tau_90) is also computed here as a check.
from torchvision import models as tvm
from torch.utils.data import Subset

ARR = os.path.join(OUT, "arrays"); os.makedirs(ARR, exist_ok=True)
SEEDS = [42, 123, 2026, 0, 1]
RESULTS = {}

class ConvNeXtOnlyModel(nn.Module):
    """ConvNeXt-Small branch of the hybrid with a head of the same width (projection, LayerNorm, mean pooling)."""
    def __init__(self, num_classes, embed_dim=384):
        super().__init__()
        self.cnn_backbone = timm.create_model("convnext_small", pretrained=True, num_classes=0, global_pool='')
        self.cnn_proj = nn.Linear(self.cnn_backbone.num_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.projection_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 128))
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        seq = self.cnn_backbone.forward_features(x).flatten(2).permute(0, 2, 1)
        z = self.norm(self.cnn_proj(seq)).mean(1)
        return self.classifier(z), self.projection_head(z)

def train_frozen(model, cfg, train_loader, val_loader, path):
    """Same loop as run_phase1_training: frozen backbone, focal + SupCon, AdamW 5e-5, 5 epochs, best val BA."""
    criterion_cls = FocalLoss(label_smoothing=cfg.LABEL_SMOOTHING).to(cfg.DEVICE)
    criterion_scl = SupervisedContrastiveLoss(temperature=cfg.SCL_TEMPERATURE).to(cfg.DEVICE)
    for p in model.cnn_backbone.parameters(): p.requires_grad = False
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.INITIAL_LR)
    best, best_epoch = 0.0, 0
    for epoch in range(cfg.HEAD_TRAIN_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad()
            logits, emb = model(inputs)
            loss = cfg.ALPHA * criterion_cls(logits, labels) + (1 - cfg.ALPHA) * criterion_scl(emb, labels)
            loss.backward(); optimizer.step()
        model.eval(); yl, yp = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                yp.extend(model(inputs.to(cfg.DEVICE))[0].argmax(1).cpu().numpy()); yl.extend(labels.numpy())
        v = balanced_accuracy_score(yl, yp)
        log(f"  [convnext] epoch {epoch+1} val_BA {v:.4f}")
        if v > best:
            best, best_epoch = v, epoch + 1; torch.save(model.state_dict(), path)
    return best_epoch

def build_tv(arch, K):
    if arch == "resnet50":
        m = tvm.resnet50(weights="IMAGENET1K_V1"); m.fc = nn.Linear(m.fc.in_features, K)
    else:
        m = tvm.efficientnet_v2_s(weights="IMAGENET1K_V1"); m.classifier[1] = nn.Linear(m.classifier[1].in_features, K)
    return m

def tv_layer(arch, m):
    return m.layer4[-1] if arch == "resnet50" else m.features[-1]

def train_tv(arch, K, cfg, train_loader, val_loader, path, seed):
    """Archived baseline recipe: full fine-tuning, cross-entropy, Adam 1e-3, 5 epochs, best validation accuracy."""
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    set_seed(seed)
    model = build_tv(arch, K).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=0.001)
    best, best_epoch = 0.0, 0
    for epoch in range(5):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad(); loss = criterion(model(inputs), labels); loss.backward(); optimizer.step()
        model.eval(); yl, yp = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                yp.extend(model(inputs.to(cfg.DEVICE)).argmax(1).cpu().numpy()); yl.extend(labels.numpy())
        v = accuracy_score(yl, yp)
        log(f"  [{arch}] epoch {epoch+1} val_acc {v:.4f}")
        if v > best:
            best, best_epoch = v, epoch + 1; torch.save(model.state_dict(), path)
    torch.backends.cudnn.deterministic = False
    return best_epoch

class RawGradCAM(GradCAM):
    """GradCAM that also keeps the raw class-activation map before ReLU and upsampling."""
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
    cam = RawGradCAM(model=wrapped, target_layers=[layer]); G, k = [], 0
    for inputs, _ in loader:
        g = cam(input_tensor=inputs.to(cfg.DEVICE), targets=[ClassifierOutputTarget(int(c)) for c in pred[k:k + len(inputs)]])
        G.append(g.astype(np.float32)); k += len(inputs)
    raw = np.concatenate(cam.raw); cam.activations_and_grads.release()
    return P, Y, np.concatenate(G), raw

def mc_dropout(model, loader, T=30):
    """Monte Carlo dropout over the dropout layers of the fusion block (the only stochastic layers of the head)."""
    model.eval(); model.fusion_module.train(); torch.manual_seed(0); out = []
    with torch.no_grad():
        for inputs, _ in loader:
            x = inputs.to(cfg.DEVICE)
            cs = model.cnn_proj(model.cnn_backbone.forward_features(x).flatten(2).permute(0, 2, 1))
            ts = model.transformer_proj(model.transformer_backbone.forward_features(x).flatten(1, 2))
            acc = 0
            for _ in range(T):
                z = model.flatten(model.pool(model.post_fusion_norm(model.fusion_module(cs, ts)).permute(0, 2, 1)))
                acc = acc + F.softmax(model.classifier(z), 1)
            out.append((acc / T).float().cpu().numpy())
    model.eval()
    return np.concatenate(out)

def default_review(name, Pv, Yv, Gv, Pt, Yt, Gt, K):
    masks, info = consensus_masks(Pv, Yv, Gv, ACTIVATION_PERCENTILE, CONSENSUS_THRESHOLD, K)
    wrong = Pv.argmax(1) != Yv
    tau90 = float(np.percentile(Pv.max(1)[wrong], 90)) if wrong.sum() >= 5 else 0.0
    base = Pt.argmax(1); r = dict(phase1=counts(Yt, base, base), mask_pixels={int(c): v for c, v in info.items()},
                                  valid=sorted(int(c) for c in masks), tau90=tau90)
    for tname, tau in (("tau_orig", TAU_ORIGINAL), ("tau90", tau90)):
        y1, y2, trig, prop = review(Pt, Gt, masks, tau, ACTIVATION_PERCENTILE)
        sl = base.copy(); s = trig & (prop == y2); sl[s] = y2[s]
        r[tname] = dict(triggered=int(trig.sum()), **counts(Yt, base, sl))
    # overlap scores |M_x & C_c| for every test image and class (-1: no valid mask) to check the offline recomputation
    S = np.full((len(Pt), K), -1, dtype=np.int32)
    for i in range(len(Pt)):
        m = instance_mask(Gt[i], ACTIVATION_PERCENTILE)
        for c, C in masks.items(): S[i, c] = int(np.sum(m & C))
    log(f"  {name}: P1 acc {r['phase1']['acc']:.4f} BA {r['phase1']['ba']:.4f} | masks {r['valid']} | "
        f"tau0.6585 {r['tau_orig']['corrected']}/{r['tau_orig']['harmed']} | tau90={tau90:.3f} {r['tau90']['corrected']}/{r['tau90']['harmed']}")
    return r, S

data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
cfg = Config()
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
    res = dict(classes=full_train_dataset.classes, train_idx_head=[int(i) for i in train_dataset.indices[:10]])
    np.save(os.path.join(ARR, f"split_seed{seed}.npy"), np.array(val_dataset.indices, dtype=np.int32))

    # ---- hybrid (archived Phase 1, unchanged)
    best_epoch, hist = run_phase1_training(cfg, train_loader, val_loader, K)
    model = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, K, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE)); model.eval()
    layer = model.cnn_backbone.stages[-1].blocks[-1].conv_dw
    Pv, Yv, Gv, Rv = collect2(LogitsOnly(model), layer, val_loader)
    Pt, Yt, Gt, Rt = collect2(LogitsOnly(model), layer, test_loader)
    MCv, MCt = mc_dropout(model, val_loader), mc_dropout(model, test_loader)
    r, S = default_review("hybrid", Pv, Yv, Gv, Pt, Yt, Gt, K); r["best_epoch"] = best_epoch; res["hybrid"] = r
    np.savez_compressed(os.path.join(ARR, f"hybrid_seed{seed}.npz"), Pv=Pv, Yv=Yv, Pt=Pt, Yt=Yt, Rv=Rv, Rt=Rt, S=S, MCv=MCv, MCt=MCt)
    del model, Gv, Gt; torch.cuda.empty_cache()

    # ---- ConvNeXt-Small alone, same head width, loss and budget
    set_seed(seed)
    m = ConvNeXtOnlyModel(K, cfg.EMBED_DIM).to(cfg.DEVICE)
    be = train_frozen(m, cfg, train_loader, val_loader, "/tmp/convnext_only.pth"); del m
    m = ConvNeXtOnlyModel(K, cfg.EMBED_DIM).to(cfg.DEVICE)
    m.load_state_dict(torch.load("/tmp/convnext_only.pth", map_location=cfg.DEVICE)); m.eval()
    layer = m.cnn_backbone.stages[-1].blocks[-1].conv_dw
    Pv, Yv, Gv, Rv = collect2(LogitsOnly(m), layer, val_loader)
    Pt, Yt, Gt, Rt = collect2(LogitsOnly(m), layer, test_loader)
    r, S = default_review("convnext", Pv, Yv, Gv, Pt, Yt, Gt, K); r["best_epoch"] = be; res["convnext"] = r
    np.savez_compressed(os.path.join(ARR, f"convnext_seed{seed}.npz"), Pv=Pv, Yv=Yv, Pt=Pt, Yt=Yt, Rv=Rv, Rt=Rt, S=S)
    del m, Gv, Gt; torch.cuda.empty_cache()

    # ---- torchvision baselines with the archived recipe
    for arch in ("resnet50", "effv2s"):
        path = f"/tmp/{arch}.pth"
        be = train_tv(arch, K, cfg, train_loader, val_loader, path, seed)
        m = build_tv(arch, K).to(cfg.DEVICE); m.load_state_dict(torch.load(path, map_location=cfg.DEVICE)); m.eval()
        Pv, Yv, Gv, Rv = collect2(m, tv_layer(arch, m), val_loader)
        Pt, Yt, Gt, Rt = collect2(m, tv_layer(arch, m), test_loader)
        r, S = default_review(arch, Pv, Yv, Gv, Pt, Yt, Gt, K); r["best_epoch"] = be; res[arch] = r
        np.savez_compressed(os.path.join(ARR, f"{arch}_seed{seed}.npz"), Pv=Pv, Yv=Yv, Pt=Pt, Yt=Yt, Rv=Rv, Rt=Rt, S=S)
        del m, Gv, Gt; torch.cuda.empty_cache()

    res["minutes"] = (time.time() - t0) / 60
    RESULTS[str(seed)] = res
    json.dump(RESULTS, open(os.path.join(OUT, "results.json"), "w"), default=float)
    log(f"seed {seed} done in {res['minutes']:.1f} min")
log("RESULTSJSON " + json.dumps(RESULTS, default=float))
log("DONE")
