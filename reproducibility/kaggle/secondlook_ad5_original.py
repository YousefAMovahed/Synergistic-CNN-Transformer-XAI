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
base_work_dir = "/kaggle/working/dataset_alzheimer1"
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
    MODEL_SAVE_PATH = f"/kaggle/working/my_model_{DATASET_NAME}_phase1.pth"
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

# ------------------------------------------------------------------ main
cfg = Config()
SEEDS = [42, 123, 2026, 0, 1]
RESULTS = {"seeds": {}, "note": "original AD5 protocol; primary tau = 0.6585"}
for seed in SEEDS:
    t0 = time.time()
    set_seed(seed)
    data_transform = transforms.Compose([transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_train_dataset = ImageFolder(root=os.path.join(cfg.DATA_DIR, 'train'), transform=data_transform)
    test_dataset = ImageFolder(root=os.path.join(cfg.DATA_DIR, 'test'), transform=data_transform)
    NUM_CLASSES, CLASS_NAMES = len(full_train_dataset.classes), full_train_dataset.classes
    assert CLASS_NAMES == test_dataset.classes, (CLASS_NAMES, test_dataset.classes)
    train_size = int(0.8 * len(full_train_dataset)); val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    log(f"=== seed {seed} classes {CLASS_NAMES} train {train_size} val {val_size} test {len(test_dataset)}")
    best_epoch, hist = run_phase1_training(cfg, train_loader, val_loader, NUM_CLASSES)

    model = HybridVisionModel(cfg.CNN_MODEL_NAME, cfg.SWIN_MODEL_NAME, NUM_CLASSES, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE))
    model.eval()
    Pv, Yv, Gv = collect(model, val_loader, cfg)
    Pt, Yt, Gt = collect(model, test_loader, cfg)
    base = Pt.argmax(1)
    res = dict(best_epoch=best_epoch, history=hist, phase1=counts(Yt, base, base))
    res["phase1"]["ece"], res["phase1"]["brier"] = ece_brier(Pt, Yt)
    log(f"  phase1 test acc {res['phase1']['acc']:.4f} BA {res['phase1']['ba']:.4f}")

    masks, info = consensus_masks(Pv, Yv, Gv, ACTIVATION_PERCENTILE, CONSENSUS_THRESHOLD, NUM_CLASSES)
    res["mask_pixels"] = {CLASS_NAMES[c]: v for c, v in info.items()}
    p_act, t_cons, optuna_used = ACTIVATION_PERCENTILE, CONSENSUS_THRESHOLD, False
    tau_primary = TAU_ORIGINAL
    if len(masks) <= 1:   # original notebook: Optuna only when the manual setting gives at most one mask
        optuna_used = True
        def objective(trial):
            tc = trial.suggest_float('consensus_threshold', 0.4, 0.95, step=0.05)
            pa = trial.suggest_int('activation_percentile', 85, 99)
            ct = trial.suggest_float('confidence_threshold', 0.5, 0.95, step=0.05)
            mk, _ = consensus_masks(Pv, Yv, Gv, pa, tc, NUM_CLASSES)
            if not mk: return 0.0
            a, b2, tr, pr = review(Pv, Gv, mk, ct, pa)
            o = Pv.argmax(1).copy(); s = tr & (pr == b2); o[s] = b2[s]
            return balanced_accuracy_score(Yv, o)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=cfg.N_TRIALS_OPTIMIZATION)
        bp = study.best_params; res["optuna_best"] = bp
        p_act, t_cons, tau_primary = bp['activation_percentile'], bp['consensus_threshold'], bp['confidence_threshold']
        masks, info = consensus_masks(Pv, Yv, Gv, p_act, t_cons, NUM_CLASSES)
    res.update(optuna_used=optuna_used, p_act=p_act, t_cons=t_cons,
               valid_masks=[CLASS_NAMES[c] for c in masks], mask_area_pct={CLASS_NAMES[c]: round(100 * float(m.mean()), 2) for c, m in masks.items()})
    wrong = Pv.max(1)[Pv.argmax(1) != Yv]
    tau_p90 = float(np.percentile(wrong, 90)) if len(wrong) else 0.99
    res["tau_p90_val"] = tau_p90
    rng = np.random.default_rng(seed)
    res["primary"], audit = evaluate_rule("primary", Pt, Yt, Gt, masks, tau_primary, p_act, rng)
    res["p90"], _ = evaluate_rule("p90", Pt, Yt, Gt, masks, tau_p90, p_act, rng)
    # validation check of the rule (enablement rule) at the primary threshold
    a, b2, tr, pr = review(Pv, Gv, masks, tau_primary, p_act)
    ov = Pv.argmax(1).copy(); s = tr & (pr == b2); ov[s] = b2[s]
    res["validation_primary"] = counts(Yv, Pv.argmax(1), ov)
    # label-shift baselines on the same checkpoint (test priors estimated without labels, EM; Saerens et al. 2002)
    train_prior = np.bincount(np.array(full_train_dataset.targets)[train_dataset.indices], minlength=NUM_CLASSES) / train_size
    q = train_prior.copy()
    for _ in range(200):
        W = Pt * (q / train_prior); W /= W.sum(1, keepdims=True); q = W.mean(0)
    res["em_prior"] = q.round(4).tolist()
    res["prior_adjust_em"] = counts(Yt, base, (Pt * q / train_prior).argmax(1))
    res["logit_adjust_balanced"] = counts(Yt, base, (np.log(Pt + 1e-12) - np.log(train_prior)).argmax(1))
    oracle = np.bincount(Yt, minlength=NUM_CLASSES) / len(Yt)
    res["prior_adjust_oracle_test_labels"] = counts(Yt, base, (Pt * oracle / train_prior).argmax(1))
    res["minutes"] = (time.time() - t0) / 60
    pr_ = res["primary"]["secondlook"]
    log(f"  masks {res['valid_masks']} tau {tau_primary} trig {res['primary']['triggered']} "
        f"SL +{pr_['corrected']}/-{pr_['harmed']} acc {pr_['acc']:.4f} BA {pr_['ba']:.4f}")
    RESULTS["seeds"][str(seed)] = res
    json.dump(audit, open(os.path.join(OUT, f"audit_seed{seed}.json"), "w"))
    np.save(os.path.join(OUT, f"P_test_seed{seed}.npy"), Pt); np.save(os.path.join(OUT, f"P_val_seed{seed}.npy"), Pv)
    json.dump(RESULTS, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    del model; torch.cuda.empty_cache()
log(json.dumps(RESULTS, default=float))
log("DONE")
