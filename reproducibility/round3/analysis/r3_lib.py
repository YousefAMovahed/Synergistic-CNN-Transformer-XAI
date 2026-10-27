# Shared offline functions for the round-3 analyses: Grad-CAM reconstruction from the raw maps saved on Kaggle,
# consensus masks and the review rule exactly as in the archived notebook.
import numpy as np, cv2
from scipy.ndimage import binary_dilation
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

P_ACT, DIL, T_CONS = 95, 3, 0.90

def upsample(raw):
    """pytorch-grad-cam: ReLU, per-image min-max scaling and bilinear resize to 224, then min-max scaling again."""
    out = np.empty((len(raw), 224, 224), np.float32)
    for i, cam in enumerate(np.maximum(raw, 0)):
        img = cam - np.min(cam); img = img / (1e-7 + np.max(img))
        img = cv2.resize(np.float32(img), (224, 224))
        img = np.maximum(img, 0); img = img - np.min(img); img = img / (1e-7 + np.max(img))
        out[i] = img
    return out

def instance_masks(G, p_act=P_ACT):
    st = np.ones((DIL, DIL))
    return np.stack([binary_dilation(g >= np.percentile(g, p_act), structure=st) for g in G])

def consensus(M, P, Y, K, t_cons=T_CONS, min_imgs=5, min_px=100, labels=None):
    """Consensus masks from the correctly classified validation images (labels may be replaced for controls)."""
    pred = P.argmax(1); lab = Y if labels is None else labels; out = {}
    correct = pred == Y
    for c in range(K):
        ids = np.where(correct & (lab == c))[0]
        if len(ids) < min_imgs: continue
        cm = M[ids].mean(0) >= t_cons
        if cm.sum() >= min_px: out[c] = cm
    return out

def scores(M, masks, K):
    S = np.full((len(M), K), -1, np.int64)
    if not masks: return S
    flat = M if M.dtype == np.float32 and M.ndim == 2 else M.reshape(len(M), -1).astype(np.float32)
    cs = sorted(masks)
    W = np.stack([masks[c].reshape(-1).astype(np.float32) for c in cs], 1)
    S[:, cs] = np.rint(flat @ W).astype(np.int64)          # exact: counts stay far below 2**24
    return S

def review(P, S, masks, tau):
    """Archived rule: trigger p1 < tau and predicted class has a valid mask; switch to y2 iff argmax_c S = y2."""
    top = np.argsort(-P, 1); y1, y2 = top[:, 0], top[:, 1]; p1 = P[np.arange(len(P)), y1]
    valid = np.array(sorted(masks)) if masks else np.array([], int)
    trig = (p1 < tau) & np.isin(y1, valid) & (len(masks) > 0)
    prop = np.full(len(P), -1)
    if len(valid):
        # ties go to the first class in iteration order (dict order = ascending class index), as in the notebook
        sub = S[:, valid]; prop_all = valid[np.argmax(sub, 1)]
        prop[trig] = prop_all[trig]
    out = y1.copy(); acc = trig & (prop == y2); out[acc] = y2[acc]
    return out, trig, prop, y1, y2

def counts(Y, base, new):
    b = int(((base != Y) & (new == Y)).sum()); c = int(((base == Y) & (new != Y)).sum())
    return dict(acc=100 * accuracy_score(Y, new), ba=100 * balanced_accuracy_score(Y, new),
                f1=100 * f1_score(Y, new, average="macro", zero_division=0), b=b, c=c, net=b - c,
                p=float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0)

def tau90(P, Y):
    wrong = P.argmax(1) != Y
    return float(np.percentile(P.max(1)[wrong], 90)) if wrong.sum() >= 5 else 0.0
