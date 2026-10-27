# Offline analyses on the five AD5 re-runs (saved probabilities and raw Grad-CAM maps):
#  (1) ordinal metrics (mean absolute grade error, quadratic weighted kappa) before and after SecondLook;
#  (2) equal-area templates: corrections/harms, exact McNemar test, and two null distributions
#      (the same equal-area templates at random positions, and attached to the other valid class);
#  (3) target-half comparison with a uniform enablement rule for every method, including a learned pair-flip rule
#      (flip an uncertain prediction y1 -> y2 for the (y1, y2) pairs that gained on half A) and the same pair rule
#      learned on the validation split of the training directory (no target labels).
import sys, json, numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score as BA, accuracy_score as ACC, cohen_kappa_score
sys.path.insert(0, '.')
from r3_lib import upsample, instance_masks, consensus, scores, review, counts, tau90
S = ["42", "123", "2026", "0", "1"]; K = 5; TAU = 0.6585
GRADE = np.array([2, 3, 0, 4, 1])           # class index (Mild, Moderate, None, Severe, Very Mild) -> ordinal grade
def ordinal(Y, p):
    g, h = GRADE[Y], GRADE[p]
    return dict(mae=float(np.abs(g - h).mean()), qwk=float(cohen_kappa_score(g, h, weights='quadratic')),
                far=int((np.abs(g - h) >= 2).sum()))
def bias_fit(L, Y, metric):
    b = np.zeros(K); cur = metric(Y, (L + b).argmax(1))
    for _ in range(3):
        for k in range(K):
            for v in np.arange(-3, 3.001, 0.25):
                bb = b.copy(); bb[k] = v; sc = metric(Y, (L + bb).argmax(1))
                if sc > cur + 1e-12: cur, b = sc, bb
    return b
def pair_rule(P, Y, tau):
    """(y1, y2) pairs among predictions with p1 < tau whose flip gains on the given labeled data."""
    top = np.argsort(-P, 1); y1, y2 = top[:, 0], top[:, 1]; trig = P.max(1) < tau; keep = set()
    for a in range(K):
        for b in range(K):
            m = trig & (y1 == a) & (y2 == b)
            if m.sum() and ((Y[m] == b).sum() - (Y[m] == a).sum()) > 0: keep.add((a, b))
    return keep
def apply_pairs(P, tau, keep):
    top = np.argsort(-P, 1); y1, y2 = top[:, 0], top[:, 1]; out = y1.copy(); trig = P.max(1) < tau
    for a, b in keep: out[trig & (y1 == a) & (y2 == b)] = b
    return out
R = {"ordinal": {}, "equal": {}, "half": {}, "pair_val": {}}
for s in S:
    z = np.load(f"r3_ad5_out/arrays/hybrid_seed{s}.npz"); Pv, Yv, Pt, Yt = z["Pv"], z["Yv"], z["Pt"], z["Yt"]
    Mv = instance_masks(upsample(z["Rv"])); Mt = instance_masks(upsample(z["Rt"])).reshape(len(Pt), -1).astype(np.float32)
    masks = consensus(Mv, Pv, Yv, K); valid = sorted(masks); St = scores(Mt, masks, K); base = Pt.argmax(1)
    sl = {name: review(Pt, St, masks, tau)[0] for name, tau in (("t06585", TAU), ("t90", tau90(Pv, Yv)))}
    # (1) ordinal metrics
    R["ordinal"][s] = {"phase1": ordinal(Yt, base), **{k: ordinal(Yt, v) for k, v in sl.items()},
                       "changes_by_distance": {k: {int(d): int(n) for d, n in zip(*np.unique(np.abs(GRADE[base] - GRADE[v])[v != base], return_counts=True))} for k, v in sl.items()}}
    # (2) equal-area templates
    pred = Pv.argmax(1); freq = {c: Mv[(pred == c) & (Yv == c)].mean(0) for c in valid}
    N = min(int(masks[c].sum()) for c in valid)
    eq = {}
    for c in valid:
        f = freq[c].ravel(); idx = np.argsort(-f, kind="stable")[:N]; m = np.zeros(f.size, bool); m[idx] = True; eq[c] = m.reshape(224, 224)
    new = review(Pt, scores(Mt, eq, K), eq, TAU)[0]; c_ = counts(Yt, base, new)
    rng = np.random.default_rng(7 + int(s)); null_pos = []
    for _ in range(200):
        mk = {c: np.roll(eq[c], (int(rng.integers(0, 224)), int(rng.integers(0, 224))), axis=(0, 1)) for c in valid}
        null_pos.append(counts(Yt, base, review(Pt, scores(Mt, mk, K), mk, TAU)[0])["net"])
    perms = []
    import itertools
    for perm in itertools.permutations(valid):
        if list(perm) == valid: continue
        mk = {c: eq[p] for c, p in zip(valid, perm)}
        perms.append(counts(Yt, base, review(Pt, scores(Mt, mk, K), mk, TAU)[0])["net"])
    R["equal"][s] = dict(N=N, b=c_["b"], c=c_["c"], net=c_["net"], p=c_["p"], null_pos_mean=float(np.mean(null_pos)),
                         null_pos_p=float((np.sum(np.array(null_pos) >= c_["net"]) + 1) / (len(null_pos) + 1)),
                         perm_mean=float(np.mean(perms)) if perms else None)
    # (3) target halves
    Lt = np.log(Pt + 1e-12); rng = np.random.default_rng(int(s)); by = [np.where(Yt == c)[0] for c in range(K)]
    meth = ("sl_always", "sl_enabled", "bias_acc", "bias_ba", "bias_ba_enabled", "pair", "pair_enabled")
    rec = {m: dict(net=[], dba=[], changed=[], on=[]) for m in meth}
    for _ in range(200):
        A = np.concatenate([rng.permutation(ix)[:len(ix) // 2] for ix in by]); mA = np.zeros(len(Yt), bool); mA[A] = True; B = ~mA
        def gain(p, m): return int((p[m] == Yt[m]).sum() - (base[m] == Yt[m]).sum()), BA(Yt[m], p[m]) - BA(Yt[m], base[m])
        def enabled(p):
            n, d = gain(p, mA); return (n > 0 and d > 0)
        x = sl["t06585"]; bb = (Lt + bias_fit(Lt[mA], Yt[mA], BA)).argmax(1)
        pr = apply_pairs(Pt, TAU, pair_rule(Pt[mA], Yt[mA], TAU))
        preds = dict(sl_always=(x, True), sl_enabled=(x, enabled(x)), bias_acc=((Lt + bias_fit(Lt[mA], Yt[mA], ACC)).argmax(1), True),
                     bias_ba=(bb, True), bias_ba_enabled=(bb, enabled(bb)), pair=(pr, True), pair_enabled=(pr, enabled(pr)))
        for m, (p, on) in preds.items():
            p = p if on else base; n, d = gain(p, B)
            rec[m]["net"].append(n); rec[m]["dba"].append(100 * d); rec[m]["changed"].append(int((p[B] != base[B]).sum())); rec[m]["on"].append(int(on))
    R["half"][s] = rec
    # pair rule learned on the validation split of the training directory (no target labels), applied to the test set
    kv = pair_rule(Pv, Yv, TAU); pv = apply_pairs(Pt, TAU, kv)
    R["pair_val"][s] = dict(pairs=sorted([list(map(int, k)) for k in kv]), **{k: v for k, v in counts(Yt, base, pv).items() if k in ("b", "c", "net")},
                            sl_always_net=counts(Yt, base, x)["net"])
    print(s, "ord", R["ordinal"][s]["phase1"], R["ordinal"][s]["t06585"], R["ordinal"][s]["t90"], "| equal", R["equal"][s],
          "| pair_val", R["pair_val"][s], flush=True)
json.dump(R, open("r10_analysis.json", "w"), indent=1, default=float)
# ---- summaries
print("\n== ordinal (mean over seeds)")
for k in ("phase1", "t06585", "t90"):
    print(k, "MAE %.3f" % np.mean([R["ordinal"][s][k]["mae"] for s in S]), "QWK %.3f" % np.mean([R["ordinal"][s][k]["qwk"] for s in S]),
          "far errors", sum(R["ordinal"][s][k]["far"] for s in S), [round(R["ordinal"][s][k]["qwk"], 3) for s in S])
print("\n== equal-area")
b = sum(R["equal"][s]["b"] for s in S); c = sum(R["equal"][s]["c"] for s in S)
print("pooled b/c", b, c, "net", b - c, "McNemar p %.2g" % binomtest(b, b + c).pvalue)
for s in S: print(s, R["equal"][s])
print("\n== target halves (sum over seeds of mean net; mean dBA; mean changes; SD of seed-mean dBA over splits; share on)")
for m in R["half"]["42"]:
    net = sum(np.mean(R["half"][s][m]["net"]) for s in S); d = np.mean([R["half"][s][m]["dba"] for s in S], 0)
    ch = np.mean([np.mean(R["half"][s][m]["changed"]) for s in S]); on = np.mean([np.mean(R["half"][s][m]["on"]) for s in S])
    print(f"{m:16s} net {net:+7.1f} | dBA {d.mean():+.2f} | changed/half {ch:5.1f} | SD {d.std():.2f} | on {on:.2f} | per seed net",
          [round(float(np.mean(R['half'][s][m]['net'])), 1) for s in S])
print("\n== pair rule from validation (no target labels)")
for s in S: print(s, R["pair_val"][s])
print("total", sum(R["pair_val"][s]["net"] for s in S), "vs SecondLook", sum(R["pair_val"][s]["sl_always_net"] for s in S))
