# Offline analyses on the five AD5 re-runs (saved probabilities and Grad-CAM maps):
#  (1) per-seed results of SecondLook with the archived threshold and with tau_90;
#  (2) CEbUTAl (Jeffrey et al., 2025): thresholds = mean normalized entropy and mean confidence of correctly
#      classified validation images outside the majority class; at test time a majority-class prediction with higher
#      uncertainty and lower confidence moves to the second-ranked class. A variant applies the rule to every class.
#  (3) target-half comparison with SecondLook and a tuned class bias: number of changed predictions (locality) and the
#      spread of the change in balanced accuracy over 200 splits (Levene test and bootstrap ratio of SDs).
import sys, json, numpy as np
from sklearn.metrics import balanced_accuracy_score as BA, accuracy_score as ACC, f1_score
from scipy.stats import levene
sys.path.insert(0, '.')
from r3_lib import upsample, instance_masks, consensus, scores, review, counts, tau90
S = ["42", "123", "2026", "0", "1"]; K = 5
TRAIN_COUNTS = np.array([266 + 57, 716 + 201, 192 + 50, 482 + 109, 129 + 30])      # Mild, Moderate, None, Severe, Very Mild
CMAJ = int(TRAIN_COUNTS.argmax())                                                 # Moderate
def nent(P): return -(P * np.log(P + 1e-12)).sum(1) / np.log(P.shape[1])
def cebutal(Pv, Yv, Pt, all_classes=False):
    pv = Pv.argmax(1); sel = (pv == Yv) & (pv != CMAJ)
    t_unc, t_con = nent(Pv[sel]).mean(), Pv[sel].max(1).mean()
    pt = Pt.argmax(1); top = np.argsort(-Pt, 1)
    trig = (nent(Pt) > t_unc) & (Pt.max(1) < t_con) & (np.ones(len(pt), bool) if all_classes else (pt == CMAJ))
    out = pt.copy(); out[trig] = top[trig, 1]
    return out, int(trig.sum()), float(t_unc), float(t_con)
def bias_fit(L, Y, metric):
    b = np.zeros(K); cur = metric(Y, (L + b).argmax(1))
    for _ in range(3):
        for k in range(K):
            for v in np.arange(-3, 3.001, 0.25):
                bb = b.copy(); bb[k] = v; sc = metric(Y, (L + bb).argmax(1))
                if sc > cur + 1e-12: cur, b = sc, bb
    return b
R = {"per_seed": {}, "cebutal": {}, "half": {}}
for s in S:
    z = np.load(f"r3_ad5_out/arrays/hybrid_seed{s}.npz"); Pv, Yv, Pt, Yt = z["Pv"], z["Yv"], z["Pt"], z["Yt"]
    Mv = instance_masks(upsample(z["Rv"])); Mt = instance_masks(upsample(z["Rt"])).reshape(len(Pt), -1).astype(np.float32)
    masks = consensus(Mv, Pv, Yv, K); St = scores(Mt, masks, K); base = Pt.argmax(1)
    row = {"phase1": counts(Yt, base, base)}
    sl = {}
    for name, tau in (("t06585", 0.6585), ("t90", tau90(Pv, Yv))):
        new, trig, *_ = review(Pt, St, masks, tau); c = counts(Yt, base, new); c.update(tau=tau, reviewed=int(trig.sum()), changed=int((new != base).sum()))
        row[name] = c; sl[name] = new
    R["per_seed"][s] = row
    for variant in (False, True):
        out, n, tu, tc = cebutal(Pv, Yv, Pt, variant)
        c = counts(Yt, base, out); c.update(triggered=n, t_unc=tu, t_con=tc)
        R["cebutal"].setdefault("all_classes" if variant else "published", {})[s] = c
    # target half
    Lt = np.log(Pt + 1e-12); rng = np.random.default_rng(int(s)); by = [np.where(Yt == c)[0] for c in range(K)]
    rec = {m: dict(net=[], dba=[], changed=[]) for m in ("sl_enabled", "bias_acc", "bias_ba")}
    new = sl["t06585"]
    for _ in range(200):
        A = np.concatenate([rng.permutation(ix)[:len(ix) // 2] for ix in by]); mA = np.zeros(len(Yt), bool); mA[A] = True; B = ~mA
        netA = int((new[mA] == Yt[mA]).sum() - (base[mA] == Yt[mA]).sum()); dA = BA(Yt[mA], new[mA]) - BA(Yt[mA], base[mA])
        preds = dict(sl_enabled=new if (netA > 0 and dA > 0) else base,
                     bias_acc=(Lt + bias_fit(Lt[mA], Yt[mA], ACC)).argmax(1), bias_ba=(Lt + bias_fit(Lt[mA], Yt[mA], BA)).argmax(1))
        for m, p in preds.items():
            rec[m]["net"].append(int((p[B] == Yt[B]).sum() - (base[B] == Yt[B]).sum()))
            rec[m]["dba"].append(100 * (BA(Yt[B], p[B]) - BA(Yt[B], base[B])))
            rec[m]["changed"].append(int((p[B] != base[B]).sum()))
    R["half"][s] = rec
    print(s, "SL", row["t06585"]["b"], row["t06585"]["c"], "| t90", round(row["t90"]["tau"], 3), row["t90"]["b"], row["t90"]["c"],
          "| CEbUTAl", {k: (R["cebutal"][k][s]["b"], R["cebutal"][k][s]["c"], R["cebutal"][k][s]["triggered"]) for k in R["cebutal"]},
          "| changed half-B mean", {m: round(float(np.mean(v["changed"])), 1) for m, v in rec.items()}, flush=True)
json.dump(R, open("r9_analysis.json", "w"), indent=1, default=float)
# pooled summaries
for k in R["cebutal"]:
    b = sum(R["cebutal"][k][s]["b"] for s in S); c = sum(R["cebutal"][k][s]["c"] for s in S)
    print("CEbUTAl", k, "b/c", b, c, "net", b - c, "acc", round(np.mean([R["cebutal"][k][s]["acc"] for s in S]), 2),
          "BA", round(np.mean([R["cebutal"][k][s]["ba"] for s in S]), 2), "F1", round(np.mean([R["cebutal"][k][s]["f1"] for s in S]), 2))
for m in ("sl_enabled", "bias_acc", "bias_ba"):
    ch = np.array([np.mean(R["half"][s][m]["changed"]) for s in S]); d = np.mean([R["half"][s][m]["dba"] for s in S], 0)
    print(m, "changed per half (mean over seeds) %.1f" % ch.mean(), [round(x, 1) for x in ch], "dBA sd %.2f" % d.std())
dsl = np.mean([R["half"][s]["sl_enabled"]["dba"] for s in S], 0); dbb = np.mean([R["half"][s]["bias_ba"]["dba"] for s in S], 0)
print("Levene SL-enabled vs bias_ba (seed-mean dBA over splits):", levene(dsl, dbb))
rng = np.random.default_rng(0); ratios = []
for _ in range(5000):
    i = rng.integers(0, len(dsl), len(dsl)); ratios.append(dbb[i].std() / max(dsl[i].std(), 1e-9))
print("SD ratio bias/SL %.2f, 95%% bootstrap [%.2f, %.2f]" % (dbb.std() / dsl.std(), np.percentile(ratios, 2.5), np.percentile(ratios, 97.5)))
