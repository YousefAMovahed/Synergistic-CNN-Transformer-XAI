# Offline analyses of the round-3 AD5 job (kernel secondlook-round3-ad5): SecondLook as a plug-in for four
# classifiers, mask controls, reversed class prior, bootstrap intervals, target-half comparison with spreads,
# per-image overrules, MC dropout, selective prediction and conformal prediction sets.
import sys, os, json, itertools
import numpy as np
from sklearn.metrics import balanced_accuracy_score as BA, accuracy_score as ACC, roc_auc_score
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3_lib import upsample, instance_masks, consensus, scores, review, counts, tau90

ARR = sys.argv[1]
OUTJ = sys.argv[2]
MODELS = ["hybrid", "convnext", "resnet50", "effv2s"]
SEEDS = [42, 123, 2026, 0, 1]
K = 5; SEV, VM, MOD = 3, 4, 1
NAMES = ["Mild", "Moderate", "None", "Severe", "Very Mild"]
DUPS = np.array(sorted(d["idx"] for d in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ad5_test_duplicates.json")))))
TRAIN_DIR = np.array([266 + 57, 716 + 201, 192 + 50, 482 + 109, 129 + 30], float)   # Mild, Moderate, None, Severe, Very Mild
TRAIN_PRIOR = TRAIN_DIR / TRAIN_DIR.sum()
rng0 = np.random.default_rng(12345)
R = {}

def load(model, seed):
    z = np.load(os.path.join(ARR, f"{model}_seed{seed}.npz"))
    d = {k: z[k] for k in z.files}
    d["Mv"] = instance_masks(upsample(d["Rv"])); d["Mt"] = instance_masks(upsample(d["Rt"]))
    d["Mt_flat"] = d["Mt"].reshape(len(d["Mt"]), -1).astype(np.float32)
    return d

def ba(y, p): return 100 * BA(y, p)

def boot_ci(Y, base, new, B=5000, rng=None):
    rng = rng or np.random.default_rng(0); n = len(Y); da, db = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        da.append(100 * ((new[i] == Y[i]).mean() - (base[i] == Y[i]).mean())); db.append(ba(Y[i], new[i]) - ba(Y[i], base[i]))
    return [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))], [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]

def reweighted(Y, base, new, keep):
    """Net change and accuracies after re-weighting the test images to the class proportions of the training directory."""
    Yk, bk, nk = Y[keep], base[keep], new[keep]
    q = np.bincount(Yk, minlength=K) / len(Yk); w = TRAIN_PRIOR[Yk] / q[Yk]; w = w * len(Yk) / w.sum()
    net = float((w * ((nk == Yk).astype(float) - (bk == Yk).astype(float))).sum())
    b = float((w * ((bk != Yk) & (nk == Yk))).sum()); c = float((w * ((bk == Yk) & (nk != Yk))).sum())
    acc0 = float(100 * (w * (bk == Yk)).sum() / w.sum()); acc1 = float(100 * (w * (nk == Yk)).sum() / w.sum())
    # resampling: 2,000 test sets of the same size drawn with the training-directory proportions
    rng = np.random.default_rng(7); by = [np.where(Yk == c)[0] for c in range(K)]
    nc = np.round(TRAIN_PRIOR * len(Yk)).astype(int); nets = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(by[c], nc[c], replace=True) for c in range(K)])
        nets.append(int((nk[idx] == Yk[idx]).sum() - (bk[idx] == Yk[idx]).sum()))
    return dict(net=net, b=b, c=c, acc0=acc0, acc1=acc1, n=int(len(Yk)), boot_lo=float(np.percentile(nets, 2.5)),
                boot_hi=float(np.percentile(nets, 97.5)), p_pos=float(np.mean(np.array(nets) > 0)))

def controls(d, masks, tau, rng, n_draw=200):
    """Mask controls on the same trigger set: permuted class labels of the masks, consensus masks from shuffled
    validation labels, random masks of the same size, the same masks at random positions, and centred discs of the
    same area."""
    Pt, Yt, Mt = d["Pt"], d["Yt"], d["Mt_flat"]; base = Pt.argmax(1)
    out = {}
    if len(masks) < 2: return out
    valid = sorted(masks)
    def net_for(mk):
        S = scores(Mt, mk, K); new, trig, prop, y1, y2 = review(Pt, S, mk, tau)
        return int((new == Yt).sum() - (base == Yt).sum()), new
    # (a) permuted assignments (all non-identity permutations of the valid classes)
    perms = [p for p in itertools.permutations(valid) if p != tuple(valid)]
    out["permuted"] = [net_for({c: masks[p] for c, p in zip(valid, perm)})[0] for perm in perms]
    # (b) consensus masks rebuilt after shuffling the labels of the correctly classified validation images
    corr = d["Pv"].argmax(1) == d["Yv"]; sh = []
    for _ in range(n_draw // 2):
        lab = d["Yv"].copy(); idx = np.where(corr)[0]; lab[idx] = rng.permutation(lab[idx])
        mk = consensus(d["Mv"], d["Pv"], d["Yv"], K, labels=lab)
        # the trigger condition (predicted class has a valid mask) is kept as in SecondLook
        sh.append(net_for(mk)[0] if len(mk) else 0)
    out["label_shuffled"] = sh
    # (c) random masks with the same number of pixels per class
    rs = []
    for _ in range(n_draw):
        mk = {}
        for c in valid:
            m = np.zeros(224 * 224, bool); m[rng.choice(224 * 224, int(masks[c].sum()), replace=False)] = True
            mk[c] = m.reshape(224, 224)
        rs.append(net_for(mk)[0])
    out["random_same_size"] = rs
    # (d) the same masks moved to random positions (circular shift)
    mv = []
    for _ in range(n_draw):
        mk = {c: np.roll(masks[c], (int(rng.integers(0, 224)), int(rng.integers(0, 224))), axis=(0, 1)) for c in valid}
        mv.append(net_for(mk)[0])
    out["random_position"] = mv
    # (e) centred discs with the area of each mask (size and central location only, no learned shape)
    yy, xx = np.mgrid[:224, :224]; r2 = (yy - 111.5) ** 2 + (xx - 111.5) ** 2
    disc = {}
    for c in valid:
        a = int(masks[c].sum()); thr = np.sort(r2.ravel())[a - 1]; disc[c] = r2 <= thr
    out["central_disc"] = [net_for(disc)[0]]
    return out

def target_half(d, sl, rng, n_splits=200):
    """Same design as the round-2 comparison, now keeping every split to report spreads and paired differences."""
    Pt, Yt, Pv, Yv = d["Pt"], d["Yt"], d["Pv"], d["Yv"]; base = Pt.argmax(1)
    Lt = np.log(Pt + 1e-12)
    def bias_fit(L, Y, metric):
        b = np.zeros(K); cur = metric(Y, (L + b).argmax(1))
        for _ in range(3):
            for k in range(K):
                for v in np.arange(-3, 3.001, 0.25):
                    bb = b.copy(); bb[k] = v; s = metric(Y, (L + bb).argmax(1))
                    if s > cur + 1e-12: cur, b = s, bb
        return b
    by = [np.where(Yt == c)[0] for c in range(K)]
    rec = {m: [] for m in ("sl_always", "sl_enabled", "bias_acc", "bias_ba")}; recba = {m: [] for m in rec}
    for _ in range(n_splits):
        A = np.concatenate([rng.permutation(ix)[:len(ix) // 2] for ix in by]); mA = np.zeros(len(Yt), bool); mA[A] = True; B = ~mA
        netA = int((sl[mA] == Yt[mA]).sum() - (base[mA] == Yt[mA]).sum()); dbaA = BA(Yt[mA], sl[mA]) - BA(Yt[mA], base[mA])
        preds = dict(sl_always=sl, sl_enabled=sl if (netA > 0 and dbaA > 0) else base,
                     bias_acc=(Lt + bias_fit(Lt[mA], Yt[mA], ACC)).argmax(1), bias_ba=(Lt + bias_fit(Lt[mA], Yt[mA], BA)).argmax(1))
        for m, p in preds.items():
            rec[m].append(int((p[B] == Yt[B]).sum() - (base[B] == Yt[B]).sum())); recba[m].append(ba(Yt[B], p[B]) - ba(Yt[B], base[B]))
    return {m: dict(net=np.array(rec[m]), dba=np.array(recba[m])) for m in rec}

def conformal(Pv, Yv, Pt, Yt, alpha=0.1):
    s = 1 - Pv[np.arange(len(Yv)), Yv]; n = len(s)
    q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    sets = (1 - Pt) <= q
    return dict(coverage=float(sets[np.arange(len(Yt)), Yt].mean()), mean_size=float(sets.sum(1).mean()),
                cov_by_class={NAMES[c]: float(sets[Yt == c, c].mean()) for c in range(K)}, q=float(q)), sets

for model in MODELS:
    R[model] = {}
    for seed in SEEDS:
        f = os.path.join(ARR, f"{model}_seed{seed}.npz")
        if not os.path.exists(f): continue
        d = load(model, seed); Pt, Yt = d["Pt"], d["Yt"]; base = Pt.argmax(1)
        masks = consensus(d["Mv"], d["Pv"], d["Yv"], K)
        S = scores(d["Mt_flat"], masks, K)
        check = bool(np.array_equal(S, d["S"]))
        t90 = tau90(d["Pv"], d["Yv"])
        r = dict(check_overlap_scores=check, valid=[NAMES[c] for c in sorted(masks)],
                 mask_px={NAMES[c]: int(m.sum()) for c, m in masks.items()}, tau90=t90, phase1=counts(Yt, base, base))
        conf = np.zeros((K, K), int)
        for a, b in zip(Yt, base): conf[a, b] += 1
        r["confusion"] = conf.tolist()
        r["severe_as_vm"] = int(conf[SEV, VM])
        keep = ~np.isin(np.arange(len(Yt)), DUPS)
        rng = np.random.default_rng(seed)
        for tname, tau in (("tau_orig", 0.6585), ("tau90", t90)):
            new, trig, prop, y1, y2 = review(Pt, S, masks, tau)
            c = counts(Yt, base, new); c["reviewed"] = int(trig.sum())
            c["ci_acc"], c["ci_ba"] = boot_ci(Yt, base, new, rng=np.random.default_rng(1))
            c["nodup"] = counts(Yt[keep], base[keep], new[keep])
            c["moves"] = {f"{NAMES[a]}->{NAMES[b]}": int(((base == a) & (new == b) & (base != new)).sum()) for a in range(K) for b in range(K) if a != b and ((base == a) & (new == b)).sum()}
            c["reversed_prior"] = reweighted(Yt, base, new, np.ones(len(Yt), bool))
            c["reversed_prior_nodup"] = reweighted(Yt, base, new, keep)
            largest = max(masks, key=lambda k: masks[k].sum()) if masks else -1
            dirn = base.copy(); s = trig & (y2 == largest); dirn[s] = y2[s]
            c["directional"] = counts(Yt, base, dirn)
            unf = base.copy(); unf[trig] = y2[trig]; c["unfiltered"] = counts(Yt, base, unf)
            # per-image overrules: reviewed images where the masks chose a class other than the largest-mask class
            ov = np.where(trig & (prop != largest))[0]
            c["overrules"] = [dict(idx=int(i), y1=NAMES[y1[i]], y2=NAMES[y2[i]], mask=NAMES[prop[i]], true=NAMES[Yt[i]],
                                   dup=bool(i in DUPS), final=NAMES[new[i]], directional=NAMES[dirn[i]]) for i in ov]
            c["controls"] = {k: dict(mean=float(np.mean(v)), lo=float(np.min(v)), hi=float(np.max(v)), n=len(v),
                                     frac_ge_sl=float(np.mean(np.array(v) >= c["net"])))
                             for k, v in controls(d, masks, tau, rng).items()}
            r[tname] = c
            if tname == "tau_orig" and model == "hybrid":
                th = target_half(d, new, np.random.default_rng(seed))
                r["target_half"] = {m: dict(net_mean=float(v["net"].mean()), net_sd=float(v["net"].std()),
                                            dba_mean=float(v["dba"].mean()), dba_sd=float(v["dba"].std()),
                                            dba_lo=float(np.percentile(v["dba"], 2.5)), dba_hi=float(np.percentile(v["dba"], 97.5)))
                                    for m, v in th.items()}
                diff = th["sl_enabled"]["dba"] - th["bias_ba"]["dba"]
                r["target_half"]["diff_dba_slen_minus_biasba"] = dict(mean=float(diff.mean()), sd=float(diff.std()),
                                                                      lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)),
                                                                      frac_pos=float((diff > 0).mean()))
                R.setdefault("_target_half_raw", {})[str(seed)] = {m: dict(net=v["net"].tolist(), dba=v["dba"].tolist()) for m, v in th.items()}
        # selective prediction: abstain on as many least-confident images as SecondLook reviewed
        n_rev = r["tau_orig"]["reviewed"]; order = np.argsort(Pt.max(1)); keep_sel = np.ones(len(Yt), bool); keep_sel[order[:n_rev]] = False
        r["selective"] = dict(abstained=int(n_rev), acc_on_rest=float(100 * (base[keep_sel] == Yt[keep_sel]).mean()),
                              auroc_error_maxprob=float(roc_auc_score(base != Yt, -Pt.max(1))))
        cf, sets = conformal(d["Pv"], d["Yv"], Pt, Yt); r["conformal"] = cf
        if "MCt" in d:
            mc = d["MCt"].argmax(1); r["mc_dropout"] = counts(Yt, base, mc)
            r["mc_dropout"]["auroc_error_mc"] = float(roc_auc_score(base != Yt, -d["MCt"].max(1)))
        R[model][str(seed)] = r
        json.dump(R, open(OUTJ, 'w'), indent=1, default=float)
        t = r["tau_orig"]; t9 = r["tau90"]
        print(f"{model:9s} {seed:5d} chk={check} P1 {r['phase1']['acc']:.2f}/{r['phase1']['ba']:.2f} SevAsVM {r['severe_as_vm']:3d} masks {r['valid']} "
              f"| 0.6585: {t['b']}/{t['c']} -> {t['acc']:.2f}/{t['ba']:.2f} | tau90={t90:.3f}: {t9['b']}/{t9['c']} -> {t9['acc']:.2f}/{t9['ba']:.2f}", flush=True)
json.dump(R, open(OUTJ, "w"), indent=1, default=float)
