# SecondLook on the round-3 new data with the documented threshold procedure (Section "Step 2: thresholds"):
# defaults p_act 95 / t_cons 0.90; if they give at most one valid mask, Optuna (TPE, 50 trials) chooses p_act in
# 85..99, t_cons in 0.40..0.95 and tau_conf in 0.50..0.95 by the validation balanced accuracy of the complete
# pipeline, and the operating threshold is tau = min(tau_conf, tau_err) (tau_err: mean confidence of misclassified
# validation images). The enablement rule is evaluated by two-fold cross-fitting on the validation split, with the
# whole procedure (including the search) repeated on one half and evaluated on the other.
import sys, os, json, io
import numpy as np, optuna
from sklearn.metrics import balanced_accuracy_score as BA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3_lib import upsample, instance_masks, consensus, scores, review, counts
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', line_buffering=True)

ARR, OUTJ = sys.argv[1], sys.argv[2]
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
SETS = {"oasis_subject": 3, "retina224": 5, "derma224": 7}
SEEDS = [42, 123, 2026]

class MaskCache:
    """Boolean instance masks per activation percentile; keeps at most three percentiles in memory."""
    def __init__(self, G): self.G, self.c = G, {}
    def __call__(self, p):
        if p not in self.c:
            if len(self.c) >= 3: self.c.pop(next(iter(self.c)))
            self.c[p] = instance_masks(self.G, p)
        return self.c[p]

def cons(M, P, Y, K, t):
    return consensus(M, P, Y, K, t_cons=t)

def scores_chunked(M, masks, K, chunk=512):
    S = np.full((len(M), K), -1, np.int64)
    if not masks: return S
    for i in range(0, len(M), chunk):
        S[i:i + chunk] = scores(M[i:i + chunk].reshape(len(M[i:i + chunk]), -1).astype(np.float32), masks, K)
    return S

def fit(Pv, Yv, cache, K, seed):
    """Documented procedure on one validation set; returns (p_act, t_cons, tau, masks, used_optuna)."""
    m = cons(cache(95), Pv, Yv, K, 0.90)
    wrong = Pv.argmax(1) != Yv
    tau_err = float(Pv.max(1)[wrong].mean()) if wrong.any() else 1.0
    if len(m) >= 2:
        return 95, 0.90, 0.6585 if False else min(tau_err, 1.0), m, False
    def objective(trial):
        t = trial.suggest_float('consensus_threshold', 0.4, 0.95, step=0.05)
        p = trial.suggest_int('activation_percentile', 85, 99)
        c = trial.suggest_float('confidence_threshold', 0.5, 0.95, step=0.05)
        mk = cons(cache(p), Pv, Yv, K, t)
        if not mk: return 0.0
        new, *_ = review(Pv, scores_chunked(cache(p), mk, K), mk, c)
        return BA(Yv, new)
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective, n_trials=50)
    b = st.best_params
    p, t, c = b['activation_percentile'], b['consensus_threshold'], b['confidence_threshold']
    return p, t, min(c, tau_err), cons(cache(p), Pv, Yv, K, t), True

def subject_vote(pred, groups, Y, K):
    ys, ps = [], []
    for g in np.unique(groups):
        i = groups == g; ps.append(np.bincount(pred[i], minlength=K).argmax()); ys.append(Y[i][0])
    return np.array(ys), np.array(ps)

R = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for name, K in SETS.items():
    if ONLY and name not in ONLY: continue
    R.setdefault(name, {})
    for model in ("xattn", "cnn"):
        for seed in SEEDS:
            f = os.path.join(ARR, f"{name}_{model}_seed{seed}.npz")
            if not os.path.exists(f): continue
            z = np.load(f); d = {k: z[k] for k in z.files}
            Gv, Gt = upsample(d["Rv"]), upsample(d["Rt"])
            cv, ct = MaskCache(Gv), MaskCache(Gt)
            Pv, Yv, Pt, Yt = d["Pv"], d["Yv"], d["Pt"], d["Yt"]; base = Pt.argmax(1)
            p, t, tau, masks, used = fit(Pv, Yv, cv, K, seed)
            if len(masks):
                new, trig, prop, y1, y2 = review(Pt, scores_chunked(ct(p), masks, K), masks, tau)
            else:
                new, trig = base.copy(), np.zeros(len(Yt), bool)
            # enablement by two-fold cross-fitting on the validation split (search repeated inside each fold)
            rng = np.random.default_rng(seed); n = len(Yv)
            if "Gv" in d:
                subj = np.unique(d["Gv"]); rng.shuffle(subj); half = np.isin(d["Gv"], subj[: len(subj) // 2])
            else:
                half = np.zeros(n, bool)
                for c in range(K):
                    ix = np.where(Yv == c)[0]; half[rng.permutation(ix)[: len(ix) // 2]] = True
            net_v, dba_v = 0, 0.0
            for A, B in ((half, ~half), (~half, half)):
                cA = MaskCache(Gv[A]); pA, tA, tauA, mA, _ = fit(Pv[A], Yv[A], cA, K, seed)
                if len(mA) < 2: continue
                cB = MaskCache(Gv[B]); nB, *_ = review(Pv[B], scores_chunked(cB(pA), mA, K), mA, tauA); bB = Pv[B].argmax(1)
                net_v += int((nB == Yv[B]).sum() - (bB == Yv[B]).sum()); dba_v += 100 * (BA(Yv[B], nB) - BA(Yv[B], bB)) / 2
            enabled = bool(net_v > 0 and dba_v > 0 and len(masks) >= 2)
            final = new if enabled else base
            r = dict(p_act=p, t_cons=t, tau=tau, optuna=used, valid=sorted(int(c) for c in masks),
                     mask_px={int(c): int(m.sum()) for c, m in masks.items()}, reviewed=int(trig.sum()),
                     phase1=counts(Yt, base, base), always=counts(Yt, base, new), val_crossfit=dict(net=net_v, dba=dba_v),
                     enabled=enabled, with_rule=counts(Yt, base, final))
            if "Gt" in d:
                ys, s0 = subject_vote(base, d["Gt"], Yt, K); _, s1 = subject_vote(new, d["Gt"], Yt, K); _, s2 = subject_vote(final, d["Gt"], Yt, K)
                r["subject"] = dict(n=int(len(ys)), phase1=counts(ys, s0, s0), always=counts(ys, s0, s1), with_rule=counts(ys, s0, s2))
            R[name][f"{model}|{seed}"] = r
            a = r["always"]
            print(f"{name:13s} {model:5s} {seed:5d} P1 {r['phase1']['acc']:.2f}/{r['phase1']['ba']:.2f} p{p} t{t:.2f} tau {tau:.3f} masks {r['valid']} "
                  f"rev {r['reviewed']:4d} always {a['b']}/{a['c']} ({a['net']:+d}, dBA {a['ba']-r['phase1']['ba']:+.2f}) "
                  f"| val {net_v:+d}/{dba_v:+.2f} -> {'ON ' if enabled else 'off'} final net {r['with_rule']['net']:+d}"
                  + (f" | subj {r['subject']['phase1']['ba']:.2f}->{r['subject']['always']['ba']:.2f}" if "subject" in r else ""))
            json.dump(R, open(OUTJ, "w"), indent=1, default=float)
