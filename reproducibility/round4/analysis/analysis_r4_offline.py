# Round-4 offline analyses from saved round-3 outputs:
#  (1) deep ensemble of the five AD5 hybrid runs (mean softmax) against Phase 1 and SecondLook;
#  (2) conformal prediction sets with label-shift weighting (oracle test priors and EM-estimated priors);
#  (3) OASIS-1: participant-cluster bootstrap of the difference between the hybrid and the ConvNeXt-only head.
import json, sys, io, numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
S = ["42", "123", "2026", "0", "1"]
R3 = json.load(open('r3_ad5_analysis.json'))
def m(y, p): return dict(acc=100 * accuracy_score(y, p), ba=100 * balanced_accuracy_score(y, p), f1=100 * f1_score(y, p, average="macro", zero_division=0))
OUT = {}
# ---- (1) ensemble
Ps = [np.load(f'r3_ad5_out/arrays/hybrid_seed{s}.npz')['Pt'] for s in S]; Y = np.load('r3_ad5_out/arrays/hybrid_seed42.npz')['Yt']
ens = np.mean(Ps, 0).argmax(1)
OUT["ensemble"] = m(Y, ens)
OUT["single_mean"] = {k: float(np.mean([m(Y, P.argmax(1))[k] for P in Ps])) for k in ("acc", "ba", "f1")}
OUT["secondlook_mean"] = {k: float(np.mean([R3["hybrid"][s]["tau_orig"][k] for s in S])) for k in ("acc", "ba", "f1")}
OUT["ensemble_sev_as_vm"] = int(((Y == 3) & (ens == 4)).sum())
print("ensemble", OUT["ensemble"], "| single mean", OUT["single_mean"], "| SecondLook mean", OUT["secondlook_mean"], "| Sev->VM", OUT["ensemble_sev_as_vm"])
# ---- (2) label-shift weighted conformal (LAC score 1 - p_y), alpha = 0.1
def em_prior(P, prior, it=300):
    q = prior.copy()
    for _ in range(it):
        W = P * (q / prior); W /= W.sum(1, keepdims=True); q = W.mean(0)
    return q
def weighted_conformal(Pv, Yv, Pt, w_class, alpha=0.1):
    """Label-shift conformal (Podkopaev & Ramdas, 2021): calibration scores weighted by w(y) = q_test(y)/q_cal(y)."""
    s = 1 - Pv[np.arange(len(Yv)), Yv]; w = w_class[Yv]
    sets = np.zeros_like(Pt, bool)
    order = np.argsort(s); s_sorted, w_sorted = s[order], w[order]
    for k in range(Pt.shape[1]):
        # threshold for candidate label k uses test weight w_class[k]
        tot = w_sorted.sum() + w_class[k]; cum = np.cumsum(w_sorted) / tot
        idx = np.searchsorted(cum, 1 - alpha)
        q = s_sorted[idx] if idx < len(s_sorted) else np.inf
        sets[:, k] = (1 - Pt[:, k]) <= q
    return sets
OUT["conformal"] = {}
for s in S:
    z = np.load(f'r3_ad5_out/arrays/hybrid_seed{s}.npz'); Pv, Yv, Pt, Yt = z['Pv'], z['Yv'], z['Pt'], z['Yt']
    qv = np.bincount(Yv, minlength=5) / len(Yv); qt = np.bincount(Yt, minlength=5) / len(Yt)
    res = {}
    for name, q in (("standard", qv), ("oracle", qt), ("em", em_prior(Pt, qv))):
        sets = weighted_conformal(Pv, Yv, Pt, q / qv)
        res[name] = dict(coverage=float(sets[np.arange(len(Yt)), Yt].mean()), size=float(sets.sum(1).mean()),
                         severe=float(sets[Yt == 3, 3].mean()))
    OUT["conformal"][s] = res
    print("conformal", s, {k: (round(v['coverage'], 3), round(v['size'], 2), round(v['severe'], 3)) for k, v in res.items()})
# ---- (3) OASIS participant-cluster bootstrap
def vote(pred, g, y):
    ys, ps = [], []
    for u in np.unique(g):
        i = g == u; ps.append(np.bincount(pred[i], minlength=3).argmax()); ys.append(y[i][0])
    return np.array(ys), np.array(ps)
OUT["oasis_boot"] = {}
for s in ["42", "123", "2026"]:
    a = np.load(f'r3_new_out/arrays/oasis_subject_xattn_seed{s}.npz'); c = np.load(f'r3_new_out/arrays/oasis_subject_cnn_seed{s}.npz')
    assert np.array_equal(a['te'], c['te'])
    g, y = a['Gt'], a['Yt']; pa, pc = a['Pt'].argmax(1), c['Pt'].argmax(1)
    subj = np.unique(g); rng = np.random.default_rng(int(s)); d_slice, d_part = [], []
    by = {u: np.where(g == u)[0] for u in subj}
    cls = {u: y[by[u][0]] for u in subj}
    groups = [np.array([u for u in subj if cls[u] == k]) for k in range(3)]
    for _ in range(2000):
        pick = np.concatenate([rng.choice(gk, len(gk), replace=True) for gk in groups])   # stratified by class
        idx = np.concatenate([by[u] for u in pick])
        gg = np.concatenate([np.full(len(by[u]), j) for j, u in enumerate(pick)])
        d_slice.append(100 * (balanced_accuracy_score(y[idx], pa[idx]) - balanced_accuracy_score(y[idx], pc[idx])))
        ya, va = vote(pa[idx], gg, y[idx]); _, vc = vote(pc[idx], gg, y[idx])
        d_part.append(100 * (balanced_accuracy_score(ya, va) - balanced_accuracy_score(ya, vc)))
    ys, vs = vote(pa, g, y); _, vcs = vote(pc, g, y)
    r = dict(n_participants={k: int(len(gk)) for k, gk in enumerate(groups)},
             slice_diff=float(100 * (balanced_accuracy_score(y, pa) - balanced_accuracy_score(y, pc))),
             slice_ci=[float(np.percentile(d_slice, 2.5)), float(np.percentile(d_slice, 97.5))], slice_p_pos=float(np.mean(np.array(d_slice) > 0)),
             part_diff=float(100 * (balanced_accuracy_score(ys, vs) - balanced_accuracy_score(ys, vcs))),
             part_ci=[float(np.percentile(d_part, 2.5)), float(np.percentile(d_part, 97.5))], part_p_pos=float(np.mean(np.array(d_part) > 0)))
    OUT["oasis_boot"][s] = r
    print("oasis", s, r)
json.dump(OUT, open('r4_offline.json', 'w'), indent=1)
