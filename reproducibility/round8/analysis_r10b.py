# (a) Equal-area templates against a random filter of the same size: among the reviewed images whose runner-up has a
#     valid template, switching the same number of images at random (exact hypergeometric probabilities).
# (b) Label budget: SecondLook with its enablement rule and the learned pair rule (with the same enablement rule) when
#     only n labeled images of half A are available (class-stratified subsample), evaluated on half B; 200 splits.
import sys, json, numpy as np
from scipy.stats import hypergeom
from sklearn.metrics import balanced_accuracy_score as BA
sys.path.insert(0, '.')
from r3_lib import upsample, instance_masks, consensus, scores, review, counts
S = ["42", "123", "2026", "0", "1"]; K = 5; TAU = 0.6585
def pair_rule(P, Y, tau):
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
R = {"equal_vs_random": {}, "budget": {}}
BUDGETS = [20, 40, 80, 160, 360]
tot = dict(n=0, b=0, c=0, Eb=0.0, Ec=0.0)
for s in S:
    z = np.load(f"r3_ad5_out/arrays/hybrid_seed{s}.npz"); Pv, Yv, Pt, Yt = z["Pv"], z["Yv"], z["Pt"], z["Yt"]
    Mv = instance_masks(upsample(z["Rv"])); Mt = instance_masks(upsample(z["Rt"])).reshape(len(Pt), -1).astype(np.float32)
    masks = consensus(Mv, Pv, Yv, K); valid = sorted(masks); base = Pt.argmax(1)
    pred = Pv.argmax(1); freq = {c: Mv[(pred == c) & (Yv == c)].mean(0) for c in valid}
    N = min(int(masks[c].sum()) for c in valid); eq = {}
    for c in valid:
        f = freq[c].ravel(); idx = np.argsort(-f, kind="stable")[:N]; m = np.zeros(f.size, bool); m[idx] = True; eq[c] = m.reshape(224, 224)
    new, trig, prop, y1, y2 = review(Pt, scores(Mt, eq, K), eq, TAU)
    cand = trig & np.isin(y2, valid)                       # images the decision step could switch
    sw = new != base; n = int(sw.sum()); good = int((cand & (Yt == y2)).sum()); bad = int((cand & (Yt == y1)).sum()); Nc = int(cand.sum())
    b = int((sw & (Yt == y2)).sum()); c = int((sw & (Yt == y1)).sum())
    R["equal_vs_random"][s] = dict(candidates=Nc, switched=n, b=b, c=c, Eb=n * good / Nc, Ec=n * bad / Nc,
                                   P_b_ge=float(hypergeom(Nc, good, n).sf(b - 1)), P_c_le=float(hypergeom(Nc, bad, n).cdf(c)))
    for k, v in (("n", n), ("b", b), ("c", c), ("Eb", n * good / Nc), ("Ec", n * bad / Nc)): tot[k] += v
    print(s, R["equal_vs_random"][s], flush=True)
    # (b) label budgets
    St = scores(Mt, masks, K); sl = review(Pt, St, masks, TAU)[0]
    rng = np.random.default_rng(int(s)); by = [np.where(Yt == c)[0] for c in range(K)]
    res = {n_: {m: dict(net=[], dba=[], on=[]) for m in ("sl", "pair")} for n_ in BUDGETS}
    for _ in range(200):
        parts = [rng.permutation(ix) for ix in by]
        A = [p[:len(p) // 2] for p in parts]; mA = np.zeros(len(Yt), bool); mA[np.concatenate(A)] = True; B = ~mA
        nA = int(mA.sum())
        for n_ in BUDGETS:
            sub = np.concatenate([a[:max(1, int(round(len(a) * min(1.0, n_ / nA))))] for a in A])
            def ok(p):
                d = int((p[sub] == Yt[sub]).sum() - (base[sub] == Yt[sub]).sum()); e = BA(Yt[sub], p[sub]) - BA(Yt[sub], base[sub])
                return d > 0 and e > 0
            pr = apply_pairs(Pt, TAU, pair_rule(Pt[sub], Yt[sub], TAU))
            for m, p in (("sl", sl), ("pair", pr)):
                on = ok(p); q = p if on else base
                res[n_][m]["net"].append(int((q[B] == Yt[B]).sum() - (base[B] == Yt[B]).sum()))
                res[n_][m]["dba"].append(100 * (BA(Yt[B], q[B]) - BA(Yt[B], base[B]))); res[n_][m]["on"].append(int(on))
    R["budget"][s] = res
    print(s, {n_: {m: (round(float(np.mean(res[n_][m]['net'])), 1), round(float(np.mean(res[n_][m]['dba'])), 2)) for m in res[n_]} for n_ in BUDGETS}, flush=True)
print("equal-area pooled: switched", tot["n"], "b/c", tot["b"], tot["c"], "random E[b]/E[c] %.1f/%.1f" % (tot["Eb"], tot["Ec"]))
print("\nlabel budget (sum over seeds of mean net on half B | mean dBA | share of splits on | min seed mean net)")
for n_ in BUDGETS:
    for m in ("sl", "pair"):
        net = sum(np.mean(R["budget"][s][n_][m]["net"]) for s in S); d = np.mean([np.mean(R["budget"][s][n_][m]["dba"]) for s in S])
        on = np.mean([np.mean(R["budget"][s][n_][m]["on"]) for s in S]); worst = min(np.mean(R["budget"][s][n_][m]["net"]) for s in S)
        lose = np.mean([np.mean(np.array(R["budget"][s][n_][m]["net"]) < 0) for s in S])
        print(f"n={n_:3d} {m:4s} net {net:+7.1f} | dBA {d:+.2f} | on {on:.2f} | worst seed {worst:+.1f} | share of splits with a loss {lose:.2f}")
json.dump(R, open("r10b_analysis.json", "w"), indent=1, default=float)
