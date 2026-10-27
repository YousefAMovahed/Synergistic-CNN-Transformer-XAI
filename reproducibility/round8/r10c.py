import sys, json, numpy as np
from sklearn.metrics import balanced_accuracy_score as BA
sys.path.insert(0, '.')
exec(open('analysis_r10b.py', encoding='utf8').read().split('R = {')[0])
out = {}
for s in S:
    z = np.load(f"r3_ad5_out/arrays/hybrid_seed{s}.npz"); Pv, Yv, Pt, Yt = z["Pv"], z["Yv"], z["Pt"], z["Yt"]; base = Pt.argmax(1)
    pv = apply_pairs(Pt, TAU, pair_rule(Pv, Yv, TAU)); rng = np.random.default_rng(int(s)); by = [np.where(Yt == c)[0] for c in range(K)]
    net, dba = [], []
    for _ in range(200):
        A = np.concatenate([rng.permutation(ix)[:len(ix) // 2] for ix in by]); B = np.ones(len(Yt), bool); B[A] = False
        net.append(int((pv[B] == Yt[B]).sum() - (base[B] == Yt[B]).sum())); dba.append(100 * (BA(Yt[B], pv[B]) - BA(Yt[B], base[B])))
    out[s] = (float(np.mean(net)), float(np.mean(dba)))
print(out, 'total', round(sum(v[0] for v in out.values()), 1), 'dBA', round(np.mean([v[1] for v in out.values()]), 2))
