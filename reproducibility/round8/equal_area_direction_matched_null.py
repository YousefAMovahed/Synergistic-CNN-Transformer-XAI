import sys, numpy as np
from scipy.stats import hypergeom
sys.path.insert(0, '.')
from r3_lib import upsample, instance_masks, consensus, scores, review
K = 5; TAU = 0.6585; SEV, VM = 3, 4
dist = set(); tot = dict(n=0, b=0, good=0, N=0, toVM=0)
for s in ["42", "123", "2026", "0", "1"]:
    z = np.load(f"r3_ad5_out/arrays/hybrid_seed{s}.npz"); Pv, Yv, Pt, Yt = z["Pv"], z["Yv"], z["Pt"], z["Yt"]
    Mv = instance_masks(upsample(z["Rv"])); Mt = instance_masks(upsample(z["Rt"])).reshape(len(Pt), -1).astype(np.float32)
    masks = consensus(Mv, Pv, Yv, K); valid = sorted(masks); base = Pt.argmax(1)
    full = review(Pt, scores(Mt, masks, K), masks, TAU)[0]; dist |= set(np.where((base != Yt) & (full == Yt))[0].tolist())
    pred = Pv.argmax(1); freq = {c: Mv[(pred == c) & (Yv == c)].mean(0) for c in valid}
    N = min(int(masks[c].sum()) for c in valid); eq = {}
    for c in valid:
        f = freq[c].ravel(); idx = np.argsort(-f, kind="stable")[:N]; m = np.zeros(f.size, bool); m[idx] = True; eq[c] = m.reshape(224, 224)
    new, trig, prop, y1, y2 = review(Pt, scores(Mt, eq, K), eq, TAU)
    sw = new != base
    if VM in valid and SEV in valid:
        cand = trig & (y1 == VM) & (y2 == SEV); n = int((sw & cand).sum()); b = int((sw & cand & (Yt == SEV)).sum())
        good = int((cand & (Yt == SEV)).sum()); Nc = int(cand.sum())
        p = hypergeom(Nc, good, n).sf(b - 1) if n else 1.0
        print(s, f"VM->Sev candidates {Nc} (correctable {good}); equal-area switched {n}, correct {b}; random E[b]={n*good/max(Nc,1):.1f}; P(b>=)={p:.3g}; switches Sev->VM {int((sw & (y1==SEV) & (y2==VM)).sum())}")
        tot['n'] += n; tot['b'] += b; tot['good'] += good; tot['N'] += Nc
    else:
        print(s, "VM or Severe without valid mask; valid:", valid)
print("pooled VM->Sev: switched", tot['n'], "correct", tot['b'], "candidate precision", round(tot['good']/tot['N'], 3))
print("distinct corrected test images over five seeds:", len(dist))
