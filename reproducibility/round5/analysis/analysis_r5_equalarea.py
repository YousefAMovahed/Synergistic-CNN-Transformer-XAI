# Equal-area templates: every valid class keeps only its N most consistently activated pixels (top-N of the
# consensus frequency map), so that the templates differ only in shape and location, not in size.
import sys, json, numpy as np
sys.path.insert(0, '.')
from r3_lib import upsample, instance_masks, consensus, scores, review, counts, tau90
S = ["42", "123", "2026", "0", "1"]; K = 5; R = {}
for s in S:
    z = np.load(f"r3_ad5_out/arrays/hybrid_seed{s}.npz")
    Mv = instance_masks(upsample(z["Rv"])); Mt = instance_masks(upsample(z["Rt"])).reshape(len(z["Pt"]), -1).astype(np.float32)
    Pv, Yv, Pt, Yt = z["Pv"], z["Yv"], z["Pt"], z["Yt"]; base = Pt.argmax(1)
    masks = consensus(Mv, Pv, Yv, K); valid = sorted(masks)
    pred = Pv.argmax(1); freq = {c: Mv[(pred == c) & (Yv == c)].mean(0) for c in valid}
    areas = {c: int(masks[c].sum()) for c in valid}
    r = dict(areas=areas)
    for tname, tau in (("t06585", 0.6585), ("t90", tau90(Pv, Yv))):
        new, *_ = review(Pt, scores(Mt, masks, K), masks, tau); r[f"consensus_{tname}"] = counts(Yt, base, new)["net"]
        for label, N in (("min", min(areas.values())), ("mean", int(np.mean(list(areas.values())))), ("max", max(areas.values()))):
            eq = {}
            for c in valid:
                f = freq[c].ravel(); idx = np.argsort(-f, kind="stable")[:N]
                m = np.zeros(f.size, bool); m[idx] = True; eq[c] = m.reshape(224, 224)
            new, trig, prop, y1, y2 = review(Pt, scores(Mt, eq, K), eq, tau)
            c_ = counts(Yt, base, new); r[f"equal_{label}_{tname}"] = c_["net"]; r[f"equal_{label}_{tname}_bc"] = (c_["b"], c_["c"])
            if label == "min" and tname == "t06585":
                r["equal_min_choice"] = {int(k): int((prop[trig] == k).sum()) for k in valid}
    R[s] = r
    print(s, r, flush=True)
tot = {k: sum(R[s][k] for s in S) for k in R["42"] if k.startswith(("consensus", "equal")) and not k.endswith(("_bc", "choice"))}
print("TOTAL", tot)
json.dump(R, open("r5_equalarea.json", "w"), indent=1, default=int)
