# SecondLook (archived settings, tau 0.6585 and tau_90) with masks and thresholds from a validation part of the AD5
# training directory, applied to the held-out part of the same directory (clean and under synthetic shifts) and to
# the official test directory.
import sys, json, numpy as np
sys.path.insert(0, '.')
from r3_lib import upsample, instance_masks, consensus, scores, review, counts, tau90
S = ["42", "123", "2026", "0", "1"]; R = {}
TAGS = ["none0", "gamma0.6", "gamma1.6", "contrast0.6", "contrast1.5", "blur2.0", "testdir"]
for s in S:
    z = np.load(f"r4b_out/arrays/inshift_seed{s}.npz")
    Mv = instance_masks(upsample(z["Rv"])); masks = consensus(Mv, z["Pv"], z["Yv"], 5); t9 = tau90(z["Pv"], z["Yv"])
    R[s] = dict(valid=sorted(int(c) for c in masks), tau90=t9)
    for tag in TAGS:
        P, Y = z[f"P_{tag}"], z[f"Y_{tag}"]; base = P.argmax(1)
        St = scores(instance_masks(upsample(z[f"R_{tag}"])), masks, 5)
        r = dict(phase1=counts(Y, base, base), sev_vm=int(((Y == 3) & (base == 4)).sum()))
        for name, tau in (("t06585", 0.6585), ("t90", t9)):
            new, trig, *_ = review(P, St, masks, tau); r[name] = counts(Y, base, new)
        R[s][tag] = r
        print(s, f"{tag:12s}", "P1 acc %.2f BA %.2f SevVM %3d" % (r["phase1"]["acc"], r["phase1"]["ba"], r["sev_vm"]),
              "| 0.6585 %d/%d (%+d)" % (r["t06585"]["b"], r["t06585"]["c"], r["t06585"]["net"]), "| t90 %d/%d (%+d)" % (r["t90"]["b"], r["t90"]["c"], r["t90"]["net"]), "masks", R[s]["valid"], flush=True)
json.dump(R, open("r4b_analysis.json", "w"), indent=1, default=float)
