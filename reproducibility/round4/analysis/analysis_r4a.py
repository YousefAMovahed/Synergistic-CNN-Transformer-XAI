# SecondLook (archived settings) on the round-4 models: hybrid trained with cross-entropy, frozen ResNet-50 and
# EfficientNetV2-S with the Phase-1 head.
import sys, json, numpy as np
sys.path.insert(0, '.')
from r3_lib import upsample, instance_masks, consensus, scores, review, counts, tau90
S = ["42", "123", "2026", "0", "1"]; R = {}
for m in ["hybridce", "resnet50head", "effv2shead"]:
    R[m] = {}
    for s in S:
        z = np.load(f"r4a_out/arrays/{m}_seed{s}.npz")
        Mv = instance_masks(upsample(z["Rv"])); Mt = instance_masks(upsample(z["Rt"]))
        masks = consensus(Mv, z["Pv"], z["Yv"], 5); St = scores(Mt, masks, 5); base = z["Pt"].argmax(1); Yt = z["Yt"]
        r = dict(phase1=counts(Yt, base, base), valid=sorted(int(c) for c in masks), sev_vm=int(((Yt == 3) & (base == 4)).sum()),
                 none_recall=float(100 * (base[Yt == 2] == 2).mean()))
        for name, tau in (("t06585", 0.6585), ("t90", tau90(z["Pv"], z["Yv"]))):
            new, trig, *_ = review(z["Pt"], St, masks, tau); r[name] = counts(Yt, base, new); r[name]["tau"] = tau
        R[m][s] = r
        print(m, s, "P1 %.2f/%.2f/%.2f" % (r["phase1"]["acc"], r["phase1"]["ba"], r["phase1"]["f1"]), "masks", r["valid"], "SevVM", r["sev_vm"],
              "| 0.6585 %d/%d -> %.2f/%.2f/%.2f" % (r["t06585"]["b"], r["t06585"]["c"], r["t06585"]["acc"], r["t06585"]["ba"], r["t06585"]["f1"]),
              "| t90=%.3f %d/%d" % (r["t90"]["tau"], r["t90"]["b"], r["t90"]["c"]), flush=True)
json.dump(R, open("r4a_analysis.json", "w"), indent=1, default=float)
