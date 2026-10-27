import json, sys, io, numpy as np
from scipy.stats import wilcoxon, binomtest
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
R = json.load(open(sys.argv[1]))
DS = ["AD5", "breastmnist", "retinamnist", "pneumoniamnist", "dermamnist"]
S = ["42", "123", "2026", "0", "1", "2", "3", "4", "5", "6"]
def v(ds, k, m="ba"): return np.array([100 * R[ds][k][s][m] for s in S])
keys = list(R["AD5"].keys())
for ds in DS:
    print(ds)
    for k in keys:
        b = v(ds, k); a = v(ds, k, "acc"); f = v(ds, k, "f1")
        extra = ""
        if k.startswith("linear"): extra = " lr " + str(sorted(set(R[ds][k][s]["lr"] for s in S)))
        print(f"   {k:16s} BA {b.mean():6.2f}±{b.std(ddof=1):5.2f}  acc {a.mean():6.2f}  F1 {f.mean():6.2f}{extra}")
def comp(name, A, B, dsets=DS):
    d = np.concatenate([A(ds) - B(ds) for ds in dsets]); w = int((d > 0).sum()); l = int((d < 0).sum())
    p = binomtest(w, w + l, 0.5).pvalue if w + l else 1; pw = wilcoxon(d).pvalue
    per = {ds: int(((A(ds) - B(ds)) > 0).sum()) for ds in dsets}
    print(f"{name:55s} wins {w} losses {l} ties {len(d)-w-l} | mean diff {d.mean():+.2f} | sign p {p:.2e} wilcoxon p {pw:.2e} | per dataset wins {per}")
best_single = lambda ds: np.maximum(v(ds, "cnn|combined"), v(ds, "swin|combined"))
print()
comp("xattn|combined vs better single backbone (per seed)", lambda ds: v(ds, "xattn|combined"), best_single)
comp("  MedMNIST only", lambda ds: v(ds, "xattn|combined"), best_single, DS[1:])
comp("xattn|combined vs cnn", lambda ds: v(ds, "xattn|combined"), lambda ds: v(ds, "cnn|combined"))
comp("xattn|combined vs swin", lambda ds: v(ds, "xattn|combined"), lambda ds: v(ds, "swin|combined"))
comp("late|combined vs better single", lambda ds: v(ds, "late|combined"), best_single)
comp("xattn|combined vs late|combined", lambda ds: v(ds, "xattn|combined"), lambda ds: v(ds, "late|combined"))
comp("xattn|ce vs late|ce", lambda ds: v(ds, "xattn|ce"), lambda ds: v(ds, "late|ce"))
comp("best xattn (either loss) vs best late (either loss)", lambda ds: np.maximum(v(ds, "xattn|combined"), v(ds, "xattn|ce")), lambda ds: np.maximum(v(ds, "late|combined"), v(ds, "late|ce")))
comp("xattn combined vs xattn ce", lambda ds: v(ds, "xattn|combined"), lambda ds: v(ds, "xattn|ce"))
comp("xattn|combined vs linear|both", lambda ds: v(ds, "xattn|combined"), lambda ds: v(ds, "linear|both"))
comp("xattn|combined vs best linear probe", lambda ds: v(ds, "xattn|combined"), lambda ds: np.max([v(ds, f"linear|{w}") for w in ("cnn", "swin", "both")], 0))
comp("linear both vs best single linear", lambda ds: v(ds, "linear|both"), lambda ds: np.maximum(v(ds, "linear|cnn"), v(ds, "linear|swin")))
for a in ["0.25", "0.75", "1.0"]:
    comp(f"alpha 0.5 vs alpha {a}", lambda ds: v(ds, "xattn|combined"), lambda ds, a=a: v(ds, f"xattn|alpha{a}"))
print("\nvalidation BA by alpha (mean over seeds):")
for ds in DS:
    print("  ", ds, {k: round(float(np.mean([100 * R[ds][k][s]["val_ba"] for s in S])), 2) for k in ["xattn|alpha0.25", "xattn|combined", "xattn|alpha0.75", "xattn|alpha1.0", "xattn|ce"]})
