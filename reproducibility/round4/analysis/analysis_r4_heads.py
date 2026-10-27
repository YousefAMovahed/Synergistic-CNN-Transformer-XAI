# Unified learning-rate grid {5e-5, 2e-4, 1e-3, 1e-2} for every head and linear probe (choice by validation BA),
# ten seeds, five datasets; includes the self-attention capacity controls and cross-entropy single backbones.
import json, numpy as np, sys, io
from scipy.stats import wilcoxon, binomtest
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
N = {}
for f in ('r4ha_out/results.json', 'r4hb_out/results.json', 'r4hc_out/results.json'): N.update(json.load(open(f)))
O = json.load(open('r3_heads_gpu/results.json'))
T = {}; T.update(json.load(open('r3_lr_a/results.json'))); T.update(json.load(open('r3_lr_b/results.json')))
S = ["42", "123", "2026", "0", "1", "2", "3", "4", "5", "6"]; DS = ["AD5", "breastmnist", "retinamnist", "pneumoniamnist", "dermamnist"]
OLD = [("xattn", "combined"), ("xattn", "ce"), ("late", "combined"), ("late", "ce"), ("cnn", "combined"), ("swin", "combined")]
NEW = [("cnn", "ce"), ("swin", "ce"), ("sa_cnn", "combined"), ("sa_cnn", "ce"), ("sa_swin", "combined"), ("sa_swin", "ce")]
def runs(ds, kind, loss, s):
    if (kind, loss) in OLD: return list(T[ds][f"{kind}|{loss}|all"][s]) + [N[ds][f"{kind}|{loss}|lr0.01"][s]]
    return [N[ds][f"{kind}|{loss}|lr{lr}"][s] for lr in (5e-05, 0.0002, 0.001, 0.01)]
def tuned(ds, kind, loss):
    return np.array([100 * max(runs(ds, kind, loss, s), key=lambda r: r["val_ba"])["ba"] for s in S])
def lin(ds, w):
    return np.array([100 * max([O[ds][f"linear|{w}"][s], N[ds][f"linear|{w}|low"][s]], key=lambda r: r["val_ba"])["ba"] for s in S])
V = {}
for ds in DS:
    V[ds] = {f"{k}|{l}": tuned(ds, k, l) for k, l in OLD + NEW}
    V[ds].update({f"lin|{w}": lin(ds, w) for w in ("cnn", "swin", "both")})
    print(ds, {k: round(v.mean(), 2) for k, v in sorted(V[ds].items(), key=lambda x: -x[1].mean())})
def comp(name, a, b):
    d = np.concatenate([V[ds][a] - (np.max([V[ds][x] for x in b], 0) if isinstance(b, list) else V[ds][b]) for ds in DS])
    w, l = int((d > 0).sum()), int((d < 0).sum())
    dsw = sum(V[ds][a].mean() > (np.max([V[ds][x] for x in b], 0).mean() if isinstance(b, list) else V[ds][b].mean()) for ds in DS)
    print(f"{name:55s} wins {w} losses {l} mean {d.mean():+.2f} wilcoxon p {wilcoxon(d).pvalue:.1e} | datasets {dsw}/5")
comp("xattn CE vs best same-capacity self-attention (per seed max)", "xattn|ce", ["sa_cnn|ce", "sa_swin|ce", "sa_cnn|combined", "sa_swin|combined"])
comp("xattn CE vs late CE", "xattn|ce", "late|ce")
comp("xattn CE vs better single (CE or combined)", "xattn|ce", ["cnn|ce", "swin|ce", "cnn|combined", "swin|combined"])
comp("xattn CE vs linear both", "xattn|ce", "lin|both")
comp("xattn CE vs best linear", "xattn|ce", ["lin|cnn", "lin|swin", "lin|both"])
comp("xattn combined vs xattn CE", "xattn|combined", "xattn|ce")
json.dump({ds: {k: v.tolist() for k, v in V[ds].items()} for ds in DS}, open('r4_heads_unified.json', 'w'))
