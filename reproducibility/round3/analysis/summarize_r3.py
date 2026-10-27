import json, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
R = json.load(open(sys.argv[1]))
S = ["42", "123", "2026", "0", "1"]
for m in [m for m in ["hybrid", "convnext", "resnet50", "effv2s"] if all(x in R.get(m, {}) for x in S)]:
    for t in ["tau_orig", "tau90"]:
        b = sum(R[m][s][t]["b"] for s in S); c = sum(R[m][s][t]["c"] for s in S)
        acc0 = np.mean([R[m][s]["phase1"]["acc"] for s in S]); ba0 = np.mean([R[m][s]["phase1"]["ba"] for s in S]); f0 = np.mean([R[m][s]["phase1"]["f1"] for s in S])
        acc1 = np.mean([R[m][s][t]["acc"] for s in S]); ba1 = np.mean([R[m][s][t]["ba"] for s in S]); f1 = np.mean([R[m][s][t]["f1"] for s in S])
        print(f"{m:9s} {t:8s} b/c {b}/{c} net {b-c:+d} | acc {acc0:.2f}->{acc1:.2f} BA {ba0:.2f}->{ba1:.2f} F1 {f0:.2f}->{f1:.2f} | per seed net "
              + " ".join(f"{R[m][s][t]['net']:+d}" for s in S))
print()
print("severe_as_vm:", {m: [R[m][s]["severe_as_vm"] for s in S if s in R[m]] for m in R if not m.startswith("_")})
print("\nCONTROLS (hybrid, tau 0.6585): net per seed")
for k in ["permuted", "label_shuffled", "random_same_size", "random_position", "central_disc"]:
    row = []
    for s in S:
        c = R["hybrid"][s]["tau_orig"]["controls"].get(k)
        row.append(f"{c['mean']:+.1f}[{c['lo']:+.0f},{c['hi']:+.0f}] ge={c['frac_ge_sl']:.2f}" if c else "--")
    print(f"  {k:17s} " + " | ".join(row))
print("  SL               " + " | ".join(f"{R['hybrid'][s]['tau_orig']['net']:+d}" for s in S))
print("  directional      " + " | ".join(f"{R['hybrid'][s]['tau_orig']['directional']['net']:+d}" for s in S))
print("  unfiltered       " + " | ".join(f"{R['hybrid'][s]['tau_orig']['unfiltered']['net']:+d}" for s in S))
for t in ["tau_orig", "tau90"]:
    print(f"\nREVERSED PRIOR ({t}), hybrid: reweighted net [boot 95%] p_pos | nodup")
    for s in S:
        a = R["hybrid"][s][t]["reversed_prior"]; d = R["hybrid"][s][t]["reversed_prior_nodup"]
        print(f"  {s:5s} net {a['net']:+.1f} (b {a['b']:.1f} c {a['c']:.1f}) acc {a['acc0']:.2f}->{a['acc1']:.2f} [{a['boot_lo']:+.0f},{a['boot_hi']:+.0f}] p+ {a['p_pos']:.3f} | nodup net {d['net']:+.1f} acc {d['acc0']:.2f}->{d['acc1']:.2f} [{d['boot_lo']:+.0f},{d['boot_hi']:+.0f}] p+ {d['p_pos']:.3f}")
    print("  total reweighted net:", round(sum(R["hybrid"][s][t]["reversed_prior"]["net"] for s in S), 1),
          " nodup:", round(sum(R["hybrid"][s][t]["reversed_prior_nodup"]["net"] for s in S), 1))
print("\nBOOTSTRAP 95% CI (hybrid, tau 0.6585): dAcc, dBA")
for s in S:
    c = R["hybrid"][s]["tau_orig"]; print(f"  {s:5s} dAcc {c['acc']-R['hybrid'][s]['phase1']['acc']:+.2f} {np.round(c['ci_acc'],2)} dBA {c['ba']-R['hybrid'][s]['phase1']['ba']:+.2f} {np.round(c['ci_ba'],2)} p={c['p']:.2e}")
print("\nTARGET HALF (hybrid)")
for s in S:
    th = R["hybrid"][s]["target_half"]
    print(f"  {s:5s} " + " | ".join(f"{m} {v['net_mean']:+.1f}±{v['net_sd']:.1f} dBA {v['dba_mean']:+.2f}±{v['dba_sd']:.2f}" for m, v in th.items() if m != "diff_dba_slen_minus_biasba")
          + f" || diff {th['diff_dba_slen_minus_biasba']}")
raw = R["_target_half_raw"]
for m in ["sl_always", "sl_enabled", "bias_acc", "bias_ba"]:
    net = np.sum([raw[s][m]["net"] for s in S], 0); dba = np.mean([raw[s][m]["dba"] for s in S], 0)
    print(f"  pooled {m:10s} total net {net.mean():+.1f} sd {net.std():.1f} [{np.percentile(net,2.5):+.0f},{np.percentile(net,97.5):+.0f}] | mean dBA {dba.mean():+.2f} sd {dba.std():.2f} [{np.percentile(dba,2.5):+.2f},{np.percentile(dba,97.5):+.2f}]")
d = np.mean([raw[s]["sl_enabled"]["dba"] for s in S], 0) - np.mean([raw[s]["bias_ba"]["dba"] for s in S], 0)
print(f"  paired diff dBA (SL-enabled - bias_ba), seed-mean per split: {d.mean():+.2f} sd {d.std():.2f} [{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}] frac>0 {np.mean(d>0):.2f}")
print("\nOVERRULES (hybrid, tau 0.6585)")
for s in S:
    for o in R["hybrid"][s]["tau_orig"]["overrules"]: print("  ", s, o)
print("\nSELECTIVE / CONFORMAL / MC")
for m in [m for m in R if not m.startswith("_")]:
    for s in [x for x in S if x in R[m]]:
        r = R[m][s]; mc = r.get("mc_dropout")
        print(f"  {m:9s} {s:5s} abstain {r['selective']['abstained']:3d} acc_rest {r['selective']['acc_on_rest']:.2f} AUROCerr {r['selective']['auroc_error_maxprob']:.3f} | conformal cov {r['conformal']['coverage']:.3f} size {r['conformal']['mean_size']:.2f} cov/class {r['conformal']['cov_by_class']}"
              + (f" | MC net {mc['net']:+d} acc {mc['acc']:.2f} BA {mc['ba']:.2f} AUROC {mc['auroc_error_mc']:.3f}" if mc else ""))
