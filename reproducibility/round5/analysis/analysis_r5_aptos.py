# RetinaMNIST-224 -> APTOS 2019 (natural shift between hospitals and cameras). SecondLook uses the documented
# threshold procedure on the RetinaMNIST validation split; as on AD5, the enablement rule and a class bias are
# tuned on half of the external test set and evaluated on the other half (200 class-stratified splits).
import sys, json, io, numpy as np, optuna
from sklearn.metrics import balanced_accuracy_score as BA, accuracy_score as ACC
sys.path.insert(0, '.')
from r3_lib import upsample, instance_masks, consensus, scores, review, counts
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', line_buffering=True)
K = 5; S = ["42", "123", "2026"]; R = {}
H = json.load(open('r5aptos_out/results.json'))["aptos"]
def chunked(M, mk):
    out = np.full((len(M), K), -1, np.int64)
    for i in range(0, len(M), 400): out[i:i + 400] = scores(M[i:i + 400].reshape(len(M[i:i + 400]), -1).astype(np.float32), mk, K)
    return out
def bias_fit(L, Y, metric):
    b = np.zeros(K); cur = metric(Y, (L + b).argmax(1))
    for _ in range(3):
        for k in range(K):
            for v in np.arange(-3, 3.001, 0.25):
                bb = b.copy(); bb[k] = v; sc = metric(Y, (L + bb).argmax(1))
                if sc > cur + 1e-12: cur, b = sc, bb
    return b
for s in S:
    z = np.load(f"r5aptos_out/arrays/aptos_xattn_seed{s}.npz")
    Pv, Yv, Pa, Ya = z["Pv"], z["Yv"], z["Pa"], z["Ya"]
    Gv, Ga = upsample(z["Rv"]), upsample(z["Ra"])
    cache = {}
    def Mv(p):
        if ("v", p) not in cache: cache[("v", p)] = instance_masks(Gv, p)
        return cache[("v", p)]
    m0 = consensus(Mv(95), Pv, Yv, K)
    wrong = Pv.argmax(1) != Yv; tau_err = float(Pv.max(1)[wrong].mean())
    if len(m0) >= 2:
        p, t, tau, masks = 95, 0.90, tau_err, m0
    else:
        def obj(tr):
            t_ = tr.suggest_float('consensus_threshold', 0.4, 0.95, step=0.05); p_ = tr.suggest_int('activation_percentile', 85, 99)
            c_ = tr.suggest_float('confidence_threshold', 0.5, 0.95, step=0.05)
            mk = consensus(Mv(p_), Pv, Yv, K, t_cons=t_)
            if not mk: return 0.0
            new, *_ = review(Pv, chunked(Mv(p_), mk), mk, c_); return BA(Yv, new)
        st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=int(s))); st.optimize(obj, n_trials=50)
        b = st.best_params; p, t = b['activation_percentile'], b['consensus_threshold']; tau = min(b['confidence_threshold'], tau_err)
        masks = consensus(Mv(p), Pv, Yv, K, t_cons=t)
    base = Pa.argmax(1)
    if len(masks):
        new, trig, prop, y1, y2 = review(Pa, chunked(instance_masks(Ga, p), masks), masks, tau)
    else:
        new, trig = base.copy(), np.zeros(len(Ya), bool)
    r = dict(p_act=p, t_cons=t, tau=tau, valid=sorted(int(c) for c in masks), reviewed=int(trig.sum()),
             phase1=counts(Ya, base, base), always=counts(Ya, base, new),
             sev_errors={f"{a}->{b_}": int(((Ya == a) & (base == b_)).sum()) for a in range(K) for b_ in range(K) if a != b_ and ((Ya == a) & (base == b_)).sum() >= 50})
    # target-half comparison
    La = np.log(Pa + 1e-12); rng = np.random.default_rng(int(s)); by = [np.where(Ya == c)[0] for c in range(K)]
    rec = {m: [] for m in ("sl_always", "sl_enabled", "bias_acc", "bias_ba")}; recba = {m: [] for m in rec}
    for _ in range(200):
        A = np.concatenate([rng.permutation(ix)[:len(ix) // 2] for ix in by]); mA = np.zeros(len(Ya), bool); mA[A] = True; B = ~mA
        netA = int((new[mA] == Ya[mA]).sum() - (base[mA] == Ya[mA]).sum()); dA = BA(Ya[mA], new[mA]) - BA(Ya[mA], base[mA])
        preds = dict(sl_always=new, sl_enabled=new if (netA > 0 and dA > 0 and len(masks) >= 2) else base,
                     bias_acc=(La + bias_fit(La[mA], Ya[mA], ACC)).argmax(1), bias_ba=(La + bias_fit(La[mA], Ya[mA], BA)).argmax(1))
        for m_, pr in preds.items():
            rec[m_].append(int((pr[B] == Ya[B]).sum() - (base[B] == Ya[B]).sum())); recba[m_].append(100 * (BA(Ya[B], pr[B]) - BA(Ya[B], base[B])))
    r["target_half"] = {m_: dict(net=float(np.mean(rec[m_])), net_sd=float(np.std(rec[m_])), dba=float(np.mean(recba[m_])), dba_sd=float(np.std(recba[m_])),
                                 frac_enabled=None) for m_ in rec}
    r["heads"] = {k: dict(internal=v["internal"]["ba"] * 100, aptos=v["aptos"]["ba"] * 100, aptos_acc=v["aptos"]["acc"] * 100) for k, v in H[s]["heads"].items()}
    R[s] = r
    a = r["always"]
    print(s, "P1 APTOS acc %.2f BA %.2f" % (r["phase1"]["acc"], r["phase1"]["ba"]), "masks", r["valid"], "p", p, "t", t, "tau %.3f" % tau,
          "rev", r["reviewed"], "always %d/%d (%+d) dBA %+.2f" % (a["b"], a["c"], a["net"], a["ba"] - r["phase1"]["ba"]),
          "| half:", {m_: (round(v["net"], 1), round(v["dba"], 2)) for m_, v in r["target_half"].items()}, "| big errors", r["sev_errors"])
json.dump(R, open("r5_aptos_analysis.json", "w"), indent=1, default=float)
for k in H["42"]["heads"]:
    print("%-16s internal BA %.2f  APTOS BA %.2f acc %.2f" % (k, np.mean([R[s]["heads"][k]["internal"] for s in S]), np.mean([R[s]["heads"][k]["aptos"] for s in S]), np.mean([R[s]["heads"][k]["aptos_acc"] for s in S])))
