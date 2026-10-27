# Short-budget results of PneumoniaMNIST and DermaMNIST with the standard review rule, recomputed from the Phase 1
# confusion matrices (archived classification reports) and the per-image audit tables of the standard-rule run.
import csv, json, numpy as np
src = open('cm.py', encoding='utf8').read()
exec(src.split('for name,lab,b,f,sup in D:')[0].split('plt.rcParams.update')[1].split('\n', 1)[1])
AUD = 'ghrepo/reproducibility/audit_tables/'
NAMES = {'DermaMNIST': {'actinic keratoses and intraepithelial carcinoma': 'AKIEC', 'basal cell carcinoma': 'BCC',
                        'benign keratosis-like lesions': 'BKL', 'dermatofibroma': 'DF', 'melanoma': 'MEL',
                        'melanocytic nevi': 'NV', 'vascular lesions': 'VASC'},
         'PneumoniaMNIST': {'normal': 'Norm', 'pneumonia': 'Pneu'}}
FILES = {'DermaMNIST': 'dermamnist_short_budget.csv', 'PneumoniaMNIST': 'pneumoniamnist_short_budget.csv'}
def metrics(M):
    tp = np.diag(M).astype(float); pred = M.sum(0); sup = M.sum(1)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0); rec = tp / sup
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    w = sup / sup.sum()
    return dict(prec=prec, rec=rec, f1=f1, acc=tp.sum() / sup.sum(), ba=rec.mean(), mf1=f1.mean(), mprec=prec.mean(),
                wprec=(prec * w).sum(), wrec=(rec * w).sum(), wf1=(f1 * w).sum())
OUT = {}
for name, lab, b, f, sup in D:
    if name not in FILES: continue
    idx = {l: i for i, l in enumerate(lab)}; M = b.copy()
    rows = list(csv.DictReader(open(AUD + FILES[name], encoding='utf8')))
    nb = nc = nw = 0
    for r in rows:
        y1, y2, mk, t = (NAMES[name][r[k]] for k in ('pred_1', 'pred_2', 'mask_class', 'true'))
        if mk != y2: continue
        M[idx[t], idx[y1]] -= 1; M[idx[t], idx[y2]] += 1
        if y1 == t: nc += 1
        elif y2 == t: nb += 1
        else: nw += 1
    assert (M >= 0).all() and list(M.sum(1)) == sup
    m0, m1 = metrics(b), metrics(M)
    print(f"== {name}: b/c/w {nb}/{nc}/{nw}, correct {np.trace(b)} -> {np.trace(M)}")
    print("  acc %.4f->%.4f  BA %.4f->%.4f  macroF1 %.4f->%.4f  macroP %.4f->%.4f  wF1 %.4f->%.4f" % (
        m0['acc'], m1['acc'], m0['ba'], m1['ba'], m0['mf1'], m1['mf1'], m0['mprec'], m1['mprec'], m0['wf1'], m1['wf1']))
    for i, l in enumerate(lab):
        print(f"  {l:6s} P {m0['prec'][i]:.4f}->{m1['prec'][i]:.4f}  R {m0['rec'][i]:.4f}->{m1['rec'][i]:.4f}  F1 {m0['f1'][i]:.4f}->{m1['f1'][i]:.4f}")
    OUT[name] = dict(M=M.tolist(), before=m0, after=m1)
json.dump(OUT, open('r10_standard.json', 'w'), indent=1, default=lambda x: x.tolist())
