# OASIS-1 participant-level split: age of the participants per class, and the comparison of the hybrid with the
# ConvNeXt-only head restricted to age-matched test participants (all participants aged 60 or older).
import json, csv, sys, io, numpy as np
from sklearn.metrics import balanced_accuracy_score as BA
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
subjects = json.load(open('oasis_subj/oasis_subjects.json'))
age = {}
for row in csv.DictReader(open('oasis_demo/oasis_cross-sectional.csv', encoding='utf8')):
    if row['ID'].endswith('_MR1'): age[row['ID'][:9]] = float(row['Age'])
def cls(folders):
    f = folders[0].lower()
    return 0 if 'non' in f else 1 if 'very' in f else 2
A = np.array([age[s] for s, _ in subjects]); C = np.array([cls(f) for _, f in subjects])
out = {"age_by_class": {}}
for k in range(3):
    a = A[C == k]; out["age_by_class"][k] = dict(n=int(len(a)), mean=float(a.mean()), sd=float(a.std(ddof=1)), min=float(a.min()), max=float(a.max()),
                                                 n_ge60=int((a >= 60).sum()))
print("age by class:", out["age_by_class"])
def vote(pred, g, y):
    ys, ps = [], []
    for u in np.unique(g):
        i = g == u; ps.append(np.bincount(pred[i], minlength=3).argmax()); ys.append(y[i][0])
    return np.array(ys), np.array(ps)
out["age_matched"] = {}
for s in ["42", "123", "2026"]:
    a = np.load(f'r3_new_out/arrays/oasis_subject_xattn_seed{s}.npz'); c = np.load(f'r3_new_out/arrays/oasis_subject_cnn_seed{s}.npz')
    g, y = a['Gt'], a['Yt']; keep = A[g] >= 60
    pa, pc = a['Pt'].argmax(1), c['Pt'].argmax(1)
    r = dict(n_participants=int(len(np.unique(g[keep]))),
             slice_hybrid=100 * BA(y[keep], pa[keep]), slice_cnn=100 * BA(y[keep], pc[keep]))
    ys, va = vote(pa[keep], g[keep], y[keep]); _, vc = vote(pc[keep], g[keep], y[keep])
    r.update(part_hybrid=100 * BA(ys, va), part_cnn=100 * BA(ys, vc))
    # also: are young controls easier? accuracy of the hybrid on CDR 0 slices by age group
    y0 = (y == 0)
    r["cdr0_acc_young"] = float(100 * (pa[y0 & (A[g] < 60)] == 0).mean()) if (y0 & (A[g] < 60)).any() else None
    r["cdr0_acc_old"] = float(100 * (pa[y0 & (A[g] >= 60)] == 0).mean())
    out["age_matched"][s] = r
    print(s, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
for k in ("slice_hybrid", "slice_cnn", "part_hybrid", "part_cnn"):
    out[k + "_mean"] = float(np.mean([out["age_matched"][s][k] for s in out["age_matched"]]))
print({k: round(v, 2) for k, v in out.items() if k.endswith("_mean")})
json.dump(out, open('r4_age.json', 'w'), indent=1)
