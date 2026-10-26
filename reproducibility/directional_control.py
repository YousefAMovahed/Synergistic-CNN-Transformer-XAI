# Reviewer Q1: does the per-image mask comparison add anything over a fixed directional rule
# ("switch to y2 only if y2 is the class with the largest consensus mask")? Exact, on the audit tables.
import csv, os, sys, io
from scipy.stats import binomtest, hypergeom
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_tables')
LARGEST = {'ad5_short_budget.csv': 'Servere', 'breastmnist_short_budget.csv': 'malignant',
           'retinamnist_short_budget.csv': '2', 'pneumoniamnist_short_budget.csv': 'pneumonia',
           'dermamnist_short_budget.csv': 'actinic keratoses and intraepithelial carcinoma',
           'posthoc_dermamnist_p2_condition.csv': 'actinic keratoses and intraepithelial carcinoma'}
def res(rows, sw):
    b = c = w = 0
    for r, s in zip(rows, sw):
        if not s or s == r['pred_1']: continue
        if r['pred_1'] == r['true']: c += 1
        elif s == r['true']: b += 1
        else: w += 1
    return b, c, w
for fn, big in LARGEST.items():
    rows = list(csv.DictReader(open(os.path.join(H, fn), encoding='utf8')))
    p2c = 'p2' in fn and 'posthoc' in fn
    sl = [r['pred_2'] if r['mask_class'] == r['pred_2'] and (not p2c or float(r['p2']) > 0.2) else None for r in rows]
    dr = [r['pred_2'] if r['pred_2'] == big and (not p2c or float(r['p2']) > 0.2) else None for r in rows]
    a, d = res(rows, sl), res(rows, dr)
    # rows where the two rules disagree
    dis = [(r, s1, s2) for r, s1, s2 in zip(rows, sl, dr) if s1 != s2]
    only_sl = res([x[0] for x in dis], [x[1] for x in dis]); only_dr = res([x[0] for x in dis], [x[2] for x in dis])
    print(f"{fn:42s} SL {a[0]}/{a[1]}/{a[2]} net {a[0]-a[1]:+d} | directional->{big[:12]} {d[0]}/{d[1]}/{d[2]} net {d[0]-d[1]:+d} | disagree {len(dis)}: SL-only {only_sl}, dir-only {only_dr}")
