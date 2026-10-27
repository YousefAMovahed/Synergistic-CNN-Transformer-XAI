# Mask comparison versus the directional rule on the four MedMNIST tasks with the standard rule (audit tables).
import csv, sys, io
from scipy.stats import binomtest
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
H = 'ghrepo/reproducibility/audit_tables/'
LARGEST = {'breastmnist_short_budget.csv': 'malignant', 'retinamnist_short_budget.csv': '2',
           'pneumoniamnist_short_budget.csv': 'pneumonia', 'dermamnist_short_budget.csv': 'actinic keratoses and intraepithelial carcinoma'}
tot_sl = tot_dr = right_sl = right_dr = n_dis = 0
for fn, big in LARGEST.items():
    rows = list(csv.DictReader(open(H + fn, encoding='utf8')))
    share = sum(r['mask_class'] == big for r in rows) / len(rows)
    fin_sl = [r['pred_2'] if r['mask_class'] == r['pred_2'] else r['pred_1'] for r in rows]
    fin_dr = [r['pred_2'] if r['pred_2'] == big else r['pred_1'] for r in rows]
    net = lambda fin: sum((f == r['true']) - (r['pred_1'] == r['true']) for f, r in zip(fin, rows))
    dis = [(r, a, b) for r, a, b in zip(rows, fin_sl, fin_dr) if a != b]
    rs = sum(a == r['true'] for r, a, b in dis); rd = sum(b == r['true'] for r, a, b in dis)
    print(f"{fn:34s} largest-mask share {share:.2f} | net SL {net(fin_sl):+d} dir {net(fin_dr):+d} | disagree {len(dis)}: SL right {rs}, dir right {rd}")
    tot_sl += net(fin_sl); tot_dr += net(fin_dr); right_sl += rs; right_dr += rd; n_dis += len(dis)
print(f"total net SL {tot_sl:+d}, directional {tot_dr:+d}; disagreements {n_dis}: SL right {right_sl}, dir right {right_dr}, "
      f"sign test p={binomtest(right_sl, right_sl + right_dr).pvalue:.3g}")
