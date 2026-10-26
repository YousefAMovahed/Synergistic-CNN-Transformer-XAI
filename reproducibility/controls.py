# Exact controls on the reviewed images of the archived short-budget runs (Table "Exact controls" of the paper).
# Each audit table lists every reviewed test image: predicted class, second-ranked class, class chosen by
# the consensus masks, true label and the logged outcome of SecondLook.
import csv, os
from scipy.stats import binomtest, hypergeom

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_tables')
RUNS = {
    'AD5': ('ad5_short_budget.csv', {'Servere', 'Very Mild'}),
    'BreastMNIST': ('breastmnist_short_budget.csv', {'malignant', 'normal, benign'}),
    'RetinaMNIST': ('retinamnist_short_budget.csv', {'0', '2', '3'}),
    'PneumoniaMNIST': ('pneumoniamnist_short_budget.csv', {'normal', 'pneumonia'}),
    'DermaMNIST': ('dermamnist_short_budget.csv', {'actinic keratoses and intraepithelial carcinoma',
                   'vascular lesions', 'melanoma', 'basal cell carcinoma', 'benign keratosis-like lesions'}),
}

def outcome(rows, switch):
    b = sum(1 for r, s in zip(rows, switch) if s and s != r['pred_1'] and r['pred_1'] != r['true'] and s == r['true'])
    c = sum(1 for r, s in zip(rows, switch) if s and s != r['pred_1'] and r['pred_1'] == r['true'])
    return b, c

total = [0] * 6
for name, (fn, valid) in RUNS.items():
    rows = list(csv.DictReader(open(os.path.join(HERE, fn), encoding='utf8')))
    sl = outcome(rows, [r['pred_2'] if r['mask_class'] == r['pred_2'] else None for r in rows])
    logged = (sum(r['logged_outcome'] == 'corrected' for r in rows), sum(r['logged_outcome'] == 'harmed' for r in rows))
    assert sl == logged, (name, sl, logged)
    unf = outcome(rows, [r['pred_2'] for r in rows])
    gate = outcome(rows, [r['pred_2'] if r['pred_2'] in valid else None for r in rows])
    N, n = len(rows), sl[0] + sl[1] + sum(r['logged_outcome'] == 'wrong_to_wrong' for r in rows)
    good = sum(1 for r in rows if r['pred_1'] != r['true'] and r['pred_2'] == r['true'])
    bad = sum(1 for r in rows if r['pred_1'] == r['true'])
    p_c = hypergeom(N, bad, n).cdf(sl[1]); p_b = hypergeom(N, good, n).sf(sl[0] - 1)
    print(f"{name:15s} reviewed {N:3d} | SecondLook {sl[0]}/{sl[1]} (net {sl[0]-sl[1]:+d}, McNemar p={binomtest(sl[0], sum(sl)).pvalue:.2g})"
          f" | unfiltered {unf[0]}/{unf[1]} (net {unf[0]-unf[1]:+d}) | class gate {gate[0]}/{gate[1]} (net {gate[0]-gate[1]:+d})"
          f" | random E {n*good/N:.1f}/{n*bad/N:.1f}, P(c<=)={p_c:.2g}, P(b>=)={p_b:.2g}")
    for k, v in enumerate(sl + unf + gate): total[k] += v
print(f"Total net: SecondLook {total[0]-total[1]:+d}, unfiltered {total[2]-total[3]:+d}, class gate {total[4]-total[5]:+d}")
