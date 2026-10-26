# Exact controls on the reviewed images of the archived short-budget runs (Table "Exact controls" of the paper).
# Each audit table lists every reviewed test image: predicted class, second-ranked class, class chosen by
# the consensus masks, true label and the logged outcome of SecondLook.
import csv, os
from scipy.stats import binomtest, hypergeom

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_tables')
STANDARD = lambda r: r['pred_2'] if r['mask_class'] == r['pred_2'] else None
MASK_DIRECT = lambda r: r['mask_class']                                    # PneumoniaMNIST, adapted rule
P2_COND = lambda r: r['pred_2'] if (r['mask_class'] == r['pred_2'] and float(r['p2']) > 0.20) else None  # DermaMNIST
DERMA_VALID = {'actinic keratoses and intraepithelial carcinoma', 'vascular lesions', 'melanoma',
               'basal cell carcinoma', 'benign keratosis-like lesions'}
RUNS = [  # (name, audit table, rule, classes with a valid mask, counted in the paper's total)
    ('AD5', 'ad5_short_budget.csv', STANDARD, {'Servere', 'Very Mild'}, True),
    ('BreastMNIST', 'breastmnist_short_budget.csv', STANDARD, {'malignant', 'normal, benign'}, True),
    ('RetinaMNIST', 'retinamnist_short_budget.csv', STANDARD, {'0', '2', '3'}, True),
    ('PneumoniaMNIST (adapted rule)', 'posthoc_pneumoniamnist_modified_rule.csv', MASK_DIRECT, {'normal', 'pneumonia'}, True),
    ('DermaMNIST (adapted rule)', 'posthoc_dermamnist_p2_condition.csv', P2_COND, DERMA_VALID, True),
    ('PneumoniaMNIST (standard rule)', 'pneumoniamnist_short_budget.csv', STANDARD, {'normal', 'pneumonia'}, False),
    ('DermaMNIST (standard rule)', 'dermamnist_short_budget.csv', STANDARD, DERMA_VALID, False),
]

def outcome(rows, switch):
    ch = [(r, s) for r, s in zip(rows, switch) if s and s != r['pred_1']]
    b = sum(1 for r, s in ch if r['pred_1'] != r['true'] and s == r['true'])
    c = sum(1 for r, s in ch if r['pred_1'] == r['true'])
    return b, c, len(ch)

total = [0] * 6
for name, fn, rule, valid, counted in RUNS:
    rows = list(csv.DictReader(open(os.path.join(HERE, fn), encoding='utf8')))
    b, c, n = outcome(rows, [rule(r) for r in rows])
    logged = (sum(r['logged_outcome'] == 'corrected' for r in rows), sum(r['logged_outcome'] == 'harmed' for r in rows))
    assert (b, c) == logged, (name, (b, c), logged)
    ub, uc, _ = outcome(rows, [r['pred_2'] for r in rows])
    gb, gc, _ = outcome(rows, [r['pred_2'] if r['pred_2'] in valid else None for r in rows])
    N = len(rows)
    good = sum(1 for r in rows if r['pred_1'] != r['true'] and r['pred_2'] == r['true'])
    bad = sum(1 for r in rows if r['pred_1'] == r['true'])
    p_c = hypergeom(N, bad, n).cdf(c); p_b = hypergeom(N, good, n).sf(b - 1)
    print(f"{name:31s} reviewed {N:3d} | SecondLook {b}/{c} (net {b-c:+d}, McNemar p={binomtest(b, b + c).pvalue:.2g})"
          f" | unfiltered {ub}/{uc} (net {ub-uc:+d}) | class gate {gb}/{gc} (net {gb-gc:+d})"
          f" | random E {n*good/N:.1f}/{n*bad/N:.1f}, P(c<=)={p_c:.2g}, P(b>=)={p_b:.2g}")
    if counted:
        for k, v in enumerate((b, c, ub, uc, gb, gc)): total[k] += v
print(f"Total net (paper table): SecondLook {total[0]-total[1]:+d}, unfiltered {total[2]-total[3]:+d}, class gate {total[4]-total[5]:+d}")
