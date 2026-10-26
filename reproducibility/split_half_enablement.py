# Enablement rule with a validation split that follows the test distribution: the AD5 test directory is split
# in two class-stratified halves; half A decides (net > 0 and balanced accuracy up), half B is evaluated.
# Exact enumeration over the hypergeometric allocation of corrected and harmed images to half A.
import itertools
from math import comb
SUPPORT = {'Mild': 49, 'Moderate': 153, 'None': 21, 'Severe': 469, 'Very Mild': 29}
K = 5
RUNS = {  # (class of the image, corrected?) -> count, from the per-image audit tables
    'archived': {('Severe', 1): 60, ('Very Mild', 0): 2},
    'seed 42': {('Severe', 1): 58, ('Very Mild', 0): 2},
    'seed 123': {('Severe', 1): 67, ('Very Mild', 0): 1, ('Moderate', 0): 25},
    'seed 2026': {('Severe', 1): 50, ('Very Mild', 0): 2},
    'seed 0': {('Moderate', 0): 41},
    'seed 1': {('Severe', 1): 79, ('Very Mild', 0): 2},
}
def hyp(N, k, n, x): return comb(k, x) * comb(N - k, n - x) / comb(N, n)
for name, ch in RUNS.items():
    groups = list(ch.items())
    ranges = []
    for (cls, corr), cnt in groups:
        N = SUPPORT[cls]; n = N // 2
        ranges.append([(x, hyp(N, cnt, n, x)) for x in range(0, cnt + 1) if hyp(N, cnt, n, x) > 0])
    p_en = 0.0; e_net_B = 0.0; e_net_B_en = 0.0
    for combo in itertools.product(*ranges):
        p = 1.0; netA = 0; dba = 0.0; netB = 0
        for ((cls, corr), cnt), (x, px) in zip(groups, combo):
            p *= px; nA = SUPPORT[cls] // 2
            netA += x if corr else -x; dba += (x if corr else -x) / nA / K
            netB += (cnt - x) if corr else -(cnt - x)
        en = netA > 0 and dba > 0
        p_en += p * en; e_net_B += p * netB; e_net_B_en += p * netB * en
    print(f"{name:9s} P(enabled on half A) = {p_en:.4f} | expected net on half B: always {e_net_B:+.1f}, under the rule {e_net_B_en:+.1f}")
