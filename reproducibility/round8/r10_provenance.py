# Source labels (public four-class collection) of the AD5 test images that matter for SecondLook, from the
# exact pixel matches computed on Kaggle (prov_out/ad5_matches.csv) and the saved outputs of the five AD5 re-runs.
import csv, json, sys, numpy as np
from collections import Counter
sys.path.insert(0, '.')
from r3_lib import upsample, instance_masks, consensus, scores, review
rows = [r for r in csv.DictReader(open('prov_out/ad5_matches.csv', encoding='utf8')) if r['ad5_split'] == 'test']
rows.sort(key=lambda r: int(r['ad5_index']))
src = np.array([r['matched_label'] for r in rows]); lab = np.array([r['ad5_label'] for r in rows])
NAMES = ['Mild', 'Moderate', 'None', 'Servere', 'Very Mild']           # class index order of the saved outputs
R = {'test_crosstab': {f'{a}->{b}': int(n) for (a, b), n in Counter(zip(lab, src)).items()}}
K = 5; out = {}
for s in ["42", "123", "2026", "0", "1"]:
    z = np.load(f"r3_ad5_out/arrays/hybrid_seed{s}.npz"); Pv, Yv, Pt, Yt = z["Pv"], z["Yv"], z["Pt"], z["Yt"]
    assert all(NAMES[y] == l for y, l in zip(Yt, lab)), 'index order differs'
    Mv = instance_masks(upsample(z["Rv"])); Mt = instance_masks(upsample(z["Rt"])).reshape(len(Pt), -1).astype(np.float32)
    masks = consensus(Mv, Pv, Yv, K); new = review(Pt, scores(Mt, masks, K), masks, 0.6585)[0]; base = Pt.argmax(1)
    sev_vm = (Yt == 3) & (base == 4); corr = (base != Yt) & (new == Yt); harm = (base == Yt) & (new != Yt)
    out[s] = {'severe_pred_vm': dict(Counter(src[sev_vm])), 'corrected': dict(Counter(src[corr])), 'harmed': dict(Counter(src[harm])),
              'severe_pred_severe': dict(Counter(src[(Yt == 3) & (base == 3)]))}
    print(s, out[s])
R['per_seed'] = out
tot = Counter()
for s in out: tot.update(out[s]['corrected'])
R['corrected_total'] = dict(tot); print('corrected, all seeds:', dict(tot))
tot = Counter()
for s in out: tot.update(out[s]['severe_pred_vm'])
print('severe predicted VM, all seeds:', dict(tot))
json.dump(R, open('r10_provenance.json', 'w'), indent=1)
