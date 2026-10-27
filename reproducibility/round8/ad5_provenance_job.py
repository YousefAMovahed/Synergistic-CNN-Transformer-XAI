# Provenance and near-duplicate check for AD5 (CPU only).
# (1) exact and near duplicates between and within the AD5 training and test directories;
# (2) matches of every AD5 image in public four-class Alzheimer MRI collections, with a cross-tabulation of labels.
import os, json, hashlib, csv, numpy as np
from collections import Counter, defaultdict
from PIL import Image
from torchvision.datasets import ImageFolder
EXT = ('.jpg', '.jpeg', '.png', '.bmp')

def find_ad5():
    for d, sub, _ in os.walk('/kaggle/input'):
        if 'train' in sub and 'test' in sub and len(os.listdir(os.path.join(d, 'train'))) == 5:
            return d
root = find_ad5(); print('AD5 root', root)
tr = ImageFolder(os.path.join(root, 'train')); te = ImageFolder(os.path.join(root, 'test'))
print(tr.classes, te.classes, len(tr), len(te))

def load(p):
    im = Image.open(p).convert('L'); a = np.asarray(im, dtype=np.uint8)
    v = np.asarray(im.resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
    v = v - v.mean(); n = np.linalg.norm(v); v = v / n if n > 0 else v
    return hashlib.md5(a.tobytes()).hexdigest(), f'{a.shape[1]}x{a.shape[0]}', v

def describe(paths):
    H, S, V = [], [], []
    for p in paths:
        h, s, v = load(p); H.append(h); S.append(s); V.append(v)
    return H, S, np.stack(V)

res = {}
tr_p = [p for p, _ in tr.samples]; tr_y = [tr.classes[y] for y in tr.targets]
te_p = [p for p, _ in te.samples]; te_y = [te.classes[y] for y in te.targets]
Htr, Str, Vtr = describe(tr_p); Hte, Ste, Vte = describe(te_p)
res['sizes'] = {'train': dict(Counter(Str)), 'test': dict(Counter(Ste))}

# ---- AD5 internal duplicates
C = Vte @ Vtr.T
nn = C.argmax(1); best = C.max(1)
exact_te = [Hte[i] in set(Htr) for i in range(len(te_p))]
res['test_vs_train'] = {
    'exact': int(sum(exact_te)),
    'near_0.99': int((best >= 0.99).sum()), 'near_0.98': int((best >= 0.98).sum()), 'near_0.95': int((best >= 0.95).sum()),
    'near_0.99_label_pairs': dict(Counter(f'{te_y[i]}<-{tr_y[nn[i]]}' for i in range(len(te_p)) if best[i] >= 0.99)),
    'near_0.98_label_pairs': dict(Counter(f'{te_y[i]}<-{tr_y[nn[i]]}' for i in range(len(te_p)) if best[i] >= 0.98)),
}
Ct = Vtr @ Vtr.T; np.fill_diagonal(Ct, -1)
bt = Ct.max(1); nt = Ct.argmax(1)
res['train_vs_train'] = {
    'exact_dupe_images': int(sum(c for c in Counter(Htr).values() if c > 1)),
    'near_0.99': int((bt >= 0.99).sum()), 'near_0.98': int((bt >= 0.98).sum()), 'near_0.95': int((bt >= 0.95).sum()),
    'near_0.99_same_label': int(sum(1 for i in range(len(tr_p)) if bt[i] >= 0.99 and tr_y[i] == tr_y[nt[i]])),
}
print(json.dumps(res, indent=1))

# ---- public four-class collections
sources = defaultdict(list)
for d, sub, files in os.walk('/kaggle/input'):
    if d.startswith(root): continue
    low = d.lower()
    if 'augment' in low and 'original' not in low: continue          # skip augmented copies, keep originals
    for f in files:
        if f.lower().endswith(EXT):
            top = d[len('/kaggle/input/'):].split('/')[0]
            sources[top].append((os.path.join(d, f), os.path.basename(d)))
res['sources'] = {}
rows = []
for name, items in sources.items():
    print('source', name, len(items))
    Hs, Ss, Vs = describe([p for p, _ in items]); ys = [lab for _, lab in items]; hs = defaultdict(list)
    for j, h in enumerate(Hs): hs[h].append(j)
    out = {'n': len(items), 'sizes': dict(Counter(Ss).most_common(5)), 'labels': dict(Counter(ys))}
    for split, P, Y, H, V in (('train', tr_p, tr_y, Htr, Vtr), ('test', te_p, te_y, Hte, Vte)):
        Cs = V @ Vs.T; j = Cs.argmax(1); b = Cs.max(1)
        ex = [bool(hs.get(H[i])) for i in range(len(P))]
        lab = [ys[hs[H[i]][0]] if ex[i] else (ys[j[i]] if b[i] >= 0.99 else None) for i in range(len(P))]
        out[split] = {'exact': int(sum(ex)), 'near_0.99': int((b >= 0.99).sum()),
                      'crosstab': dict(Counter(f'{Y[i]}->{lab[i]}' for i in range(len(P)) if lab[i] is not None))}
        for i in range(len(P)):
            rows.append([name, split, i, os.path.basename(P[i]), Y[i], int(ex[i]), round(float(b[i]), 4), ys[j[i]], lab[i] or ''])
    res['sources'][name] = out
    print(name, json.dumps(out, indent=1))

# Severe test images that the seed-42 hybrid of the five re-runs predicted as Very Mild (indices from the saved outputs)
grp = json.load(open('/kaggle/input/ad5-groups/ad5_groups.json')) if os.path.exists('/kaggle/input/ad5-groups/ad5_groups.json') else None
json.dump(res, open('/kaggle/working/ad5_provenance.json', 'w'), indent=1)
with open('/kaggle/working/ad5_matches.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['source', 'ad5_split', 'ad5_index', 'file', 'ad5_label', 'exact', 'best_corr', 'nn_label', 'matched_label']); w.writerows(rows)
print('done')
