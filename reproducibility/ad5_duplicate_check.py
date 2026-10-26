# Reviewer Q4: near-duplicate images between the AD5 training and test directories.
# Exact duplicates by MD5 of the pixel data; near duplicates by 64-bit difference hash (dHash) and,
# as a second check, by the mean absolute pixel difference of 64x64 grey thumbnails.
import os, sys, io, hashlib, numpy as np
from PIL import Image
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
ROOT = sys.argv[1] if len(sys.argv) > 1 else "dataset Alzheimer"   # folder with train/ and test/
MAP = {'MildDemented': 'Mild', 'ModerateDemented': 'Moderate', 'NonDemented': 'None',
       'SevereDemented': 'Servere', 'VeryMildDemented': 'Very Mild'}

def load(split):
    items = []
    for cls in sorted(os.listdir(os.path.join(ROOT, split))):
        d = os.path.join(ROOT, split, cls)
        for f in sorted(os.listdir(d)):
            im = Image.open(os.path.join(d, f)).convert('L')
            a = np.asarray(im)
            md5 = hashlib.md5(a.tobytes()).hexdigest()
            s = np.asarray(im.resize((9, 8), Image.BILINEAR), dtype=np.int16)
            dh = np.packbits((s[:, 1:] > s[:, :-1]).flatten())
            th = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float32)
            items.append((MAP.get(cls, cls), f, md5, dh, th, a.shape))
    return items

tr, te = load('train'), load('test')
print('train', len(tr), 'test', len(te), 'sizes train', Counter(x[5] for x in tr).most_common(2), 'test', Counter(x[5] for x in te).most_common(2))
tr_md5 = {}
for x in tr: tr_md5.setdefault(x[2], []).append(x[0])
exact = [(x[0], tr_md5[x[2]]) for x in te if x[2] in tr_md5]
print('exact pixel duplicates test->train:', len(exact), Counter((a, tuple(b)) for a, b in exact).most_common(5))
within_test = Counter(x[2] for x in te); print('exact duplicates within test:', sum(v - 1 for v in within_test.values() if v > 1))
within_train = Counter(x[2] for x in tr); print('exact duplicates within train:', sum(v - 1 for v in within_train.values() if v > 1))

TRH = np.unpackbits(np.stack([x[3] for x in tr]), axis=1)
TRT = np.stack([x[4] for x in tr]).reshape(len(tr), -1)
dmin, mad, same_cls = [], [], []
for x in te:
    h = np.unpackbits(x[3])
    d = (TRH != h).sum(1); k = int(d.argmin()); dmin.append(int(d[k]))
    m = np.abs(TRT - x[4].reshape(-1)).mean(1); j = int(m.argmin()); mad.append(float(m[j]))
    same_cls.append((tr[j][0] == x[0], tr[j][0], x[0]))
dmin, mad = np.array(dmin), np.array(mad)
for t in (0, 2, 4, 6):
    print(f'dHash Hamming <= {t}: {int((dmin <= t).sum())} of {len(te)} test images')
for t in (1, 2, 4, 8):
    sel = mad <= t
    agree = sum(1 for s, (ok, _, _) in zip(sel, same_cls) if s and ok)
    print(f'thumbnail MAD <= {t} grey levels: {int(sel.sum())} test images (nearest train image has the same class: {agree})')
print('MAD quantiles (min, 1%, 5%, median):', np.round(np.quantile(mad, [0, .01, .05, .5]), 2))
