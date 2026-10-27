# Figure: AD5 test images labeled Severe that SecondLook corrected (seed-42 re-run), with their label in the public
# four-class collection, and the most similar image of the AD5 training directory with its two labels (CPU only).
import os, hashlib, numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PICK = [377, 557, 257, 376]
def find_ad5():
    for d, sub, _ in os.walk('/kaggle/input'):
        if 'train' in sub and 'test' in sub and len(os.listdir(os.path.join(d, 'train'))) == 5: return d
root = find_ad5(); tr = ImageFolder(os.path.join(root, 'train')); te = ImageFolder(os.path.join(root, 'test'))
NAME = {'Servere': 'Severe', 'SevereDemented': 'Severe', 'VeryMildDemented': 'Very Mild', 'Very Mild': 'Very Mild',
        'MildDemented': 'Mild', 'Mild': 'Mild', 'ModerateDemented': 'Moderate', 'Moderate': 'Moderate',
        'NonDemented': 'None', 'None': 'None'}
SRC = {'VeryMildDemented': 'Very Mild Demented', 'NonDemented': 'Non Demented', 'MildDemented': 'Mild Demented',
       'ModerateDemented': 'Moderate Demented'}
src = {}
for d, sub, files in os.walk('/kaggle/input'):
    if d.startswith(root) or 'OriginalDataset' not in d: continue
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            a = np.asarray(Image.open(os.path.join(d, f)).convert('L'), dtype=np.uint8)
            src[hashlib.md5(a.tobytes()).hexdigest()] = os.path.basename(d)
def load(p): return np.asarray(Image.open(p).convert('L'), dtype=np.uint8)
def h(a): return hashlib.md5(a.tobytes()).hexdigest()
def vec(a):
    v = np.asarray(Image.fromarray(a).resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel(); v -= v.mean(); return v / (np.linalg.norm(v) + 1e-9)
TR = [load(p) for p, _ in tr.samples]; V = np.stack([vec(a) for a in TR])
fig, ax = plt.subplots(2, 4, figsize=(8.0, 4.9))
for k, i in enumerate(PICK):
    a = load(te.samples[i][0]); j = int(np.argmax(V @ vec(a))); b = TR[j]
    ax[0, k].imshow(a, cmap='gray'); ax[1, k].imshow(b, cmap='gray')
    ax[0, k].set_title(f"AD5 test: {NAME[te.classes[te.targets[i]]]}\nsource: {SRC.get(src.get(h(a)), '?')}", fontsize=8)
    ax[1, k].set_title(f"AD5 training: {NAME[tr.classes[tr.targets[j]]]}\nsource: {SRC.get(src.get(h(b)), '?')}", fontsize=8)
    print(i, te.samples[i][0], '| nearest train', tr.samples[j][0], float(V[j] @ vec(a)))
for x in ax.ravel(): x.axis('off')
plt.tight_layout(); plt.savefig('/kaggle/working/ad5_annotation_shift.png', dpi=220); plt.savefig('/kaggle/working/ad5_annotation_shift.pdf')
print('done')
