import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
src = open('cm.py', encoding='utf8').read()
# reuse the exact data block (D list) from the first plotting script
exec(src.split('for name,lab,b,f,sup in D:')[0].split('plt.rcParams.update')[1].split('\n', 1)[1])
fig, axes = plt.subplots(2, 3, figsize=(7.4, 6.0), gridspec_kw=dict(wspace=0.36, hspace=0.34, width_ratios=[1, 1, 1.3]))
D = [D[i] for i in (0, 1, 4, 2, 3, 5)]
for k, (name, lab, b, f, sup) in enumerate(D):
    ax = axes.flat[k]
    n = len(lab)
    d = f - b
    m = max(1, np.abs(d).max())
    shade = np.where(d == 0, np.nan, d).astype(float)
    ax.imshow(np.zeros_like(d), cmap="Greys", vmin=0, vmax=1)
    ax.imshow(shade, cmap="RdBu", norm=TwoSlopeNorm(0, -m, m), alpha=0.85)
    fs0 = {2: 11.0, 4: 8.6, 5: 8.2, 7: 7.0}[n]
    for i in range(n):
        for j in range(n):
            txt = f"{b[i, j]}"
            fs = fs0 - 1.0 if (n == 7 and b[i, j] >= 1000) else fs0
            if d[i, j] != 0:
                ax.text(j, i - 0.17, txt, ha="center", va="center", fontsize=fs)
                ax.text(j, i + 0.22, f"{d[i, j]:+d}", ha="center", va="center",
                        fontsize=fs, fontweight="bold",
                        color="white" if abs(d[i, j]) > 0.6 * m else "black")
            else:
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                        color="black" if i == j else "0.35")
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, lw=1.5, ec="black"))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(lab, rotation=90 if n > 5 else 0, fontsize=8)
    ax.set_yticklabels(lab, fontsize=8)
    ax.set_title(name, fontsize=10, pad=3)
    ax.set_xlabel("Predicted", fontsize=8.5, labelpad=2)
    if k % 3 == 0:
        ax.set_ylabel("True", fontsize=8.5, labelpad=2)
    ax.tick_params(length=1.5, pad=1)
fig.savefig("confusion_delta.pdf", bbox_inches="tight")
fig.savefig("confusion_delta.png", dpi=220, bbox_inches="tight")
print("ok")
