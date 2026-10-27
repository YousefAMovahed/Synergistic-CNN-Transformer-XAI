# Ten-seed matched comparison of heads on frozen features (Table tab:matched), with loss weights and linear probes;
# if the learning-rate-tuned runs are available, a second block reports every head with its learning rate chosen on
# validation balanced accuracy.
import json, os, sys, numpy as np
R = json.load(open('r3_heads_gpu/results.json'))
T = {}
for f in ('r3_lr_a/results.json', 'r3_lr_b/results.json'):
    if os.path.exists(f): T.update(json.load(open(f)))
DS = ["AD5", "breastmnist", "retinamnist", "pneumoniamnist", "dermamnist"]
S = ["42", "123", "2026", "0", "1", "2", "3", "4", "5", "6"]
ROWS = [("xattn|combined", r"Cross-attention (Phase~1)", r"combined, $\lambda=0.5$", "1.68 M"),
        ("xattn|ce", "Cross-attention", "CE", "1.68 M"),
        ("late|combined", "Late fusion", r"combined, $\lambda=0.5$", "0.94 M"),
        ("late|ce", "Late fusion", "CE", "0.94 M"),
        ("cnn|combined", "ConvNeXt-Small only", r"combined, $\lambda=0.5$", "0.50 M"),
        ("swin|combined", "Swin-Small only", r"combined, $\lambda=0.5$", "0.50 M"),
        ("xattn|alpha0.25", "Cross-attention", r"combined, $\lambda=0.25$", "1.68 M"),
        ("xattn|alpha0.75", "Cross-attention", r"combined, $\lambda=0.75$", "1.68 M"),
        ("xattn|alpha1.0", "Cross-attention", r"focal only, $\lambda=1$", "1.68 M"),
        ("linear|cnn", "Linear probe, ConvNeXt-Small", r"CE, lr tuned", "$<$0.02 M"),
        ("linear|swin", "Linear probe, Swin-Small", r"CE, lr tuned", "$<$0.02 M"),
        ("linear|both", "Linear probe, both backbones", r"CE, lr tuned", "$<$0.02 M")]
def cell(src, k, ds, bold):
    v = np.array([100 * src[ds][k][s]["ba"] for s in S]); m = f"{v.mean():.2f}"
    return (f"\\textbf{{{m}}}" if bold else m) + f"$\\pm${v.std(ddof=1):.2f}"
lines = []
groups = [ROWS[:6], ROWS[6:9], ROWS[9:]]
heads = ["Heads", r"Loss weight $\lambda$ of the cross-attention head", "Linear probes"]
best = {ds: max([r[0] for r in ROWS[:9]], key=lambda k: np.mean([R[ds][k][s]["ba"] for s in S])) for ds in DS}
for g, h in zip(groups, heads):
    lines.append(r"\multicolumn{8}{@{}l}{\textit{" + h + r"}} \\")
    for k, name, loss, par in g:
        lines.append(f"{name} & {loss} & {par} & " + " & ".join(cell(R, k, ds, best[ds] == k) for ds in DS) + r" \\")
if T and all(ds in T for ds in DS):
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{@{}l}{\textit{Every head with its learning rate chosen on validation from $\{5\times10^{-5}, 2\times10^{-4}, 10^{-3}\}$}} \\")
    for k, name, loss, par in ROWS[:6]:
        lines.append(f"{name} & {loss} & {par} & " + " & ".join(cell(T, k + '|tuned', ds, False) for ds in DS) + r" \\")
tex = r"""\begin{table}[!t]
\centering
\scriptsize
\caption{Matched comparison on frozen ConvNeXt-Small and Swin-Small features (balanced accuracy, \%, mean $\pm$ SD
over ten seeds: 42, 123, 2026 and 0--6). All heads use the optimizer, the five-epoch budget and the checkpoint rule
of Phase~1 (learning rate $5\times10^{-5}$ unless stated); ``combined'' is the focal plus supervised contrastive
loss with weight $\lambda$ on the focal term, CE is cross-entropy. The linear probes act on the mean-pooled tokens
and choose their learning rate from $\{10^{-4}, 10^{-3}, 10^{-2}\}$ on validation. Best value per dataset among
the rows with the Phase~1 learning rate in bold.}
\label{tab:matched}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}llrccccc@{}}
\toprule
\textbf{Head} & \textbf{Loss} & \textbf{Params} & \textbf{AD5} & \textbf{BreastMNIST} & \textbf{RetinaMNIST} & \textbf{PneumoniaMNIST} & \textbf{DermaMNIST} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
open("t_matched.tex", "w", encoding="utf8").write(tex)
print(tex)
