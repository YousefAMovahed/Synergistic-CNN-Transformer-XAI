# Table of heads on OASIS-1 (participant-level split) and RetinaMNIST/DermaMNIST at 224x224.
import json, numpy as np
R = json.load(open('r3_new_out/results.json'))
S = ["42", "123", "2026"]
DS = [("oasis_subject", "OASIS-1, participant split"), ("retina224", "RetinaMNIST $224^2$"), ("derma224", "DermaMNIST $224^2$")]
ROWS = [("xattn|combined", "Cross-attention (Phase~1)", "combined"), ("xattn|ce", "Cross-attention", "CE"),
        ("late|combined", "Late fusion", "combined"), ("cnn|combined", "ConvNeXt-Small only", "combined"),
        ("swin|combined", "Swin-Small only", "combined")]
best = {d: max(ROWS, key=lambda r: np.mean([R[d][s]["heads"][r[0]]["ba"] for s in S]))[0] for d, _ in DS}
lines = []
for k, name, loss in ROWS:
    cells = []
    for d, _ in DS:
        v = np.array([100 * R[d][s]["heads"][k]["ba"] for s in S]); f = np.array([100 * R[d][s]["heads"][k]["f1"] for s in S])
        b = f"\\textbf{{{v.mean():.2f}}}" if k == best[d] else f"{v.mean():.2f}"
        cells.append(f"{b}$\\pm${v.std(ddof=1):.2f} & {f.mean():.2f}")
    lines.append(f"{name} & {loss} & " + " & ".join(cells) + " \\\\")
tex = r"""\begin{table}[!t]
\centering
\scriptsize
\caption{Heads on frozen ConvNeXt-Small and Swin-Small features for OASIS-1 with a participant-level split and
for RetinaMNIST and DermaMNIST at $224\times224$ pixels (balanced accuracy, mean $\pm$ SD, and macro $F_1$, \%,
seeds 42, 123 and 2026; on OASIS-1 each seed also draws its own participant split). Same optimizer, budget and
checkpoint rule as Phase~1. Best balanced accuracy per dataset in bold.}
\label{tab:newdata}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}llcccccc@{}}
\toprule
& & \multicolumn{2}{c}{\textbf{OASIS-1, participant split}} & \multicolumn{2}{c}{\textbf{RetinaMNIST} $224^2$} & \multicolumn{2}{c}{\textbf{DermaMNIST} $224^2$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}
\textbf{Head} & \textbf{Loss} & BA & $F_1$ & BA & $F_1$ & BA & $F_1$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
open("t_newdata.tex", "w", encoding="utf8").write(tex)
print(tex)
