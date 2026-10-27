# Table: four classifiers trained on the same AD5 splits (five seeds), before and after SecondLook (tau = 0.6585).
import json, numpy as np, sys, io
from sklearn.metrics import recall_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
R = json.load(open('r3_ad5_out/results.json'))
S = ["42", "123", "2026", "0", "1"]
MOD = [("hybrid", r"Hybrid (Phase~1)", "1.68 M"), ("convnext", r"ConvNeXt-S only", "0.50 M"),
       ("resnet50", r"ResNet-50", "23.5 M"), ("effv2s", r"EffNetV2-S", "20.2 M")]
def ms(v): v = np.array(v); return f"{v.mean():.2f}$\\pm${v.std(ddof=1):.2f}"
rows = []; stats = {}
for key, name, params in MOD:
    a0 = [100 * R[s][key]["phase1"]["acc"] for s in S]; b0 = [100 * R[s][key]["phase1"]["ba"] for s in S]; f0 = [100 * R[s][key]["phase1"]["f1"] for s in S]
    a1 = [100 * R[s][key]["tau_orig"]["acc"] for s in S]; b1 = [100 * R[s][key]["tau_orig"]["ba"] for s in S]; f1 = [100 * R[s][key]["tau_orig"]["f1"] for s in S]
    bc = (sum(R[s][key]["tau_orig"]["corrected"] for s in S), sum(R[s][key]["tau_orig"]["harmed"] for s in S))
    none, sev = [], []
    for s in S:
        z = np.load(f"r3_ad5_out/arrays/{key}_seed{s}.npz"); Y = z["Yt"]; p = z["Pt"].argmax(1)
        none.append(100 * recall_score(Y, p, labels=[2], average=None)[0]); sev.append(int(((Y == 3) & (p == 4)).sum()))
    stats[key] = dict(a0=a0, b0=b0, f0=f0, a1=a1, b1=b1, f1=f1, none=none, sev=sev, bc=bc)
    rows.append(f"{name} & {params} & {ms(a0)} & {ms(b0)} & {ms(f0)} & {ms(none)} & {min(sev)}--{max(sev)} & {bc[0]}/{bc[1]} & {ms(a1)} & {ms(b1)} & {ms(f1)} \\\\")
    print(key, "acc", np.round(a0, 2), "BA", np.round(b0, 2), "F1", np.round(f0, 2), "None", np.round(none, 1), "SevVM", sev, "SL", bc, "BA1", np.round(b1, 2), "F1_1", np.round(f1, 2), "acc1", np.round(a1, 2))
A4 = json.load(open('r4a_analysis.json'))
rows.append(r"\midrule")
rows.append(r"\multicolumn{11}{@{}l}{	extit{Round 4: same splits; frozen CNN backbones with the Phase-1 head, and Phase~1 with cross-entropy}} \\")
for key, name, params in (("resnet50head", "ResNet-50, frozen", "0.99 M"), ("effv2shead", "EffNetV2-S, frozen", "0.69 M"), ("hybridce", "Hybrid, CE loss", "1.68 M")):
    g = lambda k, t: [A4[key][s][t][k] for s in S]
    sev = [A4[key][s]["sev_vm"] for s in S]; none = [A4[key][s]["none_recall"] for s in S]
    bc = (sum(A4[key][s]["t06585"]["b"] for s in S), sum(A4[key][s]["t06585"]["c"] for s in S))
    rows.append(f"{name} & {params} & {ms(g('acc','phase1'))} & {ms(g('ba','phase1'))} & {ms(g('f1','phase1'))} & {ms(none)} & {min(sev)}--{max(sev)} & {bc[0]}/{bc[1]} & {ms(g('acc','t06585'))} & {ms(g('ba','t06585'))} & {ms(g('f1','t06585'))} \\\\")
    print(key, round(np.mean(g('acc','phase1')),2), round(np.mean(g('ba','phase1')),2), round(np.mean(g('f1','phase1')),2), bc, round(np.mean(none),1))
h, c = np.array(stats["hybrid"]["b0"]), np.array(stats["convnext"]["b0"])
from scipy.stats import ttest_rel
print("hybrid - convnext BA per seed", np.round(h - c, 2), "mean", round((h - c).mean(), 2), "paired t p", ttest_rel(h, c).pvalue)
for k in ["convnext", "resnet50", "effv2s"]:
    print(k, "hybrid BA higher in", int((h > np.array(stats[k]["b0"])).sum()), "/5; pipeline F1 higher in", int((np.array(stats["hybrid"]["f1"]) > np.array(stats[k]["f0"])).sum()),
          "; pipeline BA higher in", int((np.array(stats["hybrid"]["b1"]) > np.array(stats[k]["b0"])).sum()))
tex = r"""\begin{table}[!t]
\centering
\scriptsize
\caption{Classifiers trained on the same AD5 splits with seeds 42, 123, 2026, 0 and 1 (mean $\pm$ SD over
seeds, \%), before and after SecondLook with the archived settings ($\tau=0.6585$, default masks). The hybrid and
the ConvNeXt-only model share the frozen ConvNeXt-Small backbone, the head width, the loss, the optimizer, the
budget and the checkpoint rule; in the upper block ResNet-50 and EfficientNetV2-S are fine-tuned in full with the
archived baseline recipe, in the lower block they are frozen and trained with the Phase-1 head and recipe, and
the hybrid is also trained with cross-entropy. Params: trained parameters. None: recall of the non-demented class. Sev$\to$VM: Severe test images predicted as Very Mild (range
over seeds). $b$/$c$: corrections/harms of SecondLook summed over the five seeds.}
\label{tab:fourmodels}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}lrccccccccc@{}}
\toprule
& & \multicolumn{5}{c}{\textbf{Classifier alone}} & & \multicolumn{3}{c}{\textbf{With SecondLook}} \\
\cmidrule(lr){3-7}\cmidrule(l){9-11}
\textbf{Model} & \textbf{Params} & \textbf{Acc} & \textbf{BA} & $F_1$ & \textbf{None} & \textbf{Sev$\to$VM} & $b$/$c$ & \textbf{Acc} & \textbf{BA} & $F_1$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
open("t_fourmodels.tex", "w", encoding="utf8").write(tex)
json.dump(stats, open("r3_fourmodels_stats.json", "w"))
