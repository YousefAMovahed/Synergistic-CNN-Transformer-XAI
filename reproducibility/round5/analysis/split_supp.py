# Splits the inlined manuscript (all.tex) into a concise main text and a Supplementary Material document.
# Moved blocks keep their labels; xr-hyper resolves the cross-references between the two documents, and the
# supplement numbers its sections, tables and figures with an S prefix.
import re
s = open('all.tex', encoding='utf8').read()

def env_span(text, label, env):
    k = text.index('\\label{' + label + '}')
    a = text.rindex('\\begin{' + env + '}', 0, k)
    b = text.index('\\end{' + env + '}', k) + len('\\end{' + env + '}')
    return a, b

def cut_env(label, env):
    global s
    a, b = env_span(s, label, env)
    block = s[a:b]; s = s[:a] + s[b:]
    return block

def cut_range(start, end, replacement=''):
    global s
    assert s.count(start) == 1, start
    a = s.index(start); b = s.index(end, a)
    block = s[a:b]; s = s[:a] + replacement + s[b:]
    return block

supp = {}
supp['methods'] = [cut_env('fig:baseline_arch', 'figure'), cut_env('tab:settings', 'table'), cut_env('fig:walkthrough', 'figure'),
                   cut_range('\\paragraph{Archived records}', '\\paragraph{Metrics and statistics}')]
supp['classwise'] = [cut_range('\\paragraph{Class-wise behavior of Phase~1}', '\\subsection{Five seeds on identical AD5 splits}')]
supp['robust'] = [cut_range('\\paragraph{Threshold}', '\\paragraph{Five seeds}', open('r7_threshold_summary.tex', encoding='utf8').read().strip() + '\n\n').replace('\\paragraph{Threshold}', '\\paragraph{Threshold sweep}', 1), cut_env('tab:rerun', 'table'), cut_env('tab:alternatives', 'table'),
                  cut_range('\\paragraph{Score and Grad-CAM source}', '\\paragraph{A reversed class prior}',
                            open('r5_robust_summary.tex', encoding='utf8').read().strip() + '\n\n'),
                  cut_range('\\paragraph{BreastMNIST re-runs}', '\\subsection{Participant-level OASIS-1')]
supp['sanity'] = [cut_env('fig:sanity', 'figure')]
supp['conv'] = [cut_range('\\subsection{Convergence protocol}', '\\section{Ablation Study}',
                          open('r5_conv_summary.tex', encoding='utf8').read().strip() + '\n\n%=======================================================================\n')]
supp['appendix'] = [cut_range('\\appendix', '\\bibliographystyle').replace('\\appendix', '', 1)]

# references from the main text to former appendices
s = s.replace('\\ref{app:', 'Supplementary Section~\\ref{app:')
s = s.replace('Supplementary Section~Supplementary Section~', 'Supplementary Section~')
# xr-hyper in the main preamble
assert '\\usepackage[hidelinks]{hyperref}' in s
pre_main = s
s = s.replace('\\usepackage[hidelinks]{hyperref}', '\\usepackage{xr-hyper}\n\\usepackage[hidelinks]{hyperref}\n\\externaldocument{supplement}', 1)
supp_text = '\n'.join(sum(supp.values(), []))
supp_labels = set(re.findall(r'\\label\{([^}]+)\}', supp_text)) | {'sec:supp_methods', 'sec:supp_classwise', 'sec:supp_robust', 'sec:supp_sanity', 'sec:supp_conv'}
for L in supp_labels:
    s = s.replace('\\ref{' + L + '}', '\\ref{S-' + L + '}')
s = s.replace('\\externaldocument{supplement}', '\\externaldocument[S-]{supplement}')
open('all_main.tex', 'w', encoding='utf8').write(s)

# ------------------------------------------------------------------ supplement
head = pre_main[:pre_main.index('\\begin{document}')]
head = head.replace('\\usepackage[hidelinks]{hyperref}', '\\usepackage{xr-hyper}\n\\usepackage[hidelinks]{hyperref}\n\\externaldocument{main}', 1)
body = r"""\begin{document}
\begin{frontmatter}
\title{Supplementary Material for: SecondLook: Auditable Grad-CAM-Guided Review of
Uncertain Predictions in a ConvNeXt--Swin Hybrid for Imbalanced Medical Image
Classification}
\end{frontmatter}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{section}{0}

This supplement contains the material referred to in the main text. Section,
table and figure numbers with the prefix S refer to this document; all other
references point to the main text.

\section{Further details of the methods}
\label{sec:supp_methods}
""" + "\n\n".join(supp['methods']) + r"""

\section{Class-wise short-budget results}
\label{sec:supp_classwise}
""" + "\n\n".join(supp['classwise']) + r"""

\section{Further robustness analyses on AD5}
\label{sec:supp_robust}
""" + "\n\n".join(supp['robust']) + r"""

\section{Faithfulness of the Grad-CAM maps}
\label{sec:supp_sanity}
Figure~\ref{fig:sanity} shows the deletion and insertion curves and the
parameter-randomization check summarized in Section~\ref{sec:gradcam_check}.

""" + supp['sanity'][0] + r"""

\section{Training to convergence}
\label{sec:supp_conv}
""" + supp['conv'][0] + "\n\n" + supp['appendix'][0] + r"""

\bibliographystyle{elsarticle-num}
\bibliography{cas-refs}

\end{document}
"""
main_labels = set(re.findall(r'\\label\{([^}]+)\}', s)) - supp_labels
for L in main_labels:
    body = body.replace('\\ref{' + L + '}', '\\ref{M-' + L + '}')
head = head.replace('\\externaldocument{main}', '\\externaldocument[M-]{main}')
open('all_supp.tex', 'w', encoding='utf8').write(head + body)
print('main chars', len(s), 'supp chars', len(head + body))
