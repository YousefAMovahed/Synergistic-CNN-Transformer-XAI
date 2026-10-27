# Round 8 analyses (manuscript release v1.3-secondlook)

All analyses run on CPU from saved outputs; no model was retrained.

* `ad5_provenance_job.py` (Kaggle, CPU): pixel hashes of every AD5 image against public Alzheimer MRI collections;
  outputs `ad5_provenance.json` and the per-image matches `ad5_matches.csv`.
* `r10_provenance.py`: labels in the four-class collection of the AD5 test images that SecondLook corrected
  (`r10_provenance.json`).
* `analysis_r10.py`: ordinal metrics (QWK, grade error), equal-area templates with null distributions, and the
  target-half comparison including the learned pair rule (`r10_analysis.json`).
* `analysis_r10b.py`: equal-area templates against a random filter of the same size, and the label budget of the
  pair rule and of SecondLook's enablement rule (`r10b_analysis.json`); `r10c.py`: pair rule learned on validation data,
  evaluated on the target halves.
* `r10_standard.py`, `r10_directional.py`, `confusion_figure_standard_rule.py` (with `cm.py`, `cm2.py`): short-budget
  results of PneumoniaMNIST and DermaMNIST with the standard review rule, recomputed from the archived audit tables.

Inputs: the saved probabilities and raw Grad-CAM maps of the five AD5 re-runs (`round3` outputs) and the audit tables
in `reproducibility/audit_tables`.
