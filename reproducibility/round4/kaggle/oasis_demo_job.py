# Copies the OASIS-1 cross-sectional demographics table (age, sex, CDR, MMSE) to the output.
import os, shutil
for d, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower().startswith("oasis_cross"):
            shutil.copy(os.path.join(d, f), "/kaggle/working/" + f); print("copied", f)
