# Writes the sorted participant list of the OASIS Kaggle release exactly as built by the round-3 script
# (participant id, class folder, acquisition used), so that group indices can be mapped to participant ids.
import os, json
def find_oasis():
    for d, sub, _ in os.walk("/kaggle/input"):
        low = [s.lower() for s in sub]
        if any("non" in s and "demented" in s for s in low) and any("very" in s for s in low):
            return d
root = find_oasis(); per = {}
for folder in sorted(os.listdir(root)):
    if not os.path.isdir(os.path.join(root, folder)): continue
    for fn in os.listdir(os.path.join(root, folder)):
        p = fn.split("_")
        if len(p) < 5 or not fn.lower().endswith((".jpg", ".png")): continue
        per.setdefault(p[0] + "_" + p[1], set()).add(folder)
subjects = sorted(per)
json.dump([[s, sorted(per[s])] for s in subjects], open("/kaggle/working/oasis_subjects.json", "w"))
print(len(subjects))
