# Image statistics of AD5 groups: Severe images of the training directory, Severe test images predicted correctly
# by the hybrid (seed 42), Severe test images predicted as Very Mild, and Very Mild images of both directories.
import os, json, numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
GROUPS = {"sev_as_vm": [257, 268, 270, 273, 318, 358, 370, 371, 376, 377, 412, 416, 419, 425, 430, 431, 443, 447, 449, 450, 458, 462, 463, 464, 467, 491, 499, 508, 513, 524, 528, 530, 535, 541, 556, 557, 564, 566, 567, 569, 578, 579, 580, 582, 585, 586, 589, 591, 596, 600, 602, 603, 604, 608, 611, 612, 613, 626, 627, 628, 630, 631, 632, 636, 637, 639, 641, 644, 645, 648, 649, 652, 653, 656, 657, 664, 665, 666, 671, 675, 676, 683, 685, 689], "sev_ok": [223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 269, 271, 272, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 372, 373, 374, 375, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 413, 414, 415, 417, 418, 420, 421, 422, 423, 424, 426, 427, 428, 429, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 444, 445, 446, 448, 451, 452, 453, 454, 455, 456, 457, 459, 460, 461, 465, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 492, 493, 494, 495, 496, 497, 498, 500, 501, 502, 503, 504, 505, 506, 507, 509, 510, 511, 512, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 525, 526, 527, 529, 531, 532, 533, 534, 536, 537, 538, 539, 540, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 558, 559, 560, 561, 562, 563, 565, 568, 570, 571, 572, 573, 574, 575, 576, 577, 581, 583, 584, 587, 588, 590, 592, 593, 594, 595, 597, 598, 599, 601, 605, 606, 607, 609, 610, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 629, 633, 634, 635, 638, 640, 642, 643, 646, 647, 650, 651, 654, 655, 658, 659, 660, 661, 662, 663, 667, 668, 669, 670, 672, 673, 674, 677, 678, 679, 680, 681, 682, 684, 686, 687, 688, 690, 691]}
def find():
    for d, sub, _ in os.walk("/kaggle/input"):
        if "train" in sub and "test" in sub and len(os.listdir(os.path.join(d, "train"))) == 5: return d
root = find()
tr = ImageFolder(os.path.join(root, "train")); te = ImageFolder(os.path.join(root, "test"))
print(tr.classes, te.classes)
def stats(path):
    im = Image.open(path); w, h = im.size; mode = im.mode
    a = np.asarray(im.convert("L"), dtype=np.float32)
    fg = a > 10
    rows = np.where(fg.any(1))[0]; cols = np.where(fg.any(0))[0]
    bbox = ((rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1) / a.size) if len(rows) else 0.0
    inner = a[fg] if fg.any() else np.zeros(1)
    gy, gx = np.gradient(a)
    return dict(w=w, h=h, mode=mode, mean=float(a.mean()), std=float(a.std()), fg=float(fg.mean()), bbox=float(bbox),
                fg_mean=float(inner.mean()), fg_std=float(inner.std()), sharp=float(np.sqrt(gx**2 + gy**2)[fg].mean()) if fg.any() else 0.0,
                dark_center=float((a[a.shape[0]//3:2*a.shape[0]//3, a.shape[1]//3:2*a.shape[1]//3] < 60).mean()))
def summarize(paths):
    S = [stats(p) for p in paths]
    out = {k: [float(np.mean([s[k] for s in S])), float(np.std([s[k] for s in S]))] for k in ("w", "h", "mean", "std", "fg", "bbox", "fg_mean", "fg_std", "sharp", "dark_center")}
    out["n"] = len(S); out["sizes"] = sorted({f"{s['w']}x{s['h']}" for s in S})[:6]; out["modes"] = sorted({s["mode"] for s in S})
    return out, S
sev_tr = [p for p, y in tr.samples if tr.classes[y].lower().startswith(("sev", "serv"))]
vm_tr = [p for p, y in tr.samples if "very" in tr.classes[y].lower()]
vm_te = [p for p, y in te.samples if "very" in te.classes[y].lower()]
res = {}
for name, paths in (("train_severe", sev_tr), ("test_severe_correct", [te.samples[i][0] for i in GROUPS["sev_ok"]]),
                    ("test_severe_pred_vm", [te.samples[i][0] for i in GROUPS["sev_as_vm"]]), ("train_verymild", vm_tr), ("test_verymild", vm_te)):
    res[name], S = summarize(paths); print(name, json.dumps(res[name]))
# nearest training image (pixel space, 64x64) for each test Severe image: is it a Severe or a Very Mild training image?
def small(p): return np.asarray(Image.open(p).convert("L").resize((64, 64)), dtype=np.float32).ravel() / 255.0
Xtr = np.stack([small(p) for p, _ in tr.samples]); ytr = np.array(tr.targets)
for name, idx in (("test_severe_correct", GROUPS["sev_ok"]), ("test_severe_pred_vm", GROUPS["sev_as_vm"])):
    nn_cls = []
    for i in idx:
        d = ((Xtr - small(te.samples[i][0])) ** 2).sum(1); nn_cls.append(tr.classes[int(ytr[d.argmin()])])
    from collections import Counter
    res[name + "_nn_train_class"] = dict(Counter(nn_cls)); print(name, "nearest training class", dict(Counter(nn_cls)))
json.dump(res, open("/kaggle/working/ad5_image_stats.json", "w"), indent=1)
