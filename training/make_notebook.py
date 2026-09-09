"""Generate the Kaggle fine-tuning notebook.

Kept as a generator rather than a checked-in .ipynb so the code is reviewable as
Python and the notebook is reproducible. Run it, upload the .ipynb and the
training pairs to Kaggle, and run there.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MD_INTRO = """\
# SANDHI — LoFTR fine-tuning for viewpoint robustness

**Target: viewpoint, not illumination.** That choice is measured, not assumed.

Cross-illumination looked like the obvious target — it is the project's headline
difficulty — but it cannot be labelled. Kaguya morning and evening are
independently orthorectified, so their correspondence is identity plus a
translation that must be *estimated*. Estimating it with phase correlation, the
only method independent of the matcher being trained, gives a median
half-vs-half disagreement of **5.16 px**. The pipeline already achieves
**0.513–0.796 px**. Training on labels an order of magnitude coarser than the
model would teach it our own noise, and using the pipeline's own output as
labels is circular.

Viewpoint is the opposite case:

* it is the **weakest measured axis** — real LROC NAC pairs stop matching beyond
  a ~12.3° emission-angle gap;
* its labels are **exact by construction**, because a homography applied to a
  real image gives correspondence that is applied rather than estimated.

Pixels are always genuine lunar observations. Only the geometry is imposed, and
only for training — **every reported number stays on real pairs behind the
four-control gate**, on a tile held out from training.
"""

MD_ACCEPT = """\
## Acceptance criteria — decide these before looking at results

Fine-tuned weights are adopted **only** if, on the held-out tile:

1. the four-control gate still passes (noise and flat grey must not match, and
   rolling the raw input by N must move the recovered offset by −N), and
2. the real NAC obliquity ladder improves — matching must succeed past the
   current 12.3° ceiling, and
3. cross-illumination performance does **not** regress: RMSE stays sub-pixel on
   the Kaguya and OHRC sample cases.

A model that improves obliquity while breaking illumination is a regression, not
a result. Run `pytest -m slow` and `python scripts/viewpoint.py --ladder`
locally against the exported weights before adopting them.
"""

SETUP = """\
import os, json, math, time, zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Enable the GPU accelerator in Kaggle settings."
"""

INSTALL = """\
# kornia supplies the LoFTR implementation and the pretrained outdoor weights.
!pip -q install kornia==0.8.3
from kornia.feature import LoFTR
"""

DATA = """\
# Locate the attached dataset by SEARCHING, not by assuming a mount path. The
# path depends on the dataset slug, and Kaggle may or may not have extracted the
# uploaded archive. Guessing wrong printed "0 training pairs" and then raised an
# IndexError three lines later, which reads as a code bug rather than a missing
# input (BUGS.md BUG-014).
import zipfile

def find_pairs():
    roots = [Path("/kaggle/input"), Path(".")]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("*.npz"))
            if hits:
                return hits
    for root in roots:                      # nothing loose: unpack an archive once
        for z in (sorted(root.rglob("*.zip")) if root.exists() else []):
            with zipfile.ZipFile(z) as zf:
                if any(n.endswith(".npz") for n in zf.namelist()):
                    zf.extractall("/kaggle/working/trainset")
                    return sorted(Path("/kaggle/working/trainset").rglob("*.npz"))
    return []

files = find_pairs()
if not files:
    seen = sorted(str(p) for p in Path("/kaggle/input").rglob("*")) \\
           if Path("/kaggle/input").exists() else ["/kaggle/input does not exist"]
    raise SystemExit("No .npz pairs found. Attach the sandhi-trainset dataset via "
                     "'+ Add Input' (right panel), then re-run.\\n/kaggle/input holds:\\n  "
                     + "\\n  ".join(seen[:40]))
print(f"{len(files)} training pairs under {files[0].parent}")

# Hold out a whole TILE, not random pairs. Pairs from one tile share terrain, so
# a random split would leak the same ground into train and validation and the
# validation number would be optimistic.
def tile_of(p):
    return p.stem.split("-")[0]

tiles = sorted({tile_of(f) for f in files})
print("tiles:", tiles)
assert len(tiles) >= 2, f"need >=2 tiles to hold one out, got {tiles}"
val_tile = tiles[-1]
train_files = [f for f in files if tile_of(f) != val_tile]
val_files   = [f for f in files if tile_of(f) == val_tile]
print(f"train {len(train_files)} pairs | val {len(val_files)} pairs (held-out tile {val_tile})")
"""

GT = """\
STRIDE = 8

def coarse_assignment(H, size, stride=STRIDE):
    \"\"\"GT coarse-cell assignment implied by H. Mirrors sandhi.training.\"\"\"
    import cv2
    g = size // stride
    ys, xs = np.mgrid[0:g, 0:g]
    centres = np.column_stack([(xs.ravel() + .5) * stride,
                               (ys.ravel() + .5) * stride]).astype(np.float64)
    mapped = cv2.perspectiveTransform(centres.reshape(-1, 1, 2), H).reshape(-1, 2)
    cj = np.floor(mapped[:, 0] / stride).astype(int)
    ri = np.floor(mapped[:, 1] / stride).astype(int)
    ok = (cj >= 0) & (cj < g) & (ri >= 0) & (ri < g)
    return np.arange(g * g)[ok], (ri[ok] * g + cj[ok]), g


def normalise8(img, k=31):
    \"\"\"Local contrast normalisation - the same front end the pipeline uses.\"\"\"
    import cv2
    x = img.astype(np.float32)
    mu = cv2.blur(x, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(x * x, (k, k)) - mu * mu, 1e-6))
    z = (x - mu) / sd
    lo, hi = np.percentile(z, [1, 99])
    return np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def load_pair(path, size=512):
    d = np.load(path)
    a = normalise8(d["image0"])
    b = normalise8(d["image1"])
    i, j, g = coarse_assignment(d["H"], size)
    return (torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None],
            torch.from_numpy(i).long(), torch.from_numpy(j).long(), g)
"""

LOSS = """\
def focal_bce(conf, i_idx, j_idx, g, alpha=0.25, gamma=2.0, eps=1e-6):
    \"\"\"LoFTR's coarse loss: focal BCE on the confidence matrix.

    Positives are the GT cell pairs. Negatives are every other entry, which
    hugely outnumber them - hence focal weighting rather than plain BCE.
    \"\"\"
    conf = conf.clamp(eps, 1 - eps)[0]                # (L, S)
    pos = conf[i_idx, j_idx]
    loss_pos = -alpha * (1 - pos).pow(gamma) * pos.log()

    neg_mask = torch.ones_like(conf, dtype=torch.bool)
    neg_mask[i_idx, j_idx] = False
    neg = conf[neg_mask]
    loss_neg = -(1 - alpha) * neg.pow(gamma) * (1 - neg).log()
    return loss_pos.mean() + loss_neg.mean()


@torch.no_grad()
def coarse_precision(conf, i_idx, j_idx):
    \"\"\"Fraction of GT source cells whose argmax lands on the GT target cell.

    Reported instead of loss because it is interpretable: it is the coarse-level
    match accuracy the fine stage then refines.
    \"\"\"
    pred = conf[0][i_idx].argmax(dim=1)
    return (pred == j_idx).float().mean().item()
"""

MODEL = """\
model = LoFTR(pretrained="outdoor").to(DEVICE)

# Capture the pre-threshold confidence matrix. LoFTR's public forward returns
# only thresholded keypoints, which are not differentiable; the coarse loss
# needs conf_matrix, which CoarseMatching writes into its data dict.
_captured = {}
def _hook(mod, inputs, output):
    d = inputs[-1]
    if isinstance(d, dict) and "conf_matrix" in d:
        _captured["conf"] = d["conf_matrix"]
model.coarse_matching.register_forward_hook(_hook)

def forward_conf(a, b):
    _captured.clear()
    model({"image0": a, "image1": b})
    if "conf" not in _captured:
        raise RuntimeError("conf_matrix not captured - kornia internals changed")
    return _captured["conf"]

# Freeze the CNN backbone: 96 pairs is far too little to retrain feature
# extraction, and the transformer is where viewpoint reasoning lives.
# Baseline coarse precision measured locally on one pair: 0.340 - that is the
# number fine-tuning has to beat.
for p in model.backbone.parameters():
    p.requires_grad = False
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable {sum(p.numel() for p in trainable)/1e6:.2f} M "
      f"of {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
"""

TRAIN = """\
EPOCHS = 8
LR = 1e-4
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

# NOTE: the model stays in eval() even while training, and gradients are enabled
# explicitly. In train() mode kornia's get_coarse_match expects supervision keys
# ("spv_b_ids") that only its own training harness supplies, and raises KeyError.
# eval() is also the better choice at batch size 1: BatchNorm running statistics
# are more stable than per-batch statistics computed from a single sample.
model.eval()

def run_split(files, train: bool):
    losses, precs = [], []
    for path in files:
        a, b, i_idx, j_idx, g = load_pair(path)
        a, b = a.to(DEVICE), b.to(DEVICE)
        i_idx, j_idx = i_idx.to(DEVICE), j_idx.to(DEVICE)
        with torch.set_grad_enabled(train):
            conf = forward_conf(a, b)
            loss = focal_bce(conf, i_idx, j_idx, g)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        losses.append(loss.item())
        precs.append(coarse_precision(conf.detach(), i_idx, j_idx))
    return float(np.mean(losses)), float(np.mean(precs))

# Baseline BEFORE any training, on the held-out tile. Without this the training
# curve has nothing to be compared against.
base_loss, base_prec = run_split(val_files, train=False)
print(f"baseline (pretrained)   val loss {base_loss:.4f}  coarse precision {base_prec:.3f}")

history = []
best = base_prec
for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_prec = run_split(train_files, train=True)
    va_loss, va_prec = run_split(val_files, train=False)
    sched.step()
    history.append(dict(epoch=ep, train_loss=tr_loss, train_prec=tr_prec,
                        val_loss=va_loss, val_prec=va_prec))
    flag = ""
    if va_prec > best:
        best = va_prec
        torch.save(model.state_dict(), "loftr_lunar_best.pt")
        flag = "  <- saved"
    print(f"epoch {ep:>2}  train {tr_loss:.4f}/{tr_prec:.3f}  "
          f"val {va_loss:.4f}/{va_prec:.3f}  {time.time()-t0:.0f}s{flag}")

print(f"\\nbaseline coarse precision {base_prec:.3f} -> best {best:.3f}"
      f"  ({(best-base_prec)*100:+.1f} points)")
if best <= base_prec:
    print("NO IMPROVEMENT. Do not adopt these weights.")
"""

EXPORT = """\
json.dump({"baseline_val_precision": base_prec, "best_val_precision": best,
           "held_out_tile": val_tile, "epochs": EPOCHS, "lr": LR,
           "history": history},
          open("finetune_report.json", "w"), indent=2)
print(open("finetune_report.json").read()[:600])
"""

MD_NEXT = """\
## After training

Download `loftr_lunar_best.pt` and `finetune_report.json`, then evaluate
**locally on real data** — coarse precision on warped pairs is a training
signal, not a result:

```bash
# 1. controls must still pass
python -c "from sandhi import controls; ..."   # or: sandhi controls --case ohrc

# 2. the real obliquity ladder must improve past 12.3 deg
python scripts/viewpoint.py --ladder

# 3. illumination must not regress
pytest -m slow
```

Adopt the weights only if all three hold. Wire them in by pointing
`sandhi.matching.loftr_model` at the checkpoint; keep the pretrained path as the
default until the ladder actually moves.
"""


def cell(kind, src):
    src = src.rstrip("\n").split("\n")
    src = [ln + "\n" for ln in src[:-1]] + [src[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


def build():
    cells = [
        cell("markdown", MD_INTRO),
        cell("markdown", MD_ACCEPT),
        cell("code", SETUP),
        cell("code", INSTALL),
        cell("code", DATA),
        cell("code", GT),
        cell("code", LOSS),
        cell("code", MODEL),
        cell("code", TRAIN),
        cell("code", EXPORT),
        cell("markdown", MD_NEXT),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "finetune_loftr.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}")
