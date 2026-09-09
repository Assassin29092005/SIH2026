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

> **RESULT, 2026-09-10: this approach was measured twice and rejected twice.**
> Run 2 raised warped validation precision 0.109 → 0.575 while real-pair
> matching fell 245 → 63 within one epoch. The per-epoch guard rejected all four
> epochs and no checkpoint was saved. Run 1 failed the same way with 26× less
> data, a higher learning rate, and contaminated sources.
>
> The training signal is the fault: a pair is an image and a warped copy of
> *itself*, photometrically identical, so the cheapest solution to the task is
> exact appearance matching — the precise crutch a matcher must abandon to
> survive a lunar illumination change. More data cannot fix a task that rewards
> unlearning the thing being tested. See `ROADMAP.md`.
>
> The notebook is kept because it is correct, instrumented and cheap to re-run
> against a better training signal. Never adopt its output without
> `scripts/adopt_check.py`.

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
    print("gpu:", torch.cuda.get_device_name(0),
          "| sm_%d%d" % torch.cuda.get_device_capability(0))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Enable the GPU accelerator in Kaggle settings."

# Kaggle ships a torch built for the GPU it gives you. Remember which one, so the
# next cell can tell whether pip swapped it out from under us (BUGS.md BUG-015).
TORCH_BEFORE = torch.__version__
"""

INSTALL = """\
# kornia supplies the LoFTR implementation and the pretrained outdoor weights.
#
# --no-deps is load-bearing. A plain install lets pip resolve `torch` and replace
# Kaggle's GPU-matched build with a generic PyPI wheel. Recent wheels are compiled
# for sm_70 and up, so on a P100 (sm_60) every CUDA kernel then fails with
# "no kernel image is available for execution on the device" -- and not at import,
# but on the first real forward pass, minutes later. See BUGS.md BUG-015.
!pip -q install --no-deps kornia==0.8.3 kornia_rs
from kornia.feature import LoFTR

if torch.__version__ != TORCH_BEFORE:
    raise SystemExit(f"pip replaced torch {TORCH_BEFORE} -> {torch.__version__}. "
                     "Factory-reset the session (Run > Factory reset) and re-run; "
                     "reinstalling in place will not restore the CUDA kernels.")

# Launch one real kernel before spending epochs on a build that cannot run here.
# Checking get_arch_list() alone is not decisive -- PTX entries can JIT for archs
# the list does not name -- so this measures instead of predicting.
cc = torch.cuda.get_device_capability(0)
try:
    (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()
    torch.cuda.synchronize()
    print(f"torch {torch.__version__} | CUDA kernels OK on sm_{cc[0]}{cc[1]}")
except Exception as e:
    # `from None` breaks the exception chain deliberately: IPython's traceback
    # formatter crashes on a SystemExit chained off a CUDA error and buries the
    # message under its own traceback. Measured on Kaggle, 2026-09-09.
    raise SystemExit(
        f"torch {torch.__version__} has no usable kernels for sm_{cc[0]}{cc[1]} "
        f"(built for {torch.cuda.get_arch_list()}).\\n"
        f"Switch the accelerator to GPU T4 x2 (sm_75) in Session options.\\n{e}") from None
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

# Hold out a whole SOURCE SCENE, not random pairs. Pairs from one scene share
# terrain, so a random split would leak the same ground into train and validation
# and the validation number would be optimistic.
#
# Names are "<source>_<row>_<col>_<nn>", where source is "tmc2" or
# "nac-<productid>". Splitting on "-" was correct for the old Kaguya-only naming
# and is wrong here: every TMC crop would become its own scene.
def tile_of(p):
    return p.stem.rsplit("_", 3)[0]

tiles = sorted({tile_of(f) for f in files})
print("scenes:", tiles)
assert len(tiles) >= 2, f"need >=2 scenes to hold one out, got {tiles}"

# Prefer holding out a NAC scene: it keeps the large TMC-2 source in training,
# and validating on a different NAC scene is the generalisation that matters.
# Falls back to the last scene if there is no NAC.
_nac = [t for t in tiles if t.startswith("nac")]
val_tile = _nac[-1] if _nac else tiles[-1]
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

# Freeze the CNN backbone: even ~1200 pairs is far too little to retrain feature
# extraction, and the transformer is where viewpoint reasoning lives. Note this
# was ALREADY true in the run that failed on 2026-09-09 -- freezing the backbone
# is necessary and was not sufficient; the learning rate did the damage.
# The numbers to beat are printed as the baseline below: coarse precision on the
# held-out scene, and match count on the real Kaguya pair.
for p in model.backbone.parameters():
    p.requires_grad = False
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable {sum(p.numel() for p in trainable)/1e6:.2f} M "
      f"of {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
"""

REAL_GUARD = """\
# The 2026-09-09 run improved its warped validation metric 4.5x while real-pair
# matching collapsed (OHRC 1443 -> 6). So the warped metric CANNOT be the
# checkpoint criterion. This scores the model on the genuine Kaguya
# morning/evening sample shipped in the dataset, using the same local-contrast
# normalisation the pipeline uses, and counts matches at the production
# confidence threshold -- the exact quantity that fell last time, every epoch.
import cv2

def _stretch8(img):
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

def _local_contrast(img, k=31):
    x = img.astype(np.float32)
    mu = cv2.blur(x, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(x * x, (k, k)) - mu * mu, 1e-6))
    return _stretch8((x - mu) / sd)

_found = sorted(Path("/kaggle/input").rglob("kaguya_morning.png"))
REAL_PAIR = None
if _found:
    _d = _found[0].parent
    _a = _local_contrast(cv2.imread(str(_d / "kaguya_morning.png"), cv2.IMREAD_UNCHANGED))
    _b = _local_contrast(cv2.imread(str(_d / "kaguya_evening.png"), cv2.IMREAD_UNCHANGED))
    REAL_PAIR = (torch.from_numpy(_a.astype(np.float32) / 255.)[None, None],
                 torch.from_numpy(_b.astype(np.float32) / 255.)[None, None])
    print(f"real-pair guard armed, {_a.shape} from {_d}")
else:
    print("WARNING: kaguya_morning.png not found. The real-pair guard is DISABLED "
          "and selection falls back to the warped metric, which is exactly what "
          "failed on 2026-09-09. Add samples/ to the dataset.")

def real_matches():
    \"\"\"Matches on the genuine cross-illumination pair. Higher is better.\"\"\"
    if REAL_PAIR is None:
        return float("nan")
    with torch.inference_mode():
        out = model({"image0": REAL_PAIR[0].to(DEVICE),
                     "image1": REAL_PAIR[1].to(DEVICE)})
    return int((out["confidence"] >= 0.5).sum().item())
"""

TRAIN = """\
# 4, not 8. The training set went from 96 pairs of one tile to ~1200 across five
# scenes and two sensors, so one epoch is now ~15x more gradient steps than a
# whole run was before. Fewer passes over more data is also the safer side of
# the forgetting trade the last run lost.
EPOCHS = 4
# 1e-4 destroyed real-pair matching in the 2026-09-09 run (OHRC 1443 -> 6) with
# the backbone ALREADY frozen -- so the step size itself was the fault, not which
# parameters were free to move. See BUGS.md BUG-017.
LR = 2e-5
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
base_real = real_matches()
print(f"baseline (pretrained)   val loss {base_loss:.4f}  coarse precision {base_prec:.3f}"
      f"  real-pair matches {base_real}")

history = []
best = base_prec
saved_any = False        # did any epoch clear BOTH bars?
for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_prec = run_split(train_files, train=True)
    va_loss, va_prec = run_split(val_files, train=False)
    sched.step()
    rm = real_matches()
    history.append(dict(epoch=ep, train_loss=tr_loss, train_prec=tr_prec,
                        val_loss=va_loss, val_prec=va_prec, real_matches=rm))

    # Two conditions, not one. The warped metric says the model learned the
    # task; the real-pair count says it did not forget how to match the Moon.
    # Last run the first rose for 8 straight epochs while the second collapsed.
    held = np.isnan(rm) or rm >= 0.8 * base_real
    flag = ""
    if va_prec > best and held:
        best = va_prec
        saved_any = True
        torch.save(model.state_dict(), "loftr_lunar_best.pt")
        flag = "  <- saved"
    elif va_prec > best:
        flag = f"  REJECTED: real matches {rm} < 80% of {base_real}"
    print(f"epoch {ep:>2}  train {tr_loss:.4f}/{tr_prec:.3f}  "
          f"val {va_loss:.4f}/{va_prec:.3f}  real {rm}  {time.time()-t0:.0f}s{flag}")

print(f"\\nbaseline coarse precision {base_prec:.3f} -> best {best:.3f}"
      f"  ({(best-base_prec)*100:+.1f} points)")
if not saved_any:
    print("NO CHECKPOINT SAVED. Every epoch either failed to improve the held-out")
    print(f"metric or dropped real-pair matches below {0.8*base_real:.0f} "
          f"(80% of the {base_real} baseline).")
    print("loftr_lunar_best.pt, if present, is NOT from this run.")
"""

EXPORT = """\
# checkpoint_saved is load-bearing downstream. If no epoch cleared both bars,
# loftr_lunar_best.pt is NOT from this run -- it is whatever was in the working
# directory, which on a re-run is the previous run's file. A fresh report beside
# a stale checkpoint otherwise looks entirely legitimate (BUGS.md BUG-019).
json.dump({"baseline_val_precision": base_prec, "best_val_precision": best,
           "baseline_real_matches": base_real,
           "real_match_floor": 0.8 * base_real,
           "checkpoint_saved": bool(saved_any),
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
        cell("code", REAL_GUARD),
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
