"""Helper functions for the Finale notebook (Shopee multimodal product matching)."""
import re
import unicodedata
import numpy as np
import pandas as pd
from PIL import Image

# threshold search grid (scores [0, 1] range me hain)
THR_GRID = np.round(np.arange(0.30, 1.0, 0.02), 2)


# ------------------------------------------------------------------ data / text / image
def clean_title(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\w\s]|_", " ", s)          # punctuation hatao, digits aur letters (CJK bhi) rakho
    return re.sub(r"\s+", " ", s).strip()


def load_img(img_dir, fname):
    try:
        return Image.open(f"{img_dir}/{fname}").convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224))


def l2n(E):
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-12)


def make_subset(full, n_images, seed=42):
    """Poore label_groups chunta hai (taaki positives na toote). Part C jaisa hi logic."""
    if not n_images:
        return full.copy()
    rng = np.random.default_rng(seed)
    gs = full.groupby("label_group").size()
    order = rng.permutation(gs.index.values)
    cum = gs.loc[order].cumsum().values
    keep = order[: np.searchsorted(cum, n_images) + 1]
    return full[full["label_group"].isin(keep)].reset_index(drop=True)


# ------------------------------------------------------------------ retrieval-style metric
def true_matrix(labels):
    """true[i, j] = True agar listing i aur j same product (same label_group) hain."""
    return labels[:, None] == labels[None, :]


def predict_matrix(S, thr, force=None):
    """Listing i ke liye predicted set = {j : S[i,j] >= thr} (+ forced matches). Khud ko hamesha include."""
    pred = S >= thr
    if force is not None:
        pred |= force
    np.fill_diagonal(pred, True)
    return pred


def set_metrics(pred, true):
    """Har row ka precision, recall, F1 (Kaggle metric = mean of row-wise F1)."""
    inter = (pred & true).sum(1)
    npred, ntrue = pred.sum(1), true.sum(1)
    return inter / npred, inter / ntrue, 2 * inter / (npred + ntrue)


def evaluate(S, true, thr, force=None):
    p, r, f = set_metrics(predict_matrix(S, thr, force), true)
    return dict(precision=float(p.mean()), recall=float(r.mean()), f1=float(f.mean()))


def tune_threshold(S, true, force=None, grid=THR_GRID):
    """Best threshold (max mean-F1). Sirf VAL par call karna."""
    f1s = [evaluate(S, true, t, force)["f1"] for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), float(f1s[i])


# ------------------------------------------------------------------ fusion
def fuse(S_text, S_img, w):
    """Score-level fusion: w * text + (1 - w) * image."""
    return (w * S_text + (1 - w) * S_img).astype(np.float32)


def sweep_weights(S_text, S_img, true, force=None, ws=None):
    ws = np.round(np.linspace(0, 1, 11), 1) if ws is None else ws
    rows = []
    for w in ws:
        thr, f1 = tune_threshold(fuse(S_text, S_img, w), true, force)
        rows.append(dict(w_text=float(w), thr=thr, val_F1=f1))
    return pd.DataFrame(rows)