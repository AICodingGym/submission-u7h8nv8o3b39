"""Post-processing experiments on the held-out validation cases.

  python postproc.py   # caches val probabilities once, then compares cleanup configs

Per organ: per-slice Dice (as in predict.py --val), an approximate competition-style 3D
Hausdorff score per scan volume (1 - HD/diagonal in voxel-index units; 1 if both empty,
0 if exactly one is empty), and how much predicted area each cleanup step removed.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from common import (CACHE_DIR, CKPT_DIR, CLASSES, DATA, UNet, index_slices, load_volume,
                    neighbor_index, rle_decode, to_input)

THR = 0.4
PROB_DIR = os.path.join(CACHE_DIR, "val_probs")


def cache_probs():
    if os.path.exists(os.path.join(PROB_DIR, "done")):
        return
    os.makedirs(PROB_DIR, exist_ok=True)
    dev = torch.device("cuda")
    ck = torch.load(os.path.join(CKPT_DIR, "best.pt"), map_location=dev)
    model = UNet(pretrained=False).to(dev).eval()
    model.load_state_dict(ck["model"])
    val_cases = set(ck["split"]["val_cases"])
    recs = [r for r in index_slices("train") if r["case"] in val_cases]
    df = pd.read_csv(os.path.join(DATA, "train.csv")).dropna()
    segs = {(i, c): s for i, c, s in df.itertuples(index=False)}
    vols = {}
    for r in recs:
        vols.setdefault(r["vol"], []).append(r)
    for v, vrecs in vols.items():
        h, w = vrecs[0]["h"], vrecs[0]["w"]
        assert all((r["h"], r["w"]) == (h, w) for r in vrecs)
        imgs = load_volume(vrecs)
        nb = neighbor_index([r["vol"] for r in vrecs])
        probs = np.empty((len(vrecs), 3, h, w), np.uint8)
        with torch.no_grad():
            for s in range(0, len(vrecs), 32):
                x = to_input(torch.from_numpy(imgs[nb[s:s + 32]]).to(dev))
                pr = F.interpolate(torch.sigmoid(model(x)), size=(h, w), mode="bilinear",
                                   align_corners=False)
                probs[s:s + 32] = (pr * 255).round().byte().cpu().numpy()
        gt = np.stack([np.stack([rle_decode(segs.get((r["id"], c)), (h, w)) for c in CLASSES])
                       for r in vrecs]).astype(bool)
        np.savez(os.path.join(PROB_DIR, v + ".npz"), probs=probs, gt=gt)
        print("cached", v, probs.shape, flush=True)
    open(os.path.join(PROB_DIR, "done"), "w").close()


# ---------------- cleanup steps: operate on one organ's (n, h, w) bool volume ----------------
def drop_small_2d(m, min_px):
    out, n_removed = m.copy(), 0
    for k in range(len(m)):
        if not m[k].any():
            continue
        lab, n = ndimage.label(m[k])
        sizes = np.bincount(lab.ravel())[1:]
        small = np.flatnonzero(sizes < min_px) + 1
        if len(small):
            out[k][np.isin(lab, small)] = False
            n_removed += len(small)
    return out, n_removed


def drop_small_3d(m, min_vox):
    lab, n = ndimage.label(m)  # 6-connectivity
    if n == 0:
        return m.copy(), 0
    sizes = np.bincount(lab.ravel())[1:]
    small = np.flatnonzero(sizes < min_vox) + 1
    out = m.copy()
    if len(small):
        out[np.isin(lab, small)] = False
    return out, len(small)


def drop_isolated_slices(m):
    has = m.reshape(len(m), -1).any(1)
    nb = np.zeros_like(has)
    nb[1:] |= has[:-1]
    nb[:-1] |= has[1:]
    kill = has & ~nb
    out = m.copy()
    out[kill] = False
    return out, int(kill.sum())


CONFIGS = {
    "baseline": [],
    "cc2d<50": [("2d", 50)],
    "cc2d<100": [("2d", 100)],
    "cc2d<200": [("2d", 200)],
    "cc3d<500": [("3d", 500)],
    "cc3d<2000": [("3d", 2000)],
    "isolated": [("iso", None)],
    "cc2d<100+isolated": [("2d", 100), ("iso", None)],
    "cc3d<2000+isolated": [("3d", 2000), ("iso", None)],
}


def apply(m, steps):
    stats = {}
    for kind, p in steps:
        before = m.sum()
        m, n = {"2d": lambda: drop_small_2d(m, p), "3d": lambda: drop_small_3d(m, p),
                "iso": lambda: drop_isolated_slices(m)}[kind]()
        stats[kind] = (n, int(before - m.sum()))
    return m, stats


def hausdorff_score(pred, gt):
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    # exact HD within the bounding box of both sets (nearest points always lie inside it)
    idx = np.argwhere(pred | gt)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, g = pred[sl], gt[sl]
    d1 = ndimage.distance_transform_edt(~g)[p].max()
    d2 = ndimage.distance_transform_edt(~p)[g].max()
    return 1.0 - max(d1, d2) / np.sqrt(np.sum(np.square(pred.shape)))


def main():
    cache_probs()
    files = sorted(f for f in os.listdir(PROB_DIR) if f.endswith(".npz"))
    R = {c: {k: dict(dice=[], hd=[], comps=0, px=0, slices=0, tp_px=0) for k in range(3)}
         for c in CONFIGS}
    total_pred_px = np.zeros(3, np.int64)
    for f in files:
        z = np.load(os.path.join(PROB_DIR, f))
        pred_all, gt_all = z["probs"] > THR * 255, z["gt"]
        for k in range(3):
            base, gt = pred_all[:, k], gt_all[:, k]
            total_pred_px[k] += base.sum()
            for c, steps in CONFIGS.items():
                m, st = apply(base, steps)
                r = R[c][k]
                inter = (m & gt).sum((1, 2))
                tot = m.sum((1, 2)) + gt.sum((1, 2))
                r["dice"].extend(np.where(tot == 0, 1.0, 2 * inter / np.maximum(tot, 1)))
                r["hd"].append(hausdorff_score(m, gt))
                for kind, (n, px) in st.items():
                    if kind == "iso":
                        r["slices"] += n
                    else:
                        r["comps"] += n
                removed = base & ~m
                r["px"] += int(removed.sum())
                r["tp_px"] += int((removed & gt).sum())
        print("scored", f, flush=True)

    out = {}
    print(f"\n{len(files)} validation scan volumes, threshold {THR}. "
          "Score = 0.4*Dice + 0.6*HD (approximation of the competition metric)\n")
    for c in CONFIGS:
        rows = []
        for k, cls in enumerate(CLASSES):
            r = R[c][k]
            rows.append(dict(cls=cls, dice=np.mean(r["dice"]), hd=np.mean(r["hd"]), comps=r["comps"],
                             slices=r["slices"], px=r["px"], px_frac=r["px"] / total_pred_px[k],
                             tp_px=r["tp_px"]))
        out[c] = rows
        d = np.mean([x["dice"] for x in rows])
        h = np.mean([x["hd"] for x in rows])
        print(f"== {c}: mean Dice {d:.4f}  mean HD {h:.4f}  score {0.4 * d + 0.6 * h:.4f}")
        for x in rows:
            print(f"   {x['cls']:<12} Dice {x['dice']:.4f}  HD {x['hd']:.4f}  | removed {x['comps']:>4} comps, "
                  f"{x['slices']:>3} isolated slices, {x['px']:>7} px ({100 * x['px_frac']:.2f}% of predicted; "
                  f"{x['tp_px']} px were true positives)")
    json.dump(out, open(os.path.join(CKPT_DIR, "postproc_val.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
