"""Inference.

  python predict.py --val      # Dice at ORIGINAL resolution on the held-out val cases, per organ,
                               # for several thresholds (exercises the full resize-back + RLE path)
  python predict.py --thr 0.5  # write submission.csv for data/test in test.csv row order
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import (CKPT_DIR, CLASSES, DATA, ROOT, UNet, index_slices, load_volume, neighbor_index,
                    rle_decode, rle_encode, to_input)

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.join(CKPT_DIR, "best.pt"))
p.add_argument("--val", action="store_true")
p.add_argument("--thr", type=float, default=0.5)
p.add_argument("--min-vox", type=int, default=2000,
               help="drop 3D connected components smaller than this per organ per scan (0 = off)")
p.add_argument("--out", default=os.path.join(ROOT, "submission.csv"))
args = p.parse_args()
dev = torch.device("cuda")

ck = torch.load(args.ckpt, map_location=dev)
model = UNet(pretrained=False).to(dev).eval()
model.load_state_dict(ck["model"])
print("loaded", args.ckpt, "epoch", ck["state"]["epoch"], "best", round(ck["state"]["best"], 4))


@torch.no_grad()
def predict_volumes(recs):
    """Yield (rec, probs (3,h,w) on GPU at original resolution) for each slice."""
    vols = {}
    for r in recs:
        vols.setdefault(r["vol"], []).append(r)
    for vrecs in vols.values():
        imgs = load_volume(vrecs)
        nb = neighbor_index([r["vol"] for r in vrecs])
        for s in range(0, len(vrecs), 32):
            x = to_input(torch.from_numpy(imgs[nb[s:s + 32]]).to(dev))
            pr = torch.sigmoid(model(x))
            for j, r in enumerate(vrecs[s:s + 32]):
                yield r, F.interpolate(pr[j:j + 1], size=(r["h"], r["w"]), mode="bilinear",
                                       align_corners=False)[0]


if args.val:
    val_cases = set(ck["split"]["val_cases"])
    recs = [r for r in index_slices("train") if r["case"] in val_cases]
    df = pd.read_csv(os.path.join(DATA, "train.csv"))
    segs = {(i, c): s for i, c, s in df.dropna().itertuples(index=False)}
    thrs = [0.3, 0.4, 0.5, 0.6]
    D = {t: [] for t in thrs}
    for r, pr in predict_volumes(recs):
        gt = np.stack([rle_decode(segs.get((r["id"], c)), (r["h"], r["w"])) for c in CLASSES]).astype(bool)
        prn = pr.cpu().numpy()
        for t in thrs:
            # round-trip through the submission RLE encoder so it is checked too
            b = np.stack([rle_decode(rle_encode(m), m.shape) for m in (prn > t)]).astype(bool)
            inter, tot = (b & gt).sum((1, 2)), b.sum((1, 2)) + gt.sum((1, 2))
            D[t].append(np.where(tot == 0, 1.0, 2 * inter / np.maximum(tot, 1)))
    res = {}
    for t in thrs:
        d = np.stack(D[t]).mean(0)
        res[t] = d.tolist()
        print(f"thr {t}: " + "  ".join(f"{c}={v:.4f}" for c, v in zip(CLASSES, d)) + f"  mean={d.mean():.4f}")
    json.dump(res, open(os.path.join(CKPT_DIR, "val_fullres.json"), "w"), indent=1)
else:
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    recs = index_slices("test")
    from postproc import drop_small_3d
    preds, buf, removed = {}, [], 0

    def flush():
        global removed
        if not buf:
            return
        vol = np.stack([b for _, b in buf])  # (n, 3, h, w) bool
        for k, c in enumerate(CLASSES):
            m = vol[:, k]
            if args.min_vox:
                m, n = drop_small_3d(m, args.min_vox)
                removed += n
            for (r, _), mk in zip(buf, m):
                preds[(r["id"], c)] = rle_encode(mk.astype(np.uint8)) if mk.any() else ""
        buf.clear()

    for r, pr in predict_volumes(recs):
        if buf and buf[-1][0]["vol"] != r["vol"]:
            flush()
        buf.append((r, (pr > args.thr).cpu().numpy()))
    flush()
    print(f"3D cleanup (min {args.min_vox} voxels) removed {removed} components")
    missing = [k for k in zip(test.id, test["class"]) if k not in preds]
    assert not missing, f"{len(missing)} test rows have no image, e.g. {missing[:3]}"
    test["predicted"] = [preds[k] for k in zip(test.id, test["class"])]
    test[["id", "class", "predicted"]].to_csv(args.out, index=False)
    print("wrote", args.out, len(test), "rows;", (test.predicted != "").sum(), "non-empty")
