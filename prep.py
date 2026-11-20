"""Cache all training slices (resized to SIZE) and masks as .npy for fast training."""
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from common import CACHE_DIR, CLASSES, DATA, SIZE, index_slices, load_volume, rle_decode


def work(args):
    recs, segs = args
    imgs = load_volume(recs)
    masks = np.zeros((len(recs), SIZE, SIZE), np.uint8)       # bit k = class k
    for j, r in enumerate(recs):
        for k, c in enumerate(CLASSES):
            m = rle_decode(segs.get((r["id"], c)), (r["h"], r["w"]))
            if m.any():
                m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                masks[j] |= (m << k).astype(np.uint8)
    return imgs, masks


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA, "train.csv"))
    segs = {(i, c): s for i, c, s in df.dropna().itertuples(index=False)}
    recs = index_slices("train")
    assert len(recs) == df.id.nunique(), (len(recs), df.id.nunique())
    vols = {}
    for r in recs:
        vols.setdefault(r["vol"], []).append(r)
    jobs = [(v, {k: s for k, s in segs.items() if k[0].startswith(name + "_")}) for name, v in vols.items()]
    with Pool(8) as p:
        res = p.map(work, jobs, chunksize=4)
    np.save(os.path.join(CACHE_DIR, "images.npy"), np.concatenate([r[0] for r in res]))
    np.save(os.path.join(CACHE_DIR, "masks.npy"), np.concatenate([r[1] for r in res]))
    pd.DataFrame(recs).drop(columns="path").to_csv(os.path.join(CACHE_DIR, "meta.csv"), index=False)
    print("cached", len(recs), "slices from", len(vols), "volumes")


if __name__ == "__main__":
    main()
