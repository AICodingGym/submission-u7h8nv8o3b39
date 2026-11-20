"""Shared helpers: paths, RLE, slice indexing, volume loading, model."""
import glob
import os
import re

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
HOME = os.path.expanduser("~")
CKPT_DIR = os.environ.get("UWGI_CKPT", os.path.join(HOME, "uwgi_checkpoints"))
CACHE_DIR = os.environ.get("UWGI_CACHE", "/tmp/claude-164299/-home-pjlittle-uw-madison-gi-tract-image-segmentation/56365629-45a6-46af-9826-c1360453a325/scratchpad/cache")  # regenerable via prep.py; kept off the quota-limited home
CLASSES = ["large_bowel", "small_bowel", "stomach"]
SIZE = 320          # network input resolution (square)
CTX = 2             # 2.5D: channels are slices (i-CTX, i, i+CTX)


# ---------------------------------------------------------------- RLE
# train.csv is row-major (C order) despite description.md; verified visually.
def rle_decode(s, shape):
    m = np.zeros(shape[0] * shape[1], np.uint8)
    if isinstance(s, str) and s:
        a = np.asarray(s.split(), dtype=np.int64)
        for st, ln in zip(a[0::2] - 1, a[1::2]):
            m[st:st + ln] = 1
    return m.reshape(shape)


def rle_encode(mask):
    p = np.concatenate([[0], mask.reshape(-1).astype(np.uint8), [0]])
    runs = np.where(p[1:] != p[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs))


# ---------------------------------------------------------------- indexing
_PAT = re.compile(r"slice_(\d+)_(\d+)_(\d+)_")


def index_slices(split):
    """Return list of dicts: id, case, vol, slice, path, h, w — sorted by volume/slice."""
    out = []
    for p in glob.glob(os.path.join(DATA, split, "*", "*", "scans", "*.png")):
        vol = p.split(os.sep)[-3]                     # e.g. case101_day20
        sl, w, h = map(int, _PAT.search(os.path.basename(p)).groups())
        out.append(dict(id=f"{vol}_slice_{sl:04d}", case=vol.split("_")[0], vol=vol,
                        slice=sl, path=p, h=h, w=w))
    out.sort(key=lambda d: (d["vol"], d["slice"]))
    return out


def load_volume(recs):
    """Read all slices of one volume, normalise by the volume's 99.8th percentile,
    resize to SIZE, return uint8 array (n, SIZE, SIZE)."""
    imgs = [cv2.imread(r["path"], cv2.IMREAD_UNCHANGED).astype(np.float32) for r in recs]
    hi = np.percentile(np.stack([i[::4, ::4] for i in imgs]), 99.8) + 1e-6
    out = np.empty((len(imgs), SIZE, SIZE), np.uint8)
    for k, im in enumerate(imgs):
        im = np.clip(im / hi, 0, 1) * 255
        out[k] = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA).round().astype(np.uint8)
    return out


def neighbor_index(vol_ids):
    """For each slice row i, indices of (i-CTX, i, i+CTX) clamped within its volume.
    vol_ids must be sorted so a volume's slices are contiguous and in order."""
    n = len(vol_ids)
    vol_ids = np.asarray(vol_ids)
    starts = np.zeros(n, np.int64)
    ends = np.zeros(n, np.int64)
    b = np.flatnonzero(np.r_[True, vol_ids[1:] != vol_ids[:-1], True])
    for s, e in zip(b[:-1], b[1:]):
        starts[s:e], ends[s:e] = s, e - 1
    i = np.arange(n)
    return np.stack([np.clip(i - CTX, starts, ends), i, np.clip(i + CTX, starts, ends)], 1)


# ---------------------------------------------------------------- model
class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net decoder on a timm ImageNet-pretrained encoder."""

    def __init__(self, encoder="resnet34", n_classes=3, pretrained=True):
        super().__init__()
        self.enc = timm.create_model(encoder, features_only=True, pretrained=pretrained)
        ch = self.enc.feature_info.channels()            # resnet34: 64,64,128,256,512 at /2../32
        dec = [256, 128, 64, 32, 16]
        self.blocks = nn.ModuleList()
        cin = ch[-1]
        skips = ch[:-1][::-1] + [0]
        for d, s in zip(dec, skips):
            self.blocks.append(ConvBlock(cin + s, d))
            cin = d
        self.head = nn.Conv2d(dec[-1], n_classes, 1)

    def forward(self, x):
        feats = self.enc(x)
        y = feats[-1]
        skips = feats[:-1][::-1] + [None]
        for blk, s in zip(self.blocks, skips):
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            if s is not None:
                y = torch.cat([y, s], 1)
            y = blk(y)
        return self.head(y)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_input(u8):
    """uint8 (B,3,H,W) tensor on device -> normalised float."""
    x = u8.float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)
