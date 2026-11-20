"""2.5D U-Net (ResNet34 encoder) training with patient-level split and resumable checkpoints.

Checkpoints go to ~/uwgi_checkpoints:
  last.pt   - written at every epoch end and every --ckpt-min minutes mid-epoch (atomic)
  best.pt   - weights with the best validation mean Dice
  train.log - one line per epoch (+ progress lines)
Re-running the same command resumes from last.pt automatically.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import CACHE_DIR, CKPT_DIR, CLASSES, SIZE, UNet, neighbor_index, to_input

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--val-frac", type=float, default=0.2)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--ckpt-min", type=float, default=10.0)
p.add_argument("--max-steps", type=int, default=0, help="debug: stop epoch after N steps")
args = p.parse_args()

os.makedirs(CKPT_DIR, exist_ok=True)
LOG = os.path.join(CKPT_DIR, "train.log")
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- data + split
meta = pd.read_csv(os.path.join(CACHE_DIR, "meta.csv"))
images = np.load(os.path.join(CACHE_DIR, "images.npy"))
masks = np.load(os.path.join(CACHE_DIR, "masks.npy"))
NB = neighbor_index(meta.vol.values)

split_path = os.path.join(CKPT_DIR, "split.json")
if os.path.exists(split_path):
    split = json.load(open(split_path))
else:
    cases = sorted(meta.case.unique())
    rng = np.random.default_rng(args.seed)
    val_cases = sorted(rng.choice(cases, int(round(len(cases) * args.val_frac)), replace=False).tolist())
    split = dict(val_cases=val_cases, train_cases=[c for c in cases if c not in val_cases])
    json.dump(split, open(split_path, "w"), indent=1)
tr_idx = np.flatnonzero(meta.case.isin(split["train_cases"]).values)
va_idx = np.flatnonzero(meta.case.isin(split["val_cases"]).values)
assert not set(meta.case[tr_idx]) & set(meta.case[va_idx])


def get_batch(idx):
    x = torch.from_numpy(images[NB[idx]]).to(dev, non_blocking=True)            # B,3,H,W uint8
    m = torch.from_numpy(masks[idx]).to(dev, non_blocking=True)                 # B,H,W bits
    y = torch.stack([(m >> k) & 1 for k in range(3)], 1).float()
    return to_input(x), y


def augment(x, y):
    """Random affine (rot/scale/shift) + brightness/contrast, on GPU."""
    B = x.shape[0]
    ang = (torch.rand(B, device=dev) - 0.5) * math.radians(30)
    sc = 1 + (torch.rand(B, device=dev) - 0.5) * 0.3
    tx, ty = [(torch.rand(B, device=dev) - 0.5) * 0.15 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    x = x * (1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4) + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4
    return x, y


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)) + 1)
    return 0.5 * bce + 0.5 * dice.mean()


@torch.no_grad()
def validate(model, thr=0.5):
    """Per-slice, per-class Dice at SIZE resolution. Competition convention: both empty -> 1."""
    model.eval()
    dices, pos = [], []
    for s in range(0, len(va_idx), 32):
        idx = va_idx[s:s + 32]
        x, y = get_batch(idx)
        pr = (torch.sigmoid(model(x)) > thr).float()
        inter = (pr * y).sum((2, 3))
        tot = pr.sum((2, 3)) + y.sum((2, 3))
        d = torch.where(tot == 0, torch.ones_like(tot), 2 * inter / tot.clamp(min=1))
        dices.append(d.cpu())
        pos.append((y.sum((2, 3)) > 0).cpu())
    d, pos = torch.cat(dices).numpy(), torch.cat(pos).numpy()
    model.train()
    return d.mean(0), np.array([d[pos[:, k], k].mean() for k in range(3)])


# ---------------------------------------------------------------- model / resume
model = UNet().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps_per_epoch = len(tr_idx) // args.bs
total_steps = steps_per_epoch * args.epochs
WARM = 500


def lr_at(step):
    if step < WARM:
        return args.lr * (step + 1) / WARM
    return 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + math.cos(math.pi * (step - WARM) / max(1, total_steps - WARM)))


state = dict(epoch=0, step_in_epoch=0, best=-1.0, history=[], epoch_time=0.0)
last_path, best_path = os.path.join(CKPT_DIR, "last.pt"), os.path.join(CKPT_DIR, "best.pt")
if os.path.exists(last_path):
    ck = torch.load(last_path, map_location=dev)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    state = ck["state"]
    log(f"RESUMED from last.pt at epoch {state['epoch']} step {state['step_in_epoch']}")
else:
    log(f"START train slices={len(tr_idx)} ({len(split['train_cases'])} cases)  "
        f"val slices={len(va_idx)} ({len(split['val_cases'])} cases)  steps/epoch={steps_per_epoch}  args={vars(args)}")


def save(path, with_opt=True):
    tmp = path + ".tmp"
    d = dict(model=model.state_dict(), state=state, args=vars(args), split=split)
    if with_opt:
        d["opt"] = opt.state_dict()
    torch.save(d, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- train loop
model.train()
last_ck = time.time()
while state["epoch"] < args.epochs:
    ep = state["epoch"]
    perm = np.random.default_rng(args.seed + ep).permutation(tr_idx)   # deterministic -> resumable mid-epoch
    t0 = time.time()
    run_loss, n = 0.0, 0
    for st in range(state["step_in_epoch"], steps_per_epoch):
        if args.max_steps and st >= args.max_steps:
            break
        g = ep * steps_per_epoch + st
        for pg in opt.param_groups:
            pg["lr"] = lr_at(g)
        x, y = augment(*get_batch(np.sort(perm[st * args.bs:(st + 1) * args.bs])))
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += loss.item(); n += 1
        state["step_in_epoch"] = st + 1
        if (st + 1) % 300 == 0:
            log(f"  ep {ep + 1} step {st + 1}/{steps_per_epoch} loss {run_loss / n:.4f} "
                f"lr {lr_at(g):.2e} {(time.time() - t0) / n:.3f}s/step")
        if time.time() - last_ck > args.ckpt_min * 60:
            state["epoch_time"] += time.time() - t0; t0 = time.time()   # carry partial time
            save(last_path); last_ck = time.time()
            log(f"  checkpoint saved (ep {ep + 1} step {st + 1})")
    ep_time = state["epoch_time"] + time.time() - t0
    vd, vpos = validate(model)
    mean = float(vd.mean())
    state["history"].append(dict(epoch=ep + 1, loss=run_loss / max(n, 1), dice=vd.tolist(),
                                 dice_pos=vpos.tolist(), mean=mean, sec=ep_time))
    msg = "  ".join(f"{c}={d:.4f}" for c, d in zip(CLASSES, vd))
    log(f"EPOCH {ep + 1}/{args.epochs} loss {run_loss / max(n, 1):.4f} | val dice {msg} | mean {mean:.4f} | "
        f"pos-only {' '.join(f'{v:.3f}' for v in vpos)} | {ep_time / 60:.1f} min")
    state.update(epoch=ep + 1, step_in_epoch=0, epoch_time=0.0)
    if mean > state["best"]:
        state["best"] = mean
        save(best_path, with_opt=False)
        log(f"  new best {mean:.4f} -> best.pt")
    save(last_path); last_ck = time.time()
    if args.max_steps:
        break
log(f"DONE best mean val dice {state['best']:.4f}")
