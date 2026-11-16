"""Bidirectional LSTM on whole 80-step breaths for ventilator pressure prediction.

- Inputs per timestep: raw signals (R, C, time_step, u_in, u_out) plus engineered
  features (cumulative u_in, area, lags/leads/diffs, per-breath stats, R/C one-hots).
- Same 80/20 breath_id split (seed 42) as baseline.py / lgbm_features.py.
- Loss and validation MAE are computed on inspiratory rows only (u_out == 0).
- Checkpoints every epoch to checkpoints/lstm/ (last.pt + best.pt); rerunning the
  script resumes from last.pt automatically.
- Writes submission_lstm.csv using the best checkpoint.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
SEQ_LEN = 80
EPOCHS = 200
BATCH_SIZE = 512
LR = 1e-3
CKPT_DIR = "checkpoints/lstm"
LAST_CKPT = os.path.join(CKPT_DIR, "last.pt")
BEST_CKPT = os.path.join(CKPT_DIR, "best.pt")
OUT_FILE = "submission_lstm.csv"

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    g = df.groupby("breath_id")

    df["dt"] = g["time_step"].diff().fillna(0)
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()
    df["u_in_expmean"] = df["u_in_cumsum"] / (g.cumcount() + 1)

    for k in (1, 2, 3, 4):
        df[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0)
        df[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0)
        df[f"u_in_diff{k}"] = df["u_in"] - df[f"u_in_lag{k}"]

    for stat in ("mean", "max", "std"):
        df[f"breath_u_in_{stat}"] = g["u_in"].transform(stat)

    insp_u_in = df["u_in"].where(df["u_out"] == 0, 0)
    df["insp_u_in_cumsum"] = insp_u_in.groupby(df["breath_id"]).cumsum()
    df["insp_u_in_total"] = insp_u_in.groupby(df["breath_id"]).transform("sum")

    df["RC"] = df["R"] * df["C"]
    df["R_div_C"] = df["R"] / df["C"]
    for r in (5, 20, 50):
        df[f"R_{r}"] = (df["R"] == r).astype(np.float32)
    for c in (10, 20, 50):
        df[f"C_{c}"] = (df["C"] == c).astype(np.float32)
    return df


def to_sequences(df: pd.DataFrame, cols) -> np.ndarray:
    return df[cols].to_numpy(np.float32).reshape(-1, SEQ_LEN, len(cols))


# ---------------------------------------------------------------- data
train = add_features(pd.read_csv("data/train.csv"))
test = add_features(pd.read_csv("data/test.csv"))
sample = pd.read_csv("data/sample_submission.csv")
for name, df in (("train", train), ("test", test)):
    sizes = df.groupby("breath_id").size()
    assert (sizes == SEQ_LEN).all(), f"{name}: not all breaths have {SEQ_LEN} steps"

FEATURES = [c for c in train.columns if c not in ("id", "breath_id", "pressure")]
print(f"{len(FEATURES)} features per timestep")

# Same breath_id split as the earlier models
rng = np.random.default_rng(SEED)
breaths = train["breath_id"].unique()
rng.shuffle(breaths)
val_breaths = set(breaths[: int(0.2 * len(breaths))])
is_val = train["breath_id"].isin(val_breaths).to_numpy()

# Standardise using training-split statistics only
mu = train.loc[~is_val, FEATURES].mean()
sd = train.loc[~is_val, FEATURES].std().replace(0, 1)
X_all = to_sequences((train[FEATURES] - mu) / sd, FEATURES)
X_test = to_sequences((test[FEATURES] - mu) / sd, FEATURES)
y_all = train["pressure"].to_numpy(np.float32).reshape(-1, SEQ_LEN)
m_all = (train["u_out"].to_numpy() == 0).astype(np.float32).reshape(-1, SEQ_LEN)

breath_is_val = is_val.reshape(-1, SEQ_LEN)[:, 0]
X_tr, y_tr, m_tr = X_all[~breath_is_val], y_all[~breath_is_val], m_all[~breath_is_val]
X_va, y_va, m_va = X_all[breath_is_val], y_all[breath_is_val], m_all[breath_is_val]
print(f"train breaths: {len(X_tr)}, val breaths: {len(X_va)}")

X_tr_t, y_tr_t, m_tr_t = (torch.from_numpy(a).to(device) for a in (X_tr, y_tr, m_tr))
X_va_t, y_va_t, m_va_t = (torch.from_numpy(a).to(device) for a in (X_va, y_va, m_va))


# ---------------------------------------------------------------- model
class BiLSTM(nn.Module):
    def __init__(self, n_in, hidden=(384, 256, 192, 128)):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, 128), nn.SiLU())
        layers, d = [], 128
        for h in hidden:
            layers.append(nn.LSTM(d, h, batch_first=True, bidirectional=True))
            d = 2 * h
        self.lstms = nn.ModuleList(layers)
        self.head = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.inp(x)
        for lstm in self.lstms:
            x, _ = lstm(x)
        return self.head(x).squeeze(-1)


def masked_mae(pred, y, m):
    return (torch.abs(pred - y) * m).sum() / m.sum()


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = [model(X[i:i + 2048]) for i in range(0, len(X), 2048)]
    return torch.cat(out)


model = BiLSTM(len(FEATURES)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

# ---------------------------------------------------------------- resume
os.makedirs(CKPT_DIR, exist_ok=True)
start_epoch, best_mae = 0, float("inf")
if os.path.exists(LAST_CKPT):
    ck = torch.load(LAST_CKPT, map_location=device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    torch.set_rng_state(ck["torch_rng"])
    torch.cuda.set_rng_state(ck["cuda_rng"])
    start_epoch, best_mae = ck["epoch"] + 1, ck["best_mae"]
    print(f"resumed from {LAST_CKPT}: epoch {start_epoch}, best val MAE {best_mae:.4f}")


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- train
for epoch in range(start_epoch, EPOCHS):
    t0 = time.time()
    model.train()
    perm = torch.randperm(len(X_tr_t), device=device)
    tot, n = 0.0, 0
    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        loss = masked_mae(model(X_tr_t[idx]), y_tr_t[idx], m_tr_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += loss.item() * len(idx)
        n += len(idx)
    sched.step()

    val_mae = masked_mae(predict(model, X_va_t), y_va_t, m_va_t).item()
    improved = val_mae < best_mae
    if improved:
        best_mae = val_mae
        save_atomic({"model": model.state_dict(), "epoch": epoch, "val_mae": val_mae}, BEST_CKPT)
    save_atomic({
        "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        "epoch": epoch, "best_mae": best_mae,
    }, LAST_CKPT)
    print(f"epoch {epoch + 1:3d}/{EPOCHS}  train {tot / n:.4f}  val {val_mae:.4f}"
          f"{' *' if improved else ''}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s",
          flush=True)

# ---------------------------------------------------------------- predict
best = torch.load(BEST_CKPT, map_location=device)
model.load_state_dict(best["model"])
val_mae = masked_mae(predict(model, X_va_t), y_va_t, m_va_t).item()
print(f"best epoch: {best['epoch'] + 1}")
print(f"Validation MAE (inspiratory, u_out==0): {val_mae:.4f}")

test_pred = predict(model, torch.from_numpy(X_test).to(device)).cpu().numpy().reshape(-1)
pred_by_id = pd.Series(test_pred, index=test["id"].to_numpy())
sub = sample[["id"]].copy()
sub["pressure"] = sub["id"].map(pred_by_id)
assert sub["pressure"].notna().all() and len(sub) == len(sample)
sub.to_csv(OUT_FILE, index=False)
print(f"wrote {OUT_FILE} ({len(sub)} rows)")
