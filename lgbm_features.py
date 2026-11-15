"""LightGBM with breath-level features for ventilator pressure prediction.

Changes vs baseline.py:
- Breath-level features: cumulative u_in, lags/leads, diffs, per-breath stats, R/C interactions
- Train and early-stop on inspiratory rows only (u_out == 0), matching the metric;
  features are still computed from the full breath.
- Same 80/20 breath_id split (same seed) so validation MAE is comparable to the baseline.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

SEED = 42
TARGET = "pressure"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    g = df.groupby("breath_id")

    df["step"] = g.cumcount()
    df["dt"] = g["time_step"].diff().fillna(0)
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()
    df["u_in_expmean"] = df["u_in_cumsum"] / (df["step"] + 1)

    for k in (1, 2, 3, 4):
        df[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0)
        df[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0)
        df[f"u_in_diff{k}"] = df["u_in"] - df[f"u_in_lag{k}"]
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0)
    df["u_out_lead1"] = g["u_out"].shift(-1).fillna(1)

    df["u_in_roll5_mean"] = g["u_in"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["u_in_roll5_max"] = g["u_in"].transform(lambda s: s.rolling(5, min_periods=1).max())

    for stat in ("mean", "max", "std"):
        df[f"breath_u_in_{stat}"] = g["u_in"].transform(stat)
    df["u_in_minus_mean"] = df["u_in"] - df["breath_u_in_mean"]
    df["u_in_minus_max"] = df["u_in"] - df["breath_u_in_max"]

    # Inspiratory-phase context: u_in totals during u_out == 0 only
    insp_u_in = df["u_in"].where(df["u_out"] == 0, 0)
    df["insp_u_in_cumsum"] = insp_u_in.groupby(df["breath_id"]).cumsum()
    df["insp_u_in_total"] = insp_u_in.groupby(df["breath_id"]).transform("sum")

    df["RC"] = df["R"] * df["C"]
    df["R_div_C"] = df["R"] / df["C"]
    df["R_C"] = (df["R"].astype(str) + "_" + df["C"].astype(str)).astype("category")
    return df


train = add_features(pd.read_csv("data/train.csv"))
test = add_features(pd.read_csv("data/test.csv"))
sample = pd.read_csv("data/sample_submission.csv")

# Make R_C category codes consistent between train and test
cats = sorted(set(train["R_C"].cat.categories) | set(test["R_C"].cat.categories))
train["R_C"] = train["R_C"].cat.set_categories(cats)
test["R_C"] = test["R_C"].cat.set_categories(cats)

FEATURES = [c for c in train.columns if c not in ("id", "breath_id", TARGET)]
print(f"{len(FEATURES)} features")

# Same breath_id split as baseline.py
rng = np.random.default_rng(SEED)
breaths = train["breath_id"].unique()
rng.shuffle(breaths)
val_breaths = set(breaths[: int(0.2 * len(breaths))])
is_val = train["breath_id"].isin(val_breaths)
insp = train["u_out"] == 0
tr = train[~is_val & insp]
va = train[is_val & insp]
print(f"train rows (insp): {len(tr)}, val rows (insp): {len(va)}")

model = lgb.LGBMRegressor(
    objective="regression_l1",
    learning_rate=0.05,
    num_leaves=255,
    min_child_samples=50,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    n_estimators=10000,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    tr[FEATURES], tr[TARGET],
    eval_X=(va[FEATURES],), eval_y=(va[TARGET],),
    eval_metric="l1",
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)],
)

va_pred = model.predict(va[FEATURES], num_iteration=model.best_iteration_)
mae = np.mean(np.abs(va_pred - va[TARGET].to_numpy()))
print(f"best_iteration: {model.best_iteration_}")
print(f"Validation MAE (inspiratory, u_out==0): {mae:.4f}")

imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
print("Top features by gain:\n" + (imp / imp.sum()).head(10).round(3).to_string())

# Predict all test rows (expiratory rows are unscored but must be present)
test_pred = model.predict(test[FEATURES], num_iteration=model.best_iteration_)
pred_by_id = pd.Series(test_pred, index=test["id"])
sub = sample[["id"]].copy()
sub["pressure"] = sub["id"].map(pred_by_id)
assert sub["pressure"].notna().all() and len(sub) == len(sample)
sub.to_csv("submission_features.csv", index=False)
print(f"wrote submission_features.csv ({len(sub)} rows)")
