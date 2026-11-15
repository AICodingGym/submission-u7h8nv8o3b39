"""Baseline LightGBM model for ventilator pressure prediction.

- 80/20 split by breath_id
- Features: R, C, time_step, u_in, u_out
- Validation MAE on inspiratory rows (u_out == 0)
- Writes submission.csv matching sample_submission.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

SEED = 42
FEATURES = ["R", "C", "time_step", "u_in", "u_out"]
TARGET = "pressure"

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample_submission.csv")

# 1. Split by whole breath_id
rng = np.random.default_rng(SEED)
breaths = train["breath_id"].unique()
rng.shuffle(breaths)
val_breaths = set(breaths[: int(0.2 * len(breaths))])
is_val = train["breath_id"].isin(val_breaths)
tr, va = train[~is_val], train[is_val]
print(f"train breaths: {tr.breath_id.nunique()}, val breaths: {va.breath_id.nunique()}")

# 2. Train LightGBM
params = dict(
    objective="regression_l1",
    learning_rate=0.1,
    num_leaves=255,
    n_estimators=2000,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
model = lgb.LGBMRegressor(**params)
model.fit(
    tr[FEATURES], tr[TARGET],
    eval_set=[(va[FEATURES], va[TARGET])],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)

# 3. Validation MAE on inspiratory rows only
va_pred = model.predict(va[FEATURES], num_iteration=model.best_iteration_)
insp = (va["u_out"] == 0).to_numpy()
mae_insp = np.mean(np.abs(va_pred[insp] - va[TARGET].to_numpy()[insp]))
mae_all = np.mean(np.abs(va_pred - va[TARGET].to_numpy()))
print(f"best_iteration: {model.best_iteration_}")
print(f"Validation MAE (inspiratory, u_out==0): {mae_insp:.4f}")
print(f"Validation MAE (all rows):              {mae_all:.4f}")

# 4. Predict test and write submission in sample_submission format
test_pred = model.predict(test[FEATURES], num_iteration=model.best_iteration_)
pred_by_id = pd.Series(test_pred, index=test["id"])
sub = sample[["id"]].copy()
sub["pressure"] = sub["id"].map(pred_by_id)
assert sub["pressure"].notna().all() and len(sub) == len(sample)
sub.to_csv("submission.csv", index=False)
print(f"wrote submission.csv ({len(sub)} rows)")
