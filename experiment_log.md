# Experiment Log

Validation: same 80/20 split by `breath_id` (seed 42) for every run. MAE is computed on inspiratory rows only (`u_out == 0`).

| # | Name | Key change | Val MAE (insp.) | Test MAE | Notes |
|---|------|-----------|-----------------|----------|-------|
| 1 | baseline | Raw features (R, C, time_step, u_in, u_out), row-level LightGBM | 3.754 | 3.844 | No breath history |
| 2 | lgbm_features | +36 breath-level features (cumulative u_in, lags/leads, diffs, R/C interactions); train and early-stop on inspiratory rows only | 0.532 | 0.532 | area (cumulative u_in) was the strongest feature; hit the 10k-tree cap while still improving |
| 3 | lstm | 4-layer bi-LSTM on whole 80-step breaths, 34 per-timestep features (raw + engineered), masked L1 on inspiratory rows, 200 epochs cosine LR | 0.279 | — | ~2.5 h on 1080 Ti; most gains came after LR decayed (epoch 100: 0.437); train 0.16 vs val 0.28 suggests overfitting |
