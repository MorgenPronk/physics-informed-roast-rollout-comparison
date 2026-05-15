# Final test metrics (HPO sweep)

## Seed 11

| Model | One-step R^2 | Rollout R^2 | Rollout R^2 95% CI | Rollout RMSE | Params |
|---|---:|---:|---|---:|---:|
| Mechanistic | 0.9848 | -0.4376 | [-0.6551, -0.2164] | 41.7920 | 369 |
| Physics Informed Model | 0.9848 | 0.7027 | [0.6480, 0.7611] | 19.0046 | 11250 |
| Residual LSTM | 0.8601 | 0.9633 | [0.9435, 0.9823] | 6.6774 | 6337 |
| Neural Net Baseline | 0.9996 | 0.9697 | [0.9560, 0.9805] | 6.0669 | 705 |

## Seed 23

| Model | One-step R^2 | Rollout R^2 | Rollout R^2 95% CI | Rollout RMSE | Params |
|---|---:|---:|---|---:|---:|
| Mechanistic | 0.9848 | -1.6652 | [-1.9571, -1.3859] | 56.9025 | 369 |
| Physics Informed Model | 0.9848 | 0.6985 | [0.6154, 0.7842] | 19.1372 | 11250 |
| Residual LSTM | 0.8809 | 0.9517 | [0.9230, 0.9793] | 7.6592 | 6337 |
| Neural Net Baseline | 0.9995 | 0.9414 | [0.9216, 0.9593] | 8.4401 | 705 |

## Seed 37

| Model | One-step R^2 | Rollout R^2 | Rollout R^2 95% CI | Rollout RMSE | Params |
|---|---:|---:|---|---:|---:|
| Mechanistic | 0.9848 | -0.4990 | [-0.7293, -0.2791] | 42.6750 | 369 |
| Physics Informed Model | 0.9848 | 0.6970 | [0.6234, 0.7621] | 19.1863 | 11250 |
| Residual LSTM | 0.8324 | 0.9588 | [0.9339, 0.9800] | 7.0706 | 6337 |
| Neural Net Baseline | 0.9996 | 0.9483 | [0.9276, 0.9647] | 7.9244 | 705 |

## Seed stability (rollout R^2)

| Model | min | max | range |
|---|---:|---:|---:|
| Mechanistic | -1.6652 | -0.4376 | 1.2275 |
| Physics Informed Model | 0.6970 | 0.7027 | 0.0057 |
| Residual LSTM | 0.9517 | 0.9633 | 0.0116 |
| Neural Net Baseline | 0.9414 | 0.9697 | 0.0283 |
