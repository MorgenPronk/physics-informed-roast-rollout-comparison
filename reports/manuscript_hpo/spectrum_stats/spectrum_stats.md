# Spectrum statistical-separability analysis

## Headline test rollout R²

| Model | R² | RMSE | 95% CI | Params | n roasts |
|---|---:|---:|---|---:|---:|
| nn_baseline | 0.9697 | 6.07 | [0.9560, 0.9805] | 705 | 36 |
| residual_lstm | 0.9633 | 6.68 | [0.9435, 0.9823] | 6,337 | 36 |
| multi_closure_pi | 0.9436 | 8.28 | [0.9165, 0.9717] | 14,196 | 36 |
| pi_closure_bs8 | 0.9435 | 8.28 | [0.9267, 0.9603] | 44,658 | 36 |
| pi_fixed_priors | 0.9383 | 8.66 | [0.9011, 0.9768] | 10,892 | 36 |
| scalar_tuned_priors_with_init_net | 0.9357 | 8.84 | [0.9126, 0.9597] | 367 | 36 |
| scalar_tuned_priors | 0.9346 | 8.91 | [0.9108, 0.9595] | 11 | 36 |
| residual_ff_unbounded_sweep | 0.9158 | 10.11 | [0.8869, 0.9468] | 2,433 | 36 |
| residual_ff_unbounded | 0.9098 | 10.47 | [0.8817, 0.9402] | 705 | 36 |
| residual_ff_bounded_sweep | 0.8963 | 11.23 | [0.8566, 0.9386] | 8,961 | 36 |
| residual_ff_bounded | 0.8448 | 13.73 | [0.7744, 0.9157] | 705 | 36 |
| pi_closure_sweep | 0.7027 | 19.00 | [0.6480, 0.7611] | 11,250 | 36 |
| mechanistic | -0.4376 | 41.79 | [-0.6551, -0.2164] | 369 | 36 |
| true_mechanistic | -11.0207 | 120.85 | [-11.8503, -10.2014] | 0 | 36 |

## Pairwise paired-Wilcoxon + TOST (ε=0.02) + per-roast correlation

Reading the columns:

- **median Δ**: median of per-roast differences a − b (positive => a typically higher).
- **Wilcoxon p**: paired signed-rank test for any non-zero median difference. Low p means the models differ in their per-roast scores.
- **TOST p**: equivalence test for |mean difference| < ε. Low p means the models are statistically equivalent within ±ε.
- **corr**: Pearson correlation of per-roast R² vectors. High => same roasts hard for both; low => different failure modes.

| a | b | n | median Δ R² | Δ CI95 | mean Δ | Wilcoxon p | TOST p | corr |
|---|---|---:|---:|---|---:|---:|---:|---:|
| mechanistic | pi_closure_sweep | 36 | -1.0798 | [-1.1614, -0.9705] | -1.0736 | 2.91e-11 | 1.00e+00 | +0.737 |
| mechanistic | residual_lstm | 36 | -1.2830 | [-1.4327, -1.1483] | -1.3200 | 2.91e-11 | 1.00e+00 | +0.818 |
| mechanistic | nn_baseline | 36 | -1.2428 | [-1.3857, -1.1276] | -1.3174 | 2.91e-11 | 1.00e+00 | +0.058 |
| mechanistic | pi_closure_bs8 | 36 | -1.2579 | [-1.3797, -1.1340] | -1.2982 | 2.91e-11 | 1.00e+00 | +0.424 |
| mechanistic | residual_ff_bounded | 36 | -1.2246 | [-1.3127, -1.1223] | -1.2214 | 2.91e-11 | 1.00e+00 | +0.460 |
| mechanistic | residual_ff_unbounded | 36 | -1.2469 | [-1.3702, -1.1236] | -1.2646 | 2.91e-11 | 1.00e+00 | +0.167 |
| mechanistic | true_mechanistic | 36 | +9.9990 | [+9.5866, +10.4885] | +10.4583 | 2.91e-11 | 1.00e+00 | +0.570 |
| mechanistic | scalar_tuned_priors | 36 | -1.2472 | [-1.3759, -1.1145] | -1.2942 | 2.91e-11 | 1.00e+00 | +0.564 |
| mechanistic | scalar_tuned_priors_with_init_net | 36 | -1.2451 | [-1.3760, -1.1144] | -1.2948 | 2.91e-11 | 1.00e+00 | +0.551 |
| mechanistic | pi_fixed_priors | 36 | -1.2690 | [-1.4106, -1.1390] | -1.3054 | 2.91e-11 | 1.00e+00 | +0.639 |
| mechanistic | multi_closure_pi | 36 | -1.2566 | [-1.4358, -1.1314] | -1.3048 | 2.91e-11 | 1.00e+00 | +0.500 |
| mechanistic | residual_ff_bounded_sweep | 36 | -1.2503 | [-1.3757, -1.1290] | -1.2607 | 2.91e-11 | 1.00e+00 | +0.524 |
| mechanistic | residual_ff_unbounded_sweep | 36 | -1.2496 | [-1.3768, -1.1313] | -1.2727 | 2.91e-11 | 1.00e+00 | +0.264 |
| pi_closure_sweep | residual_lstm | 36 | -0.2022 | [-0.2412, -0.1840] | -0.2465 | 2.91e-11 | 1.00e+00 | +0.889 |
| pi_closure_sweep | nn_baseline | 36 | -0.2101 | [-0.2453, -0.1763] | -0.2438 | 2.91e-11 | 1.00e+00 | -0.012 |
| pi_closure_sweep | pi_closure_bs8 | 36 | -0.1768 | [-0.2177, -0.1627] | -0.2246 | 2.91e-11 | 1.00e+00 | +0.737 |
| pi_closure_sweep | residual_ff_bounded | 36 | -0.1623 | [-0.1743, -0.1498] | -0.1479 | 1.60e-09 | 1.00e+00 | +0.851 |
| pi_closure_sweep | residual_ff_unbounded | 36 | -0.1697 | [-0.1862, -0.1560] | -0.1911 | 5.82e-11 | 1.00e+00 | +0.722 |
| pi_closure_sweep | true_mechanistic | 36 | +11.1542 | [+10.6360, +11.4584] | +11.5319 | 2.91e-11 | 1.00e+00 | +0.829 |
| pi_closure_sweep | scalar_tuned_priors | 36 | -0.1843 | [-0.2371, -0.1643] | -0.2207 | 2.91e-11 | 1.00e+00 | +0.676 |
| pi_closure_sweep | scalar_tuned_priors_with_init_net | 36 | -0.1821 | [-0.2380, -0.1630] | -0.2213 | 2.91e-11 | 1.00e+00 | +0.652 |
| pi_closure_sweep | pi_fixed_priors | 36 | -0.1952 | [-0.2471, -0.1805] | -0.2319 | 2.91e-11 | 1.00e+00 | +0.708 |
| pi_closure_sweep | multi_closure_pi | 36 | -0.1871 | [-0.2495, -0.1746] | -0.2313 | 2.91e-11 | 1.00e+00 | +0.586 |
| pi_closure_sweep | residual_ff_bounded_sweep | 36 | -0.1742 | [-0.1902, -0.1631] | -0.1872 | 2.91e-11 | 1.00e+00 | +0.918 |
| pi_closure_sweep | residual_ff_unbounded_sweep | 36 | -0.1734 | [-0.1930, -0.1642] | -0.1991 | 2.91e-11 | 1.00e+00 | +0.776 |
| residual_lstm | nn_baseline | 36 | +0.0012 | [-0.0070, +0.0119] | +0.0027 | 5.19e-01 | 3.20e-02 | -0.041 |
| residual_lstm | pi_closure_bs8 | 36 | +0.0132 | [+0.0101, +0.0252] | +0.0218 | 8.04e-07 | 6.44e-01 | +0.711 |
| residual_lstm | residual_ff_bounded | 36 | +0.0396 | [+0.0279, +0.0789] | +0.0986 | 1.46e-10 | 1.00e+00 | +0.729 |
| residual_lstm | residual_ff_unbounded | 36 | +0.0267 | [+0.0181, +0.0368] | +0.0554 | 4.32e-08 | 9.97e-01 | +0.469 |
| residual_lstm | true_mechanistic | 36 | +11.4065 | [+10.7926, +11.7307] | +11.7783 | 2.91e-11 | 1.00e+00 | +0.687 |
| residual_lstm | scalar_tuned_priors | 36 | +0.0168 | [+0.0122, +0.0323] | +0.0258 | 1.09e-05 | 8.21e-01 | +0.689 |
| residual_lstm | scalar_tuned_priors_with_init_net | 36 | +0.0160 | [+0.0117, +0.0325] | +0.0252 | 2.33e-05 | 8.02e-01 | +0.682 |
| residual_lstm | pi_fixed_priors | 36 | +0.0015 | [-0.0017, +0.0072] | +0.0146 | 2.52e-01 | 2.69e-01 | +0.768 |
| residual_lstm | multi_closure_pi | 36 | +0.0063 | [-0.0031, +0.0303] | +0.0152 | 4.25e-02 | 2.46e-01 | +0.684 |
| residual_lstm | residual_ff_bounded_sweep | 36 | +0.0240 | [+0.0146, +0.0497] | +0.0593 | 2.04e-10 | 9.99e-01 | +0.777 |
| residual_lstm | residual_ff_unbounded_sweep | 36 | +0.0191 | [+0.0109, +0.0280] | +0.0474 | 2.63e-08 | 9.90e-01 | +0.536 |
| nn_baseline | pi_closure_bs8 | 36 | +0.0162 | [+0.0099, +0.0229] | +0.0192 | 1.74e-03 | 4.63e-01 | -0.029 |
| nn_baseline | residual_ff_bounded | 36 | +0.0440 | [+0.0277, +0.0751] | +0.0959 | 1.72e-04 | 9.96e-01 | -0.120 |
| nn_baseline | residual_ff_unbounded | 36 | +0.0304 | [+0.0127, +0.0547] | +0.0527 | 2.69e-04 | 9.83e-01 | +0.002 |
| nn_baseline | true_mechanistic | 36 | +11.3965 | [+10.7830, +11.7254] | +11.7757 | 2.91e-11 | 1.00e+00 | -0.091 |
| nn_baseline | scalar_tuned_priors | 36 | +0.0166 | [+0.0098, +0.0342] | +0.0231 | 2.78e-03 | 6.12e-01 | -0.117 |
| nn_baseline | scalar_tuned_priors_with_init_net | 36 | +0.0152 | [+0.0078, +0.0373] | +0.0225 | 4.12e-03 | 5.93e-01 | -0.131 |
| nn_baseline | pi_fixed_priors | 36 | -0.0028 | [-0.0126, +0.0087] | +0.0119 | 6.36e-01 | 2.91e-01 | -0.104 |
| nn_baseline | multi_closure_pi | 36 | +0.0055 | [-0.0082, +0.0294] | +0.0125 | 3.15e-01 | 2.69e-01 | -0.170 |
| nn_baseline | residual_ff_bounded_sweep | 36 | +0.0259 | [+0.0077, +0.0474] | +0.0566 | 8.16e-04 | 9.78e-01 | -0.030 |
| nn_baseline | residual_ff_unbounded_sweep | 36 | +0.0240 | [+0.0069, +0.0421] | +0.0447 | 1.13e-03 | 9.51e-01 | -0.007 |
| pi_closure_bs8 | residual_ff_bounded | 36 | +0.0203 | [+0.0040, +0.0529] | +0.0768 | 2.16e-04 | 9.95e-01 | +0.810 |
| pi_closure_bs8 | residual_ff_unbounded | 36 | +0.0118 | [+0.0019, +0.0192] | +0.0336 | 4.76e-04 | 9.10e-01 | +0.720 |
| pi_closure_bs8 | true_mechanistic | 36 | +11.3789 | [+10.7596, +11.7247] | +11.7565 | 2.91e-11 | 1.00e+00 | +0.846 |
| pi_closure_bs8 | scalar_tuned_priors | 36 | +0.0032 | [-0.0115, +0.0101] | +0.0040 | 5.19e-01 | 4.47e-03 | +0.735 |
| pi_closure_bs8 | scalar_tuned_priors_with_init_net | 36 | +0.0002 | [-0.0125, +0.0123] | +0.0034 | 5.81e-01 | 3.02e-03 | +0.724 |
| pi_closure_bs8 | pi_fixed_priors | 36 | -0.0144 | [-0.0279, -0.0077] | -0.0072 | 8.16e-03 | 7.64e-02 | +0.766 |
| pi_closure_bs8 | multi_closure_pi | 36 | -0.0118 | [-0.0235, +0.0048] | -0.0066 | 1.66e-01 | 2.34e-02 | +0.729 |
| pi_closure_bs8 | residual_ff_bounded_sweep | 36 | +0.0085 | [-0.0016, +0.0217] | +0.0374 | 5.12e-03 | 9.35e-01 | +0.859 |
| pi_closure_bs8 | residual_ff_unbounded_sweep | 36 | +0.0056 | [-0.0023, +0.0102] | +0.0255 | 2.72e-02 | 7.23e-01 | +0.770 |
| residual_ff_bounded | residual_ff_unbounded | 36 | -0.0147 | [-0.0256, -0.0000] | -0.0432 | 1.54e-03 | 9.29e-01 | +0.866 |
| residual_ff_bounded | true_mechanistic | 36 | +11.2803 | [+10.7758, +11.6562] | +11.6797 | 2.91e-11 | 1.00e+00 | +0.871 |
| residual_ff_bounded | scalar_tuned_priors | 36 | -0.0193 | [-0.0708, -0.0001] | -0.0728 | 1.63e-03 | 9.95e-01 | +0.818 |
| residual_ff_bounded | scalar_tuned_priors_with_init_net | 36 | -0.0196 | [-0.0731, +0.0002] | -0.0734 | 2.34e-03 | 9.95e-01 | +0.805 |
| residual_ff_bounded | pi_fixed_priors | 36 | -0.0425 | [-0.0883, -0.0182] | -0.0840 | 3.67e-08 | 1.00e+00 | +0.828 |
| residual_ff_bounded | multi_closure_pi | 36 | -0.0356 | [-0.0880, -0.0138] | -0.0834 | 1.33e-05 | 9.99e-01 | +0.783 |
| residual_ff_bounded | residual_ff_bounded_sweep | 36 | -0.0170 | [-0.0312, -0.0059] | -0.0393 | 8.47e-05 | 9.45e-01 | +0.937 |
| residual_ff_bounded | residual_ff_unbounded_sweep | 36 | -0.0206 | [-0.0443, -0.0046] | -0.0512 | 1.76e-05 | 9.80e-01 | +0.907 |
| residual_ff_unbounded | true_mechanistic | 36 | +11.3287 | [+10.7773, +11.6850] | +11.7229 | 2.91e-11 | 1.00e+00 | +0.797 |
| residual_ff_unbounded | scalar_tuned_priors | 36 | -0.0151 | [-0.0256, +0.0052] | -0.0296 | 6.44e-02 | 7.96e-01 | +0.540 |
| residual_ff_unbounded | scalar_tuned_priors_with_init_net | 36 | -0.0153 | [-0.0261, -0.0001] | -0.0302 | 5.99e-02 | 8.06e-01 | +0.517 |
| residual_ff_unbounded | pi_fixed_priors | 36 | -0.0262 | [-0.0381, -0.0076] | -0.0408 | 1.45e-03 | 9.39e-01 | +0.498 |
| residual_ff_unbounded | multi_closure_pi | 36 | -0.0261 | [-0.0398, -0.0047] | -0.0402 | 2.48e-03 | 9.43e-01 | +0.454 |
| residual_ff_unbounded | residual_ff_bounded_sweep | 36 | -0.0042 | [-0.0068, -0.0022] | +0.0039 | 2.49e-02 | 1.53e-02 | +0.902 |
| residual_ff_unbounded | residual_ff_unbounded_sweep | 36 | -0.0067 | [-0.0082, -0.0056] | -0.0080 | 1.08e-04 | 6.13e-08 | +0.991 |
| true_mechanistic | scalar_tuned_priors | 36 | -11.3966 | [-11.7161, -10.7777] | -11.7525 | 2.91e-11 | 1.00e+00 | +0.715 |
| true_mechanistic | scalar_tuned_priors_with_init_net | 36 | -11.3962 | [-11.7142, -10.7773] | -11.7531 | 2.91e-11 | 1.00e+00 | +0.692 |
| true_mechanistic | pi_fixed_priors | 36 | -11.4122 | [-11.7346, -10.7904] | -11.7637 | 2.91e-11 | 1.00e+00 | +0.744 |
| true_mechanistic | multi_closure_pi | 36 | -11.4066 | [-11.7202, -10.7857] | -11.7631 | 2.91e-11 | 1.00e+00 | +0.648 |
| true_mechanistic | residual_ff_bounded_sweep | 36 | -11.3497 | [-11.6926, -10.7804] | -11.7191 | 2.91e-11 | 1.00e+00 | +0.916 |
| true_mechanistic | residual_ff_unbounded_sweep | 36 | -11.3499 | [-11.6960, -10.7810] | -11.7310 | 2.91e-11 | 1.00e+00 | +0.857 |
| scalar_tuned_priors | scalar_tuned_priors_with_init_net | 36 | +0.0003 | [-0.0006, +0.0008] | -0.0006 | 9.69e-01 | 5.80e-30 | +0.999 |
| scalar_tuned_priors | pi_fixed_priors | 36 | -0.0167 | [-0.0210, -0.0147] | -0.0112 | 1.63e-03 | 4.18e-02 | +0.965 |
| scalar_tuned_priors | multi_closure_pi | 36 | -0.0123 | [-0.0144, -0.0067] | -0.0106 | 2.33e-04 | 1.40e-03 | +0.952 |
| scalar_tuned_priors | residual_ff_bounded_sweep | 36 | +0.0150 | [+0.0003, +0.0470] | +0.0335 | 2.10e-02 | 8.73e-01 | +0.733 |
| scalar_tuned_priors | residual_ff_unbounded_sweep | 36 | +0.0088 | [-0.0059, +0.0228] | +0.0216 | 1.92e-01 | 5.60e-01 | +0.625 |
| scalar_tuned_priors_with_init_net | pi_fixed_priors | 36 | -0.0164 | [-0.0216, -0.0149] | -0.0106 | 1.63e-03 | 4.00e-02 | +0.964 |
| scalar_tuned_priors_with_init_net | multi_closure_pi | 36 | -0.0127 | [-0.0144, -0.0075] | -0.0100 | 3.86e-04 | 3.85e-04 | +0.962 |
| scalar_tuned_priors_with_init_net | residual_ff_bounded_sweep | 36 | +0.0158 | [+0.0009, +0.0489] | +0.0341 | 2.39e-02 | 8.76e-01 | +0.710 |
| scalar_tuned_priors_with_init_net | residual_ff_unbounded_sweep | 36 | +0.0090 | [-0.0063, +0.0240] | +0.0221 | 1.76e-01 | 5.80e-01 | +0.602 |
| pi_fixed_priors | multi_closure_pi | 36 | +0.0038 | [-0.0005, +0.0157] | +0.0006 | 1.92e-01 | 3.22e-05 | +0.966 |
| pi_fixed_priors | residual_ff_bounded_sweep | 36 | +0.0254 | [+0.0061, +0.0379] | +0.0447 | 5.14e-05 | 9.84e-01 | +0.737 |
| pi_fixed_priors | residual_ff_unbounded_sweep | 36 | +0.0198 | [+0.0017, +0.0296] | +0.0328 | 7.38e-03 | 8.58e-01 | +0.590 |
| multi_closure_pi | residual_ff_bounded_sweep | 36 | +0.0266 | [+0.0083, +0.0568] | +0.0441 | 1.36e-03 | 9.69e-01 | +0.649 |
| multi_closure_pi | residual_ff_unbounded_sweep | 36 | +0.0220 | [+0.0009, +0.0332] | +0.0322 | 1.53e-02 | 8.52e-01 | +0.532 |
| residual_ff_bounded_sweep | residual_ff_unbounded_sweep | 36 | -0.0023 | [-0.0047, +0.0000] | -0.0119 | 9.96e-03 | 9.08e-02 | +0.941 |
