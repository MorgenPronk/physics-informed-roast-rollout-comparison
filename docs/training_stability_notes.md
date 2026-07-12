# Training-stability notes: what we learned fitting physics through a stiff rollout

**Status: informal engineering notes.** These observations accompany the paper
but are not peer-reviewed claims. They come from the review-revision seed
campaign (`scripts/run_r1_seed_campaign.py`, artifacts in
`reports/r1_seed_campaign/`) and are shared because they are practically
useful to anyone fitting mechanistic ODE models — or hybrid PIML models —
against production time-series by autoregressive rollout. Everything here is
observed on one cohort (221 industrial coffee roasts) and one model family;
generalize with care.

## 1. Why the mechanistic baseline fails: an ablation

The paper's mechanistic baseline reaches test rollout R² ≈ −0.44 to −1.67
while a priors-anchored control with identical physics reaches ≈ 0.935. A
four-variant ablation (five seeds each) pins down why:

| Variant | Initial state | Anchored? | Per-roast init net | Test R² | Training |
|---|---|---|---|---|---|
| Baseline (paper) | probe reading + 8 °C ≈ 209 °C | no | yes | −0.44 … −1.67 | diverges (NaN), epochs 44–52 |
| Globals-only | same | no | **no** | −0.88 … −1.51 | diverges, epochs 48–49 |
| Free, correct init | 20 °C / X_b = 0.12 | no | yes | **+0.915 … +0.929** | diverges, epoch ~117 |
| Anchored control (paper) | 20 °C / 0.12 | **fixed** | frozen | +0.9346 … +0.9348 | stable to convergence |

Reading the rows top to bottom:

- **The starting values dominate.** Changing *only* the initial-state values
  from a telemetry-referenced guess to the physically correct ones (row 3 vs
  row 1) recovers ~95% of the gap. The per-roast initialization network is
  exonerated (row 2 fails like row 1 without it).
- **Anchoring buys the last ~0.01 R² and full training stability** (row 4).
  Even correctly-initialized free training eventually walks itself into a
  rollout blow-up.

## 2. The starting point was unreachable, not merely bad

The baseline's initial bean temperature is parameterized relative to the
first probe sample (which reads residual drum heat, ~200 °C, not the ~20 °C
beans). To represent the correct configuration, one scalar offset must travel
≈ 200 units. Under Adam at the swept learning rate (5.4e-4), the ~500 steps
completed before numerical divergence afford **≈ 0.3 units of travel** —
wrong by three orders of magnitude. Measured drift confirms it: the offset
moved from +8.0 to +7.7 before training died, moisture never left 0.032
(target 0.12), and the kinetic scalars visibly distorted to compensate
(h_e 35 → 26, effective-mass scale +36%).

Two distinct failure mechanisms compound here and are worth separating in
your own debugging:

1. **Optimization unreachability** — the correct basin is farther away (in
   optimizer-step units) than the training budget can reach.
2. **Numerical integration instability** — the training loss is computed by
   rolling a stiff ODE forward; when compensating parameter drift pushes the
   state out of the physically valid regime, the integration explodes, the
   gradients go NaN, and training dies mid-descent (losses were still
   falling at the point of divergence, in every seed).

## 3. Training stability across neural-network placements

Stability across the paper's placement spectrum, from the same campaign:

| Model | Learned component vs the ODE | Training outcome |
|---|---|---|
| Mechanistic baseline (unanchored) | none in dynamics; free initial states | diverges, every seed |
| Single-closure PI | MLP inside the ODE, effectively unbounded | diverges (epochs 74–88), every seed — best pre-divergence checkpoints are good (0.69–0.76) |
| Multi-closure PI | **three** MLPs inside, but softplus-bounded, sign-constrained, capped | **never diverges** |
| FF residuals (both) | MLP outside, on a **frozen** base rollout | never diverges (see gotcha #1) |
| Neural baseline | no ODE | never diverges |
| Anchored control | ODE with pinned initial states | never diverges |

The interesting inversion: the multi-closure variant carries *more* learned
capacity inside the ODE than the single-closure yet trains *more* stably,
because its closure outputs are bounded and sign-constrained. On this
evidence, what governs trainability is not how much network you put inside
the physics but **whether the learned degrees of freedom can push the
integrator out of its valid regime**. Anchored states, bounded closure
outputs, and frozen bases are the safeguards; the placement choice
determines which of them are available to you.

This is consistent with the known difficulty of fitting stiff dynamics
through unconstrained rollouts (see Kim et al., "Stiff neural ordinary
differential equations", Chaos 31, 093122, 2021, and the classical
multiple-shooting literature on ODE parameter estimation).

## 4. Gotchas we hit (so you don't)

1. **Stacking on a frozen base inherits the base's worst roast.** At seed 53
   the single-closure base trained to a fine *test* R² (0.70) while its
   rollout silently diverged on one *training* roast. Residual training on
   that base got NaN loss from epoch 1 and never produced a model. If you
   stack corrections on a frozen simulator, validate the base's rollout on
   the *training* set, not just val/test.
2. **Pooled train metrics are NaN-fragile.** One exploding roast makes the
   pooled train R² NaN even when 149/150 roasts are fine. Use per-roast
   medians alongside pooled metrics when diagnosing.
3. **Three seeds can badly underestimate seed variance.** The single-closure
   class showed a range of 0.006 across the original three seeds; six seeds
   widened it to 0.073 — a factor of ten. The class *ordering* survived, but
   any quantitative stability claim built on three seeds deserves suspicion,
   including ours.
4. **Best-checkpoint selection can mask divergence.** All the diverging
   variants still report respectable metrics because validation-selected
   checkpoints are harvested before the NaN. If you only look at final
   metrics, you will not notice that training died at epoch 50 of 300.
   Check the histories.

## 5. Reproducing these results

```bash
# Full probe family (5 variants x 5 seeds) + class extensions:
python scripts/run_r1_seed_campaign.py --phase all

# Just the ablation of section 1:
python scripts/run_r1_seed_campaign.py --phase probes

# Individual variant/seed:
python scripts/run_r1_seed_campaign.py --phase probes \
    --probe-variants free_correct_init --probe-seeds 41
```

Each run writes a JSON to `reports/r1_seed_campaign/` containing test/val/
train rollout metrics, full training history (including the first NaN
epoch), and the initial and best-epoch values of every scalar physical
parameter. Runs are resumable; existing outputs are skipped.
