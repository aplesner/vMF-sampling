# Project 1 — Tier 0/1 pre-registration

Registered before the Tier-0 sweep is run. The commit hash of the commit
adding this file is the pre-registration reference; it is recorded in
`report.html`.

## Design

- Substrate: CIFAR-10 5+5 class-incremental split, no rehearsal.
  Phase 1: 15 epochs on classes {first 5 of partition}; head extended to
  10 rows; Phase 2: 15 epochs on the other 5 classes. Evaluation every
  phase-2 epoch: old-class accuracy (argmax over all 10 logits) and
  new-class accuracy.
- Arms: tuned MLP (`mlp`, Linear-ReLU x3 x1024, AdamW) vs nGPT-style
  normalized MLP (`nmlp`, functionally normalized weight rows, learnable
  per-row eigen-LRs, unit-norm representations, cosine classifier with
  learnable logit scale, AdamW). No augmentation; inputs normalized.
- Per-arm grids (no arm is ever compared at a shared LR):
  LR in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, WD in {0, 1e-4, 1e-3, 1e-2, 1e-1}.
  5 class partitions x 3 seeds = 15 paired runs per config.
  (Grids amended 2026-08-10, see below.)
- Config selection per arm: maximize LCA (mean over the 15 phase-2 epochs
  of (old_acc + new_acc)/2) averaged over the 15 (partition, seed) pairs.
  Selection/evaluation overlap is acknowledged; a leave-one-partition-out
  re-selection is reported as a robustness check.

## Gate (Tier-0 pass criteria, from the brief)

1. Median paired LCA gain (nMLP - MLP) >= 3.0 accuracy points, with a 95%
   bootstrap CI (10 000 resamples over the 15 pairs) excluding 0.
2. Frontier dominance: on mean phase-2 curves, nMLP >= MLP on *both*
   old_acc(t) and new_acc(t) for >= 60% of the epoch grid.

If the gate fails, the failure is reported as-is. Endpoint accuracies are
reported but are not the comparison object.

## Tier-1 controls (decide whether the effect is real)

- Weight-decay dose-response: baseline MLP at its selected LR, 5 WD points
  bracketing the WD whose per-layer ||dW||/||W|| curves best overlay the
  nMLP's (least-squares match on the epoch-mean curves, layers pooled).
- LARS reference arm on the baseline MLP, own LR sweep
  {0.03, 0.1, 0.3, 1.0, 3.0} x WD {0, 1e-3}, same gate metrics vs the
  tuned AdamW baseline.

## Amendment 2026-08-10 (before any valid results existed)

Two bugs were found and fixed during pipeline validation; **no result
produced before this amendment is used anywhere**:

1. **Label-remap bug (critical).** Phase-1 training passed original CIFAR
   labels (e.g. {7,6,4,9,2}) as cross-entropy targets to a 5-output head.
   On MPS this fails silently instead of raising, so phase-1 training was
   garbage. Fixed: labels are remapped to contiguous head-row positions
   (phase-1 classes -> 0..4, phase-2 -> 5..9). All pre-fix sweep output
   was deleted.
2. **Evaluation bug.** Phase-2 metrics were computed by training all 15
   epochs first and then evaluating the final model 15 times (flat
   curves). Fixed: evaluation is interleaved per epoch.

Protocol changes, made after the fixes with only diagnostic runs in hand:

- **Augmentation on** (random crop pad-4 + horizontal flip, both phases,
  both arms). Without it the MLP memorizes (train loss 0.05, test ~29%),
  which is not a regime where stability means anything.
- **Per-arm LR grids.** The baseline MLP's old-class accuracy collapses to
  exactly 0 within one phase-2 epoch for LR >= 3e-5, while the
  scale-invariant nMLP needs ~100x larger steps. Per the program's own
  rule ("per-arm sweeping over arm-appropriate ranges is the correct
  protocol"), grids are: MLP LR in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3};
  nMLP LR in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}. WD grid unchanged and shared.

## Stated MDE

With 15 CRN-paired (partition, seed) cells and paired LCA standard
deviation expected ~2-4 points, the design detects a median paired gain of
~2-3 points; the 3.0-point gate is at the edge of detectability, so a
borderline result is reported as inconclusive, not negative.
