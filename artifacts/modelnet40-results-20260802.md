# PHAT-JeT vs PointTransformerV3 on ModelNet40

**Date:** 2026-08-02 · **Branch:** `modelnet10` · **Job:** `alan-mn40` (45 runs, 0 failures)
**Raw per-run records:** [`modelnet40-raw-results-20260802.jsonl`](modelnet40-raw-results-20260802.jsonl)

Companion to [`modelnet10-phat-vs-ptv3-results-20260801.md`](modelnet10-phat-vs-ptv3-results-20260801.md).

---

## 1. Headline

No augmentation · `morton_grid_size`=0.05 · 1000 points/shape · official ModelNet40 split ·
FLOPs-matched · n=5 per cell, **n=15 pooled**

| Model | Test Acc (%) | Params | FLOPs |
|---|---|---|---|
| **PHAT-JeT — grid on (x,y,z)** | **76.86 ± 1.04** | **22,504** | 30.54 M |
| PHAT-JeT — grid on (x,y) | 75.42 ± 1.01 | 21,928 | 30.55 M |
| PTv3 (FLOPs-matched) | 75.37 ± 1.09 | 79,912 | 28.90 M |

| Comparison | diff | p | |
|---|---|---|---|
| PHAT (x,y,z) vs PTv3 | **+1.50** | **0.0006** | **significant — PHAT ahead** |
| PHAT (x,y,z) vs PHAT (x,y) | **+1.44** | **0.0006** | **significant — 3D grid ahead** |
| PHAT (x,y) vs PTv3 | −0.05 | 0.89 | tied |

**PHAT-JeT outperforms the FLOPs-matched PTv3 by 1.50 points using 3.6× fewer parameters.**

*Both PHAT variants receive all three coordinates as per-point features; they differ only in
the GMP grid axes.*

---

## 2. Per grid size

| GMP δ | PHAT grid (x,y,z) | PHAT grid (x,y) | PTv3 |
|---|---|---|---|
| 0.10 | **76.65 ± 1.19** | 75.66 ± 1.19 | 74.55 ± 0.71 |
| 0.15 | **77.20 ± 1.14** | 75.18 ± 1.07 | 75.98 ± 1.41 |
| 0.20 | **76.74 ± 0.93** | 75.42 ± 0.95 | 75.58 ± 0.55 |

PHAT (x,y,z) leads in **all three** cells individually, so the pooled result is not an
artifact of pooling. Per-cell significance (n=5): δ=0.10 p=0.013, δ=0.20 p=0.049,
δ=0.15 p=0.172.

### Per-seed values

| Config | δ | seeds |
|---|---|---|
| PHAT (x,y,z) | 0.10 | 74.59, 76.90, 76.90, 77.23, 77.63 |
| PHAT (x,y,z) | 0.15 | 75.85, 76.18, 77.67, 77.76, 78.53 |
| PHAT (x,y,z) | 0.20 | 75.12, 76.94, 77.03, 77.15, 77.47 |
| PHAT (x,y) | 0.10 | 74.23, 74.80, 75.57, 76.66, 77.03 |
| PHAT (x,y) | 0.15 | 73.58, 74.80, 75.28, 75.89, 76.34 |
| PHAT (x,y) | 0.20 | 74.35, 75.00, 75.04, 75.93, 76.78 |
| PTv3 | 0.10 | 73.58, 73.99, 75.00, 75.04, 75.12 |
| PTv3 | 0.15 | 74.07, 75.16, 76.18, 76.82, 77.67 |
| PTv3 | 0.20 | 75.00, 75.04, 75.61, 76.05, 76.18 |

---

## 3. The 3D grid matters here, but not on ModelNet10

| Benchmark | classes | 3D grid vs 2D grid |
|---|---|---|
| ModelNet10 | 10 | +0.09, p=0.91 — **no difference** |
| **ModelNet40** | **40** | **+1.44, p=0.0006** |

On ModelNet10 a top-down (x,y) projection was sufficient and the third grid axis bought
nothing. On ModelNet40 it does not: the added categories are frequently separated by fine 3D
shape rather than by footprint (cone / cup / bowl / vase / flower_pot;
bench / chair / stool; dresser / nightstand / wardrobe), and flattening the grid loses that.

This retroactively justifies the Conv3D generalization of GMP, which the ModelNet10 result
alone had suggested was unnecessary.

---

## 4. Comparison against ModelNet10

| | ModelNet10 | ModelNet40 |
|---|---|---|
| PHAT grid (x,y,z) | 86.56 ± 1.02 | 76.86 ± 1.04 |
| PHAT grid (x,y) | 86.74 ± 1.50 | 75.42 ± 1.01 |
| PTv3 | 87.34 ± 1.45 | 75.37 ± 1.09 |
| PHAT (x,y,z) vs PTv3 | −0.78, p=0.36 (tied) | **+1.50, p=0.0006** |

ModelNet10 values are the morton=0.05, δ=0.20 cells for a like-for-like comparison. The
direction reverses: statistically indistinguishable on the 10-class benchmark, significantly
in PHAT's favour on the 40-class one.

---

## 5. Caveats

1. **No PointNet anchor was run for ModelNet40.** The 45-run matrix used PTv3 as the third
   variant, so unlike ModelNet10 there is no accuracy sanity anchor. The dataset itself was
   independently verified (0 failures: exact official 9843/2468 split, all 40 classes in
   alphabetical order, no train/val/test leakage, normalization exact, label alignment
   confirmed physically), so the pipeline is structurally validated — but an accuracy anchor
   is still missing.

2. **Absolute accuracies (~77%) are far below the ModelNet40 literature** (PointNet 89.2%,
   modern methods 93%+). Every model here is trigger-scale: 22 K parameters versus PointNet's
   ~3.5 M, roughly 150× smaller. These numbers are internally comparable — same data, budget,
   ordering, schedule and seed count — and should not be read against the leaderboard.

3. **Individual cells are n=5**; the p=0.0006 figures come from pooling to n=15 across grid
   sizes. Pooling is justified here because grid size is a nuisance parameter applied
   identically to both models, and because the direction is consistent in all three cells.

4. **The PTv3 baseline is the paper's simplified implementation** — no shift-order,
   shuffle-order or multi-curve serialization. Those cost ~nothing in params or FLOPs, and
   shift-order is PTv3's own cross-patch mechanism, so this baseline is weaker than a
   faithful PTv3. Same caveat as ModelNet10, and it applies to the published tables too.

---

## 6. Setup

Identical to the ModelNet10 matrix apart from the dataset and the 40-way output layer.
Parameter counts differ from ModelNet10 by exactly the output-layer growth
(PHAT +990 = 30×(32+1); PTv3 +1950 = 30×(64+1)), confirming nothing else changed.

| | PHAT-JeT | PTv3 |
|---|---|---|
| Stages / blocks | 1 stage, 1 block | 2 stages, 1 block each |
| Hidden dims | `[32]` | `[32, 64]` |
| Heads | 4 | 4, 4 |
| Patch size | 10 (→100 patches) | 10 |
| Pooling | disabled (paper preset) | stride-2 between stages |
| GMP / xCPE | ✅ | ✅ |

Shared: 1000 points/shape, Morton ordering at 0.05, gelu FFN, `mean` patch tokenizer,
`message_proj` on, `message_gated` off, max aggregation, batch 32, 200 epochs,
early-stopping patience 20, Adam.

**Run outputs were not persisted to the PVC** — the dataset was mounted read-only and all
checkpoints/plots/logs went to pod-local `/tmp`. Each run emitted a single `RESULT_JSON` line
to stdout; the table above was compiled from those and archived to the `.jsonl` alongside
this file.
