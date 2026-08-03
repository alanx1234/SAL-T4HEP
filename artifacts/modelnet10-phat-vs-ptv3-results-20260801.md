# PHAT-JeT vs PointTransformerV3 on ModelNet10

**Date:** 2026-08-01 · **Branch:** `modelnet10` · **Cluster:** Nautilus `cms-ml`
**Total:** 70 training runs + preprocessing, preflight, and a data sanity suite

---

## 1. Headline result (n=10 per model)

No augmentation · δ=0.2 · FLOPs-matched · 1000 points/shape · official ModelNet10 split

| Model | Test Acc (%) | 95% CI | sd | Params | FLOPs |
|---|---|---|---|---|---|
| PHAT-JeT (paper preset) | **86.61** | [85.86, 87.35] | 1.04 | **21,514** | 30.54 M |
| PTv3 (serialized, FLOPs-matched) | **88.46** | [87.61, 89.31] | 1.19 | 77,962 | 28.90 M |
| PointNet (pipeline anchor) | **89.67** | [88.60, 90.74] | 1.49 | 85,322 | 82.39 M |

### Significance (Welch t-test)

| Comparison | diff | t | df | p | |
|---|---|---|---|---|---|
| PHAT vs PTv3 | +1.85 | 3.70 | 17.7 | **0.0017** | significant |
| PHAT vs PointNet | +3.06 | 5.32 | 16.1 | **0.0001** | significant |
| PTv3 vs PointNet | +1.21 | 2.00 | 17.2 | 0.061 | not significant |

**PHAT-JeT places last, significantly.** It does achieve this with 3.6–4× fewer
parameters than either baseline.

> ### ⚠️ Absolute accuracies are below the ModelNet10 literature — this is expected
>
> Modern full-scale methods report **93–97.5%** on ModelNet10 (Grid-CNN, PointASNL,
> CurveNet). Our 86–90% is 4–8 points lower because **every model here is trigger-scale**:
> 21.5 K–85 K parameters, versus millions for published point-cloud networks. Our PointNet
> anchor in particular is ~41× smaller than the original (85 K vs ~3.5 M) and has **no
> T-Nets** — neither the 3×3 input transform nor the 64×64 feature transform.
>
> Note also that the original PointNet paper reports **ModelNet40 (89.2% OA), not
> ModelNet10**; there is no canonical PointNet ModelNet10 figure to compare against.
>
> The anchor therefore validates that **the pipeline learns correctly and the data is
> sane** (corroborated independently by the sanity suite in §4). It is *not* a reproduction
> of a published number. Comparisons here are internally consistent — same data, same
> budget, same schedule — and should not be read against the ModelNet10 leaderboard.

> **Seed count matters here.** The gap read 2.42 at n=3 and 1.28 at n=5–6 before settling
> at 1.85 at n=10. Two batches of the *identical* PHAT config differed by a full point from
> initialization alone. n=10 was committed in advance and reported as-is.

---

## 1b. ⚠️ The headline gap depends on a preprocessing parameter

`--morton_grid_size` controls the grid used to Morton-sort points *before* they reach the
model. It is **not** the GMP grid (`--grid_size`); it only decides point ordering. In §1 it
was set to 0.20 (an arbitrary choice, made to match `--grid_size`). Re-running at the script
default of **0.05** changes the conclusion:

### Full sweep at morton_grid_size = 0.05 (n=5 per cell)

| GMP δ | PHAT grid (x,y,z) | PHAT grid (x,y) | PTv3 |
|---|---|---|---|
| 0.10 | 87.62 ± 0.70 | 86.96 ± 0.69 | 87.20 ± 1.54 |
| 0.15 | 87.56 ± 0.71 | 88.06 ± 1.47 | 87.62 ± 0.65 |
| 0.20 | 86.56 ± 1.02 | 86.74 ± 1.50 | 87.34 ± 1.45 |

**Every PHAT-vs-PTv3 comparison here is non-significant** (gaps 0.06–0.78 in both
directions, p = 0.36–0.89).

### Effect of the Morton grid, like-for-like at δ=0.20

| Model | morton=0.05 | morton=0.20 | Δ | p |
|---|---|---|---|---|
| PHAT (x,y,z) | 86.56 | 86.61 (n=10) | +0.05 | 0.94 |
| PHAT (x,y) | 86.74 | 86.70 (n=10) | −0.04 | 0.96 |
| **PTv3** | **87.34** | **88.46** (n=10) | **+1.12** | 0.18 |

PHAT is insensitive to the Morton grid; PTv3 appears to gain ~1.1 points from the coarser
setting, though that is not significant at these sample sizes. **The §1 result
("PTv3 +1.85, p=0.0017") therefore rests on morton=0.20.** At the default it disappears.

> **The gap has moved four times** as design details changed: 2.42 (n=3) → 1.28 (n=5/6) →
> 1.85 (n=10) → ~0 (morton=0.05). What is stable across all of it is that the two models are
> **close**, not that either wins. Resolving the remaining ~0.8-point differences would need
> n≈35–90 per arm; the 1.12-point Morton effect would need n≈20.

### Grid on (x,y,z) vs (x,y) — PHAT runs unmodified

Height-map layout `[z, x, y]` grids on x-y only, with z carried as the leading scalar
channel — structurally identical to a jet's `[pt, eta, phi]`. This runs PHAT's **unmodified
2D jet path** (Conv2D, `coord_dim=2`); no 3D generalization at all.

| Variant | Acc (n=10, morton=0.20) | Params |
|---|---|---|
| grid on (x,y,z) — Conv3D | 86.61 ± 1.04 | 21,514 |
| grid on (x,y) — Conv2D | 86.70 ± 2.11 | 20,938 |

Difference +0.09, **p=0.907**. Adding the third grid axis buys nothing. The 576-parameter
difference is exactly 32 channels × (27−9) kernel taps, i.e. Conv3D 3³ vs Conv2D 3².

*Both variants receive all three coordinates as per-point features; they differ only in the
GMP grid axes.* z still reaches the network in the 2D variant — it just does not determine
which voxel a point falls into. That is why flattening the grid costs nothing.

---

## 2. Configurations

| | PHAT-JeT | PTv3 |
|---|---|---|
| Stages / blocks | 1 stage, 1 block | 2 stages, 1 block each |
| Hidden dims | `[32]` | `[32, 64]` |
| Heads | 4 | 4, 4 |
| Patch size | 10 (→100 patches) | 10 |
| Pooling | disabled (paper preset) | stride-2 between stages |
| Ordering | 3D Morton (δ=0.2) | 3D Morton, `assume_serialized_input` |
| GMP / xCPE | ✅ depthwise Conv3D 3³ | ✅ depthwise Conv3D 3³ |
| Patch-token global attention | ✅ | ✗ |

Shared: gelu FFN, `mean` patch tokenizer, `message_proj` on, `message_gated` off,
max aggregation, batch 32, 200 epochs, early-stopping patience 20, Adam.

**FLOPs matched to 5.4%** (30.54 M vs 28.90 M), following Appendix H. Depth and parameter
count are *not* matched — this is the paper's own protocol, and mirrors the jet setup where
PTv3 also carries ~3.3× PHAT's parameters at equal FLOPs.

---

## 3. Secondary results

### 3a. Augmentation arm (n=3) — random rotation about the up (z) axis + jitter σ=0.01

| Model | No aug | Z-up aug |
|---|---|---|
| PHAT-JeT | 86.64 ± 0.82 | 84.95 ± 2.59 |
| PTv3 | 89.06 ± 1.43 | 86.67 ± 2.32 |
| PointNet | 90.27 ± 1.54 | 86.45 ± 1.59 |

*(n=3 values, superseded by §1 for PHAT/PTv3/PointNet no-aug.)* Augmentation costs every
model 2–4 points and roughly triples seed variance, because ModelNet10 test shapes are
canonically aligned so train-time rotation creates a train/test mismatch. **Ranking is
unchanged in both arms.** No-aug is reported as the headline.

### 3b. GMP grid sweep — properly decoupled (n=3)

Ordering pinned at `morton_grid_size=0.20`; only the GMP voxel grid varies.

| GMP δ | cells across | Acc (%) |
|---|---|---|
| 0.10 | 20 | 87.22 ± 0.84 |
| 0.15 | 13 | 87.74 ± 1.93 |
| 0.20 | 10 | 87.63 ± 0.34 |
| 0.30 | 7 | 86.20 ± 1.71 |

**GMP is insensitive to grid resolution over 10–20 cells across the object**, degrading only
below ~8 — consistent with the stable band in Appendix G for 2D jets. This is the one clean
positive transfer result.

> An earlier sweep passed the same δ to both `--grid_size` and `--morton_grid_size`, so it
> moved the GMP grid *and* the point ordering together (at δ=0.15 only 0.4% of positions
> matched the δ=0.20 ordering). Those numbers are confounded and are not used.

### 3c. Measured grid occupancy (test split, no training)

| δ | grid dims | cells | occupied | pts / occupied cell |
|---|---|---|---|---|
| 0.10 | 20×20×19 | 7,600 | 385 | 2.89 |
| 0.15 | 13×14×13 | 2,366 | 193 | 5.98 |
| 0.20 | 10×10×10 | 1,000 | 106 | 10.71 |
| 0.30 | 7×7×7 | 343 | 44 | 25.30 |
| 0.40 | 5×5×5 | 125 | 24 | 46.80 |
| 0.50 | 4×4×4 | 64 | 15 | 75.16 |

Occupied cells scale as **δ^−2.06** — surface scaling, not volumetric, because ModelNet
shapes are hollow meshes. The 3D grid never becomes as sparse as a volume estimate suggests.

---

## 4. Dataset validation — 0 failures across ~60 checks

| Check | Result |
|---|---|
| Split sizes vs official ModelNet10 | 3991 train / 908 test ✅ exact, **per class** |
| On-disk mesh counts vs sampled | ✅ all 10 classes |
| Normalization | max radius = 1.000000, centroid ≤ 8e−8 ✅ |
| Duplicates within splits | none |
| **Train / val / test leakage** | **zero overlap** ✅ |
| Degenerate clouds | none |
| Label alignment (physical plausibility) | ✅ flat classes short in z, chair tall in z |
| Patch divisibility | 1000 / 10 = 100 patches, no padding ✅ |

**Preprocessing:** 1000 points/shape, area-weighted surface sampling from the official
`.off` meshes (trimesh), centred and scaled to the unit sphere. 3991 train meshes split
3592/399 (stratified, seed 42); official 908-shape test set held out.

---

## 5. Caveats worth flagging to reviewers

1. **The PTv3 baseline is not a faithful PTv3.** It omits shift-order, shuffle-order,
   multi-curve serialization (Morton only, computed once), and grid pooling with recomputed
   codes. These cost ~nothing in params or FLOPs, so "resource-constrained" does not explain
   their absence. Notably, **shift-order is PTv3's own answer to cross-patch communication** —
   the problem PHAT's patch-token attention is claimed to solve better. PTv3 won *without* it.
   This is the same baseline used in the published tables, so the issue is pre-existing.

2. **ModelNet10 cannot test GMP's claimed contribution.** The paper positions GMP's novelty
   as applying a depthwise grid encoder to the *irregular (η,φ) angular plane*, explicitly
   contrasting with methods that "operate on a regular image grid or a voxelized 3D point
   cloud." ModelNet10 is a voxelized 3D point cloud, where GMP reduces to the xCPE PTv3
   already employs. Both models here have it.

3. **A single-block model losing to hierarchical 3D architectures is not surprising.**
   PHAT's preset is 1 block with no downsampling; PTv3 gets 2 blocks plus pooling. Both are
   their respective papers' configurations at matched FLOPs.

---

## 6. Reproduction

Branch `modelnet10`. Data: `/j-jepa-vol/linformer_data/ModelNet10/points_1000`

| Job | Purpose |
|---|---|
| `reference/modelnet10-preprocess.yml` | download + surface-sample meshes |
| `reference/modelnet10-preflight.yml` | 16 port tests + params/FLOPs table + smoke run |
| `reference/modelnet10-phat-vs-ptv3-benchmark-v2.yml` | headline, aug + no-aug arms |
| `tests/test_modelnet_ports.py` | 3D port tests + 2D jet-path regression guards |

Run logs on the PVC under `modelnet10_phat_vs_ptv3_runs_v2/`, `modelnet10_followup_runs/`,
`modelnet10_lowdelta_runs/`, `modelnet10_decoupled_runs/`, `modelnet10_extraseeds_runs/`,
`modelnet10_seeds10_runs/`.

**Implementation note:** the 2D→3D port is behind `coord_dim` / `weighted_input` arguments
defaulting to the existing jet configuration. Tests pin that the jet path's forward pass,
parameter count, padding mask and Morton ordering are unchanged, so published jet results
still reproduce.
