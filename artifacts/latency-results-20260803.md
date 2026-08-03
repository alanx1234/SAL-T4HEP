# PHAT-JeT vs PointTransformerV3 — inference latency

**Date:** 2026-08-03 · **GPU:** NVIDIA RTX 2080 Ti (all measurements, same card)
**Raw records:** [`hls4ml-latency-20260803.jsonl`](hls4ml-latency-20260803.jsonl),
[`modelnet40-latency-fullpass-20260803.jsonl`](modelnet40-latency-fullpass-20260803.jsonl),
[`modelnet40-latency-synthetic-20260803.jsonl`](modelnet40-latency-synthetic-20260803.jsonl)

**Headline: PHAT-JeT is ~2x faster than the FLOPs-matched PTv3 on both benchmarks, despite
PTv3 having fewer static FLOPs.**

---

## 1. hls4ml (150 constituents, 5 classes)

Configurations are the repo's evaluation specs and reproduce the paper's Table 2:

| | params | measured FLOPs | paper Table 2 |
|---|---|---|---|
| PHAT-JeT | 6,405 | 1.29 M | 6.7 K / 1.31 M |
| PTv3 | 22,421 | 1.24 M | **22.4 K / 1.24 M** (exact) |

*(PHAT's 6,405 is the `mean` patch-tokenizer variant asserted in
`evaluate_hls4ml_multiplicity.py`; the paper's 6.7 K corresponds to `learned_pool`.)*

### Per-jet latency (ms) by batch size

| config | b=1 | b=32 | b=128 | b=1024 | b=4096 |
|---|---|---|---|---|---|
| **PHAT-JeT** | 2.0631 | 0.0746 | 0.0265 | 0.0148 | **0.0137** |
| PHAT-JeT (flash) | 2.0741 | 0.0770 | 0.0261 | 0.0149 | 0.0140 |
| PTv3 | 4.0963 | 0.1608 | 0.0572 | 0.0285 | **0.0288** |

### Speedup (PTv3 / PHAT-JeT)

| batch | 1 | 32 | 128 | 1024 | 4096 |
|---|---|---|---|---|---|
| speedup | 1.99x | 2.16x | 2.16x | 1.92x | **2.11x** |

**Flat across a 4096x range of batch sizes** — the advantage is not a small-batch artifact.

### Full pass over 50,000 jets at the evaluation batch size (4096)

| config | time (s) | throughput |
|---|---|---|
| **PHAT-JeT** | **0.677 ± 0.002** | **73,895 jets/s** |
| PHAT-JeT (flash) | 0.693 ± 0.009 | 72,107 jets/s |
| PTv3 | 1.314 ± 0.023 | 38,058 jets/s |

PHAT-JeT is **1.94x faster** on the full pass.

> **Batch size dominates absolute numbers**: 2.06 ms/jet at batch 1 versus 0.0137 ms at
> batch 4096, a 150x difference. Any latency figure is meaningless without stating the batch
> size. Batch sizes here span what the repo actually uses: training ramps 128 -> 1024 via the
> default schedule, evaluation uses 4096.

---

## 2. ModelNet40 (1000 points, 40 classes)

### Full test-set pass (2,468 objects, batch 32)

| GMP δ | PHAT grid (x,y,z) | PHAT grid (x,y) | PTv3 |
|---|---|---|---|
| 0.10 | 0.526 ± 0.001 s | 0.368 ± 0.000 s | 1.173 ± 0.007 s |
| 0.15 | 0.431 ± 0.001 s | 0.370 ± 0.001 s | 0.892 ± 0.005 s |
| 0.20 | 0.411 ± 0.002 s | 0.371 ± 0.002 s | 0.816 ± 0.006 s |

### Per-object (ms)

| GMP δ | PHAT (x,y,z) | PHAT (x,y) | PTv3 |
|---|---|---|---|
| 0.10 | 0.2130 | 0.1492 | 0.4754 |
| 0.15 | 0.1746 | 0.1498 | 0.3613 |
| 0.20 | 0.1664 | 0.1502 | 0.3306 |

### Throughput (objects/s)

| GMP δ | PHAT (x,y,z) | PHAT (x,y) | PTv3 |
|---|---|---|---|
| 0.10 | 4,695 | 6,702 | 2,103 |
| 0.15 | 5,729 | 6,678 | 2,768 |
| 0.20 | 6,010 | 6,660 | 3,025 |

PTv3 is **1.99x slower** than PHAT (x,y,z) at δ=0.20.

### Cross-check: two independent measurements agree

A separate benchmark timing one resident batch of 32 random clouds (20 warmup + 100 reps)
gives, per object at δ=0.20: PHAT (x,y,z) 0.1693 ms, PHAT (x,y) 0.1473 ms, PTv3 0.3164 ms —
within 2-4% of the full-pass numbers above, on different data and a different memory path.

---

## 3. Why PHAT is faster despite more FLOPs

| | FLOPs | measured speed |
|---|---|---|
| hls4ml | PTv3 **lower** (1.24 M vs 1.29 M) | PHAT **2.1x faster** |
| ModelNet40 | PTv3 **lower** (28.9 M vs 30.5 M) | PHAT **2.0x faster** |

PTv3's two stages plus serialized pooling incur more sequential kernel launches than FLOPs
accounting reflects. **The FLOPs-matched framing therefore understates PHAT-JeT's advantage**:
PTv3 is nominally given a compute edge it does not realise in practice.

The effect is consistent across two datasets with different input sizes (150 vs 1000 points),
patch sizes (25 vs 10) and class counts (5 vs 40), so it is architectural rather than a
property of one setup.

---

## 4. Caveats

1. **GPU wall-clock, not FPGA latency.** For the L1-trigger claim the relevant number would
   come from hls4ml synthesis; this is a GPU measurement and should be labelled as such.
2. **Flash attention should not be quoted.** It yields no speedup (0.693 s vs 0.677 s,
   marginally slower) but reports 0.97 M FLOPs instead of 1.29 M, because the flash path
   routes through Keras `MultiHeadAttention` which the profiler tallies differently. That is
   a counting artifact, and quoting it would contradict the paper's own 1.31 M.
3. **Latency varies with GMP grid spacing; static FLOPs does not.** On ModelNet40 the counter
   reports one identical value across δ ∈ {0.10, 0.15, 0.20} while measured per-object latency
   ranges 0.213 -> 0.166 ms, because the grid extent is computed at run time and static graph
   analysis cannot see it. δ=0.20 is therefore both the fastest setting and statistically tied
   on accuracy.
4. **No trained weights are involved.** Latency depends only on architecture and input shape,
   so models were built fresh; timing is unaffected by training state.

---

## 5. Method

- `@tf.function`-wrapped inference (not `model.predict`, which adds Keras/Python overhead)
- 20 warmup iterations, then 100 timed, for the fixed-batch benchmarks
- `.numpy()` after each call to force device synchronisation, so real compute is timed rather
  than asynchronous dispatch
- Full-pass benchmarks warm up on **both** batch shapes (full and the short final batch) plus
  two untimed passes, then time 5 passes; every batch is transferred host-to-device
- All configurations for a given benchmark run back-to-back in a single pod on one physical
  GPU, eliminating card-to-card variation
