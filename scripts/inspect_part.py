#!/usr/bin/env python
"""
Sweep ParT (Particle Transformer) configs to find one near ~1.3M FLOPs.
Uses the EXACT same profiling methodology as benchmark_pytorch_models.py
and scan_proj_dim.py for apples-to-apples comparison with SAL-T/Linformer.

Place in: SAL-T4HEP/scripts/inspect_part.py
Requires: SAL-T4HEP/models/parT.py (with networks.logger stub)
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

# ─── make project root importable ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.parT import ParticleTransformer


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure(model, x, v, mask, warmup=5, iters=20, device="cuda"):
    """
    Measure FLOPs, timing, and peak memory.
    Matches benchmark_pytorch_models.py methodology exactly.
    """
    model.eval()
    model.to(device)
    x = x.to(device)
    v = v.to(device) if v is not None else None
    mask = mask.to(device)

    # warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(x, v=v, mask=mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # timing
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(x, v=v, mask=mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg_per_item_ns = (sum(times) / len(times)) / x.size(0) * 1e9

    # peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            _ = model(x, v=v, mask=mask)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_mb = 0.0

    # FLOPs via profiler — exact same as benchmark_pytorch_models.py
    total_flops = None
    try:
        import os as _os
        from contextlib import redirect_stderr as _redirect_stderr
        with open(_os.devnull, "w") as _devnull, _redirect_stderr(_devnull):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
            ) as prof:
                with torch.inference_mode():
                    _ = model(x, v=v, mask=mask)
        events = prof.key_averages()
        flops = 0
        for evt in events:
            evt_flops = getattr(evt, "flops", None)
            if isinstance(evt_flops, (int, float)):
                flops += int(evt_flops)
        total_flops = flops if flops > 0 else None
    except Exception:
        total_flops = None

    return avg_per_item_ns, peak_mb, total_flops


def build_part(embed_dims, pair_embed_dims, num_heads, pair_input_dim,
               num_layers=1, num_cls_layers=1, num_classes=10, num_particles=150):
    """Build a ParT model with the given config."""
    pe = pair_embed_dims if pair_embed_dims else None
    pid = pair_input_dim if pair_embed_dims else 0

    block_params = {
        'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0,
        'scale_fc': False, 'scale_attn': False,
        'scale_heads': False, 'scale_resids': False,
    }

    model = ParticleTransformer(
        input_dim=3,
        num_classes=num_classes,
        pair_input_dim=pid,
        pair_extra_dim=0,
        remove_self_pair=True,
        use_pre_activation_pair=True,
        embed_dims=embed_dims,
        pair_embed_dims=pe,
        num_heads=num_heads,
        num_layers=num_layers,
        num_cls_layers=num_cls_layers,
        block_params=block_params,
        cls_block_params=block_params,
        fc_params=[],
        activation='gelu',
        trim=False,
        for_inference=False,
    )
    return model


def make_dummy_inputs(batch_size, num_particles, pair_input_dim):
    """Create dummy inputs matching ParT's expected format."""
    x = torch.randn(batch_size, 3, num_particles)
    v = torch.randn(batch_size, 4, num_particles) if pair_input_dim > 0 else None
    mask = torch.ones(batch_size, 1, num_particles)
    return x, v, mask


def main():
    parser = argparse.ArgumentParser(description="Sweep ParT configs for target FLOPs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_particles", type=int, default=150)
    parser.add_argument("--num_classes", type=int, default=10, help="10 for jetclass, 5 for hls4ml")
    parser.add_argument("--device", choices=["cuda", "cpu"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_flops", type=int, default=1_300_000)
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of measurement runs for mean±std")
    args = parser.parse_args()

    B = args.batch_size
    N = args.num_particles
    C = args.num_classes
    target = args.target_flops

    # pair_input_dim must be 1, 3, 4, 5, 6, or 8 (pairwise_lv_fts constraint)
    configs = [
        # (embed_dims, pair_embed_dims, num_heads, pair_input_dim, name)
        ([8],  [2],    2, 4, "d8-h2-pe2"),
        ([8],  [4],    2, 4, "d8-h2-pe4"),
        ([8],  [8],    2, 4, "d8-h2-pe8"),
        ([8],  [4,4],  2, 4, "d8-h2-pe4x4"),
        ([8],  [4],    2, 1, "d8-h2-pe4-plv1"),
        ([8],  None,   2, 0, "d8-h2-nopair"),
        ([10], [4],    2, 4, "d10-h2-pe4"),
        ([10], [2],    2, 4, "d10-h2-pe2"),
        ([10], None,   2, 0, "d10-h2-nopair"),
        ([12], [4],    2, 4, "d12-h2-pe4"),
        ([12], [2],    2, 4, "d12-h2-pe2"),
        ([12], None,   2, 0, "d12-h2-nopair"),
        ([12], [4],    4, 4, "d12-h4-pe4"),
        ([16], None,   4, 0, "d16-h4-nopair"),
        ([16], None,   2, 0, "d16-h2-nopair"),
        ([8,8],  [4],  2, 4, "d8x8-h2-pe4"),
        ([12,8], [4],  2, 4, "d12x8-h2-pe4"),
    ]

    print("=" * 100)
    print(f"ParT FLOPs Sweep — target: ~{target:,}")
    print(f"B={B}, N={N}, classes={C}, device={args.device}, runs={args.num_runs}")
    print("=" * 100)
    header = f"{'Config':<25} {'Params':>8} {'FLOPs':>14} {'ratio':>7} {'ns/evt':>10} {'peakMB':>8}"
    print(header)
    print("-" * 100)

    for embed_dims, pe, nh, pid, name in configs:
        try:
            model = build_part(embed_dims, pe, nh, pid,
                               num_classes=C, num_particles=N)
            pcount = num_params(model)
            x, v, mask = make_dummy_inputs(B, N, pid)

            # Multiple runs for stability
            flops_list = []
            times_list = []
            mems_list = []
            for _ in range(args.num_runs):
                ns, mb, flops = measure(model, x, v, mask, device=args.device)
                times_list.append(ns)
                mems_list.append(mb)
                if flops is not None:
                    flops_list.append(flops)

            if flops_list:
                flops_mean = int(np.mean(flops_list))
                flops_str = f"{flops_mean:,}"
                ratio = flops_mean / target
                ratio_str = f"{ratio:.2f}x"
            else:
                flops_str = "N/A"
                ratio_str = "N/A"
                ratio = 0

            time_mean = np.mean(times_list)
            mem_mean = np.mean(mems_list)

            marker = " ✓" if isinstance(ratio, float) and 0.85 <= ratio <= 1.15 else ""
            print(f"{name:<25} {pcount:>8,} {flops_str:>14} {ratio_str:>7} "
                  f"{time_mean:>9.1f} {mem_mean:>7.1f}{marker}")

        except Exception as e:
            print(f"{name:<25} FAILED: {e}")

    print("=" * 100)
    print(f"Target: ~{target:,} FLOPs. Configs marked ✓ are within 85-115% of target.")
    print("Use --num_classes 5 for hls4ml dataset comparison.")


if __name__ == "__main__":
    main()