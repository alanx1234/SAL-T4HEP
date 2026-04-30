#!/usr/bin/env python
"""
Train a (tiny) Particle Transformer (ParT) on jet datasets.
Adapted from train_deltanet_pytorch.py — same data loading, sorting, schedule.

Place in: SAL-T4HEP/scripts/train_part.py
Requires: SAL-T4HEP/models/parT.py (with networks.logger stub)

Key notes:
- ParT expects (N, C, P) format for features, (N, 4, P) for 4-vectors
- We compute 4-vectors from (pt, eta, phi) at data-loading time
- JetClass data is stored as (N, 3, P), needs transpose to (N, P, 3) first
- Training schedule matches SAL-T/Linformer scripts for fair comparison
"""
import os
import sys
import time
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── make project root importable ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.parT import ParticleTransformer


def parse_training_schedule(schedule, batch_size, num_epochs):
    if schedule is None or str(schedule).strip().lower() in ("", "none", "off", "false"):
        return [(batch_size, num_epochs)]
    parsed = []
    for item in str(schedule).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Schedule item '{item}' must be formatted as batch_size:epochs")
        bs, ep = item.split(":", 1)
        parsed.append((int(bs), int(ep)))
    if not parsed:
        raise ValueError("Training schedule is empty")
    return parsed


# ─── Data helpers ─────────────────────────────────────────────────────────────

def ptetaphi_to_p4(x_np):
    """
    Convert (batch, N, 3) with [pt, eta, phi] -> (batch, N, 4) with [px, py, pz, E].
    Assumes massless particles.
    """
    pt = x_np[:, :, 0]
    eta = x_np[:, :, 1]
    phi = x_np[:, :, 2]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(np.clip(eta, -10, 10))
    energy = pt * np.cosh(np.clip(eta, -10, 10))
    return np.stack([px, py, pz, energy], axis=-1)


def apply_sorting(x, sort_by):
    if sort_by == "pt":
        key = x[:, :, 0]
    elif sort_by == "eta":
        key = x[:, :, 1]
    elif sort_by == "phi":
        key = x[:, :, 2]
    elif sort_by == "delta_R":
        key = np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
    elif sort_by == "kt":
        key = x[:, :, 0] * np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
    else:
        return x
    idx = np.argsort(key, axis=1)[:, ::-1]
    return np.take_along_axis(x, idx[:, :, None], axis=1)


def prepare_part_inputs(x_np):
    """
    Prepare inputs for ParT from (batch, N, 3) ptetaphi data.
    Returns:
        x_feat: (batch, 3, N) — features in ParT's (N, C, P) format
        v:      (batch, 4, N) — 4-vectors for pairwise features
        mask:   (batch, 1, N) — 1 for real particles, 0 for padding
    """
    x_feat = x_np.transpose(0, 2, 1).astype(np.float32)
    p4 = ptetaphi_to_p4(x_np)
    v = p4.transpose(0, 2, 1).astype(np.float32)
    mask = (np.abs(x_np[:, :, 0]) > 1e-6).astype(np.float32)[:, np.newaxis, :]
    return x_feat, v, mask


def load_data(dataset, data_dir, num_particles, val_split=0.2):
    """
    Load train/val data following the project's conventions.
    Returns x_train, x_val, y_train, y_val all in (N, P, 3) format.
    """
    if dataset == "hls4ml":
        x = np.load(os.path.join(data_dir, f"x_train_robust_{num_particles}const_ptetaphi.npy"))
        y = np.load(os.path.join(data_dir, f"y_train_robust_{num_particles}const_ptetaphi.npy"))
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=val_split, random_state=42
        )
    elif dataset == "jetclass":
        x_train = np.load(os.path.join(data_dir, "train/features.npy"))
        y_train = np.load(os.path.join(data_dir, "train/labels.npy"))
        x_val = np.load(os.path.join(data_dir, "val/features.npy"))
        y_val = np.load(os.path.join(data_dir, "val/labels.npy"))
        # jetclass stores as (N, 3, P), transpose to (N, P, 3)
        x_train = x_train.transpose(0, 2, 1)
        x_val = x_val.transpose(0, 2, 1)
    elif dataset == "top":
        x_train = np.load(os.path.join(data_dir, "train/features.npy"))
        y_train = np.load(os.path.join(data_dir, "train/labels.npy"))
        x_val = np.load(os.path.join(data_dir, "val/features.npy"))
        y_val = np.load(os.path.join(data_dir, "val/labels.npy"))
    elif dataset == "QG":
        x_train = np.load(os.path.join(data_dir, "train/features.npy"))
        y_train = np.load(os.path.join(data_dir, "train/labels.npy"))
        x_val = np.load(os.path.join(data_dir, "val/features.npy"))
        y_val = np.load(os.path.join(data_dir, "val/labels.npy"))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return x_train, x_val, y_train, y_val


def load_test_data(dataset, data_dir, num_particles, mmap_mode=None):
    """Load test data following the project's conventions."""
    if dataset == "hls4ml":
        x = np.load(os.path.join(data_dir, f"x_val_robust_{num_particles}const_ptetaphi.npy"), mmap_mode=mmap_mode)
        y = np.load(os.path.join(data_dir, f"y_val_robust_{num_particles}const_ptetaphi.npy"), mmap_mode=mmap_mode)
    elif dataset == "jetclass":
        x = np.load(os.path.join(data_dir, "test/features.npy"), mmap_mode=mmap_mode)
        y = np.load(os.path.join(data_dir, "test/labels.npy"), mmap_mode=mmap_mode)
        x = x.transpose(0, 2, 1)
    elif dataset == "top":
        x = np.load(os.path.join(data_dir, "test/features.npy"), mmap_mode=mmap_mode)
        y = np.load(os.path.join(data_dir, "test/labels.npy"), mmap_mode=mmap_mode)
    elif dataset == "QG":
        x = np.load(os.path.join(data_dir, "test/features.npy"), mmap_mode=mmap_mode)
        y = np.load(os.path.join(data_dir, "test/labels.npy"), mmap_mode=mmap_mode)
    return x, y


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_v, batch_mask, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_v = batch_v.to(device)
        batch_mask = batch_mask.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x, v=batch_v, mask=batch_mask)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        _, labels = batch_y.max(1)
        correct += predicted.eq(labels).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_v, batch_mask, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_v = batch_v.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x, v=batch_v, mask=batch_mask)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            _, labels = batch_y.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_x.size(0)

    return total_loss / total, correct / total


# ─── FLOPs — matches benchmark_pytorch_models.py exactly ─────────────────────

def get_flops_profiler(model, x, v, mask, device):
    """
    Measure FLOPs using the same methodology as benchmark_pytorch_models.py:
    CPU+CUDA activities, record_shapes, profile_memory, with_flops, inference_mode.
    """
    model.eval().to(device)
    x = x.to(device)
    v = v.to(device) if v is not None else None
    mask = mask.to(device)

    # warmup
    with torch.inference_mode():
        _ = model(x, v=v, mask=mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    try:
        from contextlib import redirect_stderr
        with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
            ) as prof:
                with torch.inference_mode():
                    _ = model(x, v=v, mask=mask)

        total = 0
        for evt in prof.key_averages():
            evt_flops = getattr(evt, "flops", None)
            if isinstance(evt_flops, (int, float)):
                total += int(evt_flops)
        return total if total > 0 else None
    except Exception:
        return None


# ─── Testing ──────────────────────────────────────────────────────────────────

def run_testing(model, dataset, data_dir, save_dir, sort_by, batch_size, num_particles, device):
    logging.info("Starting testing phase...")

    x_test, y_test = load_test_data(dataset, data_dir, num_particles, mmap_mode="r")
    logging.info("Loaded TEST: x=%s, y=%s", x_test.shape, y_test.shape)

    def prepare_test_chunk(start, end):
        x_chunk = np.asarray(x_test[start:end])
        x_chunk = apply_sorting(x_chunk, sort_by)
        x_feat, v, mask = prepare_part_inputs(x_chunk)
        return (
            torch.from_numpy(x_feat).to(device),
            torch.from_numpy(v).to(device),
            torch.from_numpy(mask).to(device),
        )

    logging.info("Applied '%s' sorting to TEST set in chunks", sort_by)
    first_chunk_end = min(batch_size, len(x_test))
    x_feat_t, v_t, mask_t = prepare_test_chunk(0, first_chunk_end)

    # FLOPs (single sample)
    flops = get_flops_profiler(model, x_feat_t[:1], v_t[:1], mask_t[:1], device)
    if flops:
        logging.info("FLOPs per inference: %d", flops)
        logging.info("MACs per inference: %d", flops // 2)

    # Timing
    model.eval()
    with torch.inference_mode():
        _ = model(x_feat_t, v=v_t, mask=mask_t)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(x_feat_t, v=v_t, mask=mask_t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg_ns = np.mean(times) / first_chunk_end * 1e9
    logging.info("Avg inference time/event: %.2f ns", avg_ns)

    # GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            _ = model(x_feat_t, v=v_t, mask=mask_t)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logging.info("GPU peak memory: %.1f MB", peak_mb)

    # Predictions
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            end = min(i + batch_size, len(x_test))
            x_feat_t, v_t, mask_t = prepare_test_chunk(i, end)
            out = model(x_feat_t, v=v_t, mask=mask_t)
            all_preds.append(torch.softmax(out, dim=1).cpu().numpy())
    preds = np.vstack(all_preds)

    # Metrics
    if dataset in ("top", "QG"):
        acc = accuracy_score(y_test, (preds.ravel() > 0.5).astype(int))
        auc_m = roc_auc_score(y_test, preds.ravel())
    else:
        acc = accuracy_score(np.argmax(y_test, 1), np.argmax(preds, 1))
        auc_m = roc_auc_score(y_test, preds, average="macro", multi_class="ovo")
    logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)

    # ROC curves
    if dataset == "hls4ml":
        labels = ["q", "g", "W", "Z", "t"]
    elif dataset == "jetclass":
        labels = ["label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
                  "label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl"]
    elif dataset == "top":
        labels = ["qcd", "top"]
    elif dataset == "QG":
        labels = ["Gluon", "Quark"]
    else:
        labels = [f"class_{i}" for i in range(preds.shape[1])]

    plt.figure(figsize=(6, 6))
    one_over_fpr = {}
    for i, lab in enumerate(labels):
        if dataset in ("top", "QG"):
            fpr, tpr, _ = roc_curve(y_test, preds.ravel())
        else:
            fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i])
        roc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_val:.2f})")
        if np.max(tpr) >= 0.8:
            fpr_t = np.interp(0.8, tpr, fpr)
            one_over_fpr[lab] = 1.0 / fpr_t if fpr_t > 0 else np.nan
            plt.plot(fpr_t, 0.8, "o")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curves (ParT)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"))
    plt.close()

    for lab, val in one_over_fpr.items():
        logging.info("1/FPR@0.8 for %s: %.3f", lab, val)
    logging.info("Avg 1/FPR@0.8: %.3f", np.nanmean(list(one_over_fpr.values())))

    # Background rejection
    if dataset not in ("top", "QG"):
        rej_vals = []
        for i, lab in enumerate(labels[1:], start=1):
            if dataset == "jetclass":
                mask_bg = np.ones_like(y_test[:, 0], dtype=bool)
                bin_y = (y_test[mask_bg, i] == 1).astype(int)
            else:
                mask_bg = (y_test[:, 0] == 1) | (y_test[:, 1] == 1) | (y_test[:, i] == 1)
                bin_y = (y_test[mask_bg, i] == 1).astype(int)
            bin_s = preds[mask_bg, i]
            fpr_v, tpr_v, _ = roc_curve(bin_y, bin_s)
            idx = np.argmin(np.abs(tpr_v - 0.8))
            rej = 1.0 / fpr_v[idx] if fpr_v[idx] > 0 else np.inf
            logging.info("Bg rejection@0.8 %s: %.3f", lab, rej)
            rej_vals.append(rej)
        logging.info("Avg bg rejection@0.8: %.3f", np.nanmean(rej_vals))


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ParT (PyTorch) on jet data")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--dataset", choices=["hls4ml", "top", "QG", "jetclass"], default="hls4ml")
    p.add_argument("--sort_by", choices=["pt", "eta", "phi", "delta_R", "kt"], default="kt")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_epochs", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--schedule",
        default="128:200,256:200,512:200,1024:200,1024:200,1024:400",
        help="Comma-separated training schedule as batch_size:epochs. Use 'none' for --batch_size/--num_epochs.",
    )
    p.add_argument("--early_stopping_patience", type=int, default=40)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_particles", type=int, default=150)

    # ParT architecture
    p.add_argument("--embed_dims", type=int, nargs="+", default=[16])
    p.add_argument("--pair_embed_dims", type=int, nargs="+", default=[8])
    p.add_argument("--pair_input_dim", type=int, default=4, help="Pairwise LV features (1,3,4,5,6,8)")
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--num_cls_layers", type=int, default=1)
    p.add_argument("--no_pair_embed", action="store_true", help="Disable pair embedding entirely")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--flops_only", action="store_true", help="Build model, report FLOPs, then exit")
    p.add_argument("--test_only", action="store_true", help="Skip training and evaluate a checkpoint")
    p.add_argument("--checkpoint_path", default=None, help="Checkpoint path to load with --test_only")
    return p.parse_args()


def main():
    args = parse_args()

    # Dataset config
    if args.dataset == "hls4ml":
        num_particles = args.num_particles
        output_dim = 5
    elif args.dataset == "jetclass":
        num_particles = args.num_particles
        output_dim = 10
    elif args.dataset == "top":
        num_particles = 200
        output_dim = 1
    elif args.dataset == "QG":
        num_particles = 150
        output_dim = 1
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Save directory with trial numbering
    save_dir = os.path.join(args.save_dir, str(num_particles), args.sort_by)
    trial = 0
    while True:
        cand = os.path.join(save_dir, f"trial-{trial}")
        time.sleep(random.randint(1, 4))
        if not os.path.isdir(cand):
            save_dir = cand
            break
        trial += 1
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_dir, "train.log"),
        filemode="w", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Args: %s", args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    if args.test_only and args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required with --test_only")

    if args.test_only:
        x_train_t = torch.zeros((1, 3, num_particles), dtype=torch.float32)
        v_train_t = torch.zeros((1, 4, num_particles), dtype=torch.float32)
        mask_train_t = torch.ones((1, 1, num_particles), dtype=torch.float32)
        x_val_t = v_val_t = mask_val_t = y_val_t = None
    else:
        # ── Load data ──────────────────────────────────────────────────────
        x_train, x_val, y_train, y_val = load_data(
            args.dataset, args.data_dir, num_particles, args.val_split
        )
        logging.info("Loaded train x=%s y=%s, val x=%s y=%s",
                     x_train.shape, y_train.shape, x_val.shape, y_val.shape)

        x_train = apply_sorting(x_train, args.sort_by)
        x_val = apply_sorting(x_val, args.sort_by)

        # Prepare ParT-format inputs
        x_train_feat, v_train, mask_train = prepare_part_inputs(x_train)
        x_val_feat, v_val, mask_val = prepare_part_inputs(x_val)

        x_train_t = torch.FloatTensor(x_train_feat)
        v_train_t = torch.FloatTensor(v_train)
        mask_train_t = torch.FloatTensor(mask_train)
        y_train_t = torch.FloatTensor(y_train)

        x_val_t = torch.FloatTensor(x_val_feat)
        v_val_t = torch.FloatTensor(v_val)
        mask_val_t = torch.FloatTensor(mask_val)
        y_val_t = torch.FloatTensor(y_val)

    # ── Build model ────────────────────────────────────────────────────────
    pair_embed_dims = None if args.no_pair_embed else args.pair_embed_dims
    pair_input_dim = 0 if args.no_pair_embed else args.pair_input_dim

    block_params = {
        'dropout': args.dropout, 'attn_dropout': args.dropout,
        'activation_dropout': args.dropout,
        'scale_fc': False, 'scale_attn': False,
        'scale_heads': False, 'scale_resids': False,
    }

    model = ParticleTransformer(
        input_dim=3,
        num_classes=output_dim,
        pair_input_dim=pair_input_dim,
        pair_extra_dim=0,
        remove_self_pair=True,
        use_pre_activation_pair=True,
        embed_dims=args.embed_dims,
        pair_embed_dims=pair_embed_dims,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_cls_layers=args.num_cls_layers,
        block_params=block_params,
        cls_block_params=block_params,
        fc_params=[],
        activation='gelu',
        trim=False,
        for_inference=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Total params: %d", total_params)
    print(f"ParT model: {total_params:,} params")
    print(f"  embed_dims={args.embed_dims}, pair_embed_dims={pair_embed_dims}")
    print(f"  num_heads={args.num_heads}, layers={args.num_layers}, cls_layers={args.num_cls_layers}")

    # Measure FLOPs
    flops = get_flops_profiler(
        model, x_train_t[:1].to(device), v_train_t[:1].to(device),
        mask_train_t[:1].to(device), device
    )
    if flops:
        logging.info("FLOPs per inference: %d", flops)
        print(f"  FLOPs: {flops:,}")

    if args.flops_only:
        return

    if args.test_only:
        logging.info("Loading checkpoint: %s", args.checkpoint_path)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        run_testing(model, args.dataset, args.data_dir, save_dir,
                    args.sort_by, args.batch_size, num_particles, device)
        logging.info("Test-only evaluation complete!")
        return

    # ── Train ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    schedule = parse_training_schedule(args.schedule, args.batch_size, args.num_epochs)
    logging.info("Training schedule: %s", schedule)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    current_epoch = 0
    patience_counter = 0
    should_stop = False

    for batch_size, num_epochs in schedule:
        logging.info("Training batch_size=%d for %d epochs", batch_size, num_epochs)
        print(f"\nBatch size={batch_size}, epochs {current_epoch}->{current_epoch + num_epochs}")

        train_loader = DataLoader(
            TensorDataset(x_train_t, v_train_t, mask_train_t, y_train_t),
            batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            TensorDataset(x_val_t, v_val_t, mask_val_t, y_val_t),
            batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

        for pg in optimizer.param_groups:
            pg['lr'] = 1e-3

        for epoch in range(num_epochs):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc = validate(model, val_loader, criterion, device)

            train_losses.append(t_loss)
            val_losses.append(v_loss)
            train_accs.append(t_acc)
            val_accs.append(v_acc)

            if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
                print(f"  Epoch {current_epoch + epoch + 1}: "
                      f"t_loss={t_loss:.4f} v_loss={v_loss:.4f} v_acc={v_acc:.4f}")
                logging.info("Epoch %d: t_loss=%.4f v_loss=%.4f t_acc=%.4f v_acc=%.4f",
                             current_epoch + epoch + 1, t_loss, v_loss, t_acc, v_acc)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.early_stopping_patience:
                logging.info("Early stopping at epoch %d", current_epoch + epoch + 1)
                print(f"  Early stopping at epoch {current_epoch + epoch + 1}")
                should_stop = True
                break

        current_epoch += epoch + 1
        if should_stop:
            break

    # ── Save & plot ────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"), weights_only=True))

    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses))
    np.save(os.path.join(save_dir, "val_loss.npy"), np.array(val_losses))
    np.save(os.path.join(save_dir, "train_accuracy.npy"), np.array(train_accs))
    np.save(os.path.join(save_dir, "val_accuracy.npy"), np.array(val_accs))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train"); plt.plot(val_accs, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # ── Test ───────────────────────────────────────────────────────────────
    run_testing(model, args.dataset, args.data_dir, save_dir,
                args.sort_by, args.batch_size, num_particles, device)

    logging.info("Training complete!")
    print(f"\nDone! Results in {save_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
