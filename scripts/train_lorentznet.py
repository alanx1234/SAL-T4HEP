#!/usr/bin/env python
"""Train LorentzNet on hls4ml/top/QG/JetClass using this repo's script style."""

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.lorentznet import LorentzNet
from scripts.equivariant_utils import (
    apply_sorting,
    evaluate,
    get_flops_profiler,
    load_data,
    load_test_data,
    lorentz_scalars_from_p4,
    make_full_edges,
    make_loader,
    parse_training_schedule,
    plot_and_log_metrics,
    save_curves,
    set_optimizer_lr,
    time_inference,
    train_epoch,
    truncate_arrays,
    validate,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train LorentzNet on jet data")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--dataset", choices=["hls4ml", "top", "QG", "jetclass"], default="hls4ml")
    p.add_argument("--sort_by", choices=["pt", "eta", "phi", "delta_R", "kt", "none"], default="kt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--schedule",
        default="128:200,256:200,512:200,1024:200,1024:200,1024:400",
        help="Comma-separated training schedule as batch_size:epochs. Use 'none' for --batch_size/--num_epochs.",
    )
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_particles", type=int, default=150)
    p.add_argument("--num_particles_truncate", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=35)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--n_hidden", type=int, default=72)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--c_weight", type=float, default=5e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def forward_lorentznet(model, batch):
    _, p4, mask, _ = batch
    batch_size, n_nodes, _ = p4.shape
    flat_p4 = p4.reshape(batch_size * n_nodes, 4)
    scalars = lorentz_scalars_from_p4(p4).reshape(batch_size * n_nodes, 1)
    node_mask = mask.reshape(batch_size * n_nodes, 1).to(p4.dtype)
    edges = make_full_edges(mask)
    return model(scalars, flat_p4, edges, node_mask, n_nodes)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset == "jetclass":
        num_particles, num_classes = args.num_particles, 10
    elif args.dataset in ("top", "QG"):
        num_particles, num_classes = (200 if args.dataset == "top" else 150), 2
    else:
        num_particles, num_classes = args.num_particles, 5

    save_root = os.path.join(args.save_dir, "lorentznet", str(num_particles), args.sort_by)
    trial = 0
    while True:
        cand = os.path.join(save_root, f"trial-{trial}")
        time.sleep(random.randint(1, 4))
        if not os.path.isdir(cand):
            save_dir = cand
            break
        trial += 1
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_dir, "train.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Args: %s", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    x_train, x_val, y_train, y_val, p4_train, p4_val = load_data(
        args.dataset, args.data_dir, num_particles, args.val_split
    )
    x_train, p4_train = apply_sorting(x_train, args.sort_by, p4_train)
    x_val, p4_val = apply_sorting(x_val, args.sort_by, p4_val)
    x_train, p4_train = truncate_arrays(x_train, p4_train, args.num_particles_truncate)
    x_val, p4_val = truncate_arrays(x_val, p4_val, args.num_particles_truncate)
    logging.info("Loaded train x=%s y=%s val x=%s y=%s", x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    val_loader = make_loader(x_val, p4_val, y_val, args.batch_size, shuffle=False)

    model = LorentzNet(
        n_scalar=1,
        n_hidden=args.n_hidden,
        n_class=num_classes,
        n_layers=args.n_layers,
        c_weight=args.c_weight,
        dropout=args.dropout,
    ).to(device)
    logging.info("Total params: %d", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    first_batch = next(iter(val_loader))
    flops = get_flops_profiler(model, first_batch, device, forward_lorentznet)
    if flops:
        flops_per_event = flops // len(first_batch[-1])
        logging.info("FLOPs per inference: %d", flops_per_event)
        logging.info("MACs per inference: %d", flops_per_event // 2)

    histories = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val = float("inf")
    schedule = parse_training_schedule(args.schedule, args.batch_size, args.num_epochs)
    total_epochs = sum(ep for _, ep in schedule)
    current_epoch = 0
    logging.info("Training schedule: %s", schedule)
    for stage_idx, (stage_batch_size, stage_epochs) in enumerate(schedule):
        set_optimizer_lr(optimizer, args.lr)
        train_loader = make_loader(x_train, p4_train, y_train, stage_batch_size, shuffle=True)
        logging.info(
            "Starting stage %d/%d: batch_size=%d epochs=%d lr=%g",
            stage_idx + 1,
            len(schedule),
            stage_batch_size,
            stage_epochs,
            args.lr,
        )
        for _ in range(stage_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, forward_lorentznet)
            val_loss, val_acc = validate(model, val_loader, criterion, device, forward_lorentznet)
            current_epoch += 1
            histories["train_loss"].append(train_loss)
            histories["val_loss"].append(val_loss)
            histories["train_accuracy"].append(train_acc)
            histories["val_accuracy"].append(val_acc)
            logging.info(
                "Epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                current_epoch,
                total_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    save_curves(save_dir, histories)

    x_test, y_test, p4_test = load_test_data(args.dataset, args.data_dir, num_particles)
    x_test, p4_test = apply_sorting(x_test, args.sort_by, p4_test)
    x_test, p4_test = truncate_arrays(x_test, p4_test, args.num_particles_truncate)
    test_loader = make_loader(x_test, p4_test, y_test, args.batch_size, shuffle=False)
    avg_ns = time_inference(model, first_batch, len(first_batch[-1]), device, forward_lorentznet)
    logging.info("Avg inference time/event: %.2f ns", avg_ns)
    labels, y_onehot, probs, acc, auc_m = evaluate(model, test_loader, device, forward_lorentznet, num_classes)
    logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)
    plot_and_log_metrics(labels, y_onehot, probs, args.dataset, save_dir)


if __name__ == "__main__":
    main()
