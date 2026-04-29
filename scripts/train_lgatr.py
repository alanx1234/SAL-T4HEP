#!/usr/bin/env python
"""Train L-GATr on hls4ml/top/QG/JetClass using this repo's script style."""

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

from models.lgatr_wrapper import LGATrJetClassifier
from scripts.equivariant_utils import (
    Lion,
    apply_sorting,
    evaluate,
    get_flops_profiler,
    load_data,
    load_test_data,
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
    p = argparse.ArgumentParser(description="Train L-GATr on jet data")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--dataset", choices=["hls4ml", "top", "QG", "jetclass"], default="hls4ml")
    p.add_argument("--sort_by", choices=["pt", "eta", "phi", "delta_R", "kt", "none"], default="kt")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--schedule",
        default="128:200,256:200,512:200,1024:200,1024:200,1024:400",
        help="Comma-separated training schedule as batch_size:epochs. Use 'none' for --batch_size/--num_epochs.",
    )
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_particles", type=int, default=150)
    p.add_argument("--num_particles_truncate", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=35)
    p.add_argument("--early_stopping_patience", type=int, default=0, help="0 disables early stopping")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.2)
    p.add_argument("--lion_beta1", type=float, default=0.9)
    p.add_argument("--lion_beta2", type=float, default=0.99)
    p.add_argument("--hidden_mv_channels", type=int, default=16)
    p.add_argument("--hidden_s_channels", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--beam_spurion", choices=["xyplane", "lightlike", "spacelike", "timelike", "none"], default="xyplane")
    p.add_argument("--no_time_spurion", action="store_true")
    p.add_argument("--checkpoint_blocks", action="store_true")
    p.add_argument("--test_only", action="store_true", help="Skip training and evaluate a checkpoint")
    p.add_argument("--checkpoint_path", default=None, help="Checkpoint path to load with --test_only")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def forward_lgatr(model, batch):
    _, p4, mask, _ = batch
    return model(p4, mask=mask)


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

    save_root = os.path.join(args.save_dir, "lgatr", str(num_particles), args.sort_by)
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

    if args.test_only and args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required with --test_only")

    if args.test_only:
        x_train = y_train = p4_train = None
        val_loader = None
        n_model_nodes = args.num_particles_truncate or num_particles
        dummy_x = np.zeros((1, n_model_nodes, 3), dtype=np.float32)
        dummy_y = np.zeros((1, num_classes), dtype=np.float32)
        dummy_y[0, 0] = 1.0
        dummy_p4 = np.zeros((1, n_model_nodes, 4), dtype=np.float32)
        dummy_p4[:, :, 0] = 1.0
        first_batch = next(iter(make_loader(dummy_x, dummy_p4, dummy_y, 1, shuffle=False)))
    else:
        x_train, x_val, y_train, y_val, p4_train, p4_val = load_data(
            args.dataset, args.data_dir, num_particles, args.val_split
        )
        x_train, p4_train = apply_sorting(x_train, args.sort_by, p4_train)
        x_val, p4_val = apply_sorting(x_val, args.sort_by, p4_val)
        x_train, p4_train = truncate_arrays(x_train, p4_train, args.num_particles_truncate)
        x_val, p4_val = truncate_arrays(x_val, p4_val, args.num_particles_truncate)
        logging.info("Loaded train x=%s y=%s val x=%s y=%s", x_train.shape, y_train.shape, x_val.shape, y_val.shape)

        val_loader = make_loader(x_val, p4_val, y_val, args.batch_size, shuffle=False)
        first_batch = next(iter(val_loader))

    model = LGATrJetClassifier(
        num_classes=num_classes,
        hidden_mv_channels=args.hidden_mv_channels,
        hidden_s_channels=args.hidden_s_channels,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        beam_spurion=None if args.beam_spurion == "none" else args.beam_spurion,
        add_time_spurion=not args.no_time_spurion,
        checkpoint_blocks=args.checkpoint_blocks,
    ).to(device)
    logging.info("Total params: %d", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    lion_cls = getattr(torch.optim, "Lion", None)
    if lion_cls is not None:
        optimizer = lion_cls(
            model.parameters(),
            lr=args.lr,
            betas=(args.lion_beta1, args.lion_beta2),
            weight_decay=args.weight_decay,
        )
        logging.info("Optimizer: Lion")
    else:
        optimizer = Lion(
            model.parameters(),
            lr=args.lr,
            betas=(args.lion_beta1, args.lion_beta2),
            weight_decay=args.weight_decay,
        )
        logging.info("Optimizer: Lion (local fallback)")

    flops = get_flops_profiler(model, first_batch, device, forward_lgatr)
    if flops:
        flops_per_event = flops // len(first_batch[-1])
        logging.info("FLOPs per inference: %d", flops_per_event)
        logging.info("MACs per inference: %d", flops_per_event // 2)

    if args.test_only:
        logging.info("Loading checkpoint: %s", args.checkpoint_path)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        x_test, y_test, p4_test = load_test_data(args.dataset, args.data_dir, num_particles)
        x_test, p4_test = apply_sorting(x_test, args.sort_by, p4_test)
        x_test, p4_test = truncate_arrays(x_test, p4_test, args.num_particles_truncate)
        test_loader = make_loader(x_test, p4_test, y_test, args.batch_size, shuffle=False)
        first_test_batch = next(iter(test_loader))
        avg_ns = time_inference(model, first_test_batch, len(first_test_batch[-1]), device, forward_lgatr)
        logging.info("Avg inference time/event: %.2f ns", avg_ns)
        labels, y_onehot, probs, acc, auc_m = evaluate(model, test_loader, device, forward_lgatr, num_classes)
        logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)
        plot_and_log_metrics(labels, y_onehot, probs, args.dataset, save_dir)
        return

    histories = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val = float("inf")
    schedule = parse_training_schedule(args.schedule, args.batch_size, args.num_epochs)
    total_epochs = sum(ep for _, ep in schedule)
    current_epoch = 0
    patience_counter = 0
    should_stop = False
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
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, forward_lgatr)
            val_loss, val_acc = validate(model, val_loader, criterion, device, forward_lgatr)
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
                patience_counter = 0
            else:
                patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logging.info("Early stopping at epoch %d", current_epoch)
                should_stop = True
                break
        if should_stop:
            break

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    best_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    save_curves(save_dir, histories)

    x_test, y_test, p4_test = load_test_data(args.dataset, args.data_dir, num_particles)
    x_test, p4_test = apply_sorting(x_test, args.sort_by, p4_test)
    x_test, p4_test = truncate_arrays(x_test, p4_test, args.num_particles_truncate)
    test_loader = make_loader(x_test, p4_test, y_test, args.batch_size, shuffle=False)
    avg_ns = time_inference(model, first_batch, len(first_batch[-1]), device, forward_lgatr)
    logging.info("Avg inference time/event: %.2f ns", avg_ns)
    labels, y_onehot, probs, acc, auc_m = evaluate(model, test_loader, device, forward_lgatr, num_classes)
    logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)
    plot_and_log_metrics(labels, y_onehot, probs, args.dataset, save_dir)


if __name__ == "__main__":
    main()
