import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.profiler import ProfilerActivity, profile
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


JETCLASS_LABELS = [
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl",
]


def ptetaphi_to_p4(x_np):
    pt = x_np[:, :, 0]
    eta = x_np[:, :, 1]
    phi = x_np[:, :, 2]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(np.clip(eta, -10, 10))
    energy = pt * np.cosh(np.clip(eta, -10, 10))
    return np.stack([energy, px, py, pz], axis=-1).astype(np.float32)


def apply_sorting(x, sort_by, p4=None):
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
        return x, p4
    idx = np.argsort(key, axis=1)[:, ::-1]
    x_sorted = np.take_along_axis(x, idx[:, :, None], axis=1)
    if p4 is None:
        return x_sorted, None
    return x_sorted, np.take_along_axis(p4, idx[:, :, None], axis=1)


def labels_to_indices(y):
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int64)
    return np.argmax(y, axis=1).astype(np.int64)


def one_hot_from_indices(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=np.int64)
    out[np.arange(len(y)), y.astype(np.int64)] = 1
    return out


def _load_split(data_dir, split):
    x = np.load(os.path.join(data_dir, split, "features.npy"))
    y = np.load(os.path.join(data_dir, split, "labels.npy"))
    vectors_path = os.path.join(data_dir, split, "vectors.npy")
    p4 = np.load(vectors_path) if os.path.exists(vectors_path) else None
    return x, y, p4


def _maybe_transpose_jetclass(x, p4):
    if x.ndim == 3 and x.shape[1] in (3, 4, 7) and x.shape[1] < x.shape[2]:
        x = x.transpose(0, 2, 1)
    if p4 is not None and p4.ndim == 3 and p4.shape[1] == 4:
        p4 = p4.transpose(0, 2, 1)
    return x, p4


def load_data(dataset, data_dir, num_particles, val_split=0.2):
    if dataset == "hls4ml":
        x = np.load(os.path.join(data_dir, f"x_train_robust_{num_particles}const_ptetaphi.npy"))
        y = np.load(os.path.join(data_dir, f"y_train_robust_{num_particles}const_ptetaphi.npy"))
        p4 = ptetaphi_to_p4(x)
        x_train, x_val, y_train, y_val, p4_train, p4_val = train_test_split(
            x, y, p4, test_size=val_split, random_state=42
        )
    else:
        x_train, y_train, p4_train = _load_split(data_dir, "train")
        x_val, y_val, p4_val = _load_split(data_dir, "val")
        if dataset == "jetclass":
            x_train, p4_train = _maybe_transpose_jetclass(x_train, p4_train)
            x_val, p4_val = _maybe_transpose_jetclass(x_val, p4_val)
    if p4_train is None:
        p4_train = ptetaphi_to_p4(x_train)
    if p4_val is None:
        p4_val = ptetaphi_to_p4(x_val)
    return x_train, x_val, y_train, y_val, p4_train.astype(np.float32), p4_val.astype(np.float32)


def load_test_data(dataset, data_dir, num_particles):
    if dataset == "hls4ml":
        x = np.load(os.path.join(data_dir, f"x_val_robust_{num_particles}const_ptetaphi.npy"))
        y = np.load(os.path.join(data_dir, f"y_val_robust_{num_particles}const_ptetaphi.npy"))
        p4 = ptetaphi_to_p4(x)
    else:
        x, y, p4 = _load_split(data_dir, "test")
        if dataset == "jetclass":
            x, p4 = _maybe_transpose_jetclass(x, p4)
        if p4 is None:
            p4 = ptetaphi_to_p4(x)
    return x, y, p4.astype(np.float32)


def truncate_arrays(x, p4, num_particles_truncate):
    if num_particles_truncate is None:
        return x, p4
    return x[:, :num_particles_truncate], p4[:, :num_particles_truncate]


def make_loader(x, p4, y, batch_size, shuffle):
    mask = np.abs(p4[:, :, 0]) > 1e-8
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(p4.astype(np.float32)),
        torch.from_numpy(mask.astype(np.bool_)),
        torch.from_numpy(labels_to_indices(y)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


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


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                param.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def make_full_edges(mask):
    batch_size, n_nodes = mask.shape
    device = mask.device
    valid = mask[:, :, None] & mask[:, None, :]
    diag = torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0)
    valid = valid & ~diag
    rows, cols = torch.where(valid.reshape(batch_size * n_nodes, n_nodes))
    batch_offsets = (rows // n_nodes) * n_nodes
    return [rows, batch_offsets + cols]


def lorentz_scalars_from_p4(p4):
    psq = p4.pow(2)
    mass2 = 2 * psq[..., 0] - psq.sum(dim=-1)
    scalars = torch.sign(mass2) * torch.log(torch.abs(mass2) + 1.0)
    return scalars.unsqueeze(-1)


def get_flops_profiler(model, batch, device, forward_fn):
    model.eval().to(device)
    batch = tuple(t.to(device) for t in batch)
    try:
        with torch.no_grad():
            _ = forward_fn(model, batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                _ = forward_fn(model, batch)
        total = 0
        for evt in prof.key_averages():
            evt_flops = getattr(evt, "flops", None)
            if isinstance(evt_flops, (int, float)):
                total += int(evt_flops)
        return total if total > 0 else None
    except Exception as exc:
        logging.info("FLOPs profiling failed: %s", exc)
        return None


def train_epoch(model, loader, criterion, optimizer, device, forward_fn, log_every_batches=0, epoch=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        batch = tuple(t.to(device) for t in batch)
        labels = batch[-1]
        optimizer.zero_grad()
        logits = forward_fn(model, batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        if log_every_batches and (
            batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == n_batches
        ):
            prefix = f"Epoch {epoch} " if epoch is not None else ""
            logging.info(
                "%strain progress: batch %d/%d loss=%.4f acc=%.4f",
                prefix,
                batch_idx,
                n_batches,
                total_loss / total,
                correct / total,
            )
    return total_loss / total, correct / total


def validate(model, loader, criterion, device, forward_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            labels = batch[-1]
            logits = forward_fn(model, batch)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device, forward_fn, num_classes):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            logits = forward_fn(model, batch)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch[-1].cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    y_onehot = one_hot_from_indices(labels, num_classes)
    acc = accuracy_score(labels, probs.argmax(axis=1))
    if num_classes == 2:
        auc_m = roc_auc_score(labels, probs[:, 1])
    else:
        auc_m = roc_auc_score(y_onehot, probs, average="macro", multi_class="ovo")
    return labels, y_onehot, probs, acc, auc_m


def plot_and_log_metrics(y_labels, y_onehot, probs, dataset, save_dir):
    labels = (
        ["q", "g", "W", "Z", "t"]
        if dataset == "hls4ml"
        else JETCLASS_LABELS
        if dataset == "jetclass"
        else ["background", "signal"]
    )
    plt.figure(figsize=(6, 6))
    if probs.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_labels, probs[:, 1])
        plt.plot(fpr, tpr, label=f"signal (AUC={auc(fpr, tpr):.2f})")
        if np.max(tpr) >= 0.8:
            fpr_t = np.interp(0.8, tpr, fpr)
            logging.info("1/FPR@0.8 signal: %.3f", 1.0 / fpr_t if fpr_t > 0 else np.nan)
    else:
        one_over_fpr = {}
        for i, lab in enumerate(labels[: probs.shape[1]]):
            fpr, tpr, _ = roc_curve(y_onehot[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f"{lab} (AUC={auc(fpr, tpr):.2f})")
            if np.max(tpr) >= 0.8:
                fpr_t = np.interp(0.8, tpr, fpr)
                one_over_fpr[lab] = 1.0 / fpr_t if fpr_t > 0 else np.nan
        for lab, val in one_over_fpr.items():
            logging.info("1/FPR@0.8 for %s: %.3f", lab, val)
        if one_over_fpr:
            logging.info("Avg 1/FPR@0.8: %.3f", np.nanmean(list(one_over_fpr.values())))

        if dataset == "hls4ml":
            rejections = []
            for i, lab in enumerate(labels[2: probs.shape[1]], start=2):
                mask_bg = (y_onehot[:, 0] == 1) | (y_onehot[:, 1] == 1) | (y_onehot[:, i] == 1)
                bin_y = (y_onehot[mask_bg, i] == 1).astype(int)
                bin_s = probs[mask_bg, i]
                fpr, tpr, _ = roc_curve(bin_y, bin_s)
                idx = np.argmin(np.abs(tpr - 0.8))
                rej = 1.0 / fpr[idx] if fpr[idx] > 0 else np.inf
                logging.info("Bg rejection@0.8 %s: %.3f", lab, rej)
                rejections.append(rej)
            logging.info("Avg bg rejection@0.8: %.3f", np.nanmean(rejections))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"))
    plt.close()


def time_inference(model, batch, batch_size, device, forward_fn):
    model.eval()
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        _ = forward_fn(model, batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = forward_fn(model, batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.mean(times) / batch_size * 1e9


def save_curves(save_dir, histories):
    for name, values in histories.items():
        np.save(os.path.join(save_dir, f"{name}.npy"), np.asarray(values))
    plt.figure()
    plt.plot(histories["train_loss"], label="Train Loss")
    plt.plot(histories["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
    plt.figure()
    plt.plot(histories["train_accuracy"], label="Train Acc")
    plt.plot(histories["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()
