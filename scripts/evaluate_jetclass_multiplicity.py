#!/usr/bin/env python3
"""Evaluate JetClass checkpoints in bins of the number of real particles.

The evaluator is intentionally log-only.  Predictions are reduced online into
accuracy/confusion-matrix counts and score histograms, so no result artifacts are
written to the shared PVC.  The score histograms provide a high-resolution
approximation to one-vs-rest AUC without retaining the 20M-event prediction set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LABELS = [
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

BIN_SCHEMES = {
    "fine": [
        ("1-20", 1, 20),
        ("21-40", 21, 40),
        ("41-60", 41, 60),
        ("61-80", 61, 80),
        ("81-100", 81, 100),
        ("101-120", 101, 120),
        ("121-140", 121, 140),
        ("141-149", 141, 149),
        ("150", 150, 150),
    ],
    "coarse": [
        ("1-50", 1, 50),
        ("51-100", 51, 100),
        ("101-149", 101, 149),
        ("150", 150, 150),
    ],
    "overall": [("1-150", 1, 150)],
}

ATOMIC_BINS = [
    ("1-20", 1, 20),
    ("21-40", 21, 40),
    ("41-50", 41, 50),
    ("51-60", 51, 60),
    ("61-80", 61, 80),
    ("81-100", 81, 100),
    ("101-120", 101, 120),
    ("121-140", 121, 140),
    ("141-149", 141, 149),
    ("150", 150, 150),
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    framework: str
    root: str
    outer_indices: tuple[int, int, int] | None
    batch_size: int


MODEL_SPECS = {
    "phatjet": ModelSpec(
        "phatjet",
        "tensorflow",
        "/j-jepa-vol/jetclass_2m_benchmark_runs/ptv3_config1",
        (6, 7, 8),
        512,
    ),
    "jedi_linear": ModelSpec(
        "jedi_linear",
        "tensorflow",
        "/j-jepa-vol/jetclass_2m_benchmark_runs/jedi_linear",
        (0, 1, 2),
        2048,
    ),
    "transformer": ModelSpec(
        "transformer",
        "tensorflow",
        "/j-jepa-vol/jetclass_2m_benchmark_runs/transformer",
        (3, 4, 5),
        1024,
    ),
    "salt": ModelSpec(
        "salt",
        "tensorflow",
        "/j-jepa-vol/jetclass_2m_benchmark_runs/salt_jetclass",
        (15, 16, 17),
        1024,
    ),
    "linformer": ModelSpec(
        "linformer",
        "tensorflow",
        "/j-jepa-vol/jetclass_2m_benchmark_runs/linformer_jetclass",
        (18, 19, 20),
        1024,
    ),
    "part_small": ModelSpec(
        "part_small",
        "pytorch",
        "/j-jepa-vol/1p3mFLOPs_runs/part_d10_h2_pe4_jetclass/150/kt",
        None,
        1024,
    ),
    "pointnet": ModelSpec(
        "pointnet",
        "tensorflow",
        "/j-jepa-vol/jetclass_2m_matched_flops_runs/pointnet",
        (0, 1, 2),
        2048,
    ),
    "pointtransformer_serialized": ModelSpec(
        "pointtransformer_serialized",
        "tensorflow",
        "/j-jepa-vol/jetclass_2m_matched_flops_runs/ptv3_serialized",
        (0, 1, 2),
        512,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--data-dir",
        default="/j-jepa-vol/linformer_data/JetClass/kinematics_2m",
    )
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--score-bins", type=int, default=4096)
    parser.add_argument("--checkpoint-root", default=None)
    return parser.parse_args()


def _inner_trial(path: Path) -> int:
    values = [int(value) for value in re.findall(r"(?:^|/)trial-(\d+)(?:/|$)", path.as_posix())]
    return values[-1] if values else -1


def resolve_checkpoints(spec: ModelSpec, root_override: str | None = None) -> list[Path]:
    root = Path(root_override or spec.root)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root is missing: {root}")

    selected = []
    if spec.framework == "pytorch":
        for trial in range(3):
            path = root / f"trial-{trial}" / "best_model.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing ParT checkpoint: {path}")
            selected.append(path)
        return selected

    assert spec.outer_indices is not None
    for trial, outer_index in enumerate(spec.outer_indices):
        outer_dirs = sorted(root.glob(f"*idx-{outer_index}-trial-{trial}"))
        # Some early serialized runs used trial-numbered outer indices without
        # the benchmark's global completion-index offset.
        if not outer_dirs and spec.name == "pointtransformer_serialized":
            outer_dirs = sorted(root.glob(f"*idx-{trial + 3}-trial-{trial}"))
        candidates = []
        for outer_dir in outer_dirs:
            candidates.extend(outer_dir.rglob("best.weights.h5"))
        candidates = [path for path in candidates if path.is_file()]
        if not candidates:
            raise FileNotFoundError(
                f"No best.weights.h5 for {spec.name} trial={trial}, "
                f"outer_index={outer_index}, root={root}"
            )
        candidates.sort(key=lambda path: (_inner_trial(path), path.stat().st_mtime, path.as_posix()))
        selected.append(candidates[-1])
    return selected


def sort_by_kt(features: np.ndarray) -> np.ndarray:
    key = features[:, :, 0] * np.sqrt(
        features[:, :, 1] ** 2 + features[:, :, 2] ** 2
    )
    order = np.argsort(key, axis=1)[:, ::-1]
    return np.take_along_axis(features, order[:, :, None], axis=1)


def make_tensorflow_model(model_name: str):
    import tensorflow as tf

    if model_name == "jedi_linear":
        from models.JEDI_Linear import build_jedi_linear_classifier

        return build_jedi_linear_classifier(
            num_particles=150,
            feature_dim=3,
            embedding_dim=20,
            num_blocks=2,
            token_hidden=None,
            channel_hidden=40,
            output_dim=10,
            aggregation="mean",
            dropout_rate=0.1,
            head_hidden_dims=[20],
        )
    if model_name == "transformer":
        from models.Transformer import build_standard_transformer_classifier

        return build_standard_transformer_classifier(
            150,
            3,
            d_model=8,
            d_ff=32,
            output_dim=10,
            num_heads=2,
            convolution=False,
            use_attention_mask=False,
        )
    if model_name in {"salt", "linformer"}:
        from models.Linformer import build_linformer_transformer_classifier

        is_salt = model_name == "salt"
        return build_linformer_transformer_classifier(
            150,
            3,
            d_model=20 if is_salt else 24,
            d_ff=40 if is_salt else 32,
            output_dim=10,
            num_heads=4 if is_salt else 2,
            proj_dim=4,
            use_cpe=True,
            cpe_k=8,
            grid_size=0.2,
            cluster_E=is_salt,
            cluster_F=is_salt,
            share_EF=False,
            convolution=is_salt,
            conv_filter_heights=[1, 3, 5],
            vertical_stride=1,
            shuffle_all=0,
            shuffle_234=0,
            shuffle_34=0,
            aggregation="max",
            use_layer_norm=False,
            ffn_activation="relu",
        )
    if model_name == "pointnet":
        from models.pointNet import build_pointnet_classifier

        return build_pointnet_classifier(
            num_particles=150,
            feature_dim=3,
            output_dim=10,
            dropout_rate=0.3,
            base_width=21,
        )
    if model_name == "pointtransformer_serialized":
        from models.PointTransformer_serialized import (
            build_ptv3_serialized_jet_classifier,
        )

        return build_ptv3_serialized_jet_classifier(
            num_particles=150,
            output_dim=10,
            enc_dims=[16, 32],
            enc_layers=[1, 1],
            enc_heads=[4, 4],
            enc_patch_sizes=[25, 25],
            enc_strides=[2],
            cpe_k=8,
            grid_size=0.2,
            use_rpe=False,
            use_pool=True,
            dropout=0.0,
            aggregation="max",
            serialize_by="kt",
            assume_serialized_input=True,
        )
    if model_name == "phatjet":
        from models.PointTransformerV3TF import build_ptv3_jet_classifier

        return build_ptv3_jet_classifier(
            num_particles=150,
            output_dim=10,
            enc_dims=[16],
            enc_layers=[1],
            enc_heads=[4],
            enc_patch_sizes=[10],
            enc_strides=[2],
            cpe_k=8,
            grid_size=0.05,
            cpe_coord_mode="raw",
            use_rpe=False,
            use_cpe=True,
            use_pool=False,
            dropout=0.0,
            aggregation="max",
            ffn_activation="gelu",
            use_patch_messages=True,
            patch_tokenizer_mode="learned_pool",
            message_proj=True,
            message_gated=True,
            use_flash_attention=False,
        )
    raise ValueError(f"Unknown TensorFlow model: {model_name}")


def make_part_model():
    import torch
    from models.parT import ParticleTransformer

    block_params = {
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "activation_dropout": 0.0,
        "scale_fc": False,
        "scale_attn": False,
        "scale_heads": False,
        "scale_resids": False,
    }
    return ParticleTransformer(
        input_dim=3,
        num_classes=10,
        pair_input_dim=4,
        pair_extra_dim=0,
        remove_self_pair=True,
        use_pre_activation_pair=True,
        embed_dims=[10],
        pair_embed_dims=[4],
        num_heads=2,
        num_layers=1,
        num_cls_layers=1,
        block_params=block_params,
        cls_block_params=block_params,
        fc_params=[],
        activation="gelu",
        trim=False,
        for_inference=False,
    )


def load_models(spec: ModelSpec, checkpoints: list[Path]):
    if spec.framework == "tensorflow":
        import tensorflow as tf

        models = []
        for checkpoint in checkpoints:
            model = make_tensorflow_model(spec.name)
            try:
                model.load_weights(checkpoint)
            except Exception:
                if spec.name != "pointnet":
                    raise
                model = tf.keras.models.load_model(checkpoint, compile=False)
            models.append(model)
        return models

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for checkpoint in checkpoints:
        model = make_part_model().to(device)
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def prepare_part_inputs(features: np.ndarray):
    pt = features[:, :, 0]
    eta = features[:, :, 1]
    phi = features[:, :, 2]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(np.clip(eta, -10, 10))
    energy = pt * np.cosh(np.clip(eta, -10, 10))
    vectors = np.stack([px, py, pz, energy], axis=1).astype(np.float32)
    x = features.transpose(0, 2, 1).astype(np.float32, copy=False)
    mask = (np.abs(pt) > 1e-6).astype(np.float32)[:, None, :]
    return x, vectors, mask


def predict_models(spec: ModelSpec, models, features: np.ndarray, batch_size: int):
    if spec.framework == "tensorflow":
        return [
            np.asarray(model.predict(features, batch_size=batch_size, verbose=0))
            for model in models
        ]

    import torch

    device = next(models[0].parameters()).device
    x, vectors, mask = prepare_part_inputs(features)
    outputs = [[] for _ in models]
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch_x = torch.from_numpy(x[start:end]).to(device)
            batch_v = torch.from_numpy(vectors[start:end]).to(device)
            batch_mask = torch.from_numpy(mask[start:end]).to(device)
            for trial, model in enumerate(models):
                logits = model(batch_x, v=batch_v, mask=batch_mask)
                outputs[trial].append(torch.softmax(logits, dim=1).cpu().numpy())
    return [np.concatenate(parts, axis=0) for parts in outputs]


class BinnedMetrics:
    def __init__(self, score_bins: int):
        self.score_bins = score_bins
        shape = (len(ATOMIC_BINS), len(LABELS), score_bins)
        self.data = {
            "events": np.zeros(len(ATOMIC_BINS), dtype=np.int64),
            "correct": np.zeros(len(ATOMIC_BINS), dtype=np.int64),
            "class_counts": np.zeros(
                (len(ATOMIC_BINS), len(LABELS)), dtype=np.int64
            ),
            "confusion": np.zeros(
                (len(ATOMIC_BINS), len(LABELS), len(LABELS)), dtype=np.int64
            ),
            "positive_scores": np.zeros(shape, dtype=np.int64),
            "negative_scores": np.zeros(shape, dtype=np.int64),
        }

    @staticmethod
    def _bin_ids(counts: np.ndarray) -> np.ndarray:
        result = np.full(len(counts), -1, dtype=np.int16)
        for index, (_, low, high) in enumerate(ATOMIC_BINS):
            result[(counts >= low) & (counts <= high)] = index
        return result

    def update(
        self,
        particle_counts: np.ndarray,
        truth: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        truth_index = np.argmax(truth, axis=1)
        pred_index = np.argmax(predictions, axis=1)
        score_index = np.minimum(
            (np.clip(predictions, 0.0, 1.0) * self.score_bins).astype(np.int64),
            self.score_bins - 1,
        )
        ids = self._bin_ids(particle_counts)
        if np.any(ids < 0):
            raise ValueError("Particle counts were not covered by the atomic bins")
        self.data["events"] += np.bincount(ids, minlength=len(ATOMIC_BINS))
        correct_ids = ids[truth_index == pred_index]
        self.data["correct"] += np.bincount(
            correct_ids, minlength=len(ATOMIC_BINS)
        )
        class_key = ids * len(LABELS) + truth_index
        self.data["class_counts"] += np.bincount(
            class_key, minlength=len(ATOMIC_BINS) * len(LABELS)
        ).reshape(len(ATOMIC_BINS), len(LABELS))
        confusion_key = (
            ids * len(LABELS) * len(LABELS)
            + truth_index * len(LABELS)
            + pred_index
        )
        self.data["confusion"] += np.bincount(
            confusion_key,
            minlength=len(ATOMIC_BINS) * len(LABELS) * len(LABELS),
        ).reshape(len(ATOMIC_BINS), len(LABELS), len(LABELS))
        for class_index in range(len(LABELS)):
            score_key = ids * self.score_bins + score_index[:, class_index]
            positive = truth_index == class_index
            self.data["positive_scores"][:, class_index] += np.bincount(
                score_key[positive],
                minlength=len(ATOMIC_BINS) * self.score_bins,
            ).reshape(len(ATOMIC_BINS), self.score_bins)
            self.data["negative_scores"][:, class_index] += np.bincount(
                score_key[~positive],
                minlength=len(ATOMIC_BINS) * self.score_bins,
            ).reshape(len(ATOMIC_BINS), self.score_bins)

    @staticmethod
    def _histogram_auc(positive: np.ndarray, negative: np.ndarray) -> float:
        n_positive = int(positive.sum())
        n_negative = int(negative.sum())
        if n_positive == 0 or n_negative == 0:
            return math.nan
        negative_below = np.cumsum(negative, dtype=np.int64) - negative
        wins = np.sum(
            positive.astype(np.float64)
            * (negative_below.astype(np.float64) + 0.5 * negative)
        )
        return float(wins / (n_positive * n_negative))

    def result(self) -> dict:
        result = {}
        for scheme, bins in BIN_SCHEMES.items():
            rows = []
            for bin_index, (label, low, high) in enumerate(bins):
                atoms = [
                    index
                    for index, (_, atom_low, atom_high) in enumerate(ATOMIC_BINS)
                    if atom_low >= low and atom_high <= high
                ]
                if not atoms:
                    raise RuntimeError(f"No atomic bins found for {label}")
                events = int(self.data["events"][atoms].sum())
                positive_scores = self.data["positive_scores"][atoms].sum(axis=0)
                negative_scores = self.data["negative_scores"][atoms].sum(axis=0)
                class_counts = self.data["class_counts"][atoms].sum(axis=0)
                confusion = self.data["confusion"][atoms].sum(axis=0)
                aucs = [
                    self._histogram_auc(
                        positive_scores[class_index],
                        negative_scores[class_index],
                    )
                    for class_index in range(len(LABELS))
                ]
                valid_aucs = [value for value in aucs if math.isfinite(value)]
                rows.append(
                    {
                        "bin": label,
                        "low": low,
                        "high": high,
                        "n_events": events,
                        "accuracy": (
                            float(self.data["correct"][atoms].sum() / events)
                            if events
                            else math.nan
                        ),
                        "macro_ovr_auc": (
                            float(np.mean(valid_aucs)) if valid_aucs else math.nan
                        ),
                        "per_class_ovr_auc": {
                            name: value for name, value in zip(LABELS, aucs)
                        },
                        "class_counts": {
                            name: int(value)
                            for name, value in zip(LABELS, class_counts)
                        },
                        "confusion_matrix": confusion.tolist(),
                    }
                )
            result[scheme] = rows
        return result


def aggregate_trials(trial_results: list[dict]) -> dict:
    def mean_and_std(values):
        finite = np.asarray(
            [value for value in values if value is not None and math.isfinite(value)],
            dtype=np.float64,
        )
        if len(finite) == 0:
            return None, None
        if len(finite) == 1:
            return float(finite[0]), 0.0
        return float(finite.mean()), float(finite.std(ddof=1))

    aggregate = {}
    for scheme, bins in BIN_SCHEMES.items():
        rows = []
        for bin_index, (label, low, high) in enumerate(bins):
            accuracies = [
                trial_results[trial][scheme][bin_index]["accuracy"]
                for trial in range(len(trial_results))
            ]
            aucs = [
                trial_results[trial][scheme][bin_index]["macro_ovr_auc"]
                for trial in range(len(trial_results))
            ]
            accuracy_mean, accuracy_std = mean_and_std(accuracies)
            auc_mean, auc_std = mean_and_std(aucs)
            rows.append(
                {
                    "bin": label,
                    "low": low,
                    "high": high,
                    "n_events": trial_results[0][scheme][bin_index]["n_events"],
                    "accuracy_mean": accuracy_mean,
                    "accuracy_std": accuracy_std,
                    "macro_ovr_auc_mean": auc_mean,
                    "macro_ovr_auc_std": auc_std,
                }
            )
        aggregate[scheme] = rows
    return aggregate


def print_markdown(model_name: str, aggregate: dict) -> None:
    def format_metric(mean, std):
        if mean is None or std is None:
            return "NA"
        return f"{mean:.6f} +/- {std:.6f}"

    for scheme in ("fine", "coarse", "overall"):
        print(f"\n===== {model_name}: {scheme} multiplicity bins =====")
        print("| bin | N jets | accuracy | macro OvR AUC |")
        print("|---:|---:|---:|---:|")
        for row in aggregate[scheme]:
            print(
                f"| {row['bin']} | {row['n_events']} | "
                f"{format_metric(row['accuracy_mean'], row['accuracy_std'])} | "
                f"{format_metric(row['macro_ovr_auc_mean'], row['macro_ovr_auc_std'])} |"
            )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.score_bins < 2:
        raise ValueError("--chunk-size must be positive and --score-bins must be >= 2")

    spec = MODEL_SPECS[args.model]
    batch_size = args.batch_size or spec.batch_size
    checkpoints = resolve_checkpoints(spec, args.checkpoint_root)
    print(f"model={spec.name} framework={spec.framework}")
    for trial, checkpoint in enumerate(checkpoints):
        print(f"trial={trial} checkpoint={checkpoint}")

    if spec.framework == "tensorflow":
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

    models = load_models(spec, checkpoints)
    features_path = Path(args.data_dir) / "test" / "features.npy"
    labels_path = Path(args.data_dir) / "test" / "labels.npy"
    features = np.load(features_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    if features.ndim != 3 or features.shape[1:] != (3, 150):
        raise ValueError(f"Unexpected JetClass feature shape: {features.shape}")
    if labels.shape != (features.shape[0], len(LABELS)):
        raise ValueError(f"Unexpected JetClass label shape: {labels.shape}")
    print(
        f"test_events={len(features)} chunk_size={args.chunk_size} "
        f"batch_size={batch_size} score_bins={args.score_bins}"
    )

    accumulators = [BinnedMetrics(args.score_bins) for _ in checkpoints]
    started = time.monotonic()
    for start in range(0, len(features), args.chunk_size):
        end = min(start + args.chunk_size, len(features))
        raw = np.asarray(features[start:end])
        particle_counts = np.count_nonzero(np.abs(raw[:, 0, :]) > 1e-6, axis=1)
        if np.any((particle_counts < 1) | (particle_counts > 150)):
            raise ValueError(
                f"Out-of-range particle count in events [{start}, {end})"
            )
        x_chunk = sort_by_kt(raw.transpose(0, 2, 1)).astype(
            np.float32, copy=False
        )
        y_chunk = np.asarray(labels[start:end])
        predictions = predict_models(spec, models, x_chunk, batch_size)
        for accumulator, prediction in zip(accumulators, predictions):
            accumulator.update(particle_counts, y_chunk, prediction)
        elapsed = time.monotonic() - started
        print(
            f"progress model={spec.name} events={end}/{len(features)} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )

    trial_results = [accumulator.result() for accumulator in accumulators]
    aggregate = aggregate_trials(trial_results)
    print_markdown(spec.name, aggregate)
    payload = {
        "schema_version": 1,
        "dataset": "jetclass_2m_train_20m_test",
        "model": spec.name,
        "sort_by": "kt",
        "particle_count_definition": "count(abs(pt) > 1e-6) before sorting",
        "auc_method": "macro one-vs-rest from uniformly quantized score histograms",
        "score_bins": args.score_bins,
        "checkpoints": [str(path) for path in checkpoints],
        "trials": trial_results,
        "aggregate": aggregate,
    }
    print("MULTIPLICITY_RESULT_JSON_BEGIN")
    print(json.dumps(json_safe(payload), allow_nan=False, separators=(",", ":")))
    print("MULTIPLICITY_RESULT_JSON_END", flush=True)


if __name__ == "__main__":
    main()
