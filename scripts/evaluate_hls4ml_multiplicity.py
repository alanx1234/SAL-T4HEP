#!/usr/bin/env python3
"""Evaluate HLS4ML checkpoints in bins of the number of real particles.

This evaluator is log-only: it reads checkpoints and the validation/test arrays
from the shared PVC but writes no result files there. HLS4ML is small enough to
retain one trial's predictions in memory, allowing exact sklearn metrics in
every multiplicity bin.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_jetclass_multiplicity import (  # noqa: E402
    BIN_SCHEMES,
    aggregate_trials,
    json_safe,
    print_markdown,
    sort_by_kt,
)


LABELS = ["q", "g", "W", "Z", "t"]
SIGNAL_INDICES = (2, 3, 4)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    framework: str
    checkpoints: tuple[str, str, str]
    batch_size: int


MODEL_SPECS = {
    "phatjet": ModelSpec(
        "phatjet",
        "tensorflow",
        (
            "/j-jepa-vol/ptv3_sweeps/sweep_v1/pool0_aggmean_cpe1_ffngelu_tokmean_proj1_gate0/trial-0/150/kt/trial-0/model.weights.h5",
            "/j-jepa-vol/ptv3_sweeps/sweep_v1/pool0_aggmean_cpe1_ffngelu_tokmean_proj1_gate0/trial-1/150/kt/trial-0/model.weights.h5",
            "/j-jepa-vol/ptv3_sweeps/sweep_v1/pool0_aggmean_cpe1_ffngelu_tokmean_proj1_gate0/trial-2/150/kt/trial-0/model.weights.h5",
        ),
        2048,
    ),
    "salt": ModelSpec(
        "salt",
        "tensorflow",
        (
            "/j-jepa-vol/linformer_runs/salt_sweeps/linformer-salt-nocpe-1p3m/hls4ml/salt_nocpe/sort_kt/dmodel_20_dff_40_proj_4/trial_0/150/kt/trial-0/model.weights.h5",
            "/j-jepa-vol/linformer_runs/salt_sweeps/linformer-salt-nocpe-1p3m/hls4ml/salt_nocpe/sort_kt/dmodel_20_dff_40_proj_4/trial_1/150/kt/trial-0/model.weights.h5",
            "/j-jepa-vol/linformer_runs/salt_sweeps/linformer-salt-nocpe-1p3m/hls4ml/salt_nocpe/sort_kt/dmodel_20_dff_40_proj_4/trial_2/150/kt/trial-0/model.weights.h5",
        ),
        2048,
    ),
    "linformer": ModelSpec(
        "linformer",
        "tensorflow",
        (
            "/j-jepa-vol/linformer_runs/linformer_sweep/linformer-vanilla-nocpe-1p3m/hls4ml/vanilla_nocpe/sort_kt/dmodel_24_dff_32_proj_6/trial_0/150/kt/trial-0/model.weights.h5",
            "/j-jepa-vol/linformer_runs/linformer_sweep/linformer-vanilla-nocpe-1p3m/hls4ml/vanilla_nocpe/sort_kt/dmodel_24_dff_32_proj_6/trial_1/150/kt/trial-0/model.weights.h5",
            "/j-jepa-vol/linformer_runs/linformer_sweep/linformer-vanilla-nocpe-1p3m/hls4ml/vanilla_nocpe/sort_kt/dmodel_24_dff_32_proj_6/trial_2/150/kt/trial-0/model.weights.h5",
        ),
        2048,
    ),
    "pointnet": ModelSpec(
        "pointnet",
        "tensorflow",
        (
            "/j-jepa-vol/1p3mFLOPs_runs/pointnet/trial-0/150/kt/trial-2/model.weights.h5",
            "/j-jepa-vol/1p3mFLOPs_runs/pointnet/trial-1/150/kt/trial-3/model.weights.h5",
            "/j-jepa-vol/1p3mFLOPs_runs/pointnet/trial-2/150/kt/trial-3/model.weights.h5",
        ),
        4096,
    ),
    "part_full": ModelSpec(
        "part_full",
        "pytorch",
        (
            "/j-jepa-vol/1p3mFLOPs_runs/full_part_hls4ml/full-part-hls4ml-idx-0-trial-0/150/kt/trial-0/best_model.pt",
            "/j-jepa-vol/1p3mFLOPs_runs/full_part_hls4ml/full-part-hls4ml-idx-1-trial-1/150/kt/trial-1/best_model.pt",
            "/j-jepa-vol/1p3mFLOPs_runs/full_part_hls4ml/full-part-hls4ml-idx-2-trial-2/150/kt/trial-2/best_model.pt",
        ),
        128,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--data-dir",
        default="/j-jepa-vol/l1-jet-id/data/jetid/processed",
    )
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load all three checkpoints and run a one-event forward pass",
    )
    return parser.parse_args()


def resolve_checkpoints(spec: ModelSpec) -> list[Path]:
    checkpoints = [Path(value) for value in spec.checkpoints]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    return checkpoints


def make_tensorflow_model(model_name: str):
    if model_name == "phatjet":
        module_name = os.environ.get(
            "PHAT_MODEL_MODULE", "models.PointTransformerV3TF"
        )
        build_ptv3_jet_classifier = importlib.import_module(
            module_name
        ).build_ptv3_jet_classifier

        model = build_ptv3_jet_classifier(
            num_particles=150,
            output_dim=5,
            enc_dims=[16],
            enc_layers=[1],
            enc_heads=[4],
            enc_patch_sizes=[25],
            enc_strides=[2],
            cpe_k=8,
            grid_size=0.2,
            use_rpe=False,
            use_cpe=True,
            use_pool=False,
            dropout=0.0,
            aggregation="mean",
            ffn_activation="gelu",
            use_flash_attention=True,
            use_patch_messages=True,
            patch_tokenizer_mode="mean",
            message_proj=True,
            message_gated=False,
        )
        if model.count_params() != 6405:
            raise RuntimeError(
                f"Unexpected PHAT-JeT parameter count: {model.count_params()}"
            )
        return model
    if model_name in {"salt", "linformer"}:
        from models.Linformer import build_linformer_transformer_classifier

        is_salt = model_name == "salt"
        return build_linformer_transformer_classifier(
            150,
            3,
            d_model=20 if is_salt else 24,
            d_ff=40 if is_salt else 32,
            output_dim=5,
            num_heads=4,
            proj_dim=4 if is_salt else 6,
            use_cpe=False,
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
            output_dim=5,
            dropout_rate=0.3,
            base_width=21,
        )
    raise ValueError(f"Unknown TensorFlow model: {model_name}")


def make_part_model():
    from models.parT import ParticleTransformer

    block_params = {
        "dropout": 0.1,
        "attn_dropout": 0.1,
        "activation_dropout": 0.1,
        "scale_fc": False,
        "scale_attn": False,
        "scale_heads": False,
        "scale_resids": False,
    }
    return ParticleTransformer(
        input_dim=3,
        num_classes=5,
        pair_input_dim=4,
        pair_extra_dim=0,
        remove_self_pair=True,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=block_params,
        cls_block_params=block_params,
        fc_params=[],
        activation="gelu",
        trim=False,
        for_inference=False,
    )


def _load_tensorflow_checkpoint(model_name: str, checkpoint: Path, trial: int):
    attempts = [checkpoint]
    final_checkpoint = checkpoint.with_name("model.weights.h5")
    if final_checkpoint != checkpoint and final_checkpoint.is_file():
        attempts.append(final_checkpoint)
    errors = []
    for candidate in attempts:
        for legacy_alias in (False, True):
            model = make_tensorflow_model(model_name)
            alias = None
            load_path = candidate
            if legacy_alias:
                alias = Path("/tmp") / f"hls-{model_name}-{trial}-{candidate.stem}.h5"
                if alias.exists() or alias.is_symlink():
                    alias.unlink()
                alias.symlink_to(candidate)
                load_path = alias
            try:
                model.load_weights(load_path)
                return model, candidate, "legacy-h5-alias" if legacy_alias else "direct"
            except Exception as error:
                errors.append(
                    f"{candidate} ({'legacy alias' if legacy_alias else 'direct'}): "
                    f"{type(error).__name__}: {error}"
                )
            finally:
                if alias is not None and (alias.exists() or alias.is_symlink()):
                    alias.unlink()
    raise RuntimeError(
        f"Could not load {model_name} trial {trial}. Attempts:\n"
        + "\n".join(errors)
    )


def prepare_part_inputs(features: np.ndarray):
    pt = features[:, :, 0]
    eta = features[:, :, 1]
    phi = features[:, :, 2]
    vectors = np.stack(
        [
            pt * np.cos(phi),
            pt * np.sin(phi),
            pt * np.sinh(np.clip(eta, -10, 10)),
            pt * np.cosh(np.clip(eta, -10, 10)),
        ],
        axis=1,
    ).astype(np.float32)
    x = features.transpose(0, 2, 1).astype(np.float32, copy=False)
    mask = (np.abs(pt) > 1e-6).astype(np.float32)[:, None, :]
    return x, vectors, mask


def predict_tensorflow(model, features: np.ndarray, batch_size: int) -> np.ndarray:
    return np.asarray(model.predict(features, batch_size=batch_size, verbose=0))


def predict_part(model, features: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    x, vectors, mask = prepare_part_inputs(features)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            logits = model(
                torch.from_numpy(x[start:end]).to(device),
                v=torch.from_numpy(vectors[start:end]).to(device),
                mask=torch.from_numpy(mask[start:end]).to(device),
            )
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _binary_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(truth)) < 2:
        return math.nan
    return float(roc_auc_score(truth, scores))


def _background_rejection(
    truth_index: np.ndarray,
    predictions: np.ndarray,
    signal_index: int,
) -> float:
    # The benchmark's HLS4ML rejection uses q+g as background for W/Z/t.
    selected = np.isin(truth_index, (0, 1, signal_index))
    binary_truth = (truth_index[selected] == signal_index).astype(np.int8)
    if len(np.unique(binary_truth)) < 2:
        return math.nan
    fpr, tpr, _ = roc_curve(binary_truth, predictions[selected, signal_index])
    nearest = int(np.argmin(np.abs(tpr - 0.8)))
    return float(1.0 / fpr[nearest]) if fpr[nearest] > 0 else math.inf


def calculate_metrics(
    particle_counts: np.ndarray,
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    truth_index = np.argmax(truth, axis=1)
    finite = np.all(np.isfinite(predictions), axis=1)
    safe_predictions = np.nan_to_num(
        predictions, nan=0.0, posinf=1.0, neginf=0.0
    )
    prediction_index = np.argmax(safe_predictions, axis=1)
    result = {}
    for scheme, bins in BIN_SCHEMES.items():
        rows = []
        for label, low, high in bins:
            selected = (particle_counts >= low) & (particle_counts <= high)
            valid = selected & finite
            n_events = int(selected.sum())
            n_valid = int(valid.sum())
            class_counts = np.bincount(
                truth_index[selected], minlength=len(LABELS)
            )
            aucs = [
                _binary_auc(
                    (truth_index[valid] == class_index).astype(np.int8),
                    safe_predictions[valid, class_index],
                )
                for class_index in range(len(LABELS))
            ]
            valid_aucs = [value for value in aucs if math.isfinite(value)]
            rejections = [
                _background_rejection(
                    truth_index[valid],
                    safe_predictions[valid],
                    signal_index,
                )
                for signal_index in SIGNAL_INDICES
            ]
            finite_rejections = [
                value for value in rejections if math.isfinite(value)
            ]
            rows.append(
                {
                    "bin": label,
                    "low": low,
                    "high": high,
                    "n_events": n_events,
                    "n_valid_predictions": n_valid,
                    "n_nonfinite_predictions": n_events - n_valid,
                    "accuracy": (
                        float(
                            np.sum(
                                valid
                                & (truth_index == prediction_index)
                            )
                            / n_events
                        )
                        if n_events
                        else math.nan
                    ),
                    "roc_auc": (
                        float(np.mean(valid_aucs)) if valid_aucs else math.nan
                    ),
                    "per_class_auc": dict(zip(LABELS, aucs)),
                    "background_rejection_at_0p8": dict(
                        zip((LABELS[index] for index in SIGNAL_INDICES), rejections)
                    ),
                    "avg_background_rejection_at_0p8": (
                        float(np.mean(finite_rejections))
                        if finite_rejections
                        else math.nan
                    ),
                    "class_counts": dict(zip(LABELS, map(int, class_counts))),
                }
            )
        result[scheme] = rows
    return result


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    spec = MODEL_SPECS[args.model]
    batch_size = args.batch_size or spec.batch_size
    checkpoints = resolve_checkpoints(spec)
    print(f"model={spec.name} framework={spec.framework}", flush=True)
    if spec.name == "phatjet":
        print(
            "phat_model_module="
            + os.environ.get(
                "PHAT_MODEL_MODULE", "models.PointTransformerV3TF"
            ),
            flush=True,
        )
    for trial, checkpoint in enumerate(checkpoints):
        print(f"trial={trial} checkpoint={checkpoint}", flush=True)

    if spec.framework == "tensorflow":
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        import torch

        if not torch.cuda.is_available() and not args.load_only:
            raise RuntimeError("CUDA is required for full ParT evaluation")

    data_dir = Path(args.data_dir)
    features = np.load(
        data_dir / "x_val_robust_150const_ptetaphi.npy", mmap_mode="r"
    )
    truth = np.load(
        data_dir / "y_val_robust_150const_ptetaphi.npy", mmap_mode="r"
    )
    if features.ndim != 3 or features.shape[1:] != (150, 3):
        raise ValueError(f"Unexpected HLS4ML feature shape: {features.shape}")
    if truth.shape != (len(features), len(LABELS)):
        raise ValueError(f"Unexpected HLS4ML label shape: {truth.shape}")
    particle_counts = np.count_nonzero(
        np.abs(np.asarray(features[:, :, 0])) > 1e-6, axis=1
    )
    if np.any((particle_counts < 1) | (particle_counts > 150)):
        raise ValueError("Found particle multiplicity outside [1, 150]")
    print(
        f"test_events={len(features)} chunk_size={args.chunk_size} "
        f"batch_size={batch_size}",
        flush=True,
    )

    trial_results = []
    started = time.monotonic()
    for trial, checkpoint in enumerate(checkpoints):
        if spec.framework == "tensorflow":
            model, loaded_checkpoint, method = _load_tensorflow_checkpoint(
                spec.name, checkpoint, trial
            )
        else:
            import torch

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model = make_part_model().to(device)
            try:
                state = torch.load(
                    checkpoint, map_location=device, weights_only=True
                )
            except TypeError:
                state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state)
            model.eval()
            loaded_checkpoint = checkpoint
            method = "torch-state-dict"
        print(
            f"trial={trial} loaded_checkpoint={loaded_checkpoint} "
            f"load_method={method}",
            flush=True,
        )

        if args.load_only:
            sample = sort_by_kt(np.asarray(features[:1])).astype(
                np.float32, copy=False
            )
            output = (
                predict_tensorflow(model, sample, 1)
                if spec.framework == "tensorflow"
                else predict_part(model, sample, 1)
            )
            if output.shape != (1, len(LABELS)) or not np.all(np.isfinite(output)):
                raise RuntimeError(
                    f"Invalid load-only output for trial {trial}: {output}"
                )
        else:
            predictions = np.empty(
                (len(features), len(LABELS)), dtype=np.float32
            )
            for start in range(0, len(features), args.chunk_size):
                end = min(start + args.chunk_size, len(features))
                x_chunk = sort_by_kt(np.asarray(features[start:end])).astype(
                    np.float32, copy=False
                )
                predictions[start:end] = (
                    predict_tensorflow(model, x_chunk, batch_size)
                    if spec.framework == "tensorflow"
                    else predict_part(model, x_chunk, batch_size)
                )
                print(
                    f"progress model={spec.name} trial={trial} "
                    f"events={end}/{len(features)} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
            trial_result = calculate_metrics(
                particle_counts, np.asarray(truth), predictions
            )
            trial_results.append(trial_result)
            print(
                "HLS_MULTIPLICITY_TRIAL_JSON "
                + json.dumps(
                    json_safe(
                        {
                            "model": spec.name,
                            "trial": trial,
                            "checkpoint": str(loaded_checkpoint),
                            "metrics": trial_result,
                        }
                    ),
                    separators=(",", ":"),
                ),
                flush=True,
            )

        del model
        if spec.framework == "tensorflow":
            import tensorflow as tf

            tf.keras.backend.clear_session()
        else:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.load_only:
        print(f"LOAD_ONLY_OK model={spec.name}", flush=True)
        return
    aggregate = aggregate_trials(trial_results)
    print_markdown(spec.name, aggregate)
    print(
        "HLS_MULTIPLICITY_AGGREGATE_JSON "
        + json.dumps(
            json_safe(
                {
                    "model": spec.name,
                    "labels": LABELS,
                    "background_rejection_signals": ["W", "Z", "t"],
                    "background_classes": ["q", "g"],
                    "trials": len(trial_results),
                    "metrics": aggregate,
                }
            ),
            separators=(",", ":"),
        ),
        flush=True,
    )
    print(
        f"EVALUATION_COMPLETE model={spec.name} "
        f"events_per_trial={len(features)} trials={len(trial_results)} "
        f"elapsed_seconds={time.monotonic() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
