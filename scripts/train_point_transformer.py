#!/usr/bin/env python
"""
Train a PointTransformerV3-like TensorFlow model on jet datasets (hls4ml, top, QG, or jetclass),
profiling performance and generating ROC curves.
"""
import os
import sys

# ─── make the parent directory (project root) importable ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
		sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import logging
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# import model builders
from models.PointTransformerV3TF import build_ptv3_jet_classifier, build_jedi_ptv3_hybrid
from models.PointTransformer_serialized import build_ptv3_serialized_jet_classifier
from scripts.keras_chunked_testing import predict_in_chunks


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


def _format_log_value(value):
		if value is None:
				return "NA"
		try:
				return f"{float(value):.6f}"
		except Exception:
				return str(value)


class LoggingProgressCallback(Callback):
		def __init__(self, log_every_batches=100):
				super().__init__()
				self.log_every_batches = int(log_every_batches or 0)
				self.epoch_start_time = None
				self.batch_start_time = None

		def on_epoch_begin(self, epoch, logs=None):
				self.epoch_start_time = time.perf_counter()
				logging.info("Epoch %d begin", epoch + 1)

		def on_train_batch_begin(self, batch, logs=None):
				if self.log_every_batches:
						self.batch_start_time = time.perf_counter()

		def on_train_batch_end(self, batch, logs=None):
				if not self.log_every_batches:
						return
				if (batch + 1) == 1 or (batch + 1) % self.log_every_batches == 0:
						logs = logs or {}
						elapsed = 0.0 if self.batch_start_time is None else time.perf_counter() - self.batch_start_time
						logging.info(
								"Train batch %d end: loss=%s accuracy=%s batch_time=%.2fs",
								batch + 1,
								_format_log_value(logs.get("loss")),
								_format_log_value(logs.get("accuracy")),
								elapsed,
						)

		def on_epoch_end(self, epoch, logs=None):
				logs = logs or {}
				elapsed = 0.0 if self.epoch_start_time is None else time.perf_counter() - self.epoch_start_time
				logging.info(
						"Epoch %d end: loss=%s accuracy=%s val_loss=%s val_accuracy=%s epoch_time=%.2fs",
						epoch + 1,
						_format_log_value(logs.get("loss")),
						_format_log_value(logs.get("accuracy")),
						_format_log_value(logs.get("val_loss")),
						_format_log_value(logs.get("val_accuracy")),
						elapsed,
				)


# ---------------------------
# FLOPs computation (no mask)
# ---------------------------
def get_flops(model, input_shape):
		from tensorflow.python.framework.convert_to_constants import (
				convert_variables_to_constants_v2_as_graph,
		)

		spec_x = tf.TensorSpec(input_shape, tf.float32)

		@tf.function
		def model_fn(x):
				return model(x)

		concrete = model_fn.get_concrete_function(spec_x)
		frozen, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
		with tf.Graph().as_default() as g:
				tf.compat.v1.import_graph_def(graph_def, name="")
				run_meta = tf.compat.v1.RunMetadata()
				opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
				prof = tf.compat.v1.profiler.profile(
						graph=g, run_meta=run_meta, cmd="op", options=opts
				)
				return prof.total_float_ops


# ---------------------------
# GPU memory profiling (no mask)
# ---------------------------
def profile_gpu_memory_during_inference(model, input_data):
		tf.config.experimental.reset_memory_stats("GPU:0")

		@tf.function
		def infer(x):
				return model(x, training=False)

		_ = infer(input_data[:1])
		_ = infer(input_data)

		mem = tf.config.experimental.get_memory_info("GPU:0")
		return mem["current"] / (1024**2), mem["peak"] / (1024**2)
	
def _morton_interleave_bits_np(axes, bits: int = None) -> np.ndarray:
    """Z-order code interleaving the bits of `len(axes)` quantized coordinate arrays."""
    d = len(axes)
    if bits is None:
        # Keep the whole code inside 64 bits regardless of dimensionality.
        bits = 63 // d
    axes = [a.astype(np.uint64) for a in axes]
    z = np.zeros_like(axes[0], dtype=np.uint64)
    one = np.uint64(1)
    for i in range(bits):
        for k, a in enumerate(axes):
            z |= ((a >> np.uint64(i)) & one) << np.uint64(d * i + k)
    return z


def _morton_sort_indices_np(coords, grid_size: float, bits: int = None) -> np.ndarray:
    """
    coords: list of [B, N] coordinate arrays (2 for jets, 3 for generic point clouds).
    Returns ascending argsort indices along the point axis.
    """
    grids = []
    for a in coords:
        a_min = np.min(a, axis=1, keepdims=True)
        g = np.floor((a - a_min) / grid_size).astype(np.int64)
        grids.append(np.clip(g, 0, None).astype(np.uint64))

    morton = _morton_interleave_bits_np(grids, bits=bits)  # [B,N]
    return np.argsort(morton, axis=1)  # ascending

# ---------------------------
# Sorting helper
# ---------------------------
def apply_sorting(x, sort_by, grid_size=0.05, coord_dim=2, weighted=True):
		"""
		Sort points within each cloud.
		
		weighted=True  (jets): channels are [pt, eta, phi]; pt-based orderings available.
		weighted=False (generic): channels are the coord_dim spatial axes; only "morton"
		and "random" are meaningful, since there is no per-point weight to rank by.
		"""
		coord_start = 1 if weighted else 0
		if not weighted and sort_by in ("pt", "kt", "eta", "phi", "delta_R"):
			raise ValueError(f'sort_by="{sort_by}" requires a weight channel (jet datasets only)')
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
		elif sort_by == "morton":
		        # Jets serialize on (eta, phi); generic clouds on all coord_dim axes.
		        axes = [x[:, :, i] for i in range(coord_start, coord_start + coord_dim)]
		        idx = _morton_sort_indices_np(axes, grid_size=grid_size)  # ascending
		        return np.take_along_axis(x, idx[:, :, None], axis=1)
		elif sort_by == "random":
		    B, N, C = x.shape
		    idx = np.argsort(
		        np.random.rand(B, N), axis=1
		    )
		    return np.take_along_axis(x, idx[:, :, None], axis=1)
		else:
				return x
		idx = np.argsort(key, axis=1)[:, ::-1]
		return np.take_along_axis(x, idx[:, :, None], axis=1)


# ---------------------------
# ModelNet augmentation
# ---------------------------
def augment_point_cloud(x, jitter_sigma=0.01, rotate=True, rng=None):
		"""
		Standard ModelNet augmentation applied to [B, N, 3] clouds: a random rotation about
		the up (y) axis plus small per-point Gaussian jitter.
		"""
		rng = rng or np.random
		x = np.array(x, dtype=np.float32, copy=True)
		if rotate:
				theta = rng.uniform(0.0, 2.0 * np.pi, size=(x.shape[0],)).astype(np.float32)
				cos, sin = np.cos(theta), np.sin(theta)
				x0, x2 = x[:, :, 0].copy(), x[:, :, 2].copy()
				x[:, :, 0] = cos[:, None] * x0 + sin[:, None] * x2
				x[:, :, 2] = -sin[:, None] * x0 + cos[:, None] * x2
		if jitter_sigma > 0:
				x += rng.normal(0.0, jitter_sigma, size=x.shape).astype(np.float32)
		return x


def make_train_dataset(x, y, batch_size, augment, sort_by, grid_size, coord_dim, weighted, jitter_sigma=0.01):
		"""
		Wrap the training arrays in a tf.data pipeline. Without --augment this is a plain
		shuffled batcher. With --augment each epoch re-rotates/jitters the clouds and then
		re-applies the ordering, since rotating after sorting would break the serialization.
		"""
		ds = tf.data.Dataset.from_tensor_slices((x, y))
		ds = ds.shuffle(x.shape[0], reshuffle_each_iteration=True)
		ds = ds.batch(batch_size)

		if augment:
				def _aug(xb, yb):
						def _np_aug(xb_np):
								xb_np = augment_point_cloud(xb_np, jitter_sigma=jitter_sigma)
								return apply_sorting(xb_np, sort_by, grid_size=grid_size,
								                     coord_dim=coord_dim, weighted=weighted)
						xb = tf.numpy_function(_np_aug, [xb], tf.float32)
						xb.set_shape([None, x.shape[1], x.shape[2]])
						return xb, yb
				ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

		return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------
# Patch size utilities
# ---------------------------
def compute_stage_lengths(num_particles, enc_strides):
		"""
		Length before each stage (after previous downsamples applied).
		For stages S, and strides defined between stages: len(enc_strides) == S-1
		"""
		lengths = []
		current = num_particles
		for i in range(len(enc_strides) + 1):
				lengths.append(current)
				if i < len(enc_strides):
						stride = enc_strides[i]
						current = int(math.ceil(current / float(stride)))
		return lengths


def choose_divisible_patch_sizes(stage_lengths, preferred=[64, 32, 16, 8, 4, 2, 1]):
		patch_sizes = []
		for L in stage_lengths:
				ps = 1
				for cand in preferred:
						if L >= cand and (L % cand == 0):
								ps = cand
								break
				patch_sizes.append(ps)
		return patch_sizes


# ---------------------------
# Testing / Profiling
# ---------------------------
def run_testing(model, dataset, data_dir, save_dir, sort_by, batch_size, num_particles, morton_grid_size, num_particles_truncate=None, enc_patch_sizes=None, coord_dim=2, weighted=True):
		logging.info("Starting testing phase...")
		logging.info("Using test batch size: %d", batch_size)

		# load test set
		if dataset == "hls4ml":
				x_test = np.load(
						os.path.join(data_dir, f"x_val_robust_{num_particles}const_ptetaphi.npy"),
						mmap_mode="r",
				)
				y_test = np.load(
						os.path.join(data_dir, f"y_val_robust_{num_particles}const_ptetaphi.npy"),
						mmap_mode="r",
				)
		else:  # jetclass, top, or QG
				x_test = np.load(os.path.join(data_dir, "test/features.npy"), mmap_mode="r")
				y_test = np.load(os.path.join(data_dir, "test/labels.npy"), mmap_mode="r")
		logging.info(
				"Loaded TEST arrays for %s: %s, %s", dataset, x_test.shape, y_test.shape
		)

		n_test = x_test.shape[0]
		pad_len = 0
		if enc_patch_sizes is not None:
				max_patch = max(enc_patch_sizes)
				test_num_particles = num_particles_truncate if num_particles_truncate is not None else (
						x_test.shape[2] if dataset == "jetclass" else x_test.shape[1]
				)
				remainder = test_num_particles % max_patch
				if remainder != 0:
						pad_len = max_patch - remainder

		def prepare_test_chunk(start, end):
				x_chunk = np.asarray(x_test[start:end])
				if dataset == "jetclass":
						x_chunk = x_chunk.transpose(0, 2, 1)
				x_chunk = apply_sorting(x_chunk, sort_by, grid_size=morton_grid_size, coord_dim=coord_dim, weighted=weighted)
				if num_particles_truncate is not None:
						x_chunk = x_chunk[:, :num_particles_truncate, :]
				if pad_len:
						x_chunk = np.pad(x_chunk, ((0,0),(0,pad_len),(0,0)))
				return x_chunk.astype(np.float32, copy=False)

		x_sample = prepare_test_chunk(0, min(batch_size, n_test))
		logging.info("Applied '%s' sorting to TEST set", sort_by)
		if num_particles_truncate is not None:
				logging.info("Truncated TEST set to top-%d particles", num_particles_truncate)
		if pad_len:
				logging.info("Padded TEST set to %d particles for patch_size=%d", x_sample.shape[1], max_patch)

		# flops & macs
		num_p, feat_d = x_sample.shape[1], x_sample.shape[2]
		flops = get_flops(model, (1, num_p, feat_d))
		macs = flops // 2
		logging.info("FLOPs per inference: %d", flops)
		logging.info("MACs per inference: %d", macs)

		# timing
		_ = model.predict(x_sample, batch_size=batch_size)
		times = []
		for _ in range(20):
				t0 = time.perf_counter()
				_ = model.predict(x_sample, batch_size=batch_size)
				times.append(time.perf_counter() - t0)
		avg_ns = np.mean(times) / x_sample.shape[0] * 1e9
		logging.info("Avg inference time/event: %.2f ns", avg_ns)

		# GPU memory
		curr, peak = profile_gpu_memory_during_inference(model, x_sample)
		logging.info("GPU memory current: %.1f MB, peak: %.1f MB", curr, peak)

		# metrics
		preds = predict_in_chunks(model, n_test, batch_size, prepare_test_chunk)
		if dataset == "top" or dataset == "QG":
				acc = accuracy_score(y_test, (preds.ravel() > 0.5).astype(int))
				auc_m = roc_auc_score(y_test, preds.ravel())
		else:
				acc = accuracy_score(np.argmax(y_test, 1), np.argmax(preds, 1))
				auc_m = roc_auc_score(y_test, preds, average="macro", multi_class="ovo")
		logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)

		# ROC curves and labels
		if dataset == "hls4ml":
				labels = ["q", "g", "W", "Z", "t"]
		elif dataset == "top":
				labels = ["qcd", "top"]
		elif dataset == "QG":  # gluon is 0 and quark is 1.
				labels = ["Gluon", "Quark"]
		elif dataset == "modelnet10":
				labels = [
						"bathtub", "bed", "chair", "desk", "dresser",
						"monitor", "night_stand", "sofa", "table", "toilet",
				]
		else:
				labels = [
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

		plt.figure(figsize=(6, 6))
		one_over_fpr = {}
		for i, lab in enumerate(labels):
				if dataset == "top" or dataset == "QG":
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
		plt.xlabel("FPR")
		plt.ylabel("TPR")
		plt.title("ROC curves")
		plt.legend(loc="lower right")
		plt.tight_layout()
		plt.savefig(os.path.join(save_dir, "roc_curves.png"))
		plt.close()

		for lab, val in one_over_fpr.items():
				logging.info("1/FPR@0.8 for %s: %.3f", lab, val)
		logging.info("Avg 1/FPR@0.8: %.3f", np.nanmean(list(one_over_fpr.values())))

		# background rejection combined
		if dataset not in ("top", "QG", "modelnet10"):
				rej_vals = []
				for i, lab in enumerate(labels[1:], start=1):
						mask_bg = (
								((y_test[:, 0] == 1) | (y_test[:, 1] == 1) | (y_test[:, i] == 1))
								if dataset != "jetclass"
								else np.ones_like(y_test[:, 0], dtype=bool)
						)
						if dataset == "jetclass":
								bin_y = (y_test[mask_bg, i] == 1).astype(int)
								bin_s = preds[mask_bg, i]
						else:
								bin_y = (y_test[mask_bg, i] == 1).astype(int)
								bin_s = preds[mask_bg, i]

						fpr, tpr, _ = roc_curve(bin_y, bin_s)
						idx = np.argmin(np.abs(tpr - 0.8))
						rej = 1.0 / fpr[idx] if fpr[idx] > 0 else np.inf
						logging.info("Bg rejection@0.8 %s: %.3f", lab, rej)
						rej_vals.append(rej)
				logging.info("Avg bg rejection@0.8: %.3f", np.nanmean(rej_vals))


# ---------------------------
# Argument parsing and main
# ---------------------------
def parse_args():
		p = argparse.ArgumentParser(description="Train PointTransformerV3TF on jet data")
		p.add_argument("--data_dir", required=True)
		p.add_argument("--save_dir", required=True)
		p.add_argument(
				"--dataset", choices=["hls4ml", "top", "QG", "jetclass", "modelnet10"], default="hls4ml"
		)
		p.add_argument(
				"--sort_by",
				choices=["pt", "eta", "phi", "delta_R", "kt", "morton", "cluster",  "random"],
				default="kt",
		)
		p.add_argument(
		    "--test_sort_by",
		    choices=["pt", "eta", "phi", "delta_R", "kt", "morton", "cluster", "random"],
		    nargs="+",
		    default=None,
		    help="One or more test-time orderings to evaluate. Defaults to --sort_by."
		)
		p.add_argument("--num_points", type=int, default=1024,
				help="Points per cloud for --dataset modelnet10 (must match the preprocessed data)")
		p.add_argument("--augment", action="store_true",
				help="ModelNet only: random up-axis rotation + jitter each epoch, re-sorted after")
		p.add_argument("--jitter_sigma", type=float, default=0.01, help="Std dev of --augment jitter")
		p.add_argument("--batch_size", type=int, default=4096)
		p.add_argument("--test_batch_size", type=int, default=None)
		p.add_argument("--log_every_batches", type=int, default=100, help="Write training progress to train.log/stdout every N batches; 0 disables")
		p.add_argument("--num_epochs", type=int, default=500)
		p.add_argument(
				"--schedule",
				default="128:200,256:200,512:200,1024:200,1024:200,1024:400",
				help="Comma-separated training schedule as batch_size:epochs. Use 'none' for --batch_size/--num_epochs.",
		)
		p.add_argument("--early_stopping_patience", type=int, default=40)
		p.add_argument("--test_only", action="store_true", help="Skip training and evaluate a checkpoint")
		p.add_argument("--checkpoint_path", default=None, help="Weights path to load with --test_only")
		p.add_argument("--val_split", type=float, default=0.2)
		p.add_argument("--num_particles_truncate", type=int, default=None,
				help="If set, truncate to this many particles after sorting (e.g. 64 to keep top-64 by pt)")

		# Model hyperparameters
		p.add_argument("--enc_dims", type=int, nargs="+", default=None)
		p.add_argument("--enc_layers", type=int, nargs="+", default=None)
		p.add_argument("--enc_heads", type=int, nargs="+", default=None)
		p.add_argument("--enc_patch_sizes", type=int, nargs="+", default=None)
		p.add_argument("--enc_strides", type=int, nargs="+", default=None)
		p.add_argument("--cpe_k", type=int, default=8)
		p.add_argument("--grid_size", type=float, default=0.05, help="GeometricCPE grid size (coarser -> smaller grid)")
		p.add_argument("--morton_grid_size", type=float, default=0.05, help="Grid size for morton sorting (separate from GeometricCPE grid_size)")
		p.add_argument("--use_rpe", action="store_true")
		p.add_argument("--disable_pool", action="store_true", help="Disable GeometricPooling between stages")
		p.add_argument("--dropout", type=float, default=0.0)
		p.add_argument("--aggregation", choices=["mean", "max"], default="max")
		p.add_argument("--model_size", choices=["small", "small_2layer_no_downsamp", "small_2layer_2_downsamp", "matched", "medium", "large"], default="small")
		p.add_argument('--use_serialized_model', action='store_true', help='Use the serialized version of the PointTransformer model')
		p.add_argument('--serialize_by', choices=['morton','pt','kt'], default='morton', help='Serialization  strategy when using the serialized model')
		p.add_argument('--assume_serialized_input', action='store_true', help='Skip in-model serialization when inputs are already sorted by --sort_by')
		p.add_argument('--use_jedi_hybrid', action='store_true', help='Use JEDI-PTv3 Hybrid (O(N) global interaction instead of attention)')
		p.add_argument('--disable_cpe', action='store_true', help='Disable CPE in JEDI hybrid (for pure JEDI-style permutation invariance)')
		p.add_argument("--ffn_activation", choices=["relu", "gelu", "swish", "silu", "tanh"], default="gelu", help="Activation function for feed-forward network (relu is fastest, gelu is default)")
		p.add_argument("--jit_compile", action="store_true", help="Enable XLA JIT compilation for faster training (5-15%% speedup on modern GPUs)")
		p.add_argument("--use_flash_attention", action="store_true", help="Enable Flash Attention for faster and more memory-efficient attention (requires TensorFlow 2.11+ and compatible GPU)")
		p.add_argument("--patch_tokenizer_mode", choices=["mean","max","flatten_dense","learned_pool"], default="mean")
		p.add_argument("--message_proj", dest="message_proj", action="store_true", default=True)
		p.add_argument("--no_message_proj", dest="message_proj", action="store_false")
		p.add_argument("--message_gated", dest="message_gated", action="store_true", default=False)
		p.add_argument("--no_message_gated", dest="message_gated", action="store_false")
		g = p.add_mutually_exclusive_group()
		g.add_argument("--use_patch_messages", dest="use_patch_messages", action="store_true", default=True, help="Enable patch-message pathway (default: on)")
		g.add_argument("--no_use_patch_messages", dest="use_patch_messages", action="store_false", help="Disable patch-message pathway (patch tokenizer/proj/gate become irrelevant)")
		p.add_argument(
	    "--cpe_coord_mode",
	    choices=["raw", "pt"],
	    default="raw",
	    help='GeometricCPE coord mode: "raw"=(eta,phi), "pt"=(pt*eta, pt*phi)'
		)
		return p.parse_args()


def main():
		args = parse_args()

		test_sorts = args.test_sort_by if args.test_sort_by is not None else [args.sort_by]

		# ModelNet clouds are (x, y, z) with every point real; jets are (pt, eta, phi)
		# where the pt channel doubles as the padding indicator.
		is_generic_cloud = args.dataset == "modelnet10"
		coord_dim = 3 if is_generic_cloud else 2
		weighted_input = not is_generic_cloud
		if is_generic_cloud and args.sort_by not in ("morton", "random"):
				raise ValueError(
						f'--sort_by {args.sort_by} needs a pt channel; use "morton" (recommended) '
						'or "random" for --dataset modelnet10'
				)

		# pick num_particles & output_dim
		if args.dataset == "jetclass":
				num_particles = 150
				output_dim = 10
				loss_fn = "categorical_crossentropy"
				feature_dim = 3
		elif args.dataset == "modelnet10":
				num_particles = args.num_points
				output_dim = 10
				loss_fn = "categorical_crossentropy"
				feature_dim = 3
		elif args.dataset == "top":
				num_particles = 200
				output_dim = 1
				loss_fn = "binary_crossentropy"
				feature_dim = 3
		elif args.dataset == "QG":
				num_particles = 150
				output_dim = 1
				loss_fn = "binary_crossentropy"
				feature_dim = 3
		else:  # hls4ml
				num_particles = 150
				output_dim = 5
				loss_fn = "categorical_crossentropy"
				feature_dim = 3

		# prepare save directory
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
				level=logging.INFO,
				format="%(asctime)s %(levelname)s %(message)s",
				handlers=[
						logging.FileHandler(os.path.join(save_dir, "train.log"), mode="w"),
						logging.StreamHandler(sys.stdout),
				],
		)
		logging.info("Args: %s", args)

		if args.test_only and args.checkpoint_path is None:
				raise ValueError("--checkpoint_path is required with --test_only")

		# load train/val
		if args.test_only:
				logging.info("Skipping train/val loading for test-only evaluation")
		elif args.dataset == "hls4ml":
				x = np.load(
						os.path.join(
								args.data_dir, f"x_train_robust_{num_particles}const_ptetaphi.npy"
						)
				)
				y = np.load(
						os.path.join(
								args.data_dir, f"y_train_robust_{num_particles}const_ptetaphi.npy"
						)
				)
				x_train, x_val, y_train, y_val = train_test_split(
						x, y, test_size=args.val_split, random_state=42
				)
		else:  # jetclass, top, or QG
				x_train = np.load(os.path.join(args.data_dir, "train/features.npy"))
				y_train = np.load(os.path.join(args.data_dir, "train/labels.npy"))
				x_val = np.load(os.path.join(args.data_dir, "val/features.npy"))
				y_val = np.load(os.path.join(args.data_dir, "val/labels.npy"))

		if args.test_only:
				pass
		elif args.dataset == "jetclass":
				x_train = x_train.transpose(0, 2, 1)
				x_val = x_val.transpose(0, 2, 1)

		if not args.test_only:
				feature_dim = x_train.shape[2]
				logging.info(
						"Loaded train x=%s y=%s, val x=%s y=%s",
						x_train.shape,
						y_train.shape,
						x_val.shape,
						y_val.shape,
				)

				# apply sorting
				x_train = apply_sorting(x_train, args.sort_by, grid_size=args.morton_grid_size, coord_dim=coord_dim, weighted=weighted_input)
				x_val   = apply_sorting(x_val,   args.sort_by, grid_size=args.morton_grid_size, coord_dim=coord_dim, weighted=weighted_input)

		# truncate to top-k particles if requested (e.g. 64 instead of 128)
		num_particles_for_files = num_particles  # preserve original for filename lookup in run_testing
		if args.num_particles_truncate is not None:
				k = args.num_particles_truncate
				assert k <= num_particles, f"--num_particles_truncate={k} > num_particles={num_particles}"
				if not args.test_only:
						x_train = x_train[:, :k, :]
						x_val   = x_val[:,   :k, :]
				num_particles = k
				logging.info("Truncated to top-%d particles after sorting", k)
		else:
				num_particles_for_files = num_particles

		# select preset — CLI args take priority over preset defaults
		presets = {
			"small":  dict(enc_dims=[16], enc_layers=[1], enc_heads=[4], enc_strides=[2], enc_patch_sizes=[25], cpe_k=8, use_rpe=False),
			"small_2layer_no_downsamp": dict(enc_dims=[16, 16], enc_layers=[1, 1], enc_heads=[4, 4], enc_strides=[1, 1], enc_patch_sizes=[25, 25], cpe_k=8, use_rpe=False),
			"small_2layer_2_downsamp": dict(enc_dims=[16, 16], enc_layers=[1, 1], enc_heads=[4, 4], enc_strides=[2], enc_patch_sizes=[25, 25], cpe_k=8, use_rpe=False),
    		"matched": dict(enc_dims=[12, 16], enc_layers=[1, 1], enc_heads=[4, 4], enc_strides=[2], enc_patch_sizes=[25, 25], cpe_k=8, use_rpe=False),
    		"medium": dict(enc_dims=[12, 24, 32], enc_layers=[1, 1, 1], enc_heads=[4, 4, 4], enc_strides=[2, 2], enc_patch_sizes=[25, 25, 25], cpe_k=8, use_rpe=False),
    		"large":  dict(enc_dims=[16, 24, 32], enc_layers=[1, 1, 1], enc_heads=[4, 4, 4], enc_strides=[2, 2], enc_patch_sizes=[25, 25, 25], cpe_k=8, use_rpe=False),
    	}
		cfg = presets[args.model_size]
		# CLI args (default=None) override preset; fall back to preset only when not specified
		enc_dims        = args.enc_dims        if args.enc_dims        is not None else cfg["enc_dims"]
		enc_layers      = args.enc_layers      if args.enc_layers      is not None else cfg["enc_layers"]
		enc_heads       = args.enc_heads       if args.enc_heads       is not None else cfg["enc_heads"]
		enc_strides     = args.enc_strides     if args.enc_strides     is not None else cfg["enc_strides"]
		enc_patch_sizes = args.enc_patch_sizes if args.enc_patch_sizes is not None else cfg["enc_patch_sizes"]
		n_stages = len(enc_dims)
		if len(enc_patch_sizes) == 1 and n_stages > 1:
		    enc_patch_sizes = enc_patch_sizes * n_stages  # broadcast single value
		if len(enc_patch_sizes) != n_stages:
		    raise ValueError(f"enc_patch_sizes has len {len(enc_patch_sizes)} but expected {n_stages} (stages)")

		cpe_k = cfg["cpe_k"] if args.cpe_k is None else args.cpe_k
		use_rpe = args.use_rpe or cfg["use_rpe"]

		# zero-pad sequence to nearest multiple of patch size if needed
		# (avoids gradient shape mismatch from padding inside attention layer)
		max_patch = max(enc_patch_sizes)
		remainder = num_particles % max_patch
		if remainder != 0 and is_generic_cloud:
				# Jets tolerate zero-padding because the pt channel marks padded slots. A
				# generic cloud has no such channel, so padded rows would be indistinguishable
				# from real points sitting at the origin. Require an exact patch division.
				raise ValueError(
						f"--enc_patch_sizes {max_patch} does not divide {num_particles} points; "
						f"pick a divisor (e.g. 16, 32 or 64 for 1024) for --dataset {args.dataset}"
				)
		if remainder != 0:
				pad_len = max_patch - remainder
				if not args.test_only:
						x_train = np.pad(x_train, ((0,0),(0,pad_len),(0,0)))
						x_val   = np.pad(x_val,   ((0,0),(0,pad_len),(0,0)))
				num_particles = num_particles + pad_len
				logging.info("Padded sequence to %d particles for patch_size=%d", num_particles, max_patch)

		# build and compile model
		logging.info("Flash Attention enabled: %s", args.use_flash_attention)
		if args.use_jedi_hybrid:
			logging.info("Building JEDI-PTv3 Hybrid model (O(N) global interaction)")
			model = build_jedi_ptv3_hybrid(
				num_particles=num_particles,
				output_dim=output_dim,
				enc_dims=enc_dims,
				enc_layers=enc_layers,
				enc_strides=enc_strides,
				cpe_k=cpe_k,
				grid_size=args.grid_size,
				use_pool=(not args.disable_pool),
				use_cpe=(not args.disable_cpe),
				dropout=args.dropout,
				aggregation=args.aggregation,
				ffn_activation=args.ffn_activation,
			)
		elif args.use_serialized_model:
			model = build_ptv3_serialized_jet_classifier(
				num_particles=num_particles,
				output_dim=output_dim,
				enc_dims=enc_dims,
				enc_layers=enc_layers,
				enc_heads=enc_heads,
				enc_patch_sizes=enc_patch_sizes,
				enc_strides=enc_strides,
				cpe_k=cpe_k,
				grid_size=args.grid_size,
				use_rpe=use_rpe,
				use_pool=(not args.disable_pool),
				dropout=args.dropout,
				aggregation=args.aggregation,
				serialize_by=args.serialize_by,
				assume_serialized_input=args.assume_serialized_input,
				coord_dim=coord_dim,
				weighted_input=weighted_input,
			)
		else:
			model = build_ptv3_jet_classifier(
				num_particles=num_particles,
				output_dim=output_dim,
				enc_dims=enc_dims,
				enc_layers=enc_layers,
				enc_heads=enc_heads,
				enc_patch_sizes=enc_patch_sizes,
				enc_strides=enc_strides,
				cpe_k=cpe_k,
				grid_size=args.grid_size,
				use_rpe=use_rpe,
				use_cpe=(not args.disable_cpe),
				use_pool=(not args.disable_pool),
				dropout=args.dropout,
				aggregation=args.aggregation,
				ffn_activation=args.ffn_activation,
				use_flash_attention=args.use_flash_attention,
				use_patch_messages=args.use_patch_messages,
				patch_tokenizer_mode=args.patch_tokenizer_mode,
			    message_proj=args.message_proj,
			    message_gated=args.message_gated,
				cpe_coord_mode=args.cpe_coord_mode,
				coord_dim=coord_dim,
				weighted_input=weighted_input,
			)
		model.compile(
				optimizer=tf.keras.optimizers.Adam(),
				loss=loss_fn,
				metrics=["accuracy"],
				jit_compile=args.jit_compile,
		)
		model.summary(print_fn=lambda l: logging.info(l))
		logging.info("Total params: %d", model.count_params())

		# ── log FLOPs right after compile so we can verify config ──────────────
		flops = get_flops(model, (1, num_particles, feature_dim))
		macs = flops // 2
		logging.info("FLOPs per inference: %d", flops)
		logging.info("MACs per inference: %d", macs)
		print(f"FLOPs per inference: {flops}")
		print(f"MACs  per inference: {macs}")
		# ────────────────────────────────────────────────────────────────────────

		if args.test_only:
				logging.info("Loading checkpoint: %s", args.checkpoint_path)
				model.load_weights(args.checkpoint_path)
				for test_sort in test_sorts:
				    logging.info("=" * 60)
				    logging.info("TEST ORDERING: %s", test_sort)
				    logging.info("=" * 60)
				    run_testing(
				        model,
				        args.dataset,
				        args.data_dir,
				        save_dir,
				        test_sort,
				        args.test_batch_size or args.batch_size,
				        num_particles_for_files,
				        morton_grid_size=args.morton_grid_size,
				        num_particles_truncate=args.num_particles_truncate,
				        enc_patch_sizes=enc_patch_sizes,
				        coord_dim=coord_dim,
				        weighted=weighted_input,
				    )
				return

		# callbacks
		ckpt = ModelCheckpoint(
				os.path.join(save_dir, "best.weights.h5"),
				monitor="val_loss",
				save_best_only=True,
				save_weights_only=True,
				verbose=1,
		)
		early = EarlyStopping(
				monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True, verbose=1
		)
		progress = LoggingProgressCallback(args.log_every_batches)

		schedule = parse_training_schedule(args.schedule, args.batch_size, args.num_epochs)
		logging.info("Training schedule: %s", schedule)

		ce = 0
		histories = []
		for bs, ep in schedule:
				tf.keras.backend.set_value(model.optimizer.lr, 1e-3)
				if args.augment:
						# Re-randomize the clouds every epoch; needs a tf.data pipeline rather than
						# the static arrays, and re-sorts after augmenting so the ordering stays valid.
						train_input = make_train_dataset(
								x_train, y_train, bs, True, args.sort_by, args.morton_grid_size,
								coord_dim, weighted_input, jitter_sigma=args.jitter_sigma,
						)
						fit_kwargs = dict(x=train_input)
				else:
						fit_kwargs = dict(x=x_train, y=y_train, batch_size=bs)
				hist = model.fit(
						validation_data=(x_val, y_val),
						initial_epoch=ce,
						epochs=ce + ep,
						callbacks=[ckpt, early, progress],
						verbose=1,
						**fit_kwargs,
				)
				histories.append(hist)
				ce += ep

		# save weights and metrics
		model.save_weights(os.path.join(save_dir, "model.weights.h5"))
		train_loss = np.concatenate([h.history["loss"] for h in histories])
		val_loss = np.concatenate([h.history["val_loss"] for h in histories])
		train_acc = np.concatenate([h.history["accuracy"] for h in histories])
		val_acc = np.concatenate([h.history["val_accuracy"] for h in histories])
		np.save(os.path.join(save_dir, "train_loss.npy"), train_loss)
		np.save(os.path.join(save_dir, "val_loss.npy"), val_loss)
		np.save(os.path.join(save_dir, "train_accuracy.npy"), train_acc)
		np.save(os.path.join(save_dir, "val_accuracy.npy"), val_acc)

		# plot loss and accuracy
		plt.figure()
		plt.plot(train_loss, label="Train Loss")
		plt.plot(val_loss, label="Val Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(save_dir, "loss_curve.png"))
		plt.close()

		plt.figure()
		plt.plot(train_acc, label="Train Acc")
		plt.plot(val_acc, label="Val Acc")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
		plt.close()

		# final testing
		for test_sort in test_sorts:
		    logging.info("=" * 60)
		    logging.info("TEST ORDERING: %s", test_sort)
		    logging.info("=" * 60)
		    run_testing(
		        model,
		        args.dataset,
		        args.data_dir,
		        save_dir,
		        test_sort,
		        args.test_batch_size or args.batch_size,
		        num_particles_for_files,
		        morton_grid_size=args.morton_grid_size,
		        num_particles_truncate=args.num_particles_truncate,
		        enc_patch_sizes=enc_patch_sizes,
		        coord_dim=coord_dim,
		        weighted=weighted_input,
		    )


if __name__ == "__main__":
		main()
