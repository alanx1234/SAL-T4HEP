#!/usr/bin/env python
import os
import sys

# ─── make project root importable ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import time
import argparse
import logging

import numpy as np
import tensorflow as tf
import fastjet as fj

from models.Linformer import build_linformer_transformer_classifier
from models.LinformerBig import build_linformer_transformer_classifier_big

# import your custom layer/classes
from models.Linformer import (
    AggregationLayer,
    AttentionConvLayer,
    DynamicTanh,
    ClusteredLinformerAttention,
    LinformerTransformerBlock
)
from models.Transformer import (
    StandardMultiHeadAttention,
    StandardTransformerBlock,
    build_standard_transformer_classifier
)



def profile_gpu_memory_during_inference(model: tf.keras.Model, input_data: np.ndarray) -> tuple[float, float]:
    logging.info("Starting GPU memory profiling")
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        logging.warning("GPU memory stats not available; skipping.")
        return 0.0, 0.0

    @tf.function
    def infer(x):
        return model(x, training=False)

    _ = infer(input_data[:1]); _ = infer(input_data)
    mem = tf.config.experimental.get_memory_info("GPU:0")
    curr = mem["current"]/(1024**2)
    peak = mem["peak"]/(1024**2)
    logging.info("GPU memory profiling done: current=%.1f MB, peak=%.1f MB", curr, peak)
    return curr, peak


def get_flops(model):
    logging.info("Starting FLOPs calculation")
    input_shape = model.input_shape
    concrete_shape = tuple([1] + list(input_shape[1:]))
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    inp = tf.TensorSpec(concrete_shape, tf.float32)
    func = tf.function(model).get_concrete_function(inp)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(func)
    # Use correct context manager for new graphs
    new_graph = tf.Graph()
    with new_graph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof = tf.compat.v1.profiler.profile(graph=new_graph, run_meta=run_meta, cmd='op', options=opts)
        flops = prof.total_float_ops
    logging.info("FLOPs calculation done: %d FLOPs", flops)
    return flops


def sort_events_by_cluster(x, R, batch_size):
    logging.info("Starting cluster-based sorting with R=%.2f, batch_size=%d", R, batch_size)
    n_events, n_particles, _ = x.shape
    sorted_x = np.zeros_like(x)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    for i0 in range(0, n_events, batch_size):
        for bi, ev in enumerate(x[i0:i0+batch_size]):
            pts, etas, phis = ev[:,0], ev[:,1], ev[:,2]
            px = pts * np.cos(phis); py = pts * np.sin(phis)
            pz = pts * np.sinh(etas); E = pts * np.cosh(etas)
            ps = [fj.PseudoJet(px[j], py[j], pz[j], E[j]) for j in range(len(pts))]
            for j, pj in enumerate(ps):
                pj.set_user_index(j)
            seq = fj.ClusterSequence(ps, jet_def)
            jets = seq.inclusive_jets()
            jets.sort(key=lambda J: J.perp(), reverse=True)
            idxs = [c.user_index() for J in jets for c in J.constituents()]
            remain = [j for j in range(len(pts)) if j not in idxs]
            idxs.extend(remain)
            sorted_x[i0+bi] = ev[idxs]
    logging.info("Cluster-based sorting done")
    return sorted_x


def apply_sorting(x, sort_by, R, batch_size):
    logging.info("Starting sorting by '%s'", sort_by)
    if sort_by in ("pt","eta","phi","delta_R","kt"):
        if sort_by == "pt": key = x[:,:,0]
        elif sort_by == "eta": key = x[:,:,1]
        elif sort_by == "phi": key = x[:,:,2]
        elif sort_by == "delta_R": key = np.sqrt(x[:,:,1]**2 + x[:,:,2]**2)
        else: key = x[:,:,0] * np.sqrt(x[:,:,1]**2 + x[:,:,2]**2)
        idx = np.argsort(key, axis=1)[:, ::-1]
        sorted_x = np.take_along_axis(x, idx[:,:,None], axis=1)
    else:
        sorted_x = sort_events_by_cluster(x, R, batch_size)
    logging.info("Sorting done; data shape: %s", sorted_x.shape)
    return sorted_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hls4ml","top","jetclass","QG"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--sort_by", choices=["pt","eta","phi","delta_R","kt","cluster"], default="pt")
    parser.add_argument("--cluster_R", type=float, default=0.4)
    parser.add_argument("--cluster_batch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--aggregation", choices=["mean", "max"], default="max", help="Aggregation method for Linformer")
    parser.add_argument("--use_layer_norm", action="store_true", help="Use LayerNormalization instead of DynamicTanh")
    parser.add_argument("--ffn_activation", choices=["relu", "gelu", "swish", "silu", "tanh"], default="relu", help="Activation function for feed-forward network")
    parser.add_argument("--weights", default=None, help="Path to weights .h5 (default: save_dir/model.weights.h5 or best.weights.h5)")   
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--proj_dim", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    
    parser.add_argument("--cluster_E", action="store_true")
    parser.add_argument("--cluster_F", action="store_true")
    parser.add_argument("--share_EF", action="store_true")
    
    parser.add_argument("--convolution", action="store_true")
    parser.add_argument("--conv_filter_heights", type=int, nargs="+", default=[1,3,5])
    
    parser.add_argument("--shuffle_all", type=int, default=0)
    parser.add_argument("--shuffle_234", type=int, default=0)
    parser.add_argument("--shuffle_34", type=int, default=0)
    
    parser.add_argument("--use_cpe", action="store_true")
    parser.add_argument("--cpe_k", type=int, default=8)
    parser.add_argument("--grid_size", type=float, default=0.05)

    args = parser.parse_args()

    # Setup logging
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "test.log")
    logging.basicConfig(
        filename=log_path, filemode='w',
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cwd = os.getcwd()
    print(f"Running in directory: {cwd}")
    logging.info("Running in directory: %s", cwd)

    if args.dataset == "jetclass":
        num_particles = 150
        output_dim = 10
    elif args.dataset == "top":
        num_particles = 200
        output_dim = 1
    elif args.dataset == "QG":
        num_particles = 150
        output_dim = 1
    else:  # hls4ml
        num_particles = 150
        output_dim = 5

    logging.info("Starting data loading for dataset '%s'", args.dataset)
    
    if args.dataset == 'hls4ml':
        x = np.load(os.path.join(args.data_dir, f"x_val_robust_{num_particles}const_ptetaphi.npy"))
        y = np.load(os.path.join(args.data_dir, f"y_val_robust_{num_particles}const_ptetaphi.npy"))
    elif args.dataset == 'top':
        top_dir = os.path.join(args.data_dir, 'TopTagging', str(num_particles), 'test')
        x = np.load(os.path.join(top_dir, 'features.npy'))
        y = np.load(os.path.join(top_dir, 'labels.npy'))
    elif args.dataset == 'jetclass':
        x = np.load(os.path.join(args.data_dir, 'JetClass/kinematics/test/features.npy'))
        y = np.load(os.path.join(args.data_dir, 'JetClass/kinematics/test/labels.npy'))
        x = x.transpose(0, 2, 1)
    elif args.dataset == "QG":
        x = np.load(os.path.join(args.data_dir, 'QuarkGluon/test/features.npy'))
        y = np.load(os.path.join(args.data_dir, 'QuarkGluon/test/labels.npy'))
    
    logging.info("Data loaded: x shape=%s", x.shape)
    
    x = apply_sorting(x, args.sort_by, args.cluster_R, args.cluster_batch_size)
    x = x.astype(np.float32, copy=False)
    feat_dim = x.shape[2]
    
    conv_filter_heights = args.conv_filter_heights
    if args.num_layers > 1 and not conv_filter_heights:
        conv_filter_heights = [1, 3, 5, 7, 9]
    
    if args.num_layers > 1:
        model = build_linformer_transformer_classifier_big(
            num_particles, feat_dim,
            d_model=args.d_model,
            d_ff=args.d_ff,
            output_dim=output_dim,
            num_heads=args.num_heads,
            proj_dim=args.proj_dim,
            cluster_E=args.cluster_E,
            cluster_F=args.cluster_F,
            share_EF=args.share_EF,
            convolution=args.convolution,
            conv_filter_heights=conv_filter_heights,
            vertical_stride=1,
            num_layers=args.num_layers,
            aggregation=args.aggregation,
            use_layer_norm=args.use_layer_norm,
            ffn_activation=args.ffn_activation,
        )
    else:
        model = build_linformer_transformer_classifier(
            num_particles, feat_dim,
            d_model=args.d_model,
            d_ff=args.d_ff,
            output_dim=output_dim,
            num_heads=args.num_heads,
            proj_dim=args.proj_dim,
            cluster_E=args.cluster_E,
            cluster_F=args.cluster_F,
            share_EF=args.share_EF,
            convolution=args.convolution,
            conv_filter_heights=conv_filter_heights,
            vertical_stride=1,
            shuffle_all=args.shuffle_all,
            shuffle_234=args.shuffle_234,
            shuffle_34=args.shuffle_34,
            aggregation=args.aggregation,
            use_layer_norm=args.use_layer_norm,
            ffn_activation=args.ffn_activation,
            use_cpe=args.use_cpe,
            cpe_k=args.cpe_k,
            grid_size=args.grid_size,
        )
    
    _ = model(tf.zeros((1, num_particles, feat_dim), dtype=tf.float32), training=False)
    
    weights_path = args.weights
    if not weights_path:
        cand_model = os.path.join(args.save_dir, "model.weights.h5")
        cand_best  = os.path.join(args.save_dir, "best.weights.h5")
        weights_path = cand_model if os.path.isfile(cand_model) else cand_best
    
    logging.info("Loading weights from %s", weights_path)
    
    try:
        model.load_weights(weights_path)
        logging.info("Weights loaded successfully.")
    except Exception as e:
        logging.warning("load_weights failed: %s; retrying with skip_mismatch=True", e)
        try:
            model.load_weights(weights_path, skip_mismatch=True)
            logging.info("Weights loaded with skip_mismatch=True (some vars may be skipped).")
        except Exception as e2:
            logging.error("Failed to load weights from %s: %s", weights_path, e2)
            raise
    
    model.summary(print_fn=lambda s: logging.info(s))
    logging.info("Total parameters: %d", model.count_params())

    flops = get_flops(model)
    logging.info("FLOPs per inference: %d", flops)
    logging.info("MACs per inference: %d", flops // 2)
    logging.info("Starting inference timing (20 runs)")
    _ = model.predict(x[:args.batch_size], batch_size=args.batch_size)  

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model.predict(x[:args.batch_size], batch_size=args.batch_size)
        times.append(time.perf_counter() - t0)

    avg_ns = np.mean(times) / args.batch_size * 1e9
    logging.info("Inference timing done: avg %.2f ns/event", avg_ns)

    curr, peak = profile_gpu_memory_during_inference(model, x[:args.batch_size])
    logging.info("GPU memory current: %.1f MB, peak: %.1f MB", curr, peak)

if __name__ == "__main__":
    main()
