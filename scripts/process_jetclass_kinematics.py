#!/usr/bin/env python
"""Process JetClass ROOT files into the repo's numpy kinematics format.

Outputs:
  split/features.npy with shape (N, C, P)
  split/vectors.npy  with shape (N, 4, P), convention (E, px, py, pz)
  split/labels.npy   with shape (N, 10)
"""

import argparse
import glob
import logging
import os
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


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

SAMPLES = {
    "HToBB": "label_Hbb",
    "HToCC": "label_Hcc",
    "HToGG": "label_Hgg",
    "HToWW4Q": "label_H4q",
    "HToWW2Q1L": "label_Hqql",
    "ZToQQ": "label_Zqq",
    "WToQQ": "label_Wqq",
    "TTBar": "label_Tbqq",
    "TTBarLep": "label_Tbl",
    "ZJetsToNuNu": "label_QCD",
}


def pad(array, maxlen, value=0.0, dtype="float32"):
    array = ak.fill_none(ak.pad_none(array, maxlen, clip=True), value)
    return ak.to_numpy(ak.values_astype(array, dtype))


def read_root_chunk(path, maxlen, feature_set, entry_start=None, entry_stop=None):
    branches = [
        "part_px",
        "part_py",
        "part_pz",
        "part_energy",
        "part_deta",
        "part_dphi",
        "jet_pt",
        "jet_energy",
    ] + LABELS
    table = uproot.open(path)["tree"].arrays(
        branches,
        entry_start=entry_start,
        entry_stop=entry_stop,
        library="ak",
    )
    px = table["part_px"]
    py = table["part_py"]
    pz = table["part_pz"]
    energy = table["part_energy"]
    pt = np.hypot(px, py)

    if feature_set == "ptetaphi":
        features = np.stack(
            [
                pad(pt, maxlen),
                pad(table["part_deta"], maxlen),
                pad(table["part_dphi"], maxlen),
            ],
            axis=1,
        )
    elif feature_set == "kin7":
        part_pt_log = np.log(np.maximum(pt, 1e-8))
        part_e_log = np.log(np.maximum(energy, 1e-8))
        part_logptrel = np.log(np.maximum(pt / table["jet_pt"], 1e-8))
        part_logerel = np.log(np.maximum(energy / table["jet_energy"], 1e-8))
        part_delta_r = np.hypot(table["part_deta"], table["part_dphi"])
        features = np.stack(
            [
                pad((part_pt_log - 1.7) * 0.7, maxlen),
                pad((part_e_log - 2.0) * 0.7, maxlen),
                pad((part_logptrel + 4.7) * 0.7, maxlen),
                pad((part_logerel + 4.7) * 0.7, maxlen),
                pad((part_delta_r - 0.2) * 4.0, maxlen),
                pad(table["part_deta"], maxlen),
                pad(table["part_dphi"], maxlen),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    vectors = np.stack(
        [pad(energy, maxlen), pad(px, maxlen), pad(py, maxlen), pad(pz, maxlen)],
        axis=1,
    )
    labels = np.stack([ak.to_numpy(table[name]).astype("int64") for name in LABELS], axis=1)
    return features.astype("float32"), vectors.astype("float32"), labels


def files_for_split(raw_root, sample_type, split_dir, sample):
    pattern = os.path.join(raw_root, sample_type, split_dir, f"{sample}_*.root")
    return sorted(glob.glob(pattern))


def count_entries(path):
    return uproot.open(path)["tree"].num_entries


def allocate_outputs(output_dir, split, total, n_features, maxlen):
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        split_dir / "features.npy",
        mode="w+",
        dtype="float32",
        shape=(total, n_features, maxlen),
    )
    vectors = np.lib.format.open_memmap(
        split_dir / "vectors.npy",
        mode="w+",
        dtype="float32",
        shape=(total, 4, maxlen),
    )
    labels = np.lib.format.open_memmap(
        split_dir / "labels.npy",
        mode="w+",
        dtype="int64",
        shape=(total, len(LABELS)),
    )
    return features, vectors, labels


def process_split(args, split, split_dir, target_total):
    per_sample = target_total // len(SAMPLES)
    remainder = target_total % len(SAMPLES)
    targets = {
        sample: per_sample + (1 if idx < remainder else 0)
        for idx, sample in enumerate(SAMPLES)
    }
    n_features = 3 if args.feature_set == "ptetaphi" else 7
    features_out, vectors_out, labels_out = allocate_outputs(
        args.output_dir, split, target_total, n_features, args.maxlen
    )

    cursor = 0
    for sample, target in targets.items():
        written = 0
        files = files_for_split(args.raw_dir, args.sample_type, split_dir, sample)
        if not files:
            raise FileNotFoundError(f"No files found for {sample} in {split_dir}")
        for path in files:
            if written >= target:
                break
            n_entries = count_entries(path)
            take = min(n_entries, target - written)
            feats, vecs, labs = read_root_chunk(
                path,
                maxlen=args.maxlen,
                feature_set=args.feature_set,
                entry_start=0,
                entry_stop=take,
            )
            end = cursor + len(feats)
            features_out[cursor:end] = feats
            vectors_out[cursor:end] = vecs
            labels_out[cursor:end] = labs
            cursor = end
            written += len(feats)
            logging.info("%s %s: wrote %d/%d", split, sample, written, target)
        if written < target:
            raise RuntimeError(f"Only found {written} events for {sample}; target was {target}")

    features_out.flush()
    vectors_out.flush()
    labels_out.flush()
    logging.info("Finished %s: %d jets", split, cursor)


def shuffle_array(path, permutation, chunk_size):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    src = np.load(path, mmap_mode="r")
    dst = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=src.dtype,
        shape=src.shape,
    )
    for start in range(0, len(permutation), chunk_size):
        end = min(start + chunk_size, len(permutation))
        dst[start:end] = src[permutation[start:end]]
    dst.flush()
    del src, dst
    os.replace(tmp_path, path)


def shuffle_split(output_dir, split, seed, chunk_size):
    split_dir = Path(output_dir) / split
    labels_path = split_dir / "labels.npy"
    n_events = np.load(labels_path, mmap_mode="r").shape[0]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_events)
    logging.info("Shuffling %s with seed %d", split, seed)
    for name in ("features.npy", "vectors.npy", "labels.npy"):
        shuffle_array(split_dir / name, permutation, chunk_size)


def parse_args():
    p = argparse.ArgumentParser(description="Process JetClass ROOT kinematics to numpy")
    p.add_argument("--raw_dir", required=True, help="Directory containing Pythia/{train_100M,val_5M,test_20M}")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sample_type", default="Pythia")
    p.add_argument("--maxlen", type=int, default=150)
    p.add_argument("--feature_set", choices=["ptetaphi", "kin7"], default="ptetaphi")
    p.add_argument("--num_train", type=int, default=2_000_000)
    p.add_argument("--num_val", type=int, default=200_000)
    p.add_argument("--num_test", type=int, default=20_000_000)
    p.add_argument("--shuffle_seed", type=int, default=42)
    p.add_argument("--shuffle_chunk_size", type=int, default=50_000)
    p.add_argument("--no_shuffle", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    process_split(args, "train", "train_100M", args.num_train)
    process_split(args, "val", "val_5M", args.num_val)
    process_split(args, "test", "test_20M", args.num_test)
    if not args.no_shuffle:
        shuffle_split(args.output_dir, "train", args.shuffle_seed + 0, args.shuffle_chunk_size)
        shuffle_split(args.output_dir, "val", args.shuffle_seed + 1, args.shuffle_chunk_size)
        shuffle_split(args.output_dir, "test", args.shuffle_seed + 2, args.shuffle_chunk_size)


if __name__ == "__main__":
    main()
