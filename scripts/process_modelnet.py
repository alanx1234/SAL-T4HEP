#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_modelnet.py

Downloads ModelNet10, samples a fixed-size point cloud from each CAD mesh, and writes
NumPy (.npy) files in the same {train,val,test}/{features,labels}.npy layout used by the
JetClass/Top/QG pipelines, so the existing training scripts can load it unchanged.

Each shape becomes a [num_points, 3] cloud of (x, y, z) coordinates:
  - points are drawn area-weighted from the mesh surface (the standard PointNet protocol)
  - each cloud is centered on its centroid and scaled so the farthest point sits at radius 1

ModelNet10 ships only train/ and test/ directories, so a stratified validation split is
carved out of train (default 10%).

Output:
  <output_dir>/{train,val,test}/features.npy   float32 [n_shapes, num_points, 3]
  <output_dir>/{train,val,test}/labels.npy     float32 [n_shapes, 10]  (one-hot)
  <output_dir>/classes.json                    class name -> label index
"""
import argparse
import json
import logging
import os
import shutil
import zlib
from pathlib import Path

import numpy as np

# ModelNet10 mirrors, tried in order. The Princeton host is the canonical source but is
# frequently slow or unreachable, so keep fallbacks.
DOWNLOAD_URLS = [
    "https://3dshapenets.cs.princeton.edu/ModelNet10.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    "https://huggingface.co/datasets/Msun/modelnet10/resolve/main/ModelNet10.zip",
]

# Fixed label order; persisted to classes.json so evaluation can recover it.
CLASS_NAMES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]


def download_dataset(basedir, force_download=False):
    """Fetch and extract ModelNet10.zip into basedir, returning the extracted root."""
    import subprocess

    from dataset_utils import extract_archive

    basedir = Path(basedir)
    basedir.mkdir(parents=True, exist_ok=True)
    root = basedir / "ModelNet10"
    archive = basedir / "ModelNet10.zip"

    if force_download and root.exists():
        logging.info("Removing existing dir %s", root)
        shutil.rmtree(root)

    if root.is_dir() and any(root.glob("*/train/*.off")):
        logging.info("ModelNet10 already extracted at %s, skipping download", root)
        return root

    if force_download or not archive.exists():
        last_error = None
        for url in DOWNLOAD_URLS:
            logging.info("Downloading ModelNet10 from %s", url)
            try:
                subprocess.run(
                    ["wget", "--tries=3", "--timeout=60", url, "-O", str(archive)],
                    check=True,
                )
                break
            except subprocess.CalledProcessError as exc:  # try the next mirror
                last_error = exc
                logging.warning("Download failed from %s: %s", url, exc)
                if archive.exists():
                    archive.unlink()
        else:
            raise RuntimeError(f"All ModelNet10 mirrors failed; last error: {last_error}")
    else:
        logging.info("%s already exists, skipping download", archive)

    logging.info("Extracting %s", archive)
    extract_archive(str(archive), path=str(basedir))
    if not root.is_dir():
        raise RuntimeError(f"Extraction did not produce {root}")
    return root


def _repair_off_header(path: Path) -> Path:
    """
    Some ModelNet .off files have the vertex counts glued onto the OFF magic
    ("OFF3405 6812 0"), which trips strict parsers. Rewrite those into a temp file.
    Returns the path to load (the original when no repair is needed).
    """
    with open(path, "rb") as handle:
        head = handle.read(64)
    if not head.startswith(b"OFF") or head[3:4] in (b"\n", b"\r", b" ", b"\t"):
        return path

    logging.debug("Repairing malformed OFF header in %s", path)
    with open(path, "rb") as handle:
        body = handle.read()
    repaired = path.parent / f".repaired_{path.name}"
    with open(repaired, "wb") as handle:
        handle.write(b"OFF\n" + body[3:])
    return repaired


def sample_mesh(path: Path, num_points: int, seed: int) -> np.ndarray:
    """
    Sample `num_points` points area-weighted from the surface of one mesh, then normalize
    to a unit sphere. Returns float32 [num_points, 3].
    """
    import trimesh

    load_path = _repair_off_header(path)
    try:
        mesh = trimesh.load(str(load_path), file_type="off", force="mesh", process=False)
    finally:
        if load_path != path and load_path.exists():
            load_path.unlink()

    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
        raise ValueError(f"{path} did not load as a surface mesh")

    # trimesh.sample.sample_surface is area-weighted; seed per mesh for reproducibility.
    rng = np.random.default_rng(seed)
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=int(rng.integers(2**31 - 1)))
    points = np.asarray(points, dtype=np.float64)

    points = points - points.mean(axis=0, keepdims=True)
    radius = np.max(np.linalg.norm(points, axis=1))
    if radius > 0:
        points = points / radius
    return points.astype(np.float32)


def build_split(root: Path, split: str, num_points: int, seed: int):
    """
    Sample every mesh in <root>/<class>/<split>/. Returns (features, label_indices, names).
    """
    features, labels, names = [], [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        split_dir = root / class_name / split
        mesh_paths = sorted(split_dir.glob("*.off"))
        if not mesh_paths:
            raise RuntimeError(f"No .off files found in {split_dir}")
        logging.info("[%s/%s] sampling %d meshes", split, class_name, len(mesh_paths))

        for mesh_path in mesh_paths:
            # Seed from the file name so a given mesh always yields the same cloud,
            # independent of iteration order or how many meshes precede it. crc32 rather
            # than hash() because str hashing is salted per process.
            mesh_seed = (seed + zlib.crc32(mesh_path.name.encode())) % (2**31 - 1)
            try:
                features.append(sample_mesh(mesh_path, num_points, mesh_seed))
            except Exception as exc:
                logging.error("Skipping %s: %s", mesh_path, exc)
                continue
            labels.append(label_idx)
            names.append(mesh_path.name)

    return (
        np.stack(features, axis=0),
        np.asarray(labels, dtype=np.int64),
        names,
    )


def stratified_split(labels: np.ndarray, val_frac: float, seed: int):
    """Return (train_idx, val_idx) holding out val_frac of each class."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label_idx in np.unique(labels):
        idx = np.flatnonzero(labels == label_idx)
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(idx))))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return (
        np.sort(np.concatenate(train_idx)),
        np.sort(np.concatenate(val_idx)),
    )


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[labels]


def write_split(output_dir: Path, split: str, features: np.ndarray, labels: np.ndarray):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "features.npy", features.astype(np.float32, copy=False))
    np.save(split_dir / "labels.npy", one_hot(labels, len(CLASS_NAMES)))
    logging.info(
        "Wrote %s: features=%s labels=%s (class counts: %s)",
        split_dir,
        features.shape,
        (len(labels), len(CLASS_NAMES)),
        np.bincount(labels, minlength=len(CLASS_NAMES)).tolist(),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Sample ModelNet10 meshes into point-cloud .npy files")
    p.add_argument("--input_dir", required=True, help="Directory to download/extract ModelNet10 into")
    p.add_argument("--output_dir", required=True, help="Directory to write {train,val,test}/*.npy into")
    p.add_argument("--num_points", type=int, default=1000,
                   help="Points sampled per shape. Defaults to 1000 rather than the usual 1024 so "
                        "that the paper's patch size P=10 divides it exactly (100 patches per cloud) "
                        "and no padding is needed.")
    p.add_argument("--val_frac", type=float, default=0.1, help="Fraction of train held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--skip_download", action="store_true", help="Assume ModelNet10 is already extracted in --input_dir")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.skip_download:
        root = input_dir / "ModelNet10"
        if not root.is_dir():
            root = input_dir
    else:
        root = download_dataset(input_dir, force_download=args.force_download)
    logging.info("Using ModelNet10 root: %s", root)

    train_x, train_y, _ = build_split(root, "train", args.num_points, args.seed)
    test_x, test_y, _ = build_split(root, "test", args.num_points, args.seed)

    train_idx, val_idx = stratified_split(train_y, args.val_frac, args.seed)
    logging.info(
        "Split %d train meshes into %d train / %d val", len(train_y), len(train_idx), len(val_idx)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_split(output_dir, "train", train_x[train_idx], train_y[train_idx])
    write_split(output_dir, "val", train_x[val_idx], train_y[val_idx])
    write_split(output_dir, "test", test_x, test_y)

    with open(output_dir / "classes.json", "w") as handle:
        json.dump({name: idx for idx, name in enumerate(CLASS_NAMES)}, handle, indent=2)

    logging.info("Done. num_points=%d output=%s", args.num_points, output_dir)


if __name__ == "__main__":
    import sys

    # scripts/ must be importable for dataset_utils, matching the other process_* scripts.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
