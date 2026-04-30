import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lorentznet import LorentzNet
from scripts.equivariant_utils import (
    Lion,
    apply_sorting,
    labels_to_indices,
    lorentz_scalars_from_p4,
    make_full_edges,
    make_loader,
    parse_training_schedule,
    ptetaphi_to_p4,
    train_epoch,
)
from scripts.train_lorentznet import forward_lorentznet


def make_toy_inputs(batch_size=4, num_particles=6, num_classes=5):
    rng = np.random.default_rng(123)
    x = rng.normal(size=(batch_size, num_particles, 3)).astype("float32")
    x[..., 0] = np.abs(x[..., 0]) + 0.1
    y_idx = np.arange(batch_size) % num_classes
    y = np.eye(num_classes, dtype="int64")[y_idx]
    p4 = ptetaphi_to_p4(x)
    return x, p4, y


def test_parse_training_schedule_defaults_and_none():
    assert parse_training_schedule("128:2,256:3", batch_size=4, num_epochs=5) == [
        (128, 2),
        (256, 3),
    ]
    assert parse_training_schedule("none", batch_size=4, num_epochs=5) == [(4, 5)]


def test_ptetaphi_to_p4_and_sorting_shapes():
    x, p4, _ = make_toy_inputs()
    assert p4.shape == (4, 6, 4)
    assert np.all(p4[..., 0] >= 0)
    x_sorted, p4_sorted = apply_sorting(x, "kt", p4)
    assert x_sorted.shape == x.shape
    assert p4_sorted.shape == p4.shape


def test_labels_to_indices_handles_one_hot_and_sparse():
    assert labels_to_indices(np.array([1, 2, 0])).tolist() == [1, 2, 0]
    assert labels_to_indices(np.eye(3, dtype="int64")).tolist() == [0, 1, 2]


def test_lorentznet_forward_shape():
    x, p4, y = make_toy_inputs(batch_size=3, num_particles=5, num_classes=5)
    loader = make_loader(x, p4, y, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    model = LorentzNet(n_scalar=1, n_hidden=8, n_class=5, n_layers=2)
    out = forward_lorentznet(model, batch)
    assert out.shape == (3, 5)


def test_lorentznet_one_training_step():
    x, p4, y = make_toy_inputs(batch_size=8, num_particles=5, num_classes=5)
    loader = make_loader(x, p4, y, batch_size=4, shuffle=False)
    model = LorentzNet(n_scalar=1, n_hidden=8, n_class=5, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, acc = train_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        forward_lorentznet,
    )
    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0


def test_full_edges_respect_mask():
    mask = torch.tensor([[True, True, False], [True, False, True]])
    rows, cols = make_full_edges(mask)
    edges = set(zip(rows.tolist(), cols.tolist()))
    assert edges == {(0, 1), (1, 0), (3, 5), (5, 3)}


def test_lorentz_scalars_shape():
    _, p4, _ = make_toy_inputs(batch_size=2, num_particles=3)
    scalars = lorentz_scalars_from_p4(torch.from_numpy(p4))
    assert scalars.shape == (2, 3, 1)


def test_lion_optimizer_step_updates_parameters():
    layer = nn.Linear(3, 2)
    before = layer.weight.detach().clone()
    optimizer = Lion(layer.parameters(), lr=1e-3, weight_decay=0.0)
    loss = layer(torch.ones(4, 3)).sum()
    loss.backward()
    optimizer.step()
    assert not torch.allclose(layer.weight, before)


@pytest.mark.skipif(importlib.util.find_spec("lgatr") is None, reason="lgatr package not installed")
def test_lgatr_forward_shape_when_dependency_available():
    from models.lgatr_wrapper import LGATrJetClassifier

    _, p4, _ = make_toy_inputs(batch_size=2, num_particles=4, num_classes=5)
    mask = torch.ones(2, 4, dtype=torch.bool)
    model = LGATrJetClassifier(
        num_classes=5,
        hidden_mv_channels=2,
        hidden_s_channels=4,
        num_blocks=1,
        num_heads=1,
    )
    out = model(torch.from_numpy(p4), mask=mask)
    assert out.shape == (2, 5)


@pytest.mark.skipif(importlib.util.find_spec("lgatr") is None, reason="lgatr package not installed")
def test_lgatr_default_forward_shape_when_dependency_available():
    from models.lgatr_wrapper import LGATrJetClassifier
    import lgatr.primitives.invariants as invariants

    _, p4, _ = make_toy_inputs(batch_size=2, num_particles=150, num_classes=5)
    mask = torch.ones(2, 150, dtype=torch.bool)
    model = LGATrJetClassifier(num_classes=5)
    assert invariants.cached_einsum.__name__ == "safe_cached_einsum"
    out = model(torch.from_numpy(p4), mask=mask)
    assert out.shape == (2, 5)


def test_part_chunked_testing_runs_on_memmapped_jetclass(tmp_path):
    from models.parT import ParticleTransformer
    from scripts.train_part import run_testing

    rng = np.random.default_rng(19)
    data_dir = tmp_path / "jetclass"
    test_dir = data_dir / "test"
    test_dir.mkdir(parents=True)

    n_events = 20
    n_particles = 12
    labels_idx = np.arange(n_events) % 10
    labels = np.eye(10, dtype=np.float32)[labels_idx]
    pt = rng.uniform(0.1, 2.0, size=(n_events, n_particles)).astype("float32")
    eta = rng.normal(0.0, 0.5, size=(n_events, n_particles)).astype("float32")
    phi = rng.uniform(-np.pi, np.pi, size=(n_events, n_particles)).astype("float32")
    features = np.stack([pt, eta, phi], axis=1)
    np.save(test_dir / "features.npy", features)
    np.save(test_dir / "labels.npy", labels)

    block_params = {
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "activation_dropout": 0.0,
        "scale_fc": False,
        "scale_attn": False,
        "scale_heads": False,
        "scale_resids": False,
    }
    model = ParticleTransformer(
        input_dim=3,
        num_classes=10,
        pair_input_dim=0,
        pair_extra_dim=0,
        remove_self_pair=True,
        use_pre_activation_pair=True,
        embed_dims=[4],
        pair_embed_dims=None,
        num_heads=1,
        num_layers=1,
        num_cls_layers=1,
        block_params=block_params,
        cls_block_params=block_params,
        fc_params=[],
        activation="gelu",
        trim=False,
        for_inference=False,
    )
    save_dir = tmp_path / "part_results"
    save_dir.mkdir()

    run_testing(
        model,
        "jetclass",
        str(data_dir),
        str(save_dir),
        "kt",
        batch_size=7,
        num_particles=n_particles,
        device=torch.device("cpu"),
    )

    log_artifact = save_dir / "roc_curves.png"
    assert log_artifact.exists()


@pytest.mark.skipif(
    importlib.util.find_spec("awkward") is None or importlib.util.find_spec("uproot") is None,
    reason="JetClass ROOT processing dependencies not installed",
)
def test_jetclass_processor_outputs_repo_compatible_shapes(tmp_path):
    import awkward as ak
    import uproot

    from scripts.process_jetclass_kinematics import LABELS, SAMPLES, process_split, shuffle_split

    raw_dir = tmp_path / "raw"
    split_dir = raw_dir / "Pythia" / "train_100M"
    split_dir.mkdir(parents=True)
    rng = np.random.default_rng(11)

    for sample, positive_label in SAMPLES.items():
        counts = [2, 3]
        px_values = [rng.normal(size=n).astype("float32") for n in counts]
        py_values = [rng.normal(size=n).astype("float32") for n in counts]
        pz_values = [rng.normal(size=n).astype("float32") for n in counts]
        energy_values = [
            np.sqrt(px_values[i] ** 2 + py_values[i] ** 2 + pz_values[i] ** 2).astype("float32") + 0.1
            for i in range(len(counts))
        ]
        data = {
            "part_px": ak.Array(px_values),
            "part_py": ak.Array(py_values),
            "part_pz": ak.Array(pz_values),
            "part_energy": ak.Array(energy_values),
            "part_deta": ak.Array([rng.normal(size=n).astype("float32") for n in counts]),
            "part_dphi": ak.Array([rng.normal(size=n).astype("float32") for n in counts]),
            "jet_pt": rng.uniform(100, 200, size=len(counts)).astype("float32"),
            "jet_energy": rng.uniform(200, 300, size=len(counts)).astype("float32"),
        }
        for label in LABELS:
            data[label] = (
                np.ones(len(counts), dtype="int32")
                if label == positive_label
                else np.zeros(len(counts), dtype="int32")
            )
        with uproot.recreate(split_dir / f"{sample}_0.root") as root_file:
            root_file["tree"] = data

    out_dir = tmp_path / "processed"
    args = SimpleNamespace(
        raw_dir=str(raw_dir),
        output_dir=str(out_dir),
        sample_type="Pythia",
        maxlen=5,
        feature_set="ptetaphi",
    )
    process_split(args, "train", "train_100M", target_total=10)

    features = np.load(out_dir / "train" / "features.npy")
    vectors = np.load(out_dir / "train" / "vectors.npy")
    labels = np.load(out_dir / "train" / "labels.npy")
    assert features.shape == (10, 3, 5)
    assert vectors.shape == (10, 4, 5)
    assert labels.shape == (10, 10)
    assert np.all(labels.sum(axis=1) == 1)

    original_labels = labels.copy()
    original_features = features.copy()
    original_vectors = vectors.copy()
    shuffle_split(out_dir, "train", seed=7, chunk_size=3)
    shuffled_features = np.load(out_dir / "train" / "features.npy")
    shuffled_vectors = np.load(out_dir / "train" / "vectors.npy")
    shuffled_labels = np.load(out_dir / "train" / "labels.npy")
    assert not np.array_equal(shuffled_labels, original_labels)
    for row in range(len(shuffled_labels)):
        source_row = np.where((original_labels == shuffled_labels[row]).all(axis=1))[0][0]
        assert np.array_equal(shuffled_features[row], original_features[source_row])
        assert np.array_equal(shuffled_vectors[row], original_vectors[source_row])
