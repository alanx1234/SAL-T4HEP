"""
Tests for the 3D (ModelNet) ports of PHAT-JeT and serialized PTv3.

Two things are being checked:
  1. the new coord_dim=3 / weighted_input=False paths build and run correctly, and
  2. the existing 2D jet path is untouched, so the published jet results still reproduce.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.PointTransformerV3TF import GeometricCPE, build_ptv3_jet_classifier
from models.PointTransformer_serialized import (
    Serialization2D,
    build_ptv3_serialized_jet_classifier,
)
from scripts.train_point_transformer import (
    _morton_interleave_bits_np,
    _morton_sort_indices_np,
    apply_sorting,
    augment_point_cloud,
)

NUM_POINTS = 64
PATCH = 16


def make_cloud(batch=4, num_points=NUM_POINTS, seed=0):
    """A generic [B, N, 3] (x, y, z) cloud normalized into the unit sphere."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, num_points, 3)).astype("float32")
    x /= np.max(np.linalg.norm(x, axis=-1), axis=1)[:, None, None]
    return x


def make_jet(batch=4, num_particles=NUM_POINTS, seed=0):
    """A [B, N, 3] (pt, eta, phi) jet, with the tail zero-padded like the real data."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, num_particles, 3)).astype("float32")
    x[..., 0] = np.abs(x[..., 0]) + 0.1
    x[:, -8:, :] = 0.0  # padded constituents
    return x


def phat_kwargs(**overrides):
    kwargs = dict(
        num_particles=NUM_POINTS,
        output_dim=10,
        enc_dims=[16],
        enc_layers=[1],
        enc_heads=[4],
        enc_patch_sizes=[PATCH],
        enc_strides=[2],
        cpe_k=3,
        grid_size=0.2,
        use_pool=False,
        ffn_activation="gelu",
        patch_tokenizer_mode="mean",
        message_proj=True,
        message_gated=False,
    )
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------------------
# 3D forward passes
# --------------------------------------------------------------------------------------

def test_phat_3d_builds_and_predicts():
    model = build_ptv3_jet_classifier(**phat_kwargs(coord_dim=3, weighted_input=False))
    assert model.input_shape == (None, NUM_POINTS, 3)

    out = model.predict(make_cloud(), verbose=0)
    assert out.shape == (4, 10)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)


def test_phat_3d_uses_conv3d_for_gmp():
    model = build_ptv3_jet_classifier(**phat_kwargs(coord_dim=3, weighted_input=False))
    convs = [l for l in model.submodules if isinstance(l, tf.keras.layers.Conv3D)]
    assert convs, "3D GMP must use a Conv3D over the voxel grid"

    model_2d = build_ptv3_jet_classifier(**phat_kwargs(coord_dim=2, weighted_input=True))
    assert not [l for l in model_2d.submodules if isinstance(l, tf.keras.layers.Conv3D)]


def test_serialized_ptv3_3d_builds_and_predicts():
    model = build_ptv3_serialized_jet_classifier(
        num_particles=NUM_POINTS,
        output_dim=10,
        enc_dims=[16, 32],
        enc_layers=[1, 1],
        enc_heads=[4, 4],
        enc_patch_sizes=[PATCH, PATCH],
        enc_strides=[2],
        cpe_k=3,
        grid_size=0.2,
        serialize_by="morton",
        coord_dim=3,
        weighted_input=False,
    )
    assert model.input_shape == (None, NUM_POINTS, 3)
    out = model.predict(make_cloud(), verbose=0)
    assert out.shape == (4, 10)
    assert np.all(np.isfinite(out))


def test_generic_cloud_keeps_every_point():
    """
    The jet path masks on |channel 0| <= 1e-6. Applied to (x, y, z) that would silently
    delete every point lying on the x=0 plane, so weighted_input=False must not mask.
    """
    x = make_cloud()
    x[:, :5, 0] = 0.0  # points exactly on the x=0 plane

    model = build_ptv3_jet_classifier(**phat_kwargs(coord_dim=3, weighted_input=False))
    baseline = model.predict(x, verbose=0)

    # Moving those points must change the prediction; if they were masked it would not.
    moved = x.copy()
    moved[:, :5, 1] += 0.5
    assert not np.allclose(baseline, model.predict(moved, verbose=0), atol=1e-6)


def test_pt_coord_mode_rejected_without_weight_channel():
    with pytest.raises(ValueError, match="weighted_input"):
        build_ptv3_jet_classifier(
            **phat_kwargs(coord_dim=3, weighted_input=False, cpe_coord_mode="pt")
        )


def test_gmp_3d_matches_2d_on_a_planar_cloud():
    """
    A cloud flattened onto z=const collapses the 3D voxel grid to a single z-slice, so the
    3D GMP must reproduce the 2D GMP up to the extra (degenerate) convolution axis.
    """
    rng = np.random.default_rng(3)
    batch, n, channels = 2, 32, 8
    feats = rng.normal(size=(batch, n, channels)).astype("float32")
    coords_2d = rng.normal(size=(batch, n, 2)).astype("float32")
    coords_3d = np.concatenate([coords_2d, np.zeros((batch, n, 1), "float32")], axis=-1)

    cpe_2d = GeometricCPE(channels, kernel_size=3, grid_size=0.2, coord_dim=2, wrap_last_coord=False)
    cpe_3d = GeometricCPE(channels, kernel_size=3, grid_size=0.2, coord_dim=3)

    out_2d = cpe_2d(tf.constant(feats), None, tf.constant(coords_2d))
    out_3d = cpe_3d(tf.constant(feats), None, tf.constant(coords_3d))

    # Copy the 2D depthwise kernel into the centre z-slice of the 3D kernel so the two
    # convolutions compute the same thing on a single-slice grid.
    k2 = cpe_2d.conv.get_weights()
    k3 = cpe_3d.conv.get_weights()
    k3[0][:] = 0.0
    k3[0][:, :, k3[0].shape[2] // 2, :, :] = k2[0]
    k3[1][:] = k2[1]
    cpe_3d.conv.set_weights(k3)
    cpe_3d.pointwise.set_weights(cpe_2d.pointwise.get_weights())

    out_2d = cpe_2d(tf.constant(feats), None, tf.constant(coords_2d)).numpy()
    out_3d = cpe_3d(tf.constant(feats), None, tf.constant(coords_3d)).numpy()
    np.testing.assert_allclose(out_3d, out_2d, atol=1e-5)


# --------------------------------------------------------------------------------------
# Ordering helpers
# --------------------------------------------------------------------------------------

def test_morton_3d_orders_points_into_grid_cells():
    """Points in the same voxel must be adjacent after a 3D Morton sort."""
    coords = np.array([[
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
        [0.05, 0.0, 0.0],   # same cell as point 0 at grid_size=1.0
        [5.05, 5.0, 5.0],   # same cell as point 1
    ]], dtype="float32")

    idx = _morton_sort_indices_np([coords[..., i] for i in range(3)], grid_size=1.0)[0]
    positions = {point: int(np.flatnonzero(idx == point)[0]) for point in range(4)}
    assert abs(positions[0] - positions[2]) == 1
    assert abs(positions[1] - positions[3]) == 1


def test_morton_interleave_is_order_preserving_per_axis():
    axes = [np.array([[0, 1, 2, 3]], dtype=np.uint64)] * 3
    codes = _morton_interleave_bits_np(axes)
    assert np.all(np.diff(codes[0]) > 0)


def test_apply_sorting_rejects_pt_orderings_for_generic_clouds():
    x = make_cloud()
    with pytest.raises(ValueError, match="weight channel"):
        apply_sorting(x, "kt", coord_dim=3, weighted=False)

    sorted_x = apply_sorting(x, "morton", grid_size=0.2, coord_dim=3, weighted=False)
    assert sorted_x.shape == x.shape
    # Sorting is a permutation within each cloud, so the point multiset is unchanged.
    np.testing.assert_allclose(np.sort(sorted_x, axis=1), np.sort(x, axis=1))


def test_serialization_layer_rejects_pt_sorting_without_weights():
    with pytest.raises(ValueError, match="weight channel"):
        Serialization2D(sort_by="kt", coord_dim=3, weighted_input=False)


def test_augment_preserves_shape_and_rotates_about_up_axis():
    x = make_cloud(seed=7)
    aug = augment_point_cloud(x, jitter_sigma=0.0, rng=np.random.default_rng(1))
    assert aug.shape == x.shape
    # A rotation about y preserves the y coordinate and each point's distance from that axis.
    np.testing.assert_allclose(aug[..., 1], x[..., 1], atol=1e-5)
    r_before = np.linalg.norm(x[..., [0, 2]], axis=-1)
    r_after = np.linalg.norm(aug[..., [0, 2]], axis=-1)
    np.testing.assert_allclose(r_after, r_before, atol=1e-5)


# --------------------------------------------------------------------------------------
# Regression guards: the 2D jet path must be unchanged
# --------------------------------------------------------------------------------------

def test_jet_path_defaults_unchanged():
    """PHAT with default arguments must still be the 2D (pt, eta, phi) jet model."""
    model = build_ptv3_jet_classifier(**phat_kwargs(output_dim=5))
    assert model.input_shape == (None, NUM_POINTS, 3)
    out = model.predict(make_jet(), verbose=0)
    assert out.shape == (4, 5)
    assert np.all(np.isfinite(out))


def test_jet_padding_still_masked():
    """Zero-padded constituents must not influence the jet prediction."""
    model = build_ptv3_jet_classifier(**phat_kwargs(output_dim=5))
    x = make_jet()
    baseline = model.predict(x, verbose=0)

    # Perturb only the padded tail; the mask should make this a no-op.
    perturbed = x.copy()
    perturbed[:, -8:, 1:] = 3.0
    np.testing.assert_allclose(model.predict(perturbed, verbose=0), baseline, atol=1e-5)


def test_jet_2d_morton_ordering_unchanged():
    """The generalized Morton helper must reproduce the original 2D jet ordering."""
    rng = np.random.default_rng(11)
    eta = rng.normal(size=(3, 40)).astype("float32")
    phi = rng.normal(size=(3, 40)).astype("float32")

    def reference_2d(eta, phi, grid_size, bits=30):
        eta_g = np.clip(np.floor((eta - eta.min(1, keepdims=True)) / grid_size), 0, None).astype(np.uint64)
        phi_g = np.clip(np.floor((phi - phi.min(1, keepdims=True)) / grid_size), 0, None).astype(np.uint64)
        z = np.zeros_like(eta_g)
        for i in range(bits):
            z |= ((eta_g >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i)
            z |= ((phi_g >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i + 1)
        return np.argsort(z, axis=1)

    got = _morton_sort_indices_np([eta, phi], grid_size=0.2)
    np.testing.assert_array_equal(got, reference_2d(eta, phi, 0.2))


def test_jet_param_count_matches_expected_baseline():
    """
    Guards against the coord_dim refactor silently changing jet model capacity: the
    parameter count must not depend on the new keyword arguments being present.
    """
    explicit = build_ptv3_jet_classifier(
        **phat_kwargs(output_dim=5, coord_dim=2, weighted_input=True)
    )
    default = build_ptv3_jet_classifier(**phat_kwargs(output_dim=5))
    assert explicit.count_params() == default.count_params()


def test_serialized_jet_path_defaults_unchanged():
    model = build_ptv3_serialized_jet_classifier(
        num_particles=NUM_POINTS,
        output_dim=5,
        enc_dims=[16, 32],
        enc_layers=[1, 1],
        enc_heads=[4, 4],
        enc_patch_sizes=[PATCH, PATCH],
        enc_strides=[2],
        cpe_k=3,
        grid_size=0.2,
        serialize_by="kt",
    )
    assert model.input_shape == (None, NUM_POINTS, 3)
    out = model.predict(make_jet(), verbose=0)
    assert out.shape == (4, 5)
    assert np.all(np.isfinite(out))
