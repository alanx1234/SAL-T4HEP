import math
import tempfile
from pathlib import Path

import numpy as np

from scripts.evaluate_jetclass_multiplicity import (
    BinnedMetrics,
    ModelSpec,
    aggregate_trials,
    resolve_checkpoints,
)


def test_binned_metrics_counts_accuracy_and_auc():
    metrics = BinnedMetrics(score_bins=100)
    counts = np.array([10, 30, 55, 150])
    truth_index = np.array([0, 1, 0, 1])
    truth = np.eye(10, dtype=np.float32)[truth_index]
    predictions = np.full((4, 10), 0.001, dtype=np.float32)
    predictions[0, 0] = 0.9
    predictions[1, 0] = 0.8  # deliberately wrong
    predictions[2, 0] = 0.7
    predictions[3, 1] = 0.95

    metrics.update(counts, truth, predictions)
    result = metrics.result()

    assert [row["n_events"] for row in result["fine"]] == [
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
    ]
    assert result["coarse"][0]["n_events"] == 2
    assert result["coarse"][0]["accuracy"] == 0.5
    assert result["overall"][0]["accuracy"] == 0.75
    assert result["overall"][0]["confusion_matrix"][1][0] == 1
    assert math.isfinite(result["overall"][0]["macro_ovr_auc"])


def test_trial_aggregation_uses_sample_standard_deviation():
    metrics_a = BinnedMetrics(score_bins=10)
    metrics_b = BinnedMetrics(score_bins=10)
    counts = np.array([10, 10])
    truth = np.eye(10, dtype=np.float32)[[0, 1]]

    good = np.eye(10, dtype=np.float32)[[0, 1]]
    bad = np.eye(10, dtype=np.float32)[[1, 0]]
    metrics_a.update(counts, truth, good)
    metrics_b.update(counts, truth, bad)

    aggregate = aggregate_trials([metrics_a.result(), metrics_b.result()])
    overall = aggregate["overall"][0]
    assert overall["accuracy_mean"] == 0.5
    assert np.isclose(overall["accuracy_std"], math.sqrt(0.5))


def test_nonfinite_predictions_are_counted_and_do_not_crash_auc_histogram():
    metrics = BinnedMetrics(score_bins=10)
    counts = np.array([10, 10])
    truth = np.eye(10, dtype=np.float32)[[0, 1]]
    predictions = np.eye(10, dtype=np.float32)[[0, 1]]
    predictions[1, :] = np.nan

    metrics.update(counts, truth, predictions)
    result = metrics.result()["overall"][0]

    assert result["n_events"] == 2
    assert result["n_valid_predictions"] == 1
    assert result["n_nonfinite_predictions"] == 1
    assert result["accuracy"] == 0.5


def test_150_particle_bin_does_not_overflow_score_histogram_key():
    metrics = BinnedMetrics(score_bins=4096)
    counts = np.array([150])
    truth = np.eye(10, dtype=np.float32)[[9]]
    predictions = np.eye(10, dtype=np.float32)[[9]]

    metrics.update(counts, truth, predictions)
    result = metrics.result()

    assert result["fine"][-1]["n_events"] == 1
    assert result["fine"][-1]["accuracy"] == 1.0
    assert result["overall"][0]["n_nonfinite_predictions"] == 0


def test_checkpoint_resolver_pins_canonical_inner_trials():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for trial, (outer, inner) in enumerate(zip((10, 11, 12), (1, 0, 2))):
            canonical = (
                root
                / f"idx-{outer}-trial-{trial}"
                / "150"
                / "kt"
                / f"trial-{inner}"
                / "best.weights.h5"
            )
            canonical.parent.mkdir(parents=True)
            canonical.touch()
            newer = canonical.parents[1] / "trial-99" / "best.weights.h5"
            newer.parent.mkdir(parents=True)
            newer.touch()

        spec = ModelSpec(
            name="test_model",
            framework="tensorflow",
            root=str(root),
            outer_indices=(10, 11, 12),
            inner_trials=(1, 0, 2),
            batch_size=1,
        )
        selected = resolve_checkpoints(spec)

    assert [_inner.parent.name for _inner in selected] == [
        "trial-1",
        "trial-0",
        "trial-2",
    ]
