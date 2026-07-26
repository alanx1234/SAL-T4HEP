import math
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from scripts.evaluate_jetclass_multiplicity import (
    BinnedMetrics,
    ModelSpec,
    aggregate_trials,
    resolve_checkpoints,
)


def test_binned_metrics_counts_accuracy_auc_and_background_rejection():
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
    assert math.isfinite(result["overall"][0]["roc_auc"])
    assert "label_QCD" not in result["overall"][0][
        "background_rejection_at_0p8"
    ]


def test_background_rejection_is_inverse_fpr_at_80_percent_efficiency():
    metrics = BinnedMetrics(score_bins=100)
    counts = np.full(10, 30)
    truth_index = np.array([1] * 5 + [0] * 5)
    truth = np.eye(10, dtype=np.float32)[truth_index]
    predictions = np.zeros((10, 10), dtype=np.float32)
    predictions[:, 0] = 0.01
    predictions[:, 1] = np.array(
        [0.95, 0.90, 0.85, 0.80, 0.10, 0.99, 0.70, 0.60, 0.50, 0.40]
    )

    metrics.update(counts, truth, predictions)
    result = metrics.result()["overall"][0]

    # This deliberately contains a horizontal ROC segment at TPR=0.8.
    # np.interp matches the project's existing behavior and selects its
    # right endpoint, where FPR=1 and rejection=1.
    assert result["background_rejection_at_0p8"]["label_Hbb"] == 1.0
    assert result["avg_background_rejection_at_0p8"] == 1.0


def test_streaming_metrics_match_project_sklearn_definitions():
    score_bins = 128
    rng = np.random.default_rng(1234)
    truth_index = np.tile(np.arange(10), 100)
    truth = np.eye(10, dtype=np.float32)[truth_index]
    predictions = (
        rng.integers(0, score_bins, size=(len(truth), 10)) + 0.25
    ) / score_bins
    counts = np.full(len(truth), 30)

    metrics = BinnedMetrics(score_bins=score_bins)
    metrics.update(counts, truth, predictions)
    result = metrics.result()["overall"][0]

    expected_accuracy = accuracy_score(
        np.argmax(truth, axis=1), np.argmax(predictions, axis=1)
    )
    expected_auc = roc_auc_score(
        truth, predictions, average="macro", multi_class="ovo"
    )
    expected_rejections = {}
    for class_index, label in enumerate(
        [
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
        ][1:],
        start=1,
    ):
        fpr, tpr, _ = roc_curve(truth[:, class_index], predictions[:, class_index])
        fpr_at_efficiency = np.interp(0.8, tpr, fpr)
        expected_rejections[label] = (
            1.0 / fpr_at_efficiency if fpr_at_efficiency > 0 else math.nan
        )

    assert result["accuracy"] == expected_accuracy
    assert np.isclose(result["roc_auc"], expected_auc)
    for label, expected in expected_rejections.items():
        assert np.isclose(
            result["background_rejection_at_0p8"][label], expected
        )
    assert np.isclose(
        result["avg_background_rejection_at_0p8"],
        np.mean(list(expected_rejections.values())),
    )


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
