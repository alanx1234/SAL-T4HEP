import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.evaluate_hls4ml_multiplicity import (
    LABELS,
    _background_rejection,
    calculate_metrics,
)


def test_hls_metrics_match_project_definitions_and_exclude_q_g_rejection():
    rng = np.random.default_rng(17)
    truth_index = np.tile(np.arange(5), 200)
    truth = np.eye(5, dtype=np.float32)[truth_index]
    logits = rng.normal(size=(len(truth), 5))
    predictions = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    counts = np.full(len(truth), 30)

    result = calculate_metrics(counts, truth, predictions)["overall"][0]

    expected_accuracy = np.mean(
        np.argmax(predictions, axis=1) == truth_index
    )
    expected_auc = roc_auc_score(
        truth, predictions, average="macro", multi_class="ovo"
    )
    assert result["accuracy"] == expected_accuracy
    assert np.isclose(result["roc_auc"], expected_auc)
    assert set(result["background_rejection_at_0p8"]) == {"W", "Z", "t"}

    expected_rejections = []
    for signal_index in (2, 3, 4):
        selected = np.isin(truth_index, (0, 1, signal_index))
        binary_truth = (truth_index[selected] == signal_index).astype(int)
        fpr, tpr, _ = roc_curve(
            binary_truth, predictions[selected, signal_index]
        )
        nearest = np.argmin(np.abs(tpr - 0.8))
        expected = 1.0 / fpr[nearest]
        expected_rejections.append(expected)
        assert np.isclose(
            result["background_rejection_at_0p8"][LABELS[signal_index]],
            expected,
        )
    assert np.isclose(
        result["avg_background_rejection_at_0p8"],
        np.mean(expected_rejections),
    )


def test_background_rejection_uses_q_g_and_signal_only():
    truth_index = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    predictions = np.full((len(truth_index), 5), 0.01)
    predictions[:, 2] = [0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 1.0, 1.0]

    actual = _background_rejection(truth_index, predictions, signal_index=2)
    selected = np.isin(truth_index, (0, 1, 2))
    fpr, tpr, _ = roc_curve(
        truth_index[selected] == 2, predictions[selected, 2]
    )
    nearest = np.argmin(np.abs(tpr - 0.8))
    expected = 1.0 / fpr[nearest] if fpr[nearest] else math.inf

    assert actual == expected


def test_hls_bins_use_real_particle_counts():
    counts = np.array([10, 30, 55, 150])
    truth = np.eye(5, dtype=np.float32)[[0, 1, 2, 4]]
    predictions = truth.copy()

    result = calculate_metrics(counts, truth, predictions)

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
    assert result["overall"][0]["n_events"] == 4
    assert result["overall"][0]["accuracy"] == 1.0
