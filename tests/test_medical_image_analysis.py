"""Quick checks that need no download and no training."""

import numpy as np
import pytest
from PIL import Image

import medical_image_analysis as mia


def test_binary_metrics_by_hand():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.2, 0.9, 0.8, 0.4, 0.7, 0.95])
    m = mia.binary_metrics(y, p)
    # tn=2, fp=1, fn=1, tp=4
    assert m["confusion_matrix"] == [[2, 1], [1, 4]]
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["sensitivity"] == pytest.approx(4 / 5)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(14 / 15)


def test_majority_baseline_scores():
    y = np.array([0, 1, 1, 1])
    m = mia.binary_metrics(y, np.ones(4))
    assert m["accuracy"] == 0.75
    assert m["roc_auc"] == 0.5
    assert (m["sensitivity"], m["specificity"]) == (1.0, 0.0)


def test_bootstrap_ci_brackets_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    low, high = mia.bootstrap_auc_ci(y, p, n_boot=300)
    assert 0 <= low <= mia.roc_auc_score(y, p) <= high <= 1


def test_class_weights_balance_classes():
    y = np.array([0, 1, 1, 1])
    w = mia.class_weights(y)
    assert w[0] * 1 == pytest.approx(w[1] * 3)


def test_load_splits_shapes(tmp_path):
    path = tmp_path / "fake.npz"
    arrays = {}
    for split, n in (("train", 5), ("val", 2), ("test", 3)):
        arrays[f"{split}_images"] = np.full((n, 64, 64), 255, dtype=np.uint8)
        arrays[f"{split}_labels"] = np.zeros((n, 1), dtype=np.uint8)
    np.savez(path, **arrays)
    splits = mia.load_splits(path)
    x, y = splits["test"]
    assert x.shape == (3, 64, 64, 1) and x.dtype == np.float32 and x.max() == 1.0
    assert y.shape == (3,)


def test_load_image_crops_and_resizes(tmp_path):
    path = tmp_path / "wide.png"
    Image.fromarray(np.zeros((100, 200), dtype=np.uint8)).save(path)
    img = mia.load_image(path)
    assert img.shape == (64, 64, 1)


def test_pick_gradcam_cases():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.6, 0.1, 0.7, 0.2, 0.95])
    picks = mia.pick_gradcam_cases(y, p)
    assert picks == {"true positive": 0, "true negative": 2, "false positive": 5, "false negative": 4}


def test_cnn_and_grad_cam_shapes():
    model = mia.build_small_cnn()
    x = np.random.default_rng(0).random((2, 64, 64, 1)).astype("float32")
    prob = model.predict(x, verbose=0)
    assert prob.shape == (2, 1) and np.all((prob > 0) & (prob < 1))
    cam = mia.grad_cam(model, x[0])
    assert cam.shape == (64, 64)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6
