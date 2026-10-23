"""Pneumonia vs normal on paediatric chest X-rays (PneumoniaMNIST, 64x64).

Trains a small CNN from scratch and a frozen VGG16 feature model, scores both
on the official MedMNIST test split against a majority-class baseline, and
draws Grad-CAM heatmaps for the CNN.

This is a learning project. It is not a diagnostic tool.

Usage:
    python medical_image_analysis.py train
    python medical_image_analysis.py predict --test-index 0
    python medical_image_analysis.py predict path/to/xray.png
"""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Keep CPU use modest. Must run before TensorFlow starts its runtime.
tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("TF_THREADS", "2")))
tf.config.threading.set_inter_op_parallelism_threads(1)

# MedMNIST v2/v3, PneumoniaMNIST at 64x64 (CC BY 4.0). URL and MD5 from medmnist/info.py.
DATA_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist_64.npz?download=1"
DATA_MD5 = "8f4eceb4ccffa70c672198ea285246c6"
DATA_FILE = "pneumoniamnist_64.npz"
CLASS_NAMES = ["normal", "pneumonia"]
IMG_SIZE = 64
THRESHOLD = 0.5
SEED = 42

BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b"


# ---------------------------------------------------------------- data

def file_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(data_dir="data"):
    """Download the PneumoniaMNIST 64x64 file once and check its MD5."""
    path = Path(data_dir) / DATA_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(DATA_URL, tmp)
        tmp.replace(path)
    if file_md5(path) != DATA_MD5:
        raise ValueError(f"MD5 mismatch for {path}. Delete it and run again.")
    return path


def load_splits(path):
    """Return {split: (images, labels)} with images as float32 in [0, 1], shape (n, 64, 64, 1)."""
    splits = {}
    with np.load(path) as d:
        for split in ("train", "val", "test"):
            x = d[f"{split}_images"].astype("float32")[..., None] / 255.0
            y = d[f"{split}_labels"].ravel().astype("int64")
            splits[split] = (x, y)
    return splits


def load_image(path):
    """Read any image file as one 64x64 grey X-ray, centre-cropped to a square."""
    img = Image.open(path).convert("L")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(img, dtype="float32")[..., None] / 255.0


# ---------------------------------------------------------------- metrics

def binary_metrics(y_true, prob, threshold=THRESHOLD):
    """Accuracy, ROC AUC, sensitivity, specificity and confusion matrix for label 1 = pneumonia."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float((tp + tn) / len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def bootstrap_auc_ci(y_true, prob, n_boot=2000, seed=SEED):
    """95% percentile bootstrap interval for ROC AUC, resampling test cases."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() != y_true[idx].max():
            aucs.append(roc_auc_score(y_true[idx], prob[idx]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    return [float(low), float(high)]


def class_weights(y):
    """Weights that give both classes equal total weight in the loss."""
    counts = np.bincount(y, minlength=2)
    return {c: len(y) / (2 * counts[c]) for c in range(2)}


# ---------------------------------------------------------------- models

def build_small_cnn():
    """Four conv layers, global average pooling, one logit. Augmentation runs only in training."""
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 1))
    x = layers.RandomRotation(0.03, fill_mode="constant")(inputs)
    x = layers.RandomTranslation(0.05, 0.05, fill_mode="constant")(x)
    x = layers.RandomZoom(0.1, fill_mode="constant")(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    logit = layers.Dense(1, name="logit")(x)
    prob = layers.Activation("sigmoid", name="prob")(logit)
    return keras.Model(inputs, prob, name="small_cnn")


def train_small_cnn(splits, epochs=25, batch_size=64):
    x_train, y_train = splits["train"]
    model = build_small_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    # Pick the epoch on validation AUC. The test split is never seen here.
    stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        validation_data=splits["val"],
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights(y_train),
        callbacks=[stop],
        verbose=2,
    )
    return model, history.history


def vgg16_features(base, x, batch_size=128):
    """Pooled 512-d features from a frozen VGG16. The grey image is copied to all three channels."""
    rgb = np.repeat(x * 255.0, 3, axis=-1)
    return base.predict(keras.applications.vgg16.preprocess_input(rgb), batch_size=batch_size, verbose=0)


def train_vgg16_probe(splits):
    """Logistic regression on frozen VGG16 features. C is chosen on validation AUC."""
    base = keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg"
    )
    feats = {s: vgg16_features(base, splits[s][0]) for s in ("train", "val", "test")}
    y_train, y_val = splits["train"][1], splits["val"][1]
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
        )
        clf.fit(feats["train"], y_train)
        val_auc = roc_auc_score(y_val, clf.predict_proba(feats["val"])[:, 1])
        print(f"  VGG16 probe C={c}: val AUC {val_auc:.4f}")
        if best is None or val_auc > best[0]:
            best = (val_auc, c, clf)
    val_auc, c, clf = best
    return clf.predict_proba(feats["test"])[:, 1], {"C": c, "val_auc": float(val_auc)}


# ---------------------------------------------------------------- Grad-CAM

def grad_cam(model, image, layer_name="last_conv"):
    """Grad-CAM heatmap in [0, 1] for the predicted class, resized to the image.

    Uses the logit, not the sigmoid output, so the gradient does not vanish
    when the model is very sure.
    """
    cam_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.get_layer("logit").output])
    x = tf.convert_to_tensor(image[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, logit = cam_model(x, training=False)
        # Positive logit means pneumonia; flip the sign to explain a "normal" call.
        score = logit[:, 0] if float(logit[0, 0]) >= 0 else -logit[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], image.shape[:2], method="bilinear").numpy()[..., 0]


# ---------------------------------------------------------------- figures

def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors="#52514e")
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_samples(x, y, path, per_class=6):
    fig, axes = plt.subplots(2, per_class, figsize=(1.6 * per_class, 4.0))
    for row, label in enumerate((0, 1)):
        for col, i in enumerate(np.flatnonzero(y == label)[:per_class]):
            axes[row, col].imshow(x[i, ..., 0], cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{CLASS_NAMES[label]} (#{i})", fontsize=8, color=INK)
            axes[row, col].axis("off")
    fig.suptitle("PneumoniaMNIST test images (64x64), first six of each class", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, title in ((axes[0], "loss", "Loss (class-weighted on train)"), (axes[1], "auc", "ROC AUC")):
        ax.plot(epochs, history[key], color=BLUE, linewidth=2, label="train")
        ax.plot(epochs, history[f"val_{key}"], color=ORANGE, linewidth=2, label="validation")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("epoch")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Small CNN training curves", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, probs, results, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    colors = {"Small CNN": BLUE, "VGG16 probe": ORANGE}
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot(fpr, tpr, color=colors[name], linewidth=2, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=1.5, linestyle="--", label="Majority baseline (AUC 0.500)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC on the PneumoniaMNIST test split (n=624)", fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(results, names, path):
    fig, axes = plt.subplots(1, len(names), figsize=(3.9 * len(names), 3.6))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=12,
                    color="white" if v > cm.sum() / 2 else INK)
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name} (threshold {THRESHOLD})", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_gradcam_cases(y_true, prob, threshold=THRESHOLD):
    """Index of the most confident case of each kind: TP, TN, FP, FN (None if a kind is empty)."""
    y_pred = prob >= threshold
    margin = np.abs(prob - threshold)
    kinds = {
        "true positive": (y_true == 1) & y_pred,
        "true negative": (y_true == 0) & ~y_pred,
        "false positive": (y_true == 0) & y_pred,
        "false negative": (y_true == 1) & ~y_pred,
    }
    picks = {}
    for kind, mask in kinds.items():
        idx = np.flatnonzero(mask)
        picks[kind] = int(idx[np.argmax(margin[idx])]) if len(idx) else None
    return picks


def plot_gradcam(model, x, y_true, prob, path):
    picks = {k: v for k, v in pick_gradcam_cases(y_true, prob).items() if v is not None}
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 6.0))
    for col, (kind, i) in enumerate(picks.items()):
        cam = grad_cam(model, x[i])
        axes[0, col].imshow(x[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{kind}\ntest #{i}, true: {CLASS_NAMES[y_true[i]]}", fontsize=8, color=INK)
        axes[1, col].imshow(x[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        axes[1, col].set_title(f"P(pneumonia) = {prob[i]:.2f}", fontsize=8, color=INK)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle("Grad-CAM for the small CNN (most confident case of each kind)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- commands

def run_training(args):
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    docs, model_dir = Path(args.docs_dir), Path(args.model_dir)
    docs.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    splits = load_splits(download_data(args.data_dir))
    for name, (x, y) in splits.items():
        print(f"{name}: {len(y)} images, {int((y == 0).sum())} normal, {int((y == 1).sum())} pneumonia")
    x_test, y_test = splits["test"]
    plot_samples(x_test, y_test, docs / "sample_test_images.png")

    # Baseline: always predict the most common training class.
    majority = int(np.bincount(splits["train"][1]).argmax())
    results = {"Majority baseline": binary_metrics(y_test, np.full(len(y_test), float(majority)))}
    results["Majority baseline"]["roc_auc_95ci"] = [0.5, 0.5]
    results["Majority baseline"]["note"] = f"always predicts '{CLASS_NAMES[majority]}'"

    print("\nTraining small CNN")
    t0 = time.time()
    cnn, history = train_small_cnn(splits, epochs=args.epochs)
    cnn.save(model_dir / "small_cnn.keras")
    plot_history(history, docs / "training_curves.png")
    probs = {"Small CNN": cnn.predict(x_test, verbose=0).ravel()}
    best_epoch = int(np.argmax(history["val_auc"])) + 1
    results["Small CNN"] = binary_metrics(y_test, probs["Small CNN"])
    results["Small CNN"].update(
        roc_auc_95ci=bootstrap_auc_ci(y_test, probs["Small CNN"]),
        params=int(cnn.count_params()),
        epochs_run=len(history["loss"]),
        best_epoch=best_epoch,
        best_val_auc=float(history["val_auc"][best_epoch - 1]),
        train_seconds=round(time.time() - t0, 1),
    )

    if not args.skip_vgg16:
        print("\nExtracting VGG16 features and fitting the probe")
        t0 = time.time()
        probs["VGG16 probe"], info = train_vgg16_probe(splits)
        results["VGG16 probe"] = binary_metrics(y_test, probs["VGG16 probe"])
        results["VGG16 probe"].update(
            roc_auc_95ci=bootstrap_auc_ci(y_test, probs["VGG16 probe"]),
            train_seconds=round(time.time() - t0, 1),
            **info,
        )

    plot_roc(y_test, probs, results, docs / "roc_curve.png")
    plot_confusion(results, list(probs), docs / "confusion_matrix.png")
    plot_gradcam(cnn, x_test, y_test, probs["Small CNN"], docs / "gradcam_examples.png")

    summary = {
        "dataset": "PneumoniaMNIST 64x64 (MedMNIST v3), official splits",
        "split_sizes": {k: int(len(v[1])) for k, v in splits.items()},
        "threshold": THRESHOLD,
        "seed": SEED,
        "tensorflow": tf.__version__,
        "total_seconds": round(time.time() - start, 1),
        "results": results,
    }
    (docs / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'model':<18} {'accuracy':>8} {'ROC AUC':>8} {'95% CI':>15} {'sens':>6} {'spec':>6}")
    for name, r in results.items():
        ci = "{:.3f}-{:.3f}".format(*r["roc_auc_95ci"])
        print(f"{name:<18} {r['accuracy']:>8.3f} {r['roc_auc']:>8.3f} {ci:>15} "
              f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f}")
    print(f"\nFigures and results.json written to {docs}/, model to {model_dir}/small_cnn.keras")


def run_prediction(args):
    model = keras.models.load_model(Path(args.model_dir) / "small_cnn.keras")
    if args.test_index is not None:
        x, y = load_splits(download_data(args.data_dir))["test"]
        image, truth = x[args.test_index], CLASS_NAMES[y[args.test_index]]
    elif args.image:
        image, truth = load_image(args.image), None
    else:
        raise SystemExit("Give an image path or --test-index.")
    p = float(model.predict(image[None], verbose=0)[0, 0])
    label = CLASS_NAMES[int(p >= THRESHOLD)]
    print(f"prediction: {label}  P(pneumonia) = {p:.3f}  (threshold {THRESHOLD})")
    if truth is not None:
        print(f"true label: {truth}")
    print("Learning project only. Not for diagnosis.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="download data, train, evaluate on test, write figures to docs/")
    train.add_argument("--epochs", type=int, default=25, help="max epochs (early stopping on val AUC)")
    train.add_argument("--docs-dir", default="docs")
    train.add_argument("--skip-vgg16", action="store_true", help="train the small CNN only")
    train.set_defaults(func=run_training)

    predict = sub.add_parser("predict", help="score one image with the saved small CNN")
    predict.add_argument("image", nargs="?", help="path to a chest X-ray image")
    predict.add_argument("--test-index", type=int, help="use this image from the test split instead")
    predict.set_defaults(func=run_prediction)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
