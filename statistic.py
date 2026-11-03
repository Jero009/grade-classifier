#!/usr/bin/env python3
"""Analyze model training runs stored in model_runs.json.

Required keys per run:
    epoch_count, data_rows, training_percent, final_accuracy   (accuracy in %)

Optional keys per run (enable the extra figures):
    history:          {"loss": [...], "val_loss": [...],
                       "accuracy": [...], "val_accuracy": [...]}   (Keras-style)
    confusion_matrix: [[...], [...]]       (list of lists)
    class_names:      ["a", "b", ...]

Usage:
    python analyze_runs.py                 # show figures
    python analyze_runs.py --save out/     # also save PNGs to out/
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HISTORY_FILE = "model_runs.json"
REQUIRED = ["epoch_count", "data_rows", "training_percent", "final_accuracy"]
TOP_N = 5

BLUE, ORANGE, RED, GREEN = "#4da6ff", "#ffb84d", "#ff5555", "#55dd88"


# ----------------------------------------------------------------- loading --
def load_runs(path=HISTORY_FILE):
    p = Path(path)
    if not p.exists():
        print(f"'{path}' does not exist. Run the training script first!")
        return None
    try:
        runs = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("Failed to parse the JSON file.")
        return None
    if not runs:
        print("Run history is empty.")
        return None

    df = pd.DataFrame(runs)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        print(f"Missing columns: {sorted(missing)}")
        return None

    df.insert(0, "run", range(1, len(df) + 1))  # just an ID, order carries no meaning
    return df


# ----------------------------------------------------------------- helpers --
def pick(history, *names):
    """Return the first existing series from a history dict."""
    for n in names:
        if isinstance(history, dict) and history.get(n):
            return np.asarray(history[n], dtype=float)
    return None


def to_percent(a):
    """Keras logs accuracy as 0-1; convert so it matches final_accuracy (%)."""
    if a is None:
        return None
    return a * 100 if np.nanmax(a) <= 1.5 else a


def finish(fig, name, save_dir):
    fig.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{name}.png", dpi=150)


# ----------------------------------------------------------------- figures --
def print_summary(df):
    print("\n--- ALL RUNS ---")
    print(df.drop(columns=["history", "confusion_matrix", "class_names"],
                  errors="ignore").to_string(index=False))
    print("-" * 60)
    top = df.nlargest(min(TOP_N, len(df)), "final_accuracy")
    print(f"\n--- TOP {len(top)} RUNS ---")
    print(top[["run"] + REQUIRED].to_string(index=False))
    print("-" * 60)


def fig_overview(df, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Run overview", fontsize=14)

    # 1. All runs ranked by accuracy (order is by score, not by when it ran)
    ax = axes[0, 0]
    ranked = df.sort_values("final_accuracy", ascending=False)
    colors = [GREEN if i == 0 else BLUE for i in range(len(ranked))]
    ax.bar([f"#{r}" for r in ranked["run"]], ranked["final_accuracy"],
           color=colors, alpha=0.85)
    ax.set_title("Runs ranked by accuracy (green = best)")
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, ranked["final_accuracy"].min() - 5),
                ranked["final_accuracy"].max() + 2)
    ax.grid(True, axis="y", color="#333333", linestyle="--", alpha=0.7)

    # 2. Epochs vs accuracy bubble (size = data rows, color = training %)
    ax = axes[0, 1]
    max_rows = df["data_rows"].max()
    sizes = (df["data_rows"] / max_rows if max_rows > 0 else 1) * 300 + 30
    sc = ax.scatter(df["epoch_count"], df["final_accuracy"], s=sizes,
                    c=df["training_percent"], cmap="viridis",
                    alpha=0.75, edgecolors="white")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Training %")
    ax.set_title("Epochs vs accuracy (size = data rows)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.7)

    # 3. Data rows vs accuracy (learning curve)
    ax = axes[1, 0]
    ax.scatter(df["data_rows"], df["final_accuracy"], color=ORANGE, s=60, alpha=0.85)
    ax.set_title("Data rows vs accuracy (individual runs)")
    ax.set_xlabel("Data rows")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.7)

    # 4. Training % vs accuracy
    ax = axes[1, 1]
    ax.scatter(df["training_percent"], df["final_accuracy"], color=ORANGE, s=60, alpha=0.85)
    ax.set_title("Training % vs accuracy (individual runs)")
    ax.set_xlabel("Training split (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.7)

    finish(fig, "overview", save_dir)


def fig_top_runs(df, save_dir):
    top = df.nlargest(min(TOP_N, len(df)), "final_accuracy").iloc[::-1]
    labels = [f"#{r.run}  ({int(r.epoch_count)} ep, {int(r.data_rows)} rows, "
              f"{r.training_percent:g}% train)" for r in top.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.7 * len(top)))
    bars = ax.barh(labels, top["final_accuracy"], color=BLUE, alpha=0.85)
    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    lo = top["final_accuracy"].min()
    ax.set_xlim(max(0, lo - 5), top["final_accuracy"].max() + 3)
    ax.set_title(f"Top {len(top)} runs")
    ax.set_xlabel("Accuracy (%)")
    ax.grid(True, axis="x", color="#333333", linestyle="--", alpha=0.7)
    finish(fig, "top_runs", save_dir)


def fig_curves(df, save_dir):
    """Training/validation curves for the best and worst run."""
    if "history" not in df.columns:
        print("No 'history' logged yet, skipping training curves.")
        return
    with_hist = df[df["history"].apply(lambda h: isinstance(h, dict) and len(h) > 0)]
    if with_hist.empty:
        print("No 'history' logged yet, skipping training curves.")
        return

    best = with_hist.loc[with_hist["final_accuracy"].idxmax()]
    worst = with_hist.loc[with_hist["final_accuracy"].idxmin()]
    picks = [("Best", best)] if best["run"] == worst["run"] else [("Best", best), ("Worst", worst)]

    fig, axes = plt.subplots(len(picks), 2, figsize=(14, 4.5 * len(picks)), squeeze=False)
    fig.suptitle("Training vs validation curves", fontsize=14)

    for row, (tag, run) in enumerate(picks):
        h = run["history"]
        for col, (title, train_names, val_names, pct) in enumerate([
            ("Loss", ("loss",), ("val_loss",), False),
            ("Accuracy (%)", ("accuracy", "acc"), ("val_accuracy", "val_acc"), True),
        ]):
            ax = axes[row, col]
            tr, va = pick(h, *train_names), pick(h, *val_names)
            if pct:
                tr, va = to_percent(tr), to_percent(va)
            if tr is not None:
                ax.plot(range(1, len(tr) + 1), tr, color=BLUE, label="Train")
            if va is not None:
                ax.plot(range(1, len(va) + 1), va, color=ORANGE, label="Validation")
                if not pct:  # mark the epoch with the lowest validation loss
                    e = int(np.argmin(va))
                    ax.axvline(e + 1, color=GREEN, linestyle=":", label=f"Min val loss (ep {e + 1})")
            ax.set_title(f"{tag} run #{int(run['run'])}: {title} "
                         f"(final {run['final_accuracy']:.2f}%)")
            ax.set_xlabel("Epoch")
            ax.grid(True, color="#333333", linestyle="--", alpha=0.7)
            if tr is not None or va is not None:
                ax.legend(facecolor="#222222", edgecolor="none")

    finish(fig, "curves", save_dir)


def fig_gap(df, save_dir):
    """Train - validation accuracy gap per run (overfitting indicator)."""
    if "history" not in df.columns:
        return
    rows = []
    for r in df.itertuples():
        tr = to_percent(pick(r.history, "accuracy", "acc"))
        va = to_percent(pick(r.history, "val_accuracy", "val_acc"))
        if tr is not None and va is not None:
            rows.append((r.run, tr[-1] - va[-1]))
    if not rows:
        return

    rows.sort(key=lambda t: t[1], reverse=True)
    runs, gaps = zip(*rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([f"#{r}" for r in runs], gaps,
           color=[RED if g > 5 else BLUE for g in gaps], alpha=0.85)
    ax.axhline(0, color="white", linewidth=0.8)
    ax.set_title("Train - validation accuracy gap (red = gap > 5 points, likely overfitting)")
    ax.set_xlabel("Run ID (sorted by gap)")
    ax.set_ylabel("Gap (percentage points)")
    ax.grid(True, axis="y", color="#333333", linestyle="--", alpha=0.7)
    finish(fig, "gap", save_dir)


def fig_confusion(df, save_dir):
    """Confusion matrix of the best run that logged one."""
    if "confusion_matrix" not in df.columns:
        return
    has = df[df["confusion_matrix"].notna()]
    if has.empty:
        return
    best = has.loc[has["final_accuracy"].idxmax()]
    cm = np.asarray(best["confusion_matrix"], dtype=float)
    n = cm.shape[0]
    names = best.get("class_names")
    names = names if isinstance(names, list) and len(names) == n else [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(1.2 * n + 4, 1.0 * n + 3.5))
    im = ax.imshow(cm, cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "white")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix, best run #{int(best['run'])} ({best['final_accuracy']:.2f}%)")
    finish(fig, "confusion", save_dir)


# -------------------------------------------------------------------- main --
def analyze_model_runs(path=HISTORY_FILE, save_dir=None):
    df = load_runs(path)
    if df is None:
        return

    print_summary(df)

    plt.style.use("dark_background")
    save_dir = Path(save_dir) if save_dir else None

    fig_overview(df, save_dir)
    fig_top_runs(df, save_dir)
    fig_curves(df, save_dir)
    fig_gap(df, save_dir)
    fig_confusion(df, save_dir)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model training runs.")
    parser.add_argument("--file", default=HISTORY_FILE, help="path to runs JSON")
    parser.add_argument("--save", default=None, help="directory to save PNGs into")
    args = parser.parse_args()
    analyze_model_runs(args.file, args.save)