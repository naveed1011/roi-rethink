#!/usr/bin/env python3
"""Generate the README figures from the real code paths.

Run:
    python scripts/make_figures.py --out assets

This writes:
    assets/synthetic_samples.png   a grid of generated scans with GT boxes
    assets/loop_example.png        turn-1 input -> turn-2 marked ROI
    assets/eval_mock.png           metrics from the offline MockVLM run

The example loop uses a MockVLM whose bbox hint is deliberately set to the
ground truth, so the figure can show a *successful* marked ROI. That is an
illustration of the mechanism, not a model result; the metrics figure shows
what the blind mock actually scores.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import matplotlib  # noqa: E402
except ImportError:
    sys.exit("matplotlib is required for this script: pip install matplotlib")

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from roi_rethink.agent import RethinkAgent  # noqa: E402
from roi_rethink.eval import evaluate  # noqa: E402
from roi_rethink.synthetic import SyntheticAnomalyDataset  # noqa: E402
from roi_rethink.vlm import MockVLM  # noqa: E402

GREEN = "#2e7d32"
BLUE = "#1f4e79"
ORANGE = "#c55a11"


def _draw_gt(ax, bbox, label=None):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    ax.add_patch(
        Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=GREEN, linewidth=2.0, linestyle=(0, (4, 2)),
        )
    )
    if label:
        ax.text(
            x1, max(0, y1 - 5), label,
            color="white", fontsize=9, va="bottom",
            bbox=dict(facecolor=GREEN, edgecolor="none", pad=2.0),
        )


def make_sample_grid(out_dir, n=8, seed=11):
    ds = SyntheticAnomalyDataset(seed=seed)
    samples = [s for s in ds.generate(n * 3) if s.has_anomaly][:n]
    if not samples:
        raise RuntimeError("no anomalous samples generated")

    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, sample in zip(axes, samples):
        ax.imshow(sample.image)
        _draw_gt(ax, sample.bbox, sample.anomaly_kind)
        ax.set_title(f"planted {sample.anomaly_kind}", fontsize=10)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle(
        "Synthetic scans with planted anomalies (dashed green = ground-truth box)",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "synthetic_samples.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def make_loop_example(out_dir, seed=5):
    sample = next(
        s for s in SyntheticAnomalyDataset(seed=seed).generate(10)
        if s.has_anomaly and min(s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]) >= 45
    )
    # Scripted hint = GT: this is an oracle illustration of the loop UI,
    # not a claim about model accuracy.
    agent = RethinkAgent(MockVLM(hint_bbox=sample.bbox))
    traj = agent.run(sample.image, sample.prompt)
    if traj.marked_image is None:
        raise RuntimeError("the example loop did not produce a marked image")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
    axes[0].imshow(sample.image)
    axes[0].set_title("Turn 1: raw scan + query", fontsize=11)
    axes[1].imshow(traj.marked_image)
    axes[1].set_title("Turn 2: marked ROI returned by the environment", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        "The mark-and-rethink loop (example box scripted to the GT for illustration)",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "loop_example.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def make_metrics_figure(out_dir, n=10, seed=7):
    samples = SyntheticAnomalyDataset(seed=seed).generate(n)
    result = evaluate(
        RethinkAgent(MockVLM(hint_bbox=(110, 110, 170, 170))),
        samples,
        verbose=False,
    )
    m = result["metrics"]

    keys = [
        "valid_mark_rate",
        "mean_format",
        "mean_reward",
        "detection_accuracy",
        "mean_iou_positive",
        "iou_at_0_5",
    ]
    labels = [
        "valid mark rate",
        "format score",
        "mean reward",
        "detection accuracy",
        "mean IoU (positive)",
        "IoU@0.5",
    ]
    values = [m[k] for k in keys]
    colors = [BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE]

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("score (0–1)")
    ax.set_title(
        f"Offline MockVLM run (n={n}, seed={seed}): protocol vs. actual grounding",
        fontsize=11,
    )
    for bar, value in zip(bars, values):
        ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va="center", fontsize=9)
    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, color=BLUE, label="protocol / structure"),
            Rectangle((0, 0), 1, 1, color=ORANGE, label="detection / localization"),
        ],
        loc="lower right", frameon=False, fontsize=8,
    )
    ax.text(
        0.0, -0.22,
        "The mock is blind and always marks a fixed box: structure scores are high, "
        "localization is chance.",
        transform=ax.transAxes, fontsize=8, color="#444444",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "eval_mock.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets")
    p.add_argument("--seed", type=int, default=7, help="eval seed")
    p.add_argument("--n", type=int, default=10, help="eval sample count")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    make_sample_grid(args.out)
    make_loop_example(args.out)
    metrics = make_metrics_figure(args.out, n=args.n, seed=args.seed)
    print("\nMock metrics used for eval_mock.png:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
