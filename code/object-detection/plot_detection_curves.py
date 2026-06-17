"""Combined precision-recall and F1-confidence figure for all four detectors.

Top row: precision-recall curves. Bottom row: F1-confidence curves. One column
per model (YOLO26, YOLO11n, YOLOv8n, YOLO26+DraxNet), each panel showing the
all-class, active-TB, and latent-TB curves.

Curve data sources:
  - YOLO26 and YOLO26+DraxNet: the committed CSVs under figures/ (their trained
    weights are not distributed, so the curves cannot be recomputed here).
  - YOLO11n and YOLOv8n: extracted live from their best.pt via Ultralytics val
    (curves_results), since their weights are present.

Output: figures/detection_curves.pdf (two-column figure for the manuscript).

Run from the repo root (uses the project venv):
    code/_mlx/.venv/bin/python code/object-detection/plot_detection_curves.py
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "figures"
DATA = REPO / "dataset/object-detection/dataset.yaml"
IMG = 512

# class index -> curve label used in the figures (active = ActiveTuberculosis,
# obsolete = ObsoletePulmonaryTuberculosis).
CLASS_LABEL = {0: "active", 1: "obsolete"}
CURVE_STYLE = {  # label -> (display, color)
    "all": ("All classes", "#222222"),
    "active": ("Active TB", "#1f77b4"),
    "obsolete": ("Latent TB", "#d62728"),
}

# (column title, key); CSV models read figures/{pr,f1}_{key}.csv, live models
# are recomputed from best.pt.
MODELS = [
    ("YOLO26", "yolo26", None),
    ("YOLO11n", "yolo11n", "code/object-detection/yolo11n/results/weights/best.pt"),
    ("YOLOv8n", "yolov8n", "code/object-detection/yolov8n/results/weights/best.pt"),
    ("YOLO26 + DraxNet", "draxnet", None),
]


def read_csv_curves(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    out: dict[str, tuple[list[float], list[float]]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            c = row["curve"]
            out.setdefault(c, ([], []))
            out[c][0].append(float(row["x"]))
            out[c][1].append(float(row["y"]))
    return out


def live_curves(weights: str):
    """Return (pr, f1) curve dicts {label: (x, y)} from a trained checkpoint."""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(REPO / weights))
    res = model.val(data=str(DATA), imgsz=IMG, device=0, plots=False, verbose=False)
    names = list(res.curves)  # ['Precision-Recall(B)', 'F1-Confidence(B)', ...]
    cr = res.curves_results

    def build(idx):
        x, y = cr[idx][0], np.asarray(cr[idx][1])  # x:(n,), y:(nc, n)
        x = np.asarray(x)
        d = {"all": (x.tolist(), y.mean(0).tolist())}
        for ci, lab in CLASS_LABEL.items():
            if ci < y.shape[0]:
                d[lab] = (x.tolist(), y[ci].tolist())
        return d

    pr = build(names.index("Precision-Recall(B)"))
    f1 = build(names.index("F1-Confidence(B)"))
    return pr, f1


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(MODELS), figsize=(12, 5.2), sharex="row", sharey="row")

    for col, (title, key, weights) in enumerate(MODELS):
        if weights is None:
            pr = read_csv_curves(FIGDIR / f"pr_{key}.csv")
            f1 = read_csv_curves(FIGDIR / f"f1_{key}.csv")
        else:
            pr, f1 = live_curves(weights)

        for row, curves, (xlab, ylab) in (
            (0, pr, ("Recall", "Precision")),
            (1, f1, ("Confidence", "F1")),
        ):
            ax = axes[row][col]
            for lab, (disp, color) in CURVE_STYLE.items():
                if lab in curves:
                    ax.plot(curves[lab][0], curves[lab][1], color=color, lw=1.4, label=disp)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(ylab, fontsize=10)
            ax.set_xlabel(xlab, fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = FIGDIR / "detection_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
