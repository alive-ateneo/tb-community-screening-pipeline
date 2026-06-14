"""Compute paper-format classification table from confusion matrices.

Uses the same numpy-based per-class precision/recall/f1 logic as
confusion_matrix_calc.py, extended to:
 - load each model's confusion_matrix.csv (or reconstruct flipnet's CM
   from per-class metrics in its screenshot)
 - read AUC values from metrics.csv
 - compute paper-style Sensitivity (TB recall) and Specificity (recall
   for combined non-TB, i.e. healthy + sick non-TB)
 - print a LaTeX-ready row per model
"""
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "classification"

CLASSES = ["healthy", "sick-non-tb", "tb"]
TB_IDX = 2

# Display name and parameter count (millions) used in the paper. Counts are the trained
# 3-class-head models (1000-class ImageNet classifier replaced by a 3-class head),
# matching the paper's classification table and appendix.
MODELS = [
    ("flipnet",                       "FlipR",                       2.8),
    ("efficientnet-b0",               "EfficientNet-B0",             4.0),
    ("mobilenetv3-large",             "MobileNetV3-Large",           4.2),
    ("draxnet-on-mobilenetv3-large",  "Drax-MobileNetV3-Large",      4.8),
    ("densenet121",                   "DenseNet-121",                7.0),
    ("resnet18",                      "ResNet-18",                  11.2),
    ("draxnet",                       "DraxNet",                    16.5),
    ("resnet50",                      "ResNet-50",                  23.5),
    ("convnext-tiny",                 "ConvNeXt-Tiny",              27.8),
]


def load_cm_csv(model_dir: Path) -> np.ndarray:
    """Load a 3x3 confusion matrix CSV (rows=actual, cols=predicted)."""
    candidates = [
        model_dir / "confusion_matrix.csv",
        model_dir / "confusion_matrix (1).csv",
    ]
    for path in candidates:
        if path.exists():
            rows = []
            with path.open() as fh:
                reader = csv.reader(fh)
                header = next(reader)
                col_order = header[1:]
                index = [col_order.index(c) for c in CLASSES]
                row_buf = {}
                for row in reader:
                    row_buf[row[0]] = [int(row[1 + i]) for i in index]
            cm = np.array([row_buf[c] for c in CLASSES], dtype=int)
            return cm
    raise FileNotFoundError(f"No confusion_matrix.csv in {model_dir}")


def reconstruct_flipnet_cm() -> np.ndarray:
    """Reconstruct flipnet (FlipR) CM from per-class metrics in the screenshot.

    Screenshot (Per-class) values:
      healthy:     support 570, sens 0.99649 -> TP=568, spec 0.99565 -> FP=3
      sick-non-tb: support 570, sens 0.99298 -> TP=566, spec 0.98696 -> FP=9
      tb:          support 120, sens 0.94167 -> TP=113, spec 0.99912 -> FP=1
    Solving the linear system over the off-diagonal cells gives a unique CM.
    """
    cm = np.array([
        [568,   2,   0],
        [  3, 566,   1],
        [  0,   7, 113],
    ], dtype=int)
    return cm


def per_class_metrics(cm: np.ndarray) -> dict:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) == 0, 0, tp / (tp + fp))
        recall = np.where((tp + fn) == 0, 0, tp / (tp + fn))
        f1 = np.where((precision + recall) == 0, 0,
                      2 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def paper_metrics(cm: np.ndarray) -> dict:
    """Compute the metrics reported in the paper.

    sensitivity = recall for TB class
    specificity = recall for the combined non-TB class (paper definition)
    """
    m = per_class_metrics(cm)
    accuracy = np.diag(cm).sum() / cm.sum()
    sens_tb = m["recall"][TB_IDX]
    # Combined non-TB: collapse healthy + sick-non-tb into a single "non-tb"
    # class and compute recall for that aggregated class.
    non_tb_rows = [i for i in range(len(CLASSES)) if i != TB_IDX]
    non_tb_correct = sum(cm[i, j] for i in non_tb_rows for j in non_tb_rows)
    non_tb_total = sum(cm[i].sum() for i in non_tb_rows)
    spec_tb = non_tb_correct / non_tb_total
    avg_prec = m["precision"].mean()
    avg_rec = m["recall"].mean()
    avg_f1 = m["f1"].mean()
    return {
        "accuracy": accuracy,
        "sens_tb": sens_tb,
        "spec_tb": spec_tb,
        "avg_prec": avg_prec,
        "avg_rec": avg_rec,
        "avg_f1": avg_f1,
        "per_class": m,
    }


def read_auc_tb(model_dir: Path) -> float | None:
    metrics_path = model_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0] == "auc_tb":
                return float(parts[1])
    # flipnet has a different format; parse free-form text
    with metrics_path.open() as fh:
        text = fh.read()
    if "AUROC" in text:
        for line in text.splitlines():
            if "AUROC" in line:
                return float(line.split("-")[-1].strip())
    return None


def main():
    print(f"{'Model':28s} {'AUC_TB':>7s} {'Acc':>7s} {'Sens':>7s} {'Spec':>7s} "
          f"{'AvgPrec':>8s} {'AvgRec':>8s} {'AvgF1':>7s}")
    print("-" * 92)
    rows = []
    for slug, display, params_m in MODELS:
        model_dir = ROOT / slug
        if slug == "flipnet":
            cm = reconstruct_flipnet_cm()
            auc_tb = 0.9984  # from screenshot Per-class table for tb auroc
        else:
            cm = load_cm_csv(model_dir)
            auc_tb = read_auc_tb(model_dir)
        pm = paper_metrics(cm)
        print(f"{display:28s} {auc_tb:.4f}  "
              f"{pm['accuracy']*100:6.2f}  {pm['sens_tb']*100:6.2f}  "
              f"{pm['spec_tb']*100:6.2f}  "
              f"{pm['avg_prec']*100:7.2f}  {pm['avg_rec']*100:7.2f}  "
              f"{pm['avg_f1']*100:6.2f}")
        rows.append((display, params_m, auc_tb, pm))

    # LaTeX rows
    print()
    print("% LaTeX rows for paper Table 2")
    for display, params_m, auc_tb, pm in rows:
        params_str = f"{params_m:.1f}M"
        print(f"{display:28s} & {auc_tb:.4f} & "
              f"{pm['accuracy']*100:.2f} & {pm['sens_tb']*100:.2f} & "
              f"{pm['spec_tb']*100:.2f} & "
              f"{pm['avg_prec']*100:.2f} & {pm['avg_rec']*100:.2f} & "
              f"{pm['avg_f1']*100:.2f} & {params_str} \\\\")


if __name__ == "__main__":
    main()
