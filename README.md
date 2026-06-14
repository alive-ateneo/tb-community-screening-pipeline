# Lightweight Tuberculosis Screening with Natural Language Explanations for Active Case Finding

A lightweight chest X-ray pipeline for community tuberculosis (TB) screening that pairs compact classification and localization models with a plain-language description of their findings, built for PyTorch and Ultralytics.

Full paper: "Lightweight Tuberculosis Screening with Natural Language Explanations for Active Case Finding" (See manuscript folder)

### Overview

This pipeline targets community-based active case finding, where high-volume screening happens in remote locations with limited compute, limited connectivity, and no radiologist on site. The system reads each chest X-ray for presumptive TB, localizes suspicious lesions, and then states what the models found in language an on-site reader without radiology training can act on. Every component is kept small enough to run on modest hardware.

The pipeline has three stages. Classification introduces FlipR, a symmetry-gated network that matches the strongest baselines while using the fewest parameters. Localization adds a custom backbone that raises active-TB lesion accuracy over base YOLO26. Explanation turns the structured model output into a short report-style summary and checks that summary against the structured record.

---

### Models

This repository contains the following:

- **FlipR** (`code/classification/flipr`) - a 3-stage-ablated ResNet18 with a learned lateral-asymmetry gate after `layer2`, trained from scratch, the lightest classifier in the study at 2.8M parameters.
- **Classification baselines** (`code/classification/`) - DraxNet, Drax-MobileNetV3-Large, ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, MobileNetV3-Large, and ConvNeXt-Tiny, all trained through the shared engine.
- **YOLO26** (`code/object-detection/yolo26`) - the base detector for active and obsolete TB lesions.
- **DraxNet-YOLO26** (`code/object-detection/draxnet-yolo26`) - a custom-backbone detector that improves active-lesion accuracy over base YOLO26.
- **Explanation module** (`code/explanation`) - converts detector output into a report-style summary and applies a deterministic faithfulness check.

The eight non-FlipR models share one engine, `code/_mlx`, selected by a command-line flag. Each model folder holds only its run command, config, and results.

---

### Repository structure

```
.
├── code/
│   ├── _mlx/                  shared engine for classification and detection
│   ├── classification/
│   │   ├── flipr/             standalone FlipR module (own training stack)
│   │   ├── draxnet/           run.sh + config.yaml + results/
│   │   ├── resnet18/          (one folder per baseline)
│   │   └── ...
│   ├── object-detection/
│   │   ├── yolo26/
│   │   └── draxnet-yolo26/
│   └── explanation/           natural-language explanation module
├── dataset/
│   ├── classification/        train/ val/ test/ over {healthy, sick-non-tb, tb}/
│   └── object-detection/      images/ labels/ over {train, val}/ + dataset.yaml
├── manuscript/                LaTeX source for the paper
├── figures/
└── README.md
```

---

### Setup

1. Install Python 3.10 or newer.
2. Install each module independently, since the modules pin their own dependencies:
   - Shared engine: `pip install -r code/_mlx/requirements.txt`
   - FlipR: `cd code/classification/flipr && uv sync`
   - Explanation: `cd code/explanation && uv sync` (add the `inference` extra for the optional language model: `uv pip install -e ".[inference]"`)
3. Place the dataset as described below.

---

### Dataset

The study uses TBX11K, re-split into three classes. Place the data under `dataset/` in the layout the code expects:

```
dataset/
├── classification/
│   ├── train/   {healthy, sick-non-tb, tb}/*.png
│   ├── val/     {healthy, sick-non-tb, tb}/*.png
│   └── test/    {healthy, sick-non-tb, tb}/*.png
└── object-detection/
    ├── images/  {train, val}/*.png
    ├── labels/  {train, val}/*.txt        YOLO format
    └── dataset.yaml                         2 classes: ActiveTuberculosis, ObsoletePulmonaryTuberculosis
```

---

### Steps for training

Each model is reproduced by its own `run.sh`, which retrains from scratch and writes artifacts into that model's `results/`. All runs use seed 42 and select the best-validation checkpoint.

1. Train and benchmark a classification baseline through the shared engine:
   ```bash
   cd code/classification/resnet18 && ./run.sh
   ```
2. Train FlipR through its own stack:
   ```bash
   cd code/classification/flipr && ./run.sh
   ```
3. Train a detector:
   ```bash
   cd code/object-detection/yolo26 && ./run.sh
   ```
4. Generate an explanation from detector output (see `code/explanation/README.md`).

Each model folder records its settings in `config.yaml`.

---

### Citation

```bibtex
@article{morales2026lightweighttb,
  author  = {Morales, Irish Danielle and Alampay, Raphael B. and Saulog, Ruben A. and Abu, Patricia Angela R.},
  title   = {Lightweight Tuberculosis Screening with Natural Language Explanations for Active Case Finding},
  year    = {2026}
}
```

For questions, please contact [irish.morales@alumni.ateneo.edu](mailto:irish.morales@alumni.ateneo.edu).
