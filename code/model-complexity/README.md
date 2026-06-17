# Model complexity: parameters and multiply-adds

Computes the trainable parameter count and the multiply-add (MAC) count for
every model in the paper, for both the classification and detection modules.
The counts are static (architecture only): no GPU, no trained checkpoints, and
no dataset are required.

## Convention

We report **MACs (multiply-adds, a.k.a. Mult-Adds)**, counting one fused
multiply-add as one operation. This matches the convention used by the
lightweight backbones we compare against (MobileNetV3, ShuffleNet). The
relationship to FLOPs is:

    FLOPs = 2 x MACs

This factor of two is a common source of confusion. Two things to keep in
mind:

- Ultralytics' own `model.info()` prints "GFLOPs" that are **already doubled**
  (verified in `ultralytics.utils.torch_utils.get_flops`, which multiplies the
  thop MAC count by 2). So an Ultralytics "GFLOPs" value is `2 x MACs`, twice
  the MACs reported here. The detection script prints both columns so the
  relationship is visible.
- MAC counters (fvcore, thop, ptflops) all count one multiply-add as one
  operation, but some label the column "FLOPs". We use fvcore for both modules
  and call the result MACs, which is what it actually counts.

All counts use the training resolution `512 x 512`.

## Scripts

| Script | Models | Extra dependency |
| --- | --- | --- |
| `params_flops_classification.py` | FlipR, EfficientNet-B0, MobileNetV3-Large, Drax-MobileNetV3-Large, DenseNet-121, ResNet-18, DraxNet, ResNet-50, ConvNeXt-Tiny | none beyond torch/torchvision/fvcore |
| `params_flops_detection.py` | YOLO26, YOLO26 + DraxNet | the pinned `ultralytics` fork |

## Reproduce

```bash
# from the repo root
uv venv .venv-complexity --python 3.11
source .venv-complexity/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install fvcore
python code/model-complexity/params_flops_classification.py

# detection additionally needs the pinned ultralytics fork
uv pip install "ultralytics @ git+https://github.com/ralampay/ultralytics"
python code/model-complexity/params_flops_detection.py
```

The classification script imports the model definitions directly from the repo
(`code/_mlx` and `code/classification/flipr/src`) and does not require the
training-only dependencies (opencv, rich, matplotlib). The detection script
builds the two detectors from the same YAML configs the training pipeline
resolves (`yolo26.yaml`, `draxnet-yolo26.yaml`).

## Notes

- Detection class count is `nc = 2` (`ActiveTuberculosis`,
  `ObsoletePulmonaryTuberculosis`), from
  `dataset/object-detection/dataset.yaml`.
- The committed detection config passes the bare model name `yolo26` /
  `draxnet-yolo26` with no `n/s/m/l/x` scale suffix, so Ultralytics resolves
  the **nano** head ("no model scale passed. Assuming scale='n'"). The script
  reports this scale as `n*` to flag that it was assumed, not explicit. For
  YOLO26 + DraxNet, only the YOLO head scales; the DraxNet backbone is fixed.
- Parameter counts reproduce the values stated in the manuscript
  (FlipR 2.8M, EfficientNet-B0 4.0M, MobileNetV3-Large 4.2M,
  Drax-MobileNetV3-Large 4.8M, DraxNet 16.5M).
