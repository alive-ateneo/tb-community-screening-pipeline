"""Parameter and multiply-add (MAC) counts for the detection models.

Reports, for the two detectors in the paper (YOLO26 and YOLO26 with the
DraxNet backbone), the trainable parameter count and the number of
multiply-add operations (MACs, a.k.a. Mult-Adds) for a single forward pass at
the training resolution.

We report MACs, counting one fused multiply-add as one operation, so that the
detection numbers use the same convention as the classification table. Note
that Ultralytics' own `model.info()` prints "GFLOPs" that are already doubled
(GFLOPs = 2 x MACs, verified in ultralytics.utils.torch_utils.get_flops), so
those values are twice the MACs reported here. We count MACs directly with
fvcore to keep one convention across both modules.

Inputs: 1 x 3 x 512 x 512 (the training resolution), nc = 2 detection classes
(ActiveTuberculosis, ObsoletePulmonaryTuberculosis), randomly initialised
weights. The model graphs are built from the same YAML configs the training
pipeline resolves (`yolo26.yaml`, `draxnet-yolo26.yaml`) in the ralampay
ultralytics fork that this repo pins.

Dependencies: torch, fvcore, and the pinned ultralytics fork
(`ultralytics @ git+https://github.com/ralampay/ultralytics`). No GPU and no
trained checkpoints are required.

Run from the repo root:
    python code/model-complexity/params_flops_detection.py
"""

from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)

IMG = 512  # training resolution
NC = 2  # ActiveTuberculosis, ObsoletePulmonaryTuberculosis

# (display name, model YAML the pipeline resolves via MODEL_ALIASES)
#
# Note: yolov8.yaml and yolo11.yaml are the standard lightweight detection
# baselines added under code/object-detection/{yolov8n,yolo11n}. Like yolo26,
# the bare base YAML carries no n/s/m/l/x suffix, so Ultralytics resolves the
# nano head ("Assuming scale='n'"); the scale column reports this as n*. They
# train from scratch under the same protocol, so their complexity belongs in
# the same table once their runs are populated.
MODELS = [
    ("YOLO26", "yolo26.yaml"),
    ("YOLO26 + DraxNet", "draxnet-yolo26.yaml"),
    ("YOLOv8n", "yolov8.yaml"),
    ("YOLO11n", "yolo11.yaml"),
]


def main() -> None:
    import torch
    from fvcore.nn import FlopCountAnalysis
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    x = torch.randn(1, 3, IMG, IMG)
    print(f"Detection complexity at input (1, 3, {IMG}, {IMG}), nc={NC}")
    print(f"{'Model':20} {'scale':>6} {'Params (M)':>11} {'MACs (G)':>10} {'Ultra GFLOPs':>13}")
    print("-" * 64)
    for name, cfg in MODELS:
        yolo = YOLO(cfg, task="detect")
        net = yolo.model  # DetectionModel (nn.Module)
        # Rebuild the head for nc=2 if the YAML default differs.
        if getattr(net, "nc", None) != NC:
            net = YOLO(cfg, task="detect").model
        net.eval()
        # Bare "yolo26"/"draxnet-yolo26" (no n/s/m/l/x suffix) resolves to nano:
        # Ultralytics warns "no model scale passed. Assuming scale='n'".
        scale = (net.yaml.get("scale") or "n*") if isinstance(getattr(net, "yaml", None), dict) else "n*"
        params = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            fca = FlopCountAnalysis(net, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            macs = fca.total()  # MACs (one multiply-add = one op)
        ultra_gflops = get_flops(net, IMG)  # Ultralytics convention = 2 x MACs
        print(f"{name:20} {str(scale):>6} {params / 1e6:11.3f} {macs / 1e9:10.3f} {ultra_gflops:13.3f}")
    print("\nMACs = multiply-adds (one fused multiply-add = one op). FLOPs = 2 x MACs.")
    print("Ultralytics GFLOPs already apply the x2 convention, so they should be ~2 x MACs (G).")
    print("scale 'n*' = nano assumed: bare 'yolo26'/'draxnet-yolo26' pass no n/s/m/l/x suffix.")


if __name__ == "__main__":
    main()
