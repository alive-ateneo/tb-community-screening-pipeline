"""Parameter and multiply-add (MAC) counts for the classification models.

Reports, for every classifier in the paper, the trainable parameter count and
the number of multiply-add operations (MACs, a.k.a. Mult-Adds) for a single
forward pass at the training resolution. We report MACs rather than FLOPs to
match the convention used by the lightweight backbones we compare against
(MobileNetV3, ShuffleNet), where one fused multiply-add counts as one
operation. FLOPs under the "2 ops per multiply-add" convention are simply
twice these values. Counts are computed with fvcore, which counts one
multiply-add as one operation, so the numbers below are MACs.

Inputs: 1 x 3 x 512 x 512 (the training resolution), 3 output classes,
randomly initialised weights (parameter and MAC counts do not depend on the
trained values).

Dependencies: torch, torchvision, fvcore. No GPU and no trained checkpoints
are required. The model definitions are imported directly from the repo so the
training-only dependencies (opencv, rich, matplotlib, ...) are not needed; the
small shim below registers the `mlx` namespace packages by path so the leaf
model modules import without executing the heavier package __init__ chain.

Run from the repo root:
    python code/model-complexity/params_flops_classification.py
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[2]
MLX_ROOT = REPO_ROOT / "code" / "_mlx"
FLIPR_SRC = REPO_ROOT / "code" / "classification" / "flipr" / "src"

INPUT_SHAPE = (1, 3, 512, 512)  # training resolution, 3-channel CXR
NUM_CLASSES = 3


def _register_namespace(name: str, path: Path) -> None:
    """Expose `name` as a package rooted at `path` without running its __init__.

    The model definitions use absolute imports (mlx.core.exceptions,
    mlx.modes.image_classification.models.blocks). Pre-seeding sys.modules with
    path-only namespace packages lets those imports resolve the leaf modules we
    need while skipping the training-only side effects in the real __init__.py
    files (opencv, rich, matplotlib).
    """
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules.setdefault(name, module)


def _setup_imports() -> None:
    for namespace, rel in [
        ("mlx", "mlx"),
        ("mlx.core", "mlx/core"),
        ("mlx.modes", "mlx/modes"),
        ("mlx.modes.image_classification", "mlx/modes/image_classification"),
        ("mlx.modes.image_classification.models", "mlx/modes/image_classification/models"),
    ]:
        _register_namespace(namespace, MLX_ROOT / rel)
    sys.path.insert(0, str(FLIPR_SRC))


def build_models() -> dict:
    _setup_imports()

    draxnet = importlib.import_module("mlx.modes.image_classification.models.draxnet")
    drax_mobilenet = importlib.import_module("mlx.modes.image_classification.models.drax_mobilenet")
    lighttbnet = importlib.import_module("mlx.modes.image_classification.models.lighttbnet")
    standard = importlib.import_module("mlx.modes.image_classification.models.standard")
    from tb_classifier.models.classifier import TBClassifier  # FlipR

    def std(name):
        return standard.build_standard_model(
            name, num_classes=NUM_CLASSES, colored=True, pretrained=False, config={}
        )

    # Ordered roughly by parameter count (smallest first).
    return {
        "LightTBNet": lambda: lighttbnet.build_lighttbnet(
            num_classes=NUM_CLASSES, colored=True, pretrained=False, config={}
        ),
        "FlipR": lambda: TBClassifier(num_classes=NUM_CLASSES, pretrained=False),
        "EfficientNet-B0": lambda: std("efficientnet_b0"),
        "MobileNetV3-Large": lambda: std("mobilenet_v3_large"),
        "Drax-MobileNetV3-Large": lambda: drax_mobilenet.build_drax_mobilenet_v3_large(
            num_classes=NUM_CLASSES, colored=True, pretrained=False, config={}
        ),
        "DenseNet-121": lambda: std("densenet121"),
        "ResNet-18": lambda: std("resnet18"),
        "DraxNet": lambda: draxnet.build_draxnet(
            num_classes=NUM_CLASSES, colored=True, pretrained=False, config={}
        ),
        "ResNet-50": lambda: std("resnet50"),
        "ConvNeXt-Tiny": lambda: std("convnext_tiny"),
    }


def main() -> None:
    import torch
    from fvcore.nn import FlopCountAnalysis

    x = torch.randn(*INPUT_SHAPE)
    print(f"Classification complexity at input {tuple(INPUT_SHAPE)}, {NUM_CLASSES} classes")
    print(f"{'Model':24} {'Params (M)':>11} {'MACs (G)':>10} {'Mult-Adds/img (M)':>18}")
    print("-" * 66)
    for name, factory in build_models().items():
        model = factory().eval()
        params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            fca = FlopCountAnalysis(model, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            macs = fca.total()  # fvcore counts one multiply-add as one operation
        print(f"{name:24} {params / 1e6:11.3f} {macs / 1e9:10.3f} {macs / 1e6:18.1f}")
    print("\nMACs = multiply-adds (one fused multiply-add = one op). FLOPs = 2 x MACs.")


if __name__ == "__main__":
    main()
