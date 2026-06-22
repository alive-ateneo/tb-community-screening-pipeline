"""On-device (CPU) runtime benchmark for the paper's models.

Companion to ``params_flops_classification.py`` / ``params_flops_detection.py``.
Those scripts report the static complexity (parameters and multiply-adds); this
one measures what a model costs to run on a CPU-only device, the setting an
active case-finding deployment without a GPU would face:

    - CPU latency   single-image forward-pass time (ms), median over many runs,
                    at both 1 thread (portable worst case) and N threads;
    - peak memory   peak resident set size of the inference process (MB),
                    reported absolute and net of the bare runtime baseline;
    - on-disk size  size of the committed trained checkpoint (MB), the artifact
                    a device would actually ship;
    - MACs          multiply-adds per image (G), recomputed here so the table is
                    self-contained (matches the params_flops_* scripts).

These runtime numbers are hardware dependent (CPU model, thread count, BLAS
kernels, framework), so they are NOT comparable across papers; what is
comparable is the static complexity. The value of this table is the
*internally controlled* comparison: every model is measured on the same host,
framework, and input, so the relative ordering (for example, that latency
tracks MACs, not parameters) is robust even though the absolute milliseconds
are specific to this host.

Timing and memory depend on the architecture and the float32 compute dtype, not
on the trained weight *values*, so models are built with random initialisation
(the same import paths as the params_flops_* scripts). The on-disk column is the
one number that needs the real checkpoint, and it is read off the committed file.
No GPU and no dataset are required.

Each model is benchmarked in its own subprocess so the peak-RSS reading
(``/proc/self/status`` VmHWM) is that model's alone and the CPU allocator does
not carry state between models. Run from the repo root with the mlx venv (it
provides torch, fvcore, and the pinned ultralytics fork):

    # classification models (default)
    code/_mlx/.venv/bin/python code/model-complexity/benchmark_ondevice.py

    # detection models
    code/_mlx/.venv/bin/python code/model-complexity/benchmark_ondevice.py --task detection
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

INPUT_SHAPE = (1, 3, 512, 512)  # training resolution, 3-channel CXR

WARMUP = 8
ITERS = 40
THREADS_MULTI = 6  # physical cores of the benchmark host (i7-8750H: 6C/12T)

CLS = REPO_ROOT / "code" / "classification"
OD = REPO_ROOT / "code" / "object-detection"

# Committed trained checkpoint per model: the deployable on-disk artifact.
# Classification: the eight mlx models save a pure float32 state_dict (.pth) and
# FlipR a Lightning checkpoint (.ckpt) carrying weights plus a little metadata,
# so all are ~= params x 4 bytes. Detection: Ultralytics saves best.pt in FP16,
# so detection sizes are ~= params x 2 bytes (half the classification rule).
CHECKPOINTS = {
    "classification": {
        "LightTBNet": CLS / "lighttbnet/results/lighttbnet.pth",
        "FlipR": CLS / "flipr/experiments/results/best.ckpt",
        "EfficientNet-B0": CLS / "efficientnet-b0/results/efficientnet_b0.pth",
        "MobileNetV3-Large": CLS / "mobilenetv3-large/results/mobilenet_v3_large.pth",
        "Drax-MobileNetV3-Large": CLS / "drax-mobilenetv3-large/results/drax_mobilenet_v3_large.pth",
        "DenseNet-121": CLS / "densenet121/results/densenet121.pth",
        "ResNet-18": CLS / "resnet18/results/resnet18.pth",
        "DraxNet": CLS / "draxnet/results/draxnet.pth",
        "ResNet-50": CLS / "resnet50/results/resnet50.pth",
        "ConvNeXt-Tiny": CLS / "convnext-tiny/results/convnext_tiny.pth",
    },
    "detection": {
        "YOLO26": OD / "yolo26/results/weights/best.pt",
        "YOLO11n": OD / "yolo11n/results/weights/best.pt",
        "YOLOv8n": OD / "yolov8n/results/weights/best.pt",
        "YOLO26 + DraxNet": OD / "draxnet-yolo26/results/weights/best.pt",
    },
}


def build_registry(task: str) -> dict:
    """Return {display name: zero-arg factory building the nn.Module}."""
    if task == "classification":
        from params_flops_classification import build_models
        return build_models()
    if task == "detection":
        from ultralytics import YOLO
        # Ordered to match the manuscript detection table.
        order = [
            ("YOLO26", "yolo26.yaml"),
            ("YOLO11n", "yolo11.yaml"),
            ("YOLOv8n", "yolov8.yaml"),
            ("YOLO26 + DraxNet", "draxnet-yolo26.yaml"),
        ]
        return {name: (lambda c=cfg: YOLO(c, task="detect").model) for name, cfg in order}
    raise ValueError(f"unknown task {task!r}")


def _peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB (Linux VmHWM)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024  # kB -> MB
    return float("nan")


def _time_samples(model, x, threads: int) -> list[float]:
    """Single-image forward-pass latencies (ms) at the given thread count."""
    import torch

    torch.set_num_threads(threads)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        samples = []
        for _ in range(ITERS):
            start = time.perf_counter()
            model(x)
            samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples


def _summarize(samples: list[float]) -> dict:
    n = len(samples)
    return {
        "median": samples[n // 2],
        "p10": samples[max(0, int(0.10 * n))],
        "p90": samples[min(n - 1, int(0.90 * n))],
    }


def benchmark_one(name: str, task: str) -> dict:
    """Measure one model in this process and return its row as a dict."""
    import torch
    from fvcore.nn import FlopCountAnalysis

    factory = build_registry(task)[name]
    x = torch.randn(*INPUT_SHAPE)

    model = factory().eval()
    params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        macs = fca.total()

    lat_1t = _summarize(_time_samples(model, x, threads=1))
    lat_nt = _summarize(_time_samples(model, x, threads=THREADS_MULTI))

    ckpt = CHECKPOINTS[task][name]
    disk_mb = ckpt.stat().st_size / 1e6 if ckpt.exists() else float("nan")

    return {
        "model": name,
        "params_m": params / 1e6,
        "macs_g": macs / 1e9,
        "disk_mb": disk_mb,
        "lat_1t_ms": lat_1t,
        "lat_nt_ms": lat_nt,
        "peak_rss_mb": _peak_rss_mb(),
    }


def benchmark_baseline(task: str) -> float:
    """Peak RSS (MB) of a process that loads the runtime and runs nn.Identity.

    Subtracting this from a model's peak RSS isolates the model's own weight +
    activation footprint from the fixed runtime cost. The detection runtime
    additionally imports ultralytics, so its baseline is measured with that
    import to keep the subtraction honest.
    """
    import torch

    if task == "detection":
        import ultralytics  # noqa: F401

    x = torch.randn(*INPUT_SHAPE)
    identity = torch.nn.Identity().eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            identity(x)
    return _peak_rss_mb()


def _emit(payload: dict) -> None:
    print("JSON " + json.dumps(payload))


def _parse_child(stdout: str) -> dict:
    """Recover the JSON payload from a child's stdout (ignores other noise)."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("JSON "):
            return json.loads(line[5:])
    raise RuntimeError("no JSON payload in child output")


def _run_child(args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"child failed: {' '.join(args)}")
    return _parse_child(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="classification", choices=["classification", "detection"])
    parser.add_argument("--model", help="benchmark one model in-process, emit JSON")
    parser.add_argument("--baseline", action="store_true", help="emit runtime baseline peak RSS")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.baseline:
        _emit({"baseline_rss_mb": benchmark_baseline(args.task)})
        return
    if args.model:
        _emit(benchmark_one(args.model, args.task))
        return

    # Driver: one subprocess per model (clean per-model peak RSS), plus baseline.
    baseline_rss = _run_child(["--task", args.task, "--baseline"])["baseline_rss_mb"]
    rows = [_run_child(["--task", args.task, "--model", name]) for name in build_registry(args.task)]

    import csv

    print(f"\nOn-device CPU benchmark ({args.task})  input {tuple(INPUT_SHAPE)}, batch 1, {ITERS} timed iters")
    print(f"Multi-thread column uses {THREADS_MULTI} CPU threads.")
    print(f"Bare runtime baseline peak RSS: {baseline_rss:.0f} MB\n")
    print(f"{'Model':24}{'Params(M)':>10}{'MACs(G)':>9}{'Disk(MB)':>9}"
          f"{'Lat@1T(ms)':>13}{f'Lat@{THREADS_MULTI}T(ms)':>13}{'PeakRSS(MB)':>12}{'NetRSS(MB)':>11}")
    print("-" * 101)
    for r in rows:
        l1, ln = r["lat_1t_ms"], r["lat_nt_ms"]
        net = r["peak_rss_mb"] - baseline_rss
        print(
            f"{r['model']:24}{r['params_m']:>10.2f}{r['macs_g']:>9.2f}{r['disk_mb']:>9.1f}"
            f"{l1['median']:>9.1f}±{(l1['p90']-l1['p10'])/2:>2.0f}"
            f"{ln['median']:>9.1f}±{(ln['p90']-ln['p10'])/2:>2.0f}"
            f"{r['peak_rss_mb']:>12.0f}{net:>11.0f}"
        )
    print("\nLatency = median single-image forward pass; ± is half the p10-p90 spread.")
    print("NetRSS = PeakRSS minus the bare runtime baseline (weights + activations).")
    print("MACs = multiply-adds (one fused multiply-add = one op). FLOPs = 2 x MACs.")

    csv_path = args.csv or str(HERE / f"ondevice_benchmark_{args.task}.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "params_m", "macs_g", "disk_mb",
            "lat_1t_median_ms", "lat_1t_p10_ms", "lat_1t_p90_ms",
            f"lat_{THREADS_MULTI}t_median_ms", f"lat_{THREADS_MULTI}t_p10_ms", f"lat_{THREADS_MULTI}t_p90_ms",
            "peak_rss_mb", "net_rss_mb", "baseline_rss_mb",
        ])
        for r in rows:
            l1, ln = r["lat_1t_ms"], r["lat_nt_ms"]
            writer.writerow([
                r["model"], f"{r['params_m']:.4f}", f"{r['macs_g']:.4f}", f"{r['disk_mb']:.3f}",
                f"{l1['median']:.3f}", f"{l1['p10']:.3f}", f"{l1['p90']:.3f}",
                f"{ln['median']:.3f}", f"{ln['p10']:.3f}", f"{ln['p90']:.3f}",
                f"{r['peak_rss_mb']:.1f}", f"{r['peak_rss_mb'] - baseline_rss:.1f}", f"{baseline_rss:.1f}",
            ])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
