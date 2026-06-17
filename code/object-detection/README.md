# Object detection

This stage localizes TB lesions on a chest X-ray and labels each as active or obsolete tuberculosis. It holds two detectors, both run through the shared engine `code/_mlx`, which wraps Ultralytics. YOLO26 is the base detector. DraxNet-YOLO26 swaps in a custom backbone that raises active-lesion accuracy.

### Models

| Folder | mlx model | Backbone |
| --- | --- | --- |
| `yolo26` | `yolo26` | base YOLO26 |
| `draxnet-yolo26` | `draxnet-yolo26` | custom DraxNet backbone |
| `yolov8n` | `yolov8n` | YOLOv8 nano baseline |
| `yolo11n` | `yolo11n` | YOLO11 nano baseline |

`yolov8n` and `yolo11n` are standard lightweight detection baselines. Like `yolo26`, the bare model name resolves to the nano scale (Ultralytics reports "Assuming scale='n'"). They follow the same from-scratch, 100-epoch, 512x512 protocol. Their `results/` are not yet populated.

### Running a model

Each folder has a `run.sh` that trains the detector and writes artifacts into its `results/`. Weights are not distributed, so the script retrains from scratch.

```bash
cd yolo26 && ./run.sh
```

The detectors read `../../../dataset/object-detection/dataset.yaml`, which defines two classes, `ActiveTuberculosis` and `ObsoletePulmonaryTuberculosis`, over a YOLO-format `images/` and `labels/` tree split into `train/` and `val/`.

The configs record 100 epochs, which is confirmed from the saved training logs. Batch size and image size reflect engine defaults and were not separately logged for the paper run, so the configs mark them as such. Treat them as the most likely settings rather than verified ones.

### Results

Each folder keeps its Ultralytics outputs in `results/`: `results.csv`, the per-class mAP tables (`per_class_map*.csv`), confusion matrices, and the precision, recall, F1, and loss curves. The debug batch previews from training are dropped to keep the folder small.
