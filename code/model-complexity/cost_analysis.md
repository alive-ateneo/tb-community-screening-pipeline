# Deployment Cost Analysis: Pipeline vs Commercial TB CAD

Companion to the on-device benchmark (`benchmark_ondevice.py`,
`ondevice_benchmark_classification.csv`, `ondevice_benchmark_detection.csv`).
Deployed configuration: FlipR classifier + YOLOv8n detector (the detector runs
only on flagged images, so per-study figures below are an upper bound).

## 1. Compute cost of the pipeline (derived, reproducible)

Single-thread combined latency on the benchmark host (Intel Core i7-8750H,
batch 1, 512x512, no GPU):

    FlipR 244.3 ms + YOLOv8n 124.3 ms = 368.6 ms / study  -> 2.71 studies/s

| Config | Latency/study | Throughput | vCPU-hours / 1000 studies |
|---|---|---|---|
| Single-thread (1 vCPU, cost-optimal) | 368.6 ms | 2.71/s | 0.102 |
| 6-thread (6 vCPU, latency-optimal)   | 121.7 ms | 8.22/s | 0.203 |

Single-thread is cost-optimal: the 6-thread run is 3.0x faster in latency but
uses 2.0x more vCPU-hours (sub-linear scaling). For batch screening, parallelize
across studies, not within one.

At a commodity cloud CPU rate (~$0.045/vCPU-hour, e.g. AWS c7i):
- Raw compute ~ **$0.0046 per 1000 studies** (under one US cent / 1000).
- With 10x real-world overhead, still under **$0.05 / 1000 studies**.

Loose external floors (official pricing, accessed 2026-06-22), generic vision
APIs and NOT TB CAD, for order of magnitude only:
- Google Cloud Vision: $1.50 / 1000 images
- AWS Rekognition: $0.25 - 1.00 / 1000 images

## 2. Worked scenario: 100,000-study/year ACF program

100,000 x 0.3686 s = 10.2 vCPU-hours of compute per year.

| Deployment | One-time | Recurring/year | Notes |
|---|---|---|---|
| Cloud CPU ($0.045/vCPU-hr) | -- | ~$0.46 compute | Bill dominated by instance uptime, not inference |
| Edge device (offline, ~$300 mini-PC, 3-yr life) | $300 | ~$100 amortized + ~$7 power = ~$107 | Single-thread capacity ~234k studies/day, one device covers the program many times over |
| Commercial, volume-priced (qXR/Lunit, ~$1.40/screen mid-range) | -- | ~$140,000 | Per-screen license scales linearly with volume |
| Commercial, perpetual bundle (CAD4TB GDF) | ~$16,700 | support/renewal | Cheap per-screen at high volume, fixed entry cost + vendor lock |

The contrast is structural: the pipeline's marginal cost per study is effectively
zero and capacity is hardware-bound, whereas volume-priced commercial CAD scales
linearly with screening volume. Field ACF already runs CAD on a laptop the team
owns, so the realistic marginal cost is the license, not the compute.

## 3. Commercial cost evidence: what is citable

No commercial TB CAD vendor publishes a list price (confirmed; Bashir et al. 2022
states prices are negotiable and were "until recently not publicly disclosed").

Firmest public figures, Stop TB Partnership Global Drug Facility concessional
bundles (2021): CAD4TB ~$16,700, InferRead ~$5,000 (perpetual license + hardware
+ install + training + 1yr support). qXR and Lunit are not in the catalogue.
VERIFY against the live GDF Diagnostics catalogue before citing (research fetch
was TLS-blocked; figures trace to the 2021 announcement).

Multi-product per-screen model (Bashir et al. 2022, PLOS ONE, Pakistan, USD
FY20-21), the only true cross-product comparison:

| Product | Per-screen (ACF) | License model |
|---|---|---|
| InferRead | $0.19 - 2.73 | Perpetual |
| CAD4TB | $0.25 - 2.28 | Perpetual |
| Lunit | $0.94 - 1.64 | Volume-capped |
| qXR | $0.95 - 1.85 | Volume-capped |
| Radiologist | $0.93 | -- |

Empirical field cost-per-case (single-product, single-country, NOT mutually
comparable, different denominators/years/perspectives):
- qXR Nigeria: $9,000/yr license + $75,000 X-ray device (Garg et al. 2025, PLOS Dig Health)
- CAD4TB Zambia: $13.31/screen (Jo et al. 2021, PLOS ONE)
- qXR India: ~$0.36/case interpreted, cost-saving vs radiologist (Raval et al. 2025, Front Digit Health)

## 4. Honesty boundaries (do NOT)

- Do NOT put pipeline compute cost and commercial per-screen cost in one
  head-to-head table: different denominators, base years, currencies, perspectives.
- Do NOT claim a lower unit price for the same service. Across all sources,
  compute is never the cost driver; license + imaging hardware + program logistics
  dominate. The pipeline's advantage is structural (no per-study license,
  CPU/edge-deployable, ~$0 marginal cost), not a cheaper unit price.
- Do NOT cite accuracy papers (Qin, Worodria) for cost: they contain none.
- Do NOT cite WHO normative documents for product-level prices: they carry none.

## References

- Bashir et al. 2022, PLOS ONE 17(12):e0277393, doi:10.1371/journal.pone.0277393
- Jo et al. 2021, PLOS ONE 16(9):e0256531, doi:10.1371/journal.pone.0256531
- Garg et al. 2025, PLOS Digital Health 4(6):e0000894, doi:10.1371/journal.pdig.0000894
- Raval et al. 2025, Frontiers in Digital Health 7:1629127, doi:10.3389/fdgth.2025.1629127
- Stop TB Partnership GDF, 2021 (concessional bundle prices; verify current catalogue)
- Cloud pricing: AWS c7i / Rekognition, Google Cloud Vision pricing pages (accessed 2026-06-22)
