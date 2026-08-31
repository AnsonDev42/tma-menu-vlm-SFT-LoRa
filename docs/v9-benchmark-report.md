# From private menu data to an auditable VLM benchmark

## Executive summary

This repository turns a production menu-understanding task into a reproducible
vision-language-model workflow: private dataset compilation, Luna baselines,
Qwen3-VL LoRA fine-tuning, evaluation, artifact handling, and GPU inference.

The biggest result is model progress. The pilot LoRA test produced **0.7167 item
F1**, **0.6446 structural F1**, and **66.67% valid JSON**. The expanded v9 run
reached **0.8794**, **0.7242**, and **100%** respectively. Luna remains the target
at **0.9766 item F1** and **0.7559 structural F1**.

We also tested BF16 + FlashAttention 2 on an NVIDIA A40. Item and OCR F1 stayed
close on the 48-example robustness set, but structural F1 fell from **0.6780 to
0.3462**. That quality drop matters more than the throughput gain and remains
the main inference follow-up.

The next work should improve the model first: add harder training examples, tune
the LoRA/vision adaptation boundary, and train more directly for menu structure.
A controlled FA2/batch matrix can then optimize the winning checkpoint.

## Model progress

The first LoRA run used a small pilot dataset; v9 used a much larger private
dataset and broader evaluation. Because the inventories differ, the table shows
the project's progression rather than a controlled experiment.

| Run | Split | Menus | Item F1 | Structural F1 | Valid JSON |
|---|---|---:|---:|---:|---:|
| Pilot LoRA | Validation | 3 | 0.9808 | 0.8192 | 100% |
| Pilot LoRA | Test | 3 | 0.7167 | 0.6446 | 66.67% |
| v9 LoRA | Validation | 13 | 0.7872 | 0.7119 | 92.31% |
| v9 LoRA | Test | 12 | 0.8794 | 0.7242 | 100% |
| Luna | v9 test | 12 | **0.9766** | **0.7559** | 100% |

## What was built

The work went beyond a one-off training script:

- A deterministic compiler converts one menu image plus numbered OCR lines into
  compact section/item JSON conversations suitable for multimodal SFT.
- Immutable split assignments prevent validation/test leakage; augmented
  derivatives stay attached to their canonical source split.
- Luna baselines are imported only with approved response provenance and bound
  to the exact dataset manifest.
- BF16 LoRA training freezes the vision tower while targeting the language
  attention/MLP layers and multimodal merger.
- Frozen test gates bind dataset, model revision, adapter, and baseline hashes so
  a renamed output cannot silently consume the primary test twice.
- Adapter-only bundles carry checksums, configuration, metrics, logs, hardware
  evidence, and provenance without publishing private images or model outputs.
- Batched inference records preprocessing, vision encoding, prefill, TTFT,
  decode duration, aggregate decode throughput, and peak memory.
- W&B observability runs offline with public configuration and aggregate metrics
  only; private payloads and paths are excluded.
- CUDA placement fails closed: CPU/disk/meta offload and a missing auditable
  tensor placement are rejected, as is any attention backend other than FA2 for
  the accelerated path.

## Experimental setup

| Component | Pinned value |
|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` |
| Model revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Dataset | v9, 500 records: 375 train / 13 validation / 12 test plus 52 / 48 robustness derivatives |
| Training | BF16 LoRA; frozen vision blocks |
| GPU | NVIDIA A40, 48 GB nominal VRAM |
| Pre-FA runtime | Torch 2.13.0, Transformers 5.16.1, default attention, batch 1 |
| FA2 runtime | Torch 2.10.0+cu128, official FlashAttention 2.8.3 wheel, BF16 |
| Decoding | Greedy, maximum 4,096 new tokens |

Torch 2.10 was used for the accelerated evaluation because an official prebuilt
FlashAttention wheel was available. INT8 was intentionally not executed.

## Frozen clean-test comparison

All three quality columns below use the same 12 canonical test menus. Luna is a
reference baseline and has no local latency/throughput telemetry.

| Metric | Luna baseline | LoRA pre-FA, batch 1 | LoRA FA2, batch 2 |
|---|---:|---:|---:|
| Item F1 | **0.9766** | 0.8794 | 0.8049 |
| OCR-line F1 | **0.9619** | 0.9330 | 0.8515 |
| Dish-name accuracy | **0.8044** | 0.6991 | 0.7857 |
| Section F1 | **0.7692** | 0.7576 | 0.6780 |
| Structural F1 | **0.7559** | 0.7242 | 0.6944 |
| Schema-valid rate | 100% | 100% | 91.67% |
| Parseable rate | 100% | 100% | 91.67% |
| Recorded throughput | n/a | 16.82 tok/s | 20.73 aggregate decode tok/s |
| Mean request latency | n/a | 55.79 s | 125.66 s |
| Amortized wall time / completed example | n/a | 55.79 s | 62.83 s |

The clean run does **not** support a claim that FA2 improved end-to-end speed or
preserved accuracy. Aggregate token throughput rose 1.23×, but effective
examples/second fell about 11%, request latency increased, and several quality
metrics regressed. Amortized wall time is mean batch wall time divided by batch
size; each request still waits for the full batch. The old throughput includes
the complete `generate()` call, while the new decode metric
separates TTFT and excludes the first generated token, so token-rate comparison
is directional. Output-length variation and batch padding also affect the result.

## Matched robustness comparison

The pre-FA and FA2 runs below both cover all 48 robustness derivatives. Luna was
not evaluated on this robustness inventory, so no Luna column is presented.

| Metric | LoRA pre-FA, batch 1 | LoRA FA2, batch 16 | Change |
|---|---:|---:|---:|
| Item F1 | 0.8212 | 0.8134 | -0.0078 |
| OCR-line F1 | 0.8860 | 0.8743 | -0.0117 |
| Dish-name accuracy | 0.8194 | 0.7990 | -0.0205 |
| Section F1 | 0.6320 | 0.6473 | +0.0153 |
| Structural F1 | 0.6780 | 0.3462 | -0.3318 |
| Schema-valid rate | 91.67% | 93.75% | +2.08 pp |
| Parseable rate | 91.67% | 93.75% | +2.08 pp |
| Recorded throughput | 16.51 tok/s | 36.95 aggregate decode tok/s | 2.24× directional |
| Mean request latency | 65.08 s | 563.86 s | 8.66× longer |
| Amortized wall time / completed example | 65.08 s | 35.24 s | **1.85× capacity** |

Batch 16 completed in three batches without OOM. During the live run, the
operator observed roughly 94–99% GPU utilization in `nvidia-smi`. Framework
telemetry reported a 28.29 GB peak; an operator-observed late-decode device
allocation reached **38,767 MiB (37.86 GiB) of 46,068 MiB (44.99 GiB)** as the
KV cache grew. These live observations were recorded during the private Runpod
session rather than derived from the published metric JSON. The late peak is the
safer capacity signal and argues against blindly increasing the batch further.

The item/OCR changes are small enough to motivate further testing, not to claim
equivalence. The structural-F1 drop is material and is the main quality concern.
It may reflect batching/length interactions, a runtime difference, or ordinary
generation variance across the changed execution path. Only a controlled rerun
can distinguish those explanations.

## What the result demonstrates

1. A 4B multimodal adapter can be trained and evaluated through a reproducible,
   provenance-aware workflow without committing private data or weights.
2. Batched FA2 inference can saturate an A40 and nearly double completed
   robustness examples per unit time at batch 16.
3. GPU occupancy alone is not the objective: TTFT was 9.96 seconds in the
   batch-16 robustness run, and long sequences made each batch wait for its
   slowest member.
4. Throughput and quality must be reported together. The current evidence shows
   a capacity win alongside a structural-quality regression that remains open.
5. Operational proof matters: the run used offline W&B telemetry, checksum-bound
   artifacts, a hard cost cap, and explicit pod teardown.

## Next steps

1. Collect and label harder training menus around recurring structural errors.
2. Tune LoRA rank, learning rate, target modules, and selective vision unfreezing.
3. Try curriculum and structure-weighted training for sections, ownership, and
   valid JSON.
4. Explore human-checked Luna distillation on training or newly collected data.
5. Compare stronger base models, then run the controlled FA2/batch sweep on the
   best-quality checkpoint.
6. Quantize that checkpoint to INT8 and 4-bit only if the complete F1 and valid
   JSON suite holds up.

## Reproducing the public workflow

See [training and evaluation](training-and-evaluation.md),
[private rerun procedure](rerun-private-training.md),
[Runpod operations](runpod.md), and the
[artifact-bundle contract](artifact-bundle.md). The repository publishes code,
contracts, aggregate metrics, and synthetic fixtures only; private images, OCR,
annotations, predictions, adapters, and credentials remain external.
