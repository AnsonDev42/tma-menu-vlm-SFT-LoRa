# From private menu data to an auditable VLM benchmark

## Executive summary

This repository turns a production menu-understanding task into a reproducible
vision-language-model workflow: immutable data releases, deterministic
compilation, provenance-bound Luna baselines, parameter-efficient Qwen3-VL
fine-tuning, frozen evaluation gates, private artifact handling, and instrumented
GPU inference.

The current v9 experiment fine-tuned `Qwen/Qwen3-VL-4B-Instruct` with a BF16
LoRA adapter, then tested batched inference with FlashAttention 2 on an NVIDIA
A40. On the matched 48-example robustness set, FA2 at batch 16 increased
end-to-end evaluation capacity from roughly **0.92 to 1.70 examples/minute**
(**1.85×**) while item F1 moved from **0.8212 to 0.8134**. The result is useful,
but not a clean attribution to FlashAttention: the runtime, attention backend,
batch size, and telemetry implementation changed together.

The strongest next experiment is therefore a controlled matrix on the same
examples: FA2 on/off × batch 1/2/4/8/16, with identical decoding and the current
phase-level telemetry.

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

1. Run a controlled FA2 on/off matrix over the same robustness examples and
   identical Torch/Transformers/decoding versions.
2. Sweep batch 1/2/4/8/16 and report examples/minute, aggregate tokens/second,
   TTFT, peak late-decode VRAM, and every quality metric.
3. Bucket menus by prompt size and expected output length so short generations
   do not wait behind 4,096-token outliers.
4. Inspect the structural-F1 failures by example and determine whether they come
   from section assignment, truncated JSON, padding, or batch-dependent decoding.
5. Compare the LoRA errors with Luna on the canonical test set and prioritize
   data/target changes where Luna retains a clear advantage.
6. Keep INT8 as a separate hardware-specific experiment rather than mixing it
   into the FA2/BF16 comparison.

## Reproducing the public workflow

See [training and evaluation](training-and-evaluation.md),
[private rerun procedure](rerun-private-training.md),
[Runpod operations](runpod.md), and the
[artifact-bundle contract](artifact-bundle.md). The repository publishes code,
contracts, aggregate metrics, and synthetic fixtures only; private images, OCR,
annotations, predictions, adapters, and credentials remain external.
