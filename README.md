# TMA Menu VLM Fine-Tuning

Public tooling for building, training, and evaluating a private Qwen3-VL LoRA
adapter at TMA's current visual grouping seam:

```text
one menu image + numbered Azure OCR lines -> compact {"s": [...], "i": [...]}
```

The repository contains code, contracts, and generated synthetic fixtures only.
Real images, OCR, annotations, predictions, credentials, datasets, adapters, and
checkpoints must remain outside Git.

## Training TMA's own menu model

This repo is where we teach TMA to read menus with its own model. It turns our
private menu images and OCR into a reproducible training set, fine-tunes
Qwen3-VL with LoRA, and checks the result against Luna. The private menus and
model weights stay out of Git; the code, data contracts, evaluation tools, and
synthetic examples live here.

The latest experiment asked a practical question: how many menus can we process
at once without making the model worse? On an NVIDIA A40, BF16 + FlashAttention
2 at batch 16 finished menus at **1.85× the rate** of the old batch-1 run on the
48-example robustness set, reaching **36.95 aggregate decode tok/s**. Item F1
barely moved from **0.8212 to 0.8134**, while schema validity improved from
**91.67% to 93.75%**.

### Final benchmark

The clean test uses the same 12 menus for Luna, the original LoRA run, and the
new FA2 run. Luna is still the quality target; it is an API baseline, so there is
no local GPU speed number for it.

| Clean test | Luna | LoRA pre-FA | LoRA FA2, batch 2 |
|---|---:|---:|---:|
| Item F1 | **0.9766** | 0.8794 | 0.8049 |
| OCR-line F1 | **0.9619** | 0.9330 | 0.8515 |
| Structural F1 | **0.7559** | 0.7242 | 0.6944 |
| Valid JSON schema | 100% | 100% | 91.67% |
| Throughput | n/a | 16.82 tok/s | 20.73 aggregate tok/s |

The token rates are directional, not apples-to-apples: the old metric covers the
full `generate()` call, while the new metric isolates aggregate batch decoding.

The robustness test uses 48 harder, augmented menus. We have matched pre-FA and
FA2 runs for this set, but no Luna robustness run yet.

| Robustness test | LoRA pre-FA, batch 1 | LoRA FA2, batch 16 |
|---|---:|---:|
| Menus per minute | 0.92 | **1.70** |
| Item F1 | 0.8212 | 0.8134 |
| OCR-line F1 | 0.8860 | 0.8743 |
| Structural F1 | **0.6780** | 0.3462 |
| Valid JSON schema | 91.67% | **93.75%** |
| Request latency | 65.08 s | 563.86 s |

The FA2 batch-16 path completed **1.85× more menus per minute**, but every
individual menu waited longer for its batch. This is an observed execution-path
result, not an isolated batching or FlashAttention speedup: the runtime,
attention backend, batch size, and telemetry changed together. It was not a free
win either. Structural F1 fell sharply, so the next job is to find out whether
long outputs, padding, or batch-dependent decoding is scrambling section
structure. Faster is useful; faster and wrong is not.

The FA2 run kept the A40 around 94–99% utilized and peaked at 37.86 GiB of its
44.99 GiB VRAM during late decoding. The private Runpod session cost about
$0.38 before its final partial billing hour and was deleted after the artifacts
were retrieved. These are operator observations; the quality and timing numbers
above come from the saved aggregate metrics.

The full [v9 benchmark report](docs/v9-benchmark-report.md) has the methodology,
limitations, and next experiments.

## Model setup

- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Training: BF16 LoRA on the language layers and multimodal merger
- Vision encoder: frozen
- Dataset: 500 private menu records with fixed train, validation, test, and
  robustness splits
- Evaluation: greedy decoding against the same compact section/item JSON schema

See the upstream [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
[TRL VLM guidance](https://huggingface.co/docs/trl/sft_trainer#training-vision-language-models),
and [PEFT LoRA interface](https://huggingface.co/docs/peft/en/package_reference/lora).

## Local development and synthetic proof

Only the lightweight base/dev dependencies are installed on a Mac; no model is
downloaded:

```bash
uv sync --extra dev

scratch="$(mktemp -d)"
uv run menu-vlm synthetic-release --output "$scratch/release"
uv run menu-vlm validate-release --release "$scratch/release"
uv run menu-vlm compile --release "$scratch/release" --output "$scratch/compiled"
uv run menu-vlm validate-dataset --dataset "$scratch/compiled"
uv run menu-vlm preflight \
  --config configs/qwen3-vl-4b-lora.json \
  --dataset "$scratch/compiled" --no-download

uv run pytest
uv run ruff check .
uv run mypy src
./scripts/scan-public.sh
```

Repeated builds of the same release are byte-stable. The compiler copies images
into the generated external dataset, writes checksummed JSONL conversations, and
routes validation/test derivatives to separate robustness files. It never
resplits an immutable release.

## Private release

Set explicit external paths. `.env` is ignored and may hold local path/credential
configuration, but commands accept paths and secrets through CLI/environment too:

```bash
cp .env.example .env
export MENU_DATASET_PATH=/absolute/private/menu-silver-enhanced-release
export MENU_COMPILED_PATH=/absolute/private/qwen3-vl-sft-v1

uv run menu-vlm validate-release --release "$MENU_DATASET_PATH"
uv run menu-vlm compile --release "$MENU_DATASET_PATH" --output "$MENU_COMPILED_PATH"
uv run menu-vlm approve-luna-response-provenance \
  --dataset "$MENU_COMPILED_PATH" \
  --tma-data-root /absolute/private/tma-menu-parser-data \
  --evaluation-run COMPLETED_CURRENT_LUNA_RUN \
  --canonical-baseline /absolute/private/luna-test-predictions.jsonl \
  --output /absolute/private/luna-response-provenance.json
uv run menu-vlm import-luna-baseline \
  --dataset "$MENU_COMPILED_PATH" \
  --tma-data-root /absolute/private/tma-menu-parser-data \
  --evaluation-run COMPLETED_CURRENT_LUNA_RUN \
  --response-provenance /absolute/private/luna-response-provenance.json \
  --response-provenance-sidecar /absolute/private/luna-response-provenance.json.sha256 \
  --output /absolute/private/luna-test-predictions.jsonl
uv run menu-vlm package --source "$MENU_COMPILED_PATH" \
  --archive /absolute/private/qwen3-vl-sft-v1.tar.gz
```

The contracts are in [docs/release-and-dataset.md](docs/release-and-dataset.md).

## Train and evaluate

The heavyweight environment is intended for a Linux GPU pod:

```bash
uv sync --extra train --frozen --no-install-package flash-attn
uv sync --extra train --frozen
uv run menu-vlm preflight --config configs/qwen3-vl-4b-lora.json \
  --dataset "$MENU_COMPILED_PATH"
uv run menu-vlm train --config configs/qwen3-vl-4b-lora.json \
  --dataset "$MENU_COMPILED_PATH" --output /workspace/tma-menu-vlm/run/training
```

The frozen training extra pins the Torchvision backend required by Qwen3-VL's
image/video processor alongside the pinned Torch runtime. Runpod bootstrap
imports both Qwen3-VL processor classes before touching private training data or
loading model weights.

Executable prediction/evaluation commands are in
[docs/training-and-evaluation.md](docs/training-and-evaluation.md). Selection is
validation-only: structural F1 (including printed note text and ownership), item
F1, then loss. `evaluate-test` requires the saved Luna baseline and uses a fixed
gate store keyed to dataset, pinned model revision, and adapter content; the gate
records the baseline hash, and changing an output filename cannot consume the
same frozen test twice.

## Runpod and returned artifacts

[docs/runpod.md](docs/runpod.md) covers authenticated launch, checksum transfer,
detached training, monitoring, retrieval, verification, and cleanup. Scripts
require explicit paths and obtain secrets only from the environment or existing
CLI configuration.

For a step-by-step repeat run with a new private release such as v9, use
[docs/rerun-private-training.md](docs/rerun-private-training.md). It covers
immutable-release validation, a fresh Luna baseline, bounded Runpod execution,
validation-only iteration, the one-shot primary-test gate, returned-adapter
reconstruction, and secure cleanup.

The adapter-only return contract is in [docs/artifact-bundle.md](docs/artifact-bundle.md).
The base model is never merged into the bundle.
