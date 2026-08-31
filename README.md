# TMA Menu VLM Fine-Tuning

Public tooling for building, training, and evaluating a private Qwen3-VL LoRA
adapter at TMA's current visual grouping seam:

```text
one menu image + numbered Azure OCR lines -> compact {"s": [...], "i": [...]}
```

The repository contains code, contracts, and generated synthetic fixtures only.
Real images, OCR, annotations, predictions, credentials, datasets, adapters, and
checkpoints must remain outside Git.

## What is pinned

- Base: `Qwen/Qwen3-VL-4B-Instruct`
- Revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Prompt: `menu-v2-vision-v1`, byte-for-byte TMA production system/user shape;
  the 4,284-byte system prompt includes exactly one terminal LF and has SHA-256
  `c4c4466bbdf49eb066bab6486bd9c9a0bf9230aeafb2da60b0ab02cd617fa476`
- Training: seeded BF16 LoRA; one controlled 4-bit QLoRA fallback after OOM
- Vision policy: vision blocks frozen; loaded language attention/MLP and exact
  multimodal merger modules are resolved to exact PEFT targets
- Split policy: preserve an immutable release's assignments; only explicitly
  unsplit generic input may be split, deterministically by restaurant with
  canonical-source fallback

The model revision is an Apache-2.0 4B vision-language checkpoint. The training
implementation follows the model's native processor/chat-template interface and
TRL's VLM dataset contract. See the upstream [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
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
detached training, monitoring, retrieval, verification, cleanup, and the hard
`$15` ceiling. Scripts require explicit paths and obtain secrets only from the
environment or existing CLI configuration.

For a step-by-step repeat run with a new private release such as v9, use
[docs/rerun-private-training.md](docs/rerun-private-training.md). It covers
immutable-release validation, a fresh Luna baseline, bounded Runpod execution,
validation-only iteration, the one-shot primary-test gate, returned-adapter
reconstruction, and secure cleanup.

The adapter-only return contract is in [docs/artifact-bundle.md](docs/artifact-bundle.md).
The base model is never merged into the bundle.
