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

### Model progress

The first version trained on a small pilot dataset. Its validation score looked
great, but the test exposed the real problem: it did not generalize yet. The
current v9 run uses a much larger private dataset and a broader evaluation set.

| Run | Split | Menus | Item F1 | Structural F1 | Valid JSON |
|---|---|---:|---:|---:|---:|
| v1 LoRA, pilot dataset | Validation | 3 | **0.9808** | **0.8192** | 100% |
| v1 LoRA, pilot dataset | Test | 3 | 0.7167 | 0.6446 | 66.67% |
| v9 LoRA, expanded dataset | Validation | 13 | 0.7872 | 0.7119 | 92.31% |
| v9 LoRA, expanded dataset | Test | 12 | **0.8794** | **0.7242** | **100%** |
| Luna baseline | v9 test | 12 | 0.9766 | 0.7559 | 100% |

The pilot and v9 sets are different sizes, so this is project history rather
than a controlled benchmark. The direction is still encouraging: test item F1
moved from **0.7167 to 0.8794**, structural F1 from **0.6446 to 0.7242**, and
valid JSON from **66.67% to 100%**. Luna is still ahead, but the gap is smaller.

### Did faster inference hurt quality?

We also ran the same 48 robustness menus through the original inference path and
the new BF16 + FlashAttention 2 path.

| Robustness metric | Original | FA2 |
|---|---:|---:|
| Item F1 | **0.8212** | 0.8134 |
| OCR-line F1 | **0.8860** | 0.8743 |
| Structural F1 | **0.6780** | 0.3462 |
| Valid JSON | 91.67% | **93.75%** |

Item/OCR F1 and valid JSON stayed close, but structural F1 did not. We are not
treating the faster path as a free win until we understand that drop.

The full [v9 benchmark report](docs/v9-benchmark-report.md) keeps the inference
timings, hardware details, limitations, and next experiments.

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

## Next experiments

- **Feed it harder menus.** Add more dense, multilingual, multi-column, low-light,
  and unusual-price layouts instead of repeating the easy cases we already solve.
- **Turn mistakes into better training data.** Use recurring error patterns to
  collect and label new examples for missed headers, merged dishes, split dishes,
  notes, and section ownership.
- **Tune what LoRA can learn.** Sweep rank, alpha, learning rate, and target
  modules; compare language-only adaptation with the multimodal merger and a
  small, carefully unfrozen slice of the vision encoder.
- **Teach structure more directly.** Try curriculum training from simple to dense
  menus and give extra weight to section boundaries, item ownership, and valid
  JSON—not just token-level imitation.
- **Learn from the stronger baseline.** Explore human-checked Luna outputs on the
  training pool or newly collected unlabeled menus for distillation and
  pseudo-labeling, especially on difficult menu structure.
- **Try a stronger base.** Compare the current 4B checkpoint with newer or larger
  Qwen-VL variants and measure whether the quality gain is worth the extra GPU
  cost.
- **Compress the winner.** Once the best-quality checkpoint is chosen, test INT8
  and 4-bit weights and keep them only if item, OCR, and structural F1 survive.

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
