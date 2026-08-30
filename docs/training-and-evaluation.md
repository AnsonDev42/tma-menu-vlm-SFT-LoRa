# Training and evaluation

## Baseline

`configs/qwen3-vl-4b-lora.json` pins the model revision, package versions, seeds,
three-epoch ceiling, batch/accumulation settings, LoRA rank, and selection policy.
`preflight --no-download` validates these invariants against a checked-in Qwen3-VL
module profile on a Mac. Full preflight additionally checks installed dependency
versions and validates every configured suffix against the loaded model.

The first run uses BF16 LoRA. Scoped language selectors and the exact loaded
`model.visual.merger.linear_fc1/2` modules expand to exact module names before
they are passed to PEFT. A post-injection check refuses suffix collisions or any
unexpected target; `visual.blocks` and patch embedding remain frozen. VLM
truncation and packing are disabled (`max_length=null`) so image tokens cannot be
removed. Training uses the native `AutoProcessor`, TRL's conversational
prompt/completion VLM dataset/collator, and `completion_only_loss=true`; TRL's
unsupported VLM `assistant_only_loss` mode is disabled. Gradient checkpointing,
epoch validation/saves, one-epoch early stopping, and a hard failure on
non-finite train/eval loss remain enabled.

## Predict and validate checkpoints

For each saved checkpoint:

```bash
uv run menu-vlm predict \
  --config configs/qwen3-vl-4b-lora.json \
  --dataset /workspace/private-dataset \
  --split-file validation.jsonl \
  --adapter /workspace/run/training/checkpoints/checkpoint-N \
  --output /workspace/run/checkpoint-N.predictions.jsonl

uv run menu-vlm evaluate \
  --references /workspace/private-dataset/validation.jsonl \
  --predictions /workspace/run/checkpoint-N.predictions.jsonl \
  --output /workspace/run/checkpoint-N.metrics.json
```

Build `checkpoints.jsonl` with `checkpoint`, `structural_f1`, `item_f1`, and
`eval_loss`, then select without touching test:

```bash
uv run menu-vlm select-checkpoint --metrics /workspace/run/checkpoints.jsonl \
  --output /workspace/run/checkpoint-selection.json
```

The evaluator parses and schema-checks compact JSON, validates OCR references,
matches items/sections independent of array order and generated section IDs, and
reports parse/schema rates, OCR precision/recall/F1, item precision/recall/F1,
section precision/recall/F1, section assignment accuracy, normalized dish-name
accuracy, printed-note precision/recall/F1, normalized note-text accuracy,
note-owner accuracy, aggregate structural F1, merge/split errors, latency,
throughput, and peak memory. Note order is irrelevant; missing notes, wrong text,
or attaching a note to the wrong matched owner reduces structural F1. Robustness
uses the same metrics on its separate split. Pass saved
Luna predictions with `--luna-predictions` for an identical baseline report.
For the primary frozen test this argument is mandatory; the Runpod workflow never
falls back to a test report without the saved baseline.

## Consume frozen test once

After validation selects a checkpoint:

```bash
uv run menu-vlm evaluate-test \
  --references /workspace/private-dataset/test.jsonl \
  --predictions /workspace/run/test.predictions.jsonl \
  --dataset-sha256 DATASET_SHA256 \
  --checkpoint /workspace/run/selected-adapter \
  --gate-store /workspace/tma-menu-vlm/frozen-test-gates \
  --luna-predictions /workspace/run/luna-test-predictions.jsonl \
  --output /workspace/run/test.metrics.json
```

The fixed gate store is outside the training/evaluation output tree and writes one
gate keyed by dataset SHA-256, pinned model ID and
revision, and the hashes of the selected adapter config/weights. Replaying the
same identity with a different metrics output path or predictions file still
fails. The gate also records the required Luna prediction SHA-256, while keeping
the one-shot identity bound to the candidate adapter rather than offering a new
test attempt when a baseline file changes. A different adapter content hash
creates a different identity. Use validation—not frozen test—for iteration.

Mac inference is optional. The authoritative run is on the Runpod environment;
the 16 GB Mac path is limited to build/validation/preflight unless a compatible
quantized runtime is separately proven.
