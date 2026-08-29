# Training and evaluation

## Baseline

`configs/qwen3-vl-4b-lora.json` pins the model revision, package versions, seeds,
three-epoch ceiling, batch/accumulation settings, LoRA rank, and selection policy.
`preflight --no-download` validates these invariants against a checked-in Qwen3-VL
module profile on a Mac. Full preflight additionally checks installed dependency
versions and validates every configured suffix against the loaded model.

The first run uses BF16 LoRA. Only language attention/MLP and
`visual.merger.linear_fc*` match adapter targets; `visual.blocks` and patch
embedding are forbidden. PEFT freezes all unadapted base parameters. VLM
truncation and packing are disabled (`max_length=null`) so image tokens cannot be
removed. Training uses the native `AutoProcessor`, assistant-only loss,
gradient checkpointing, epoch validation/saves, one-epoch early stopping, and a
hard failure on non-finite train/eval loss.

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
accuracy, aggregate structural F1, merge/split errors, latency, throughput, and
peak memory. Robustness uses the same metrics on its separate split. Pass saved
Luna predictions with `--luna-predictions` for an identical baseline report.

## Consume frozen test once

After validation selects a checkpoint:

```bash
uv run menu-vlm evaluate-test \
  --references /workspace/private-dataset/test.jsonl \
  --predictions /workspace/run/test.predictions.jsonl \
  --dataset-sha256 DATASET_SHA256 \
  --checkpoint /workspace/run/selected-adapter \
  --output /workspace/run/test.metrics.json
```

The command writes both metrics and a `.once.json` gate bound to dataset hash and
checkpoint. It refuses to overwrite either. Test is therefore not an iteration
loop; create a new immutable dataset/run identity for another test evaluation.

Mac inference is optional. The authoritative run is on the Runpod environment;
the 16 GB Mac path is limited to build/validation/preflight unless a compatible
quantized runtime is separately proven.
