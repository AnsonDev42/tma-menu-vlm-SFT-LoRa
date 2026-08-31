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
  --output /workspace/run/checkpoint-N.predictions.jsonl \
  --batch-size 2 --inference-mode bf16

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

Prediction is deliberately CUDA-only. BF16 and INT8 both request
`flash_attention_2`, pin the entire model to `cuda:0`, and fail if Transformers
reports CPU/disk offload or a different attention implementation. INT8 uses the
bitsandbytes CUDA backend (`load_in_8bit=True`); it cannot silently run on macOS.
`--batch-size` batches complete multimodal conversations and preserves input
order and exact example count. Each row records image preprocessing, vision
encoder, LLM prefill, time to first token, decode duration, generated-token
count, decode-token count, decode throughput, total latency, and peak CUDA memory.
Decode tokens deliberately exclude the first token attributed to TTFT; a
one-token completion therefore has zero decode tokens and zero decode throughput.
Per-example throughput uses that example's post-TTFT tokens, while batch
throughput sums post-TTFT tokens across the batch and divides by the one shared
decode interval. Aggregate telemetry counts shared phase timings once per batch.
Evaluation files
carry SHA-256 provenance for their exact references and predictions.

Optional W&B logging requires `uv sync --extra train --extra telemetry`. Pass a
public-safe project name with `--wandb-project`. Only the pinned public model
configuration, inference mode, requested batch/token limits, example count, and
aggregate scalar telemetry are sent. Dataset paths, split names, example IDs,
images, prompts, predictions, and adapter paths are never logged.

For the frozen BF16/INT8 benchmark, run both modes with the same compiled split,
adapter, batch size, and token limit, then evaluate each prediction file against
the same references. Compare the resulting structural and performance metrics
only after confirming their `provenance.references_sha256` values are identical
and equal the corresponding `files_sha256` entry in the compiled manifest. Run
this for both `test.jsonl` (clean) and `robustness_test.jsonl`; keep all payloads
and outputs in the private Runpod root.

## Consume frozen test once

After validation selects a checkpoint:

```bash
uv run menu-vlm evaluate-test \
  --references /workspace/private-dataset/test.jsonl \
  --predictions /workspace/run/test.predictions.jsonl \
  --dataset-sha256 DATASET_SHA256 \
  --dataset-manifest /workspace/private-dataset/manifest.json \
  --checkpoint /workspace/run/selected-adapter \
  --gate-store /workspace/tma-menu-vlm/frozen-test-gates \
  --luna-predictions /workspace/run/luna-test-predictions.jsonl \
  --output /workspace/run/test.metrics.json
```

The fixed gate store is outside the training/evaluation output tree and writes one
gate keyed by dataset SHA-256, the exact compiled manifest file SHA-256, pinned model ID and
revision, and the hashes of the selected adapter config/weights. Replaying the
same identity with a different metrics output path or predictions file still
fails. The gate also records the required Luna prediction SHA-256, while keeping
the one-shot identity bound to the candidate adapter rather than offering a new
test attempt when a baseline file changes. A different adapter content hash
creates a different identity. Use validation—not frozen test—for iteration.
The returned artifact also carries the fixed-name approved Luna response
provenance manifest. Bundle creation verifies its aggregate anchor and the exact
compiled dataset manifest schema/accounting before binding either to the frozen
gate. The provenance baseline digest must also equal the returned Luna prediction
bytes, so importer and bundle success both prove the same baseline identity.

Mac inference is optional. The authoritative run is on the Runpod environment;
the 16 GB Mac path is limited to build/validation/preflight unless a compatible
quantized runtime is separately proven.
