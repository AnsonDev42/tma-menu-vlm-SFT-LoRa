# Re-run private training with a new release (for example v9)

This runbook takes a completed private silver release through compilation, a
bounded Runpod Qwen3-VL-4B LoRA run, evaluation, retrieval, and cleanup. It is
written for a new release such as `menu-silver-enhanced-v9`; replace paths and
the completed Luna evaluation run ID, but do not change the public repository
to hold any private material.

The public repository may contain code, synthetic fixtures, and this guide only.
Keep menu images, OCR, labels, release manifests, raw provider responses, Luna
predictions, Runpod credentials, returned adapters, and metric JSON outside Git.
Use an ignored local directory or another private location.

## What is and is not a v9 input

`menu-vlm compile` accepts a finished immutable silver release, not a raw
collection of annotations. The release must pass `validate-release` and contain
its release `manifest.json`, split assignment, lineage, records, exclusions, and
hashed images as described in [release-and-dataset.md](release-and-dataset.md).

Existing v9 split assignments are immutable. The compiler keeps source families
together: primary validation and test use source images only, while derivatives
of those same source families are routed to the separate robustness files. Do
not manually move images between JSONL files to make a metric look better.

## 1. Prepare a private local run directory

From a clean checkout of this repository, choose a private run ID and explicit
absolute paths. The example uses a directory below `.local/`; ensure it is
ignored in your own checkout before placing private files there.

```bash
uv sync --extra dev

export RUN_ID=qwen3-vl-4b-v9-20260830
export PRIVATE_ROOT=/absolute/private/menu-vlm-runs/$RUN_ID
export MENU_RELEASE_PATH=/absolute/private/menu-silver-enhanced-v9
export MENU_COMPILED_PATH=$PRIVATE_ROOT/compiled
export MENU_ARCHIVE_PATH=$PRIVATE_ROOT/compiled.tar.gz
export TMA_DATA_ROOT=/absolute/private/tma-menu-parser-data
export LUNA_EVALUATION_RUN=COMPLETED_CURRENT_LUNA_RUN_FOR_V9
export LUNA_REPLAY_BASELINE=$PRIVATE_ROOT/replay/luna-test-predictions.jsonl
export LUNA_BASELINE=$PRIVATE_ROOT/luna-test-predictions.jsonl
export LUNA_PROVENANCE=$PRIVATE_ROOT/luna-response-provenance.json
export LUNA_TRUST_ROOT=$PRIVATE_ROOT/luna-trust-root.json
export RESULT_ROOT=$PRIVATE_ROOT/results

mkdir -p "$PRIVATE_ROOT" "$RESULT_ROOT"
```

Never commit these environment values, a Runpod API key, or any files under
`$PRIVATE_ROOT`. A local `.env` is acceptable only when it is ignored.

## 2. Compile and preflight the v9 dataset locally

Validate the immutable release before compilation. The second validator checks
the compiler output, including source-family split isolation and all file hashes.

```bash
uv run menu-vlm validate-release --release "$MENU_RELEASE_PATH"
uv run menu-vlm compile --release "$MENU_RELEASE_PATH" --output "$MENU_COMPILED_PATH"
uv run menu-vlm validate-dataset --dataset "$MENU_COMPILED_PATH"
uv run menu-vlm preflight \
  --config configs/qwen3-vl-4b-lora.json \
  --dataset "$MENU_COMPILED_PATH" \
  --no-download
uv run menu-vlm package --source "$MENU_COMPILED_PATH" --archive "$MENU_ARCHIVE_PATH"
```

Record the printed dataset/archive checksums. They identify this run and should
be copied into your private run notes. A failed validator is a stop condition:
repair or rebuild the release, rather than editing compiled JSONL by hand.

## 3. Import a fresh Luna baseline for v9's frozen test split

The frozen primary test compares the adapter with the current production Luna
seam. Therefore the baseline must come from a completed current-Luna evaluation
that matches the compiled v9 test image hashes and OCR exactly. Reusing a
baseline from a different release or test split is rejected.

First materialize the replay into a staging directory. The command accepts only
one completed matching evaluation and writes both the baseline and its SHA-256
sidecar. Read the printed `prediction_sha256`, compare it with your private run
notes, and paste that exact 64-character value as `LUNA_BASELINE_SHA256`. Do not
derive or substitute the value inside the trust-root command: this explicit
operator checkpoint is the independent approval step.

```bash
mkdir -p "$(dirname "$LUNA_REPLAY_BASELINE")"
uv run menu-vlm materialize-luna-baseline \
  --dataset "$MENU_COMPILED_PATH" \
  --tma-data-root "$TMA_DATA_ROOT" \
  --evaluation-run "$LUNA_EVALUATION_RUN" \
  --output "$LUNA_REPLAY_BASELINE"

export LUNA_BASELINE_SHA256=PASTE_EXACT_MATERIALIZED_SHA256
uv run menu-vlm create-luna-trust-root \
  --dataset "$MENU_COMPILED_PATH" \
  --tma-data-root "$TMA_DATA_ROOT" \
  --evaluation-run "$LUNA_EVALUATION_RUN" \
  --baseline "$LUNA_REPLAY_BASELINE" \
  --baseline-sha256 "$LUNA_BASELINE_SHA256" \
  --output "$LUNA_TRUST_ROOT"

uv run menu-vlm approve-luna-response-provenance \
  --dataset "$MENU_COMPILED_PATH" \
  --tma-data-root "$TMA_DATA_ROOT" \
  --evaluation-run "$LUNA_EVALUATION_RUN" \
  --canonical-baseline "$LUNA_REPLAY_BASELINE" \
  --trust-root "$LUNA_TRUST_ROOT" \
  --trust-root-sidecar "$LUNA_TRUST_ROOT.sha256" \
  --output "$LUNA_PROVENANCE"

uv run menu-vlm import-luna-baseline \
  --dataset "$MENU_COMPILED_PATH" \
  --tma-data-root "$TMA_DATA_ROOT" \
  --evaluation-run "$LUNA_EVALUATION_RUN" \
  --response-provenance "$LUNA_PROVENANCE" \
  --response-provenance-sidecar "$LUNA_PROVENANCE.sha256" \
  --trust-root "$LUNA_TRUST_ROOT" \
  --trust-root-sidecar "$LUNA_TRUST_ROOT.sha256" \
  --output "$LUNA_BASELINE"

cmp "$LUNA_REPLAY_BASELINE" "$LUNA_BASELINE"
uv run menu-vlm verify-luna-trust-root \
  --dataset "$MENU_COMPILED_PATH" \
  --baseline "$LUNA_BASELINE" \
  --baseline-sidecar "$LUNA_BASELINE.sha256" \
  --response-provenance "$LUNA_PROVENANCE" \
  --response-provenance-sidecar "$LUNA_PROVENANCE.sha256" \
  --trust-root "$LUNA_TRUST_ROOT" \
  --trust-root-sidecar "$LUNA_TRUST_ROOT.sha256"
rm "$LUNA_REPLAY_BASELINE" "$LUNA_REPLAY_BASELINE.sha256"
```

The import command produces `$LUNA_BASELINE.sha256`. Keep the baseline,
provenance, trust root, and all three sidecars private. The root binds the exact
compiled dataset manifest bytes and dataset identity, Luna evaluation run,
replayed baseline digest, and pinned TMA runtime contract. Mutation, reuse with
a different v9 build or run, a missing/wrong sidecar, or an occupied output path
is a hard failure. These files are required inputs to the private Runpod path
and are deliberately included only in the returned private adapter bundle.

## 4. Launch a bounded Runpod pod

Authenticate through either `runpodctl` or the already-authorized Runpod MCP,
inspect live availability immediately before launch, and use an official PyTorch
image. The first baseline configuration is one secure 48 GB A40, up to three
epochs, with 20 GB container storage and 100 GB workspace storage.

```bash
export RUNPOD_API_KEY=... # shell environment only
export RUNPOD_MAX_COST_USD=10 # hard budget for this run; must be <= 15
runpodctl update
runpodctl version
runpodctl user
runpodctl gpu list --include-unavailable

scripts/runpod/launch-pod.sh \
  "$RUN_ID" TEMPLATE_ID "NVIDIA A40" 8 20 100 EU-SE-1
```

When using the authenticated Runpod MCP instead, create the equivalent secure
pod with GPU type `NVIDIA A40`, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, a 20 GB container disk, a
100 GB `/workspace` volume, and your public SSH key. Do not copy an MCP
credential into a shell environment. Read the resulting pod record and retain
only its direct SSH host, port, and pod ID in private run notes.

Replace `TEMPLATE_ID` and `EU-SE-1` with a currently available official template
and data center. The launcher obtains the live GPU price, enforces
`RUNPOD_MAX_COST_USD` (USD 10 in this example; never more than USD 15) across
compute and storage, writes a guard receipt, and arms a
host-local deletion watchdog. Keep the launching Mac awake until cleanup; the
watchdog cannot run while it is asleep. Save the returned `POD_ID` and guard
receipt path:

```bash
export POD_ID=RETURNED_POD_ID
export GUARD_RECEIPT=/absolute/private/path/RETURNED_POD_ID.json
export REMOTE_ROOT=/workspace/$RUN_ID
```

Do not make a pod public or expose an application port. SSH is sufficient.

## 5. Transfer, train, and monitor

Wait until SSH is ready, then transfer public code and private inputs directly
to the pod. The transfer excludes `.git`, `.venv`, and `.env`, disables rsync
owner/group preservation for Runpod volumes, and validates the private Luna
evidence sidecars before writing remotely.

```bash
scripts/runpod/transfer-to-pod.sh \
  "$POD_ID" "$PWD" "$MENU_ARCHIVE_PATH" \
  "$LUNA_BASELINE" "$LUNA_BASELINE.sha256" \
  "$LUNA_PROVENANCE" "$LUNA_PROVENANCE.sha256" \
  "$LUNA_TRUST_ROOT" "$LUNA_TRUST_ROOT.sha256" \
  "$REMOTE_ROOT"

scripts/runpod/run-training.sh \
  "$POD_ID" "$REMOTE_ROOT" "$(basename "$MENU_ARCHIVE_PATH")" \
  "$(basename "$LUNA_BASELINE")" "$(basename "$LUNA_BASELINE").sha256" \
  "$(basename "$LUNA_PROVENANCE")" "$(basename "$LUNA_PROVENANCE").sha256" \
  "$(basename "$LUNA_TRUST_ROOT")" "$(basename "$LUNA_TRUST_ROOT").sha256" \
  "$GUARD_RECEIPT"
```

For an MCP-created pod, use the returned direct SSH host and port. On macOS's
built-in rsync, omit `--info=progress2` and add `--no-owner --no-group` (Runpod
workspace volumes reject ownership preservation). Transfer the public checkout
without `.git`, `.venv`, `.env`, caches, or private data; transfer the archive
and Luna/trust-root evidence files only to the private remote directory; then
run `sha256sum -c` against every transferred sidecar before starting work.

Monitor until the remote log contains `TRAIN_EVAL_BUNDLE_DONE`:

```bash
runpodctl pod logs "$POD_ID" --follow
# Or: use `runpodctl ssh info "$POD_ID"`, then tail "$REMOTE_ROOT/train.log".
```

The on-pod workflow is fixed: it verifies inputs, imports the pinned GPU
environment, runs the Qwen processor import check, trains BF16 LoRA (one QLoRA
fallback only after a CUDA OOM), scores every checkpoint on validation, selects
by structural F1 then item F1 then validation loss, evaluates robustness, and
consumes the primary test once. Any checksum failure, non-finite loss, second
OOM, or test-gate failure stops the run.

`run-training.sh` consumes the local CLI guard receipt. If the pod was created
through MCP and no such receipt exists, start `on-pod-train.sh` over SSH in a
detached `setsid timeout 6h` process instead. That timeout is a secondary cap,
not a substitute for account billing controls; retrieve the result and delete
the pod through MCP immediately after verification.

## 6. Retrieve and verify the returned adapter

Retrieve before deleting anything remote. This command checks the archive
sidecar and safely extracts the bundle locally:

```bash
scripts/runpod/retrieve.sh "$POD_ID" "$REMOTE_ROOT" "$RESULT_ROOT"
```

The important local files are:

```text
$RESULT_ROOT/artifact-bundle/artifact-manifest.json
$RESULT_ROOT/artifact-bundle/files/luna_trust_root/luna-trust-root.json
$RESULT_ROOT/artifact-bundle/files/luna_trust_root_sidecar/luna-trust-root.json.sha256
$RESULT_ROOT/artifact-bundle/files/adapter_config/adapter_config.json
$RESULT_ROOT/artifact-bundle/files/adapter_weights/adapter_model.safetensors
$RESULT_ROOT/artifact-bundle/files/selected_validation_metrics/selected.validation.metrics.json
$RESULT_ROOT/artifact-bundle/files/test_metrics/test.metrics.json
$RESULT_ROOT/artifact-bundle/files/robustness_test_metrics/robustness-test.metrics.json
```

`retrieve.sh` already runs `verify-archive`; repeat it if you copy the archive
elsewhere:

```bash
uv run menu-vlm verify-archive \
  --archive "$RESULT_ROOT/artifact-bundle.tar.gz" \
  --output "$RESULT_ROOT/reverified-bundle"
```

Do not call the adapter a Luna replacement based only on training loss or
validation. Compare frozen `test.metrics.json` with its embedded `luna_baseline`,
and require parse/schema/OCR-reference validity as well as structural and item
metrics. Treat robustness results as a separate perturbation signal.

## 7. Evaluate again without invalidating the protocol

The primary test gate is intentionally one-shot per dataset, model revision, and
adapter content. Do not bypass it by calling the generic evaluator on
`test.jsonl`, deleting a gate file, or changing output filenames. For iteration,
evaluate validation and robustness splits; for a new v9 adapter, create a new
immutable run identity and let the normal workflow consume its own test once.

To use a returned adapter for an additional validation-only GPU evaluation,
reconstruct the two adapter files into one private directory first:

```bash
export RETURNED_BUNDLE=$RESULT_ROOT/artifact-bundle
export ADAPTER_DIR=$PRIVATE_ROOT/reconstructed-adapter
mkdir -p "$ADAPTER_DIR"
cp "$RETURNED_BUNDLE/files/adapter_config/adapter_config.json" "$ADAPTER_DIR/"
cp "$RETURNED_BUNDLE/files/adapter_weights/adapter_model.safetensors" "$ADAPTER_DIR/"
```

Then run `predict` and `evaluate` on a GPU environment against `validation.jsonl`
or a robustness split. A 16 GB Mac is suitable for release/dataset validation,
archive verification, and no-download preflight; it is not the authoritative
full Qwen3-VL inference environment unless a separately tested compatible
quantized setup is available.

```bash
uv run menu-vlm predict \
  --config configs/qwen3-vl-4b-lora.json \
  --dataset "$MENU_COMPILED_PATH" \
  --split-file validation.jsonl \
  --adapter "$ADAPTER_DIR" \
  --output "$PRIVATE_ROOT/evaluation/validation.predictions.jsonl"

uv run menu-vlm evaluate \
  --references "$MENU_COMPILED_PATH/validation.jsonl" \
  --predictions "$PRIVATE_ROOT/evaluation/validation.predictions.jsonl" \
  --output "$PRIVATE_ROOT/evaluation/validation.metrics.json"
```

## 8. Clean up after verification

Only after the returned archive verifies locally, permanently remove the exact
remote root and pod, then confirm that the pod is absent:

```bash
scripts/runpod/cleanup.sh "$POD_ID" "$REMOTE_ROOT" --delete-pod
runpodctl pod list
```

If you used a separate Runpod network volume, delete it only after verifying the
returned archive. It has independent storage billing. Preserve the local bundle,
its SHA-256 sidecar, the run's private checksums, and a concise aggregate metric
summary for future comparisons.

## Failure recovery

| Symptom | Safe response |
| --- | --- |
| `validate-release` or `validate-dataset` fails | Stop. Repair the upstream release or rebuild it; do not hand-edit compiled JSONL. |
| Luna import rejects the run | Produce a fresh completed current-Luna evaluation that exactly matches the compiled v9 test split and OCR. |
| SSH is not ready | Wait and retry transfer. Do not start training until the input transfer finishes. |
| `Qwen3VLVideoProcessor` import fails | Use a checkout containing the merged Torchvision portability fix and rerun the pod bootstrap from a clean remote root. |
| First training attempt OOMs | The on-pod workflow performs its one pinned QLoRA fallback. A second OOM stops the run; do not start an uncontrolled sweep. |
| Frozen test has weak metrics | Keep the bundle and use validation/robustness evidence to diagnose. Do not rerun the same frozen test identity. |
| Retrieval fails | Do not delete the remote root or pod until archive retrieval and `verify-archive` succeed. |
