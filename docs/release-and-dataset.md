# Release and compiled dataset contract

## Immutable silver-enhanced input

`menu-vlm compile` accepts an explicit directory with format
`menu-silver-enhanced-release-v1`:

```text
manifest.json       format/version/counts and files_sha256
records.json        included source_reference and silver_augmented records
excluded.json       every excluded derivative and reason
mapping.json        canonical source IDs and original lineage
lineage.json        derivative -> canonical source, split, transform, preservation
splits.json         train/validation/test document assignments and lists
images/             release-hashed images
audit/              optional release-hashed source audit
```

The validator requires the declared hashes to be the exact release file
inventory (excluding `manifest.json`) and checks every declared file hash before
reading labels. It then requires exact identity/count accounting, path-safe
unique IDs, source/derivative image
hashes, content-preserving lineage, inherited derivative annotations, identical
source/derivative split assignment, and non-empty source examples in all three
splits. Optional files such as audit evidence may exist only when included in
that exact hashed inventory; unhashed extras and declared-but-absent files fail
validation. Compiled output likewise accounts for every generated file.

Existing assignments are immutable. `--allow-unsplit` is accepted only for the
generic `menu-canonical-unsplit-v1` contract used by public fixtures or future
unassigned imports. It groups canonical sources by non-empty `restaurant_id`,
falls back to canonical `document_id` with a warning, splits whole groups from a
seed, and makes each derivative inherit its source assignment.

## Projection to the Luna seam

Canonical OCR span order defines the 1-based line indices. Each canonical label
is projected to the exact compact response:

- section: `id`, `h: {t,l}`, and shared `notes: [{t,l}]`;
- item: all owned non-note OCR lines `l`, visible name `n`, first name line anchor
  `a`, deterministic supervised confidence `c=1.0`, section reference `s`, and
  dish-specific notes;
- descriptions, prices, calories, and variants contribute OCR evidence to item
  `l` but are not emitted as downstream enrichment fields;
- unknown OCR IDs, an item without OCR evidence, invalid anchors, dangling
  sections, and out-of-range lines are projection failures.

Projection failures are never coerced. `projection-failures.json` records each
ID and reason; the manifest proves `input_records = compiled_records + failures`.
If failures empty a primary split, compilation stops.

## Compiled output

One source image or derivative is one example. Train contains source images and
approved derivatives. Primary validation/test contain source images only;
derivatives from those source groups go to `robustness_validation.jsonl` and
`robustness_test.jsonl`.

Every JSONL record contains relative `image`, `messages`, compact `target`,
source lineage, split/role, OCR count, and prompt version/hash. The message order
matches production: system; user text with page index and numbered OCR; user
image placeholder; compact assistant JSON. Training resolves the top-level image
column through TRL's native VLM collator, avoiding manually constructed image
tokens. The vendored system prompt is preserved byte-for-byte, including exactly
one terminal LF: 4,284 UTF-8 bytes with SHA-256
`c4c4466bbdf49eb066bab6486bd9c9a0bf9230aeafb2da60b0ab02cd617fa476`.

`manifest.json` records the source release hash, split assignments, prompt hash,
counts, every generated file SHA-256, and an aggregate dataset SHA-256. Repeated
compiles are byte-identical.

## Imported current-Luna test baseline

`menu-vlm import-luna-baseline` takes the compiled dataset, an explicit TMA data
root, a completed evaluation run ID, and the approved fixed-name private
`luna-response-provenance.json` plus its checksum sidecar. It requires the
run/report model to be
`current-tma-core:gpt-5.6-luna`, exact completed document/reference/prediction
accounting, and a one-to-one image SHA-256 match with compiled `test.jsonl`.

For every matched page, `_evaluation.call_path` must be exactly
`baselines/tma-core/<cache-sha256>/call.json`; absolute paths, traversal, symlink
escapes, cache-key mismatches, or another subtree fail. The call's input image
hash and exact TMA OCR digest must match the reference, its contract must match
the report baseline contract, and its cache key is recomputed from those inputs.
The contract itself is pinned to the current TMA extraction adapter, provider
settings, excluded stages, runtime-version inventory, and safe hashed
`src/**/*.py` source inventory. The checked-in
`src/menu_vlm/current_tma_contract_v1.json` artifact is checksum-pinned by the
importer and supplies the independent approval anchor; a self-consistent forged
inventory, digest set, or runtime-version set therefore fails.
The reference OCR text and order must also reproduce the compiled user prompt.
Every run metadata field must exactly equal its shared report field. The report
execution and prediction execution must be identical successful-call objects. Their
cost, rate, original-call timing, provider timing, and fallback count are bound to
the corresponding exact call record, report baseline, and materialized result
fields. The call record has exactly seven successful-call keys; even an optional
`error: null` field is rejected. The prediction must identify the same call and
`tma.json` result with a stopped, error-free provider response. That alias-safe
materialized result must have the current typed result inventory, repeat the
exact contract, contain exact typed item/result/note/location structures whose
lengths agree with the vision and fallback counts, and equal the prediction's
embedded TMA result. Its processed-image SHA-256 and dimensions are recomputed
from the bound compiled source image with the approved current-TMA processor;
copied or fabricated processed-image evidence fails. Report, document, execution,
prediction-execution, and call records also use their exact audited key inventories
and reject booleans, strings, and floats in integer result IDs and page-index
fields. Before any provider response is parsed, the importer verifies the exact
raw bytes against the approved provenance manifest. That canonical manifest is
checksum verified, must match the public production aggregate anchor, and binds
all and only the compiled test/report documents one-to-one by image hash, cache
key, exact call and raw-response paths, call identity, and raw-response SHA-256.
There is no production fallback. Approval generation independently reproduces
the predictions and requires the canonical fixed baseline digest before writing
the private manifest and sidecar. Private paths and per-document hashes remain
outside Git. Only the sibling `provider.raw.json` is read, and it must be an error-free typed
OpenAI 3.0.0 `ChatCompletion` envelope parsed in strict mode with Pydantic 2.13.4,
the approved exact model, null logprobs, nonnegative
token accounting, approved service-tier values, exact typed URL-citation
annotations, no moderation payload, allowed envelope/choice/message keys, and
exactly one index-zero stopped Luna assistant text response. Refusal or
tool/function calls fail, and
the reference OCR and compact JSON are also parsed in strict mode, and the compact
output must reference only the compiled OCR line range. Completed-run metadata is
also closed: `schema_error` is null, timestamps and dataset versions are strings,
and overall/per-document metrics and count objects have the approved exact typed
schemas with bool-safe integer accounting.

The importer writes prediction JSONL ordered by compiled example ID and an exact
`<output>.sha256` sidecar. Both are private test evidence: keep them outside Git,
transfer them directly to the pod together with the private provenance manifest,
and return both fixed-name files in the adapter bundle. The public synthetic smoke
uses a separate pinned synthetic-only provenance aggregate accepted only for the
explicit public synthetic dataset version.
