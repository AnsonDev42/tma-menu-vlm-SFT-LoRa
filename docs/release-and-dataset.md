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
tokens.

`manifest.json` records the source release hash, split assignments, prompt hash,
counts, every generated file SHA-256, and an aggregate dataset SHA-256. Repeated
compiles are byte-identical.
