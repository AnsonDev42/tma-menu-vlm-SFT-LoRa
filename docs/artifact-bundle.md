# Adapter artifact bundle

The return bundle is adapter-only and must contain exactly these roles:

| Role | Evidence |
| --- | --- |
| `adapter_config`, `adapter_weights` | Selected PEFT adapter |
| `processor_provenance` | Native processor/template snapshot or hashed provenance |
| `dependency_lock`, `training_config` | Exact software and run configuration |
| `dataset_manifest` | Private-free release/split hashes and prompt provenance |
| `training_logs` | Loss, stop, and failure history |
| `checkpoint_selection` | Structural F1 -> item F1 -> loss rationale |
| `predictions`, `metrics` | Frozen audit predictions and order-insensitive report |
| `hardware`, `commands` | GPU/runtime facts and replay commands |

The artifact spec also requires model ID/revision, dataset SHA-256, seed, hardware,
and commands. `menu-vlm bundle` copies only explicit files, hashes each role, and
writes `artifact-manifest.json`. `menu-vlm package` creates a deterministic tarball,
an internal file manifest, and an external SHA-256 sidecar. `verify-archive`
rejects checksum drift, path traversal, links, unsupported entries, and missing or
extra files before exposing the returned bundle.

Adapters and private evaluation output are not public release assets and are
ignored by Git. Do not merge the base model into the bundle.
