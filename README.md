# TMA Menu VLM Fine-Tuning

Open-source workspace for experiments that fine-tune vision-language models
for menu understanding.

This repository currently defines only the project and data interface. It does
not yet select a model, training framework, or training pipeline.

## Private dataset contract

Training data is private and must remain outside this Git repository. Set
`MENU_DATASET_PATH` to the absolute path of the local dataset directory:

```bash
cp .env.example .env
export MENU_DATASET_PATH=/absolute/path/to/menu-training-data
```

The directory's internal schema will be documented when the training pipeline
is introduced. Do not copy menu images, annotations, dataset archives, signed
URLs, credentials, or other private TMA data into this repository.

When used from the TMA workspace, the expected local layout is:

```text
tma/
├── ml/menu-vlm-finetune/       # this public repository
└── .local/menu-training-data/  # ignored private data
```

For that layout:

```bash
export MENU_DATASET_PATH="$(git rev-parse --show-toplevel)/.local/menu-training-data"
```

## Status

Bootstrap only. Training commands, dependencies, data validation, and model
artifacts will be added in later changes.
