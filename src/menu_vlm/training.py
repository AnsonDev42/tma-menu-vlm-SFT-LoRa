import importlib.metadata
import math
import platform
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .compiler import validate_compiled_dataset
from .constants import MODEL_ID, MODEL_REVISION
from .jsonio import read_json, read_jsonl, sha256_file, write_json

_DRY_MODULES = (
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.self_attn.k_proj",
    "model.language_model.layers.0.self_attn.v_proj",
    "model.language_model.layers.0.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.self_attn.q_proj",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.patch_embed.proj",
)

_PINNED_TRAIN_PACKAGES = {
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.2",
    "datasets": "5.0.1",
    "peft": "0.20.0",
    "pillow": "12.3.0",
    "torch": "2.13.0",
    "transformers": "5.16.1",
    "trl": "1.12.0",
}


def preflight_config(
    config_path: Path,
    *,
    no_download: bool,
    module_names: Iterable[str] | None = None,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    config = _validated_config(config_path)
    dataset = validate_compiled_dataset(dataset_path) if dataset_path is not None else None
    names = tuple(module_names) if module_names is not None else _DRY_MODULES
    target_report = validate_module_targets(config, names)
    installed = _installed_training_packages()
    if not no_download:
        missing_or_wrong = {
            name: {"expected": expected, "installed": installed.get(name)}
            for name, expected in _PINNED_TRAIN_PACKAGES.items()
            if name != "bitsandbytes" or config["mode"] == "qlora"
            if installed.get(name) != expected
        }
        if missing_or_wrong:
            raise ValueError(f"Pinned training dependencies are unavailable: {missing_or_wrong}")
    return {
        "valid": True,
        "no_download": no_download,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "mode": config["mode"],
        "image_token_safe": config["max_length"] is None and config["packing"] is False,
        "vision_tower_frozen": config["freeze_vision_tower"],
        "matched_families": target_report["matched_families"],
        "matched_module_count": len(target_report["matched_modules"]),
        "matched_modules": target_report["matched_modules"],
        "dataset_format": "conversational_prompt_completion",
        "loss_scope": "completion_only",
        "sft_config": {
            "completion_only_loss": True,
            "assistant_only_loss": False,
        },
        "dataset": dataset,
        "installed_training_packages": installed,
        "source": "static pinned Qwen3-VL module profile" if no_download else "loaded model",
    }


def validate_module_targets(config: dict[str, Any], module_names: Iterable[str]) -> dict[str, Any]:
    names = tuple(module_names)
    selectors = config["target_module_selectors"]
    language = selectors["language"]
    language_prefix = str(language["prefix"])
    language_suffixes = tuple(str(suffix) for suffix in language["suffixes"])
    language_matches = sorted(
        name
        for name in names
        if name.startswith(language_prefix)
        and any(name.endswith(f".{suffix}") for suffix in language_suffixes)
    )
    missing_suffixes = [
        suffix
        for suffix in language_suffixes
        if not any(name.endswith(f".{suffix}") for name in language_matches)
    ]
    if missing_suffixes:
        raise ValueError(f"Configured language LoRA targets do not exist: {missing_suffixes}")
    projector_targets = tuple(str(name) for name in selectors["projector"]["exact"])
    projector_matches = sorted(name for name in names if name in projector_targets)
    missing_projectors = sorted(set(projector_targets) - set(projector_matches))
    if missing_projectors:
        raise ValueError(f"Configured projector LoRA targets do not exist: {missing_projectors}")
    matched = sorted([*language_matches, *projector_matches])
    forbidden_matches = [
        name
        for name in matched
        if name.startswith("model.visual.") and name not in projector_targets
    ]
    if forbidden_matches:
        raise ValueError(f"LoRA targets enter the frozen vision tower: {forbidden_matches}")
    matched_families = ["language", "projector"]
    if not projector_matches or not language_matches:
        raise ValueError("Loaded targets must cover language_model and visual.merger modules")
    return {
        "matched_modules": matched,
        "matched_families": sorted(matched_families),
        "language_modules": language_matches,
        "projector_modules": projector_matches,
    }


def validate_peft_target_modules(
    selected_modules: Iterable[str], targeted_modules: Iterable[str]
) -> dict[str, Any]:
    """Prove PEFT injected exactly the resolved loaded-model modules and no suffix collision."""
    selected = tuple(sorted(set(selected_modules)))
    targeted = tuple(sorted(set(targeted_modules)))
    missing = [name for name in selected if not any(actual.endswith(name) for actual in targeted)]
    unexpected = [
        actual
        for actual in targeted
        if not any(actual == name or actual.endswith(f".{name}") for name in selected)
    ]
    if missing or unexpected or len(targeted) != len(selected):
        raise ValueError(
            f"PEFT targeted modules diverge from exact selection; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return {"selected": list(selected), "targeted": list(targeted)}


def run_training(config_path: Path, dataset_path: Path, output: Path) -> dict[str, Any]:
    """Execute the pinned TRL/PEFT baseline. Imports are lazy for no-download Mac checks."""
    config = _validated_config(config_path)
    dataset_report = validate_compiled_dataset(dataset_path)
    preflight_config(config_path, no_download=False, dataset_path=dataset_path)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    import torch  # type: ignore[import-not-found]
    from datasets import Dataset, Image  # type: ignore[import-not-found]
    from peft import LoraConfig, TaskType  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForMultimodalLM,
        AutoProcessor,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    seed = int(config["seed"])
    random.seed(seed)
    set_seed(seed, deterministic=True)
    train_rows = trainer_rows(dataset_path, "train.jsonl")
    validation_rows = trainer_rows(dataset_path, "validation.jsonl")
    train_dataset = Dataset.from_list(train_rows).cast_column("image", Image())
    validation_dataset = Dataset.from_list(validation_rows).cast_column("image", Image())

    quantization = None
    if config["mode"] == "qlora":
        if platform.system() != "Linux":
            raise ValueError("QLoRA mode is supported only in the Linux Runpod environment")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    processor = AutoProcessor.from_pretrained(
        config["model_id"], revision=config["model_revision"], trust_remote_code=False
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization,
        trust_remote_code=False,
    )
    module_report = validate_module_targets(config, (name for name, _ in model.named_modules()))
    if config["freeze_vision_tower"]:
        model.model.visual.requires_grad_(False)
    model.config.use_cache = False
    lora = LoraConfig(
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=module_report["matched_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    training_args = SFTConfig(
        output_dir=str(output / "checkpoints"),
        seed=seed,
        data_seed=int(config["data_seed"]),
        full_determinism=True,
        num_train_epochs=float(config["max_epochs"]),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        max_length=None,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        bf16=True,
        tf32=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
        peft_config=lora,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=int(config["early_stopping_patience"])),
            _finite_loss_callback(TrainerCallback),
        ],
    )
    targeted_modules = getattr(trainer.model, "targeted_module_names", ())
    injection_report = validate_peft_target_modules(
        module_report["matched_modules"], targeted_modules
    )
    trainable_vision = [
        name
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad and ".visual." in name and ".visual.merger." not in name
    ]
    if trainable_vision:
        raise ValueError(f"Frozen vision tower has trainable parameters: {trainable_vision}")
    train_result = trainer.train()
    adapter_path = output / "final-adapter"
    trainer.save_model(str(adapter_path))
    processor_path = output / "processor"
    processor.save_pretrained(processor_path)
    report = {
        "schema_version": "1.0",
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "mode": config["mode"],
        "seed": seed,
        "dataset_sha256": dataset_report["dataset_sha256"],
        "adapter_path": str(adapter_path.relative_to(output)),
        "processor_path": str(processor_path.relative_to(output)),
        "processor_files_sha256": _directory_hashes(processor_path),
        "module_targets": module_report,
        "peft_injection": injection_report,
        "train_metrics": train_result.metrics,
        "checkpoint_selection_pending": True,
    }
    write_json(output / "training-report.json", report)
    write_json(output / "training-config.json", config)
    return report


def _finite_loss_callback(base: type[Any]) -> Any:
    """Construct a Transformers callback lazily so base installs stay lightweight."""

    class FiniteLossCallback(base):  # type: ignore[misc]
        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Any = None,
            **_: Any,
        ) -> Any:
            for key in ("loss", "eval_loss"):
                value = logs.get(key) if isinstance(logs, dict) else None
                if isinstance(value, int | float) and not math.isfinite(value):
                    raise FloatingPointError(
                        f"Non-finite {key} at step {state.global_step}: {value}"
                    )
            return control

    return FiniteLossCallback()


def _validated_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a JSON object")
    required = {
        "model_id",
        "model_revision",
        "mode",
        "seed",
        "data_seed",
        "max_epochs",
        "target_module_selectors",
        "freeze_vision_tower",
        "max_length",
        "completion_only_loss",
        "gradient_checkpointing",
        "selection",
    }
    if not required.issubset(config):
        raise ValueError(f"Training config is missing keys: {sorted(required - set(config))}")
    if config["model_id"] != MODEL_ID or config["model_revision"] != MODEL_REVISION:
        raise ValueError("Training config does not pin the approved Qwen3-VL model revision")
    if config["mode"] not in {"bf16", "qlora"}:
        raise ValueError("Training mode must be bf16 or qlora")
    if config["max_length"] is not None:
        raise ValueError("max_length must be null so truncation cannot remove image tokens")
    config.setdefault("packing", False)
    if config["packing"] is not False:
        raise ValueError("VLM sequence packing is disabled for the baseline")
    if config["completion_only_loss"] is not True:
        raise ValueError("Baseline must train on prompt/completion completion-only loss")
    _validate_target_selectors(config["target_module_selectors"])
    if not config["freeze_vision_tower"]:
        raise ValueError("The first run must freeze the vision tower")
    if not (0 < int(config["max_epochs"]) <= 3):
        raise ValueError("The first run may use at most three epochs")
    if config["selection"] != ["structural_f1:max", "item_f1:max", "eval_loss:min"]:
        raise ValueError("Checkpoint selection policy has drifted")
    return config


def trainer_rows(dataset_path: Path, filename: str) -> list[dict[str, Any]]:
    root = dataset_path.resolve()
    rows = []
    for row in read_jsonl(root / filename):
        image_path = (root / row["image"]).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"Training image is missing or escapes dataset: {row['example_id']}")
        messages = row["messages"]
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Compiled conversation shape drifted: {row['example_id']}")
        rows.append(
            {
                "image": str(image_path),
                "prompt": messages[:2],
                "completion": messages[2:],
            }
        )
    if not rows:
        raise ValueError(f"Training input is empty: {filename}")
    return rows


def _validate_target_selectors(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"language", "projector"}:
        raise ValueError("target_module_selectors must define language and projector")
    language = value["language"]
    projector = value["projector"]
    if not isinstance(language, dict) or set(language) != {"prefix", "suffixes"}:
        raise ValueError("Language target selector is invalid")
    if language["prefix"] != "model.language_model.":
        raise ValueError("Language target selector must stay scoped to model.language_model")
    expected_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    if set(language["suffixes"]) != expected_suffixes:
        raise ValueError("Language target suffix contract has drifted")
    if not isinstance(projector, dict) or set(projector) != {"exact"}:
        raise ValueError("Projector target selector is invalid")
    if set(projector["exact"]) != {
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
    }:
        raise ValueError("Projector targets must be exact visual merger modules")


def _installed_training_packages() -> dict[str, str]:
    result = {}
    for package in _PINNED_TRAIN_PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
