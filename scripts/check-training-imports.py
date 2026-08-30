#!/usr/bin/env python3
"""Verify the frozen Qwen3-VL training import seam without loading a model."""

import json
from importlib.metadata import version

import torch
import torchvision
from transformers import Qwen3VLProcessor, Qwen3VLVideoProcessor


def _require_version(package: str, expected: str) -> str:
    observed = version(package)
    if observed != expected:
        raise RuntimeError(f"{package} version {observed} does not match locked {expected}")
    return observed


def main() -> None:
    report = {
        "torch": _require_version("torch", "2.13.0"),
        "torchvision": _require_version("torchvision", "0.28.0"),
        "cuda_available": torch.cuda.is_available(),
        "processor": Qwen3VLProcessor.__name__,
        "video_processor": Qwen3VLVideoProcessor.__name__,
        "torchvision_import": torchvision.__name__,
        "valid": True,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
