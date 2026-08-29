from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PROMPT_VERSION = "menu-v2-vision-v1"
DATASET_FORMAT = "tma-menu-vlm-sft-v1"
RELEASE_FORMAT = "menu-silver-enhanced-release-v1"
SPLITS = ("train", "validation", "test")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPOSITORY_ROOT / "prompts" / f"{PROMPT_VERSION}.txt"
