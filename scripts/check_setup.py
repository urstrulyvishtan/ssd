#!/usr/bin/env python3
"""Verify SSD setup: env vars, optional import test. Run from repo root."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> None:
    os.chdir(REPO_ROOT)
    errors = []

    if not os.environ.get("SSD_HF_CACHE"):
        errors.append(
            "SSD_HF_CACHE is not set. Set it to your HuggingFace hub directory "
            "(e.g. /path/to/huggingface/hub). See .env.example."
        )
    if not os.environ.get("SSD_DATASET_DIR"):
        errors.append(
            "SSD_DATASET_DIR is not set. Set it to your processed datasets directory, "
            "or create it with: python scripts/get_data_from_hf.py --num-samples 10000. See .env.example."
        )

    if errors:
        print("Setup check failed:\n")
        for e in errors:
            print(f"  • {e}")
        print("\nCopy .env.example to .env, fill in paths, then: source .env")
        sys.exit(1)

    print("Environment variables OK.")
    if "--import" in sys.argv:
        print("Checking import ...")
        try:
            from ssd import LLM  # noqa: F401
            print("Import OK.")
        except Exception as e:
            print(f"Import failed: {e}")
            if sys.platform != "linux":
                print("Note: Full inference requires Linux with NVIDIA GPU and CUDA.")
            sys.exit(1)


if __name__ == "__main__":
    main()
