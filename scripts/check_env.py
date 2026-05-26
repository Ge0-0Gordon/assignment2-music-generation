"""Report the Phase 0 project environment for CSE253 Assignment 2."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys


DEPENDENCIES = [
    "torch",
    "miditok",
    "symusic",
    "mido",
    "midiutil",
    "music21",
    "transformers",
    "nbformat",
    "pandas",
    "numpy",
    "matplotlib",
    "tqdm",
]


def dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print("phase0_environment_report")
    print(f"cwd={os.getcwd()}")
    print(f"python_executable={sys.executable}")
    print(f"python_version={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print("dependencies:")
    for name in DEPENDENCIES:
        print(f"  {name}: {dependency_available(name)}")

    if dependency_available("torch"):
        import torch

        print("torch:")
        print(f"  version: {torch.__version__}")
        print(f"  cuda_version: {torch.version.cuda}")
        print(f"  cuda_available: {torch.cuda.is_available()}")
        print(f"  cuda_device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"  gpu_name: {torch.cuda.get_device_name(0)}")
        else:
            print("  gpu_name: N/A")


if __name__ == "__main__":
    main()
