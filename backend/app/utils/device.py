"""
Device Utilities

Handles CPU/GPU selection for training and inference.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """
    Return the best available device.

    Priority:
        CUDA -> CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_device_name() -> str:
    """
    Return the name of the active device.
    """

    device = get_device()

    if device.type == "cuda":
        return torch.cuda.get_device_name(0)

    return "CPU"


def get_device_info() -> dict:
    """
    Return detailed information about the current device.
    """

    device = get_device()

    info = {
        "device": str(device),
        "device_name": get_device_name(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }

    if device.type == "cuda":

        properties = torch.cuda.get_device_properties(0)

        info.update(
            {
                "memory_gb": round(
                    properties.total_memory / (1024 ** 3),
                    2,
                ),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )

    return info


def print_device_info() -> None:
    """
    Print current hardware information.
    """

    info = get_device_info()

    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)

    print(f"Device          : {info['device']}")
    print(f"Name            : {info['device_name']}")
    print(f"CUDA Available  : {info['cuda_available']}")
    print(f"CUDA Devices    : {info['cuda_device_count']}")

    if info["device"] == "cuda":
        print(f"Memory (GB)     : {info['memory_gb']}")
        print(f"Capability      : {info['capability']}")

    print("=" * 60)


def move_to_device(*objects):
    """
    Move tensors or models to the active device.

    Example
    -------
    model = move_to_device(model)

    images, labels = move_to_device(images, labels)
    """

    device = get_device()

    moved = [obj.to(device) for obj in objects]

    if len(moved) == 1:
        return moved[0]

    return tuple(moved)


if __name__ == "__main__":

    print_device_info()