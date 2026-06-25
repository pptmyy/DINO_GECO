from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def extract_dinov3_adapter_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return trainable DINOv3Adapter weights, excluding the frozen DINOv3 encoder."""
    backbone = getattr(module, "backbone", module)
    return {
        key: value.detach().cpu()
        for key, value in backbone.state_dict().items()
        if not key.startswith("dinov3.")
    }


def save_dinov3_adapter_checkpoint(
    path: str | Path,
    module: torch.nn.Module,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    checkpoint = {
        "adapter": extract_dinov3_adapter_state_dict(module),
        "metadata": dict(metadata or {}),
    }
    torch.save(checkpoint, Path(path))


def load_dinov3_adapter_checkpoint(
    module: torch.nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    checkpoint = torch.load(Path(path), map_location=map_location)
    state_dict = checkpoint.get("adapter", checkpoint)
    backbone = getattr(module, "backbone", module)
    return backbone.load_state_dict(state_dict, strict=False)
