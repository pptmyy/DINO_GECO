from typing import Callable, Dict, Iterable, Optional

import torch


def normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    has_module = all(k.startswith("module.") for k in state_dict.keys())
    if has_module:
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _format_keys(keys: Iterable[str], limit: int = 20) -> str:
    keys = list(keys)
    shown = keys[:limit]
    suffix = "" if len(keys) <= limit else f" ... (+{len(keys) - limit} more)"
    return ", ".join(shown) + suffix


def load_model_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    allow_partial_load: bool = False,
    logger: Optional[Callable[[str], None]] = None,
    context: str = "checkpoint",
) -> None:
    """Load model weights and fail by default on structure mismatches."""

    state_dict = normalize_checkpoint_keys(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    if logger is not None:
        if missing_keys:
            logger(f"{context}: missing keys: {_format_keys(missing_keys)}")
        if unexpected_keys:
            logger(f"{context}: unexpected keys: {_format_keys(unexpected_keys)}")

    if (missing_keys or unexpected_keys) and not allow_partial_load:
        details = []
        if missing_keys:
            details.append(f"missing keys: {_format_keys(missing_keys)}")
        if unexpected_keys:
            details.append(f"unexpected keys: {_format_keys(unexpected_keys)}")
        raise RuntimeError(
            f"Model structure does not match {context}; " + "; ".join(details) + ". "
            "Use --allow-partial-load only when this mismatch is intentional."
        )
