import re
from collections.abc import Mapping
from pathlib import Path

import torch


_PREFIXES = ("module.", "model.", "backbone.")
_BLOCK_PATTERN = re.compile(r"^layers\.(\d+)\.blocks\.(\d+)\.(.+)$")
_DOWNSAMPLE_PATTERN = re.compile(r"^layers\.(\d+)\.downsample\.(.+)$")


def _state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("The checkpoint must contain a state dictionary")
    for key in ("model", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("No model state dictionary was found in the checkpoint")


def _strip_prefix(key):
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def _target_keys(source_key):
    if source_key.startswith("patch_embed."):
        return ((source_key, "patch_embed"),)
    match = _BLOCK_PATTERN.match(source_key)
    if match:
        stage, block, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
        if 0 <= stage <= 3:
            return (
                (f"encoder_stages.{stage}.{block}.{suffix}", "encoder"),
                (f"decoder_stages.{3 - stage}.blocks.{block}.{suffix}", "decoder"),
            )
    match = _DOWNSAMPLE_PATTERN.match(source_key)
    if match:
        stage, suffix = int(match.group(1)), match.group(2)
        if 0 <= stage <= 2:
            return ((f"downsamples.{stage}.{suffix}", "downsample"),)
    return ()


def _adapt(source, target):
    if tuple(source.shape) == tuple(target.shape):
        return source.to(dtype=target.dtype)
    return None


def load_vmamba_small_pretrained(model, checkpoint_path):
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    source_state = _state_dict(torch.load(path, map_location="cpu"))
    target_state = model.state_dict()
    assignments = {}
    components = {}
    for original_key, source_tensor in source_state.items():
        if not torch.is_tensor(source_tensor):
            continue
        source_key = _strip_prefix(str(original_key))
        for target_key, component in _target_keys(source_key):
            target_tensor = target_state.get(target_key)
            if target_tensor is None:
                continue
            compatible = _adapt(source_tensor, target_tensor)
            if compatible is not None:
                assignments[target_key] = compatible
                components[target_key] = component

    component_counts = {
        name: sum(value == name for value in components.values())
        for name in ("patch_embed", "encoder", "downsample", "decoder")
    }
    missing = [name for name, count in component_counts.items() if count == 0]
    if missing:
        raise RuntimeError(
            "Pretrained mapping did not cover: " + ", ".join(missing)
        )

    expected_stages = {
        "encoder_stages": 4,
        "downsamples": 3,
        "decoder_stages": 4,
    }
    for prefix, count in expected_stages.items():
        for stage in range(count):
            if not any(key.startswith(f"{prefix}.{stage}.") for key in assignments):
                raise RuntimeError(f"Pretrained mapping did not cover {prefix}.{stage}")

    backbone_prefixes = (
        "patch_embed.",
        "encoder_stages.",
        "downsamples.",
        "decoder_stages.",
    )
    backbone_numel = sum(
        tensor.numel()
        for key, tensor in target_state.items()
        if key.startswith(backbone_prefixes)
    )
    loaded_numel = sum(target_state[key].numel() for key in assignments)
    coverage = loaded_numel / backbone_numel
    if coverage < 0.5:
        raise RuntimeError(f"Pretrained backbone coverage is too low: {coverage:.2%}")

    model.load_state_dict(assignments, strict=False)
    return {
        "checkpoint": str(path),
        "loaded_tensors": len(assignments),
        "backbone_coverage": coverage,
        "components": component_counts,
    }


__all__ = ["load_vmamba_small_pretrained"]
