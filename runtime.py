"""Explicit model configuration and strict, vocabulary-aware checkpoint loading."""

import hashlib
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "vision_model": "vit_large_patch16_224", "img_size": 224, "patch_size": 16,
    "hidden_size": 512, "vocab_size": 20938, "max_seq_len": 6001,
    "tissue_type": 5, "num_hidden_layers": 6, "num_attention_heads": 8,
    "intermediate_size": 1024, "attention_backend": "auto",
}


def load_config(path):
    with open(path, encoding="utf-8-sig") as stream:
        supplied = json.load(stream)
    if not isinstance(supplied, dict):
        raise ValueError("Model configuration must be a JSON object.")
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError("Unknown model configuration keys: " + ", ".join(sorted(unknown)))
    config = {**DEFAULT_CONFIG, **supplied}
    for key in set(DEFAULT_CONFIG) - {"vision_model", "attention_backend"}:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if config["hidden_size"] % config["num_attention_heads"]:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")
    if config["attention_backend"] not in ("auto", "flash", "sdpa", "math"):
        raise ValueError("attention_backend must be auto, flash, sdpa or math.")
    if config["vision_model"] != "vit_large_patch16_224":
        raise ValueError("This release supports the original UNI ViT-L architecture.")
    return config


def vocabulary_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_vocabulary(path):
    with open(path, encoding="utf-8-sig") as stream:
        vocabulary = json.load(stream)
    if not isinstance(vocabulary, dict) or vocabulary.get("CLS") != 0:
        raise ValueError("Vocabulary must be a gene->integer JSON object with CLS=0.")
    values = list(vocabulary.values())
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError("Vocabulary IDs must be integers.")
    if sorted(values) != list(range(len(values))):
        raise ValueError("Vocabulary IDs must be unique and contiguous from 0.")
    return vocabulary


def _checkpoint_state(payload):
    """Accept release dictionaries and legacy bare state dictionaries."""
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state dictionary.")
    state = payload.get("state_dict", payload.get("model_state_dict", payload))
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError("Checkpoint state keys must be strings.")
        while key.startswith("module."):
            key = key[len("module."):]
        # The original public repository used self.spatialtip; local weights use self.stGPT.
        if key.startswith("spatialtip."):
            key = "stGPT." + key[len("spatialtip."):]
        if key in normalized:
            raise ValueError("Checkpoint key collision after prefix normalization: " + key)
        normalized[key] = value
    return normalized


def load_model(config_path, vocabulary_path, checkpoint=None,
               vision_checkpoint=None, device="cpu"):
    import torch
    from spatialtip_model import SpatialTIPModel

    config = load_config(config_path)
    vocabulary = load_vocabulary(vocabulary_path)
    if len(vocabulary) != config["vocab_size"]:
        raise ValueError("Vocabulary length and config vocab_size differ. Use the checkpoint's original vocabulary.")
    if checkpoint is None and vision_checkpoint is None:
        raise ValueError("Provide a complete SpatialTIP checkpoint or licensed UNI weights for new training.")
    if checkpoint is not None and vision_checkpoint is not None:
        raise ValueError("A full SpatialTIP checkpoint already contains the vision encoder; choose one initialization.")
    payload = None
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected = payload.get("vocabulary_sha256") if isinstance(payload, dict) else None
        if expected is not None and expected != vocabulary_sha256(vocabulary_path):
            raise ValueError("Vocabulary SHA256 does not match the checkpoint.")
        saved_config = payload.get("config") if isinstance(payload, dict) else None
        if saved_config is not None:
            for key in DEFAULT_CONFIG:
                if key not in ("attention_backend", "max_seq_len") and key in saved_config and saved_config[key] != config[key]:
                    raise ValueError("Checkpoint and requested architecture differ: " + key)
    model = SpatialTIPModel(**config, vision_checkpoint=vision_checkpoint)
    if payload is not None:
        model.load_state_dict(_checkpoint_state(payload), strict=True)
    model.config_dict = dict(config)
    return model.to(torch.device(device))
