"""Test that CRFConfig defaults point to the fine-tuned checkpoint."""
import os
from pathlib import Path

import pytest


def test_crf_config_points_to_fine_tuned_weights():
    """CRFConfig.model_weights_path must resolve to the fine-tuned checkpoint."""
    from data_structuring.config import CRFConfig

    cfg = CRFConfig()
    resolved = Path(cfg.model_weights_path).resolve()

    assert resolved.name == "CRF_bank_v1_best.safetensors", (
        f"Expected fine-tuned weights file, got: {resolved.name}"
    )
    assert "address-fine-tune" in str(resolved), (
        f"Weights path should reference address-fine-tune directory: {resolved}"
    )
    assert resolved.exists(), f"Fine-tuned weights file not found: {resolved}"


def test_crf_config_keeps_original_architecture_config():
    """model_config_path must still point to the original architecture JSON."""
    from data_structuring.config import CRFConfig

    cfg = CRFConfig()
    resolved = Path(cfg.model_config_path).resolve()

    assert resolved.name == "CRF_with_MLP_EPOCH_1.config.json", (
        f"Config JSON should remain the original architecture file, got: {resolved.name}"
    )
    assert resolved.exists(), f"Architecture config file not found: {resolved}"
