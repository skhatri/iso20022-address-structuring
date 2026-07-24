import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure scripts and module paths are accessible
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "scripts"))
sys.path.append(str(repo_root / "iso20022-address-structuring"))

from data_structuring.components.models.crf_base import BaseCRF
from scripts.onnx_viterbi import ONNXViterbiDecoder, viterbi_decode


class ConcreteCRF(BaseCRF):
    def forward(self, emissions, tags, mask=None):
        pass


def test_onnx_viterbi_no_pytorch_import_in_module():
    """Verify that scripts/onnx_viterbi does not import PyTorch directly."""
    import scripts.onnx_viterbi as ov
    assert "torch" not in ov.__file__
    with open(ov.__file__, "r", encoding="utf-8") as f:
        content = f.read()
    assert "import torch" not in content
    assert "from torch" not in content


def test_onnx_viterbi_decode_parity():
    """Verify 100% tag sequence parity between PyTorch BaseCRF.viterbi_decode and ONNX viterbi_decode."""
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 16
    seq_len = 25
    num_tags = 10

    crf = ConcreteCRF(num_tags=num_tags)
    emissions_pt = torch.randn(batch_size, seq_len, num_tags)
    
    # Create mask with varying sequence lengths
    mask_pt = torch.ones(batch_size, seq_len, dtype=torch.uint8)
    for b in range(batch_size):
        valid_len = np.random.randint(5, seq_len + 1)
        mask_pt[b, valid_len:] = 0

    # PyTorch reference decoding
    expected_tags = crf.viterbi_decode(emissions_pt, mask_pt)

    # NumPy ONNX decoding
    emissions_np = emissions_pt.numpy()
    transitions_np = crf.transitions.detach().numpy()
    start_trans_np = crf.start_transitions.detach().numpy()
    end_trans_np = crf.end_transitions.detach().numpy()
    mask_np = mask_pt.numpy().astype(bool)

    actual_tags = viterbi_decode(
        emissions=emissions_np,
        transitions=transitions_np,
        mask=mask_np,
        start_transitions=start_trans_np,
        end_transitions=end_trans_np,
    )

    assert len(actual_tags) == len(expected_tags)
    for b in range(batch_size):
        assert actual_tags[b] == expected_tags[b], f"Mismatch at batch index {b}"


def test_onnx_viterbi_class_interface():
    """Verify ONNXViterbiDecoder class interface with 2D single sequence input."""
    torch.manual_seed(123)
    seq_len = 15
    num_tags = 7

    crf = ConcreteCRF(num_tags=num_tags)
    emissions_pt = torch.randn(1, seq_len, num_tags)
    mask_pt = torch.ones(1, seq_len, dtype=torch.uint8)

    expected_tags = crf.viterbi_decode(emissions_pt, mask_pt)[0]

    decoder = ONNXViterbiDecoder(
        transitions=crf.transitions.detach().numpy(),
        start_transitions=crf.start_transitions.detach().numpy(),
        end_transitions=crf.end_transitions.detach().numpy(),
    )

    actual_tags = decoder.decode(emissions_pt[0].numpy())
    assert actual_tags == expected_tags
