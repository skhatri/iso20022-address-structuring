import sys
from pathlib import Path
import pytest

# Ensure scripts and module paths are accessible
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "scripts"))
sys.path.append(str(repo_root / "iso20022-address-structuring"))

from scripts.benchmark_onnx_latency import run_benchmark, generate_markdown_report, find_first_existing


def test_benchmark_onnx_latency_execution(tmp_path):
    """Verify that run_benchmark executes cleanly and returns structured benchmark results and report."""
    default_models = [
        repo_root / "iso20022-address-structuring/resources/models/CRF_with_MLP_EPOCH_1.safetensors",
        repo_root / "iso20022-address-structuring-resources/models/CRF_with_MLP_EPOCH_1.safetensors",
    ]
    default_configs = [
        repo_root / "iso20022-address-structuring/resources/models/CRF_with_MLP_EPOCH_1.config.json",
        repo_root / "iso20022-address-structuring-resources/models/CRF_with_MLP_EPOCH_1.config.json",
    ]

    model_path = find_first_existing(default_models)
    config_path = find_first_existing(default_configs)
    onnx_path = repo_root / "test.onnx"
    output_report = tmp_path / "test_onnx_latency_report.md"

    results, report_content = run_benchmark(
        model_path=model_path,
        config_path=config_path,
        onnx_path=onnx_path,
        batch_sizes=[1, 16],
        seq_len=20,
        warmup=2,
        iterations=5,
        output_report=output_report,
    )

    assert len(results) >= 2, "Expected at least 2 benchmark results for PyTorch and ONNX CPU"
    assert output_report.exists(), "Output markdown report file should be created"

    # Check metrics fields in results
    required_keys = {"provider", "batch_size", "avg_latency_ms", "p95_latency_ms", "throughput", "speedup"}
    for res in results:
        assert required_keys.issubset(res.keys())
        assert res["avg_latency_ms"] > 0
        assert res["p95_latency_ms"] > 0
        assert res["throughput"] > 0
        assert res["speedup"] > 0

    assert "# Task 8: ONNX vs PyTorch Latency & Throughput Benchmark Report" in report_content
    assert "| Runtime / Provider | Batch Size |" in report_content


def test_generate_markdown_report_structure():
    """Verify that generate_markdown_report correctly formats mock benchmark metrics."""
    mock_results = [
        {
            "provider": "PyTorch CPU",
            "batch_size": 1,
            "avg_latency_ms": 2.5,
            "p95_latency_ms": 2.8,
            "throughput": 400.0,
            "speedup": 1.0,
        },
        {
            "provider": "ONNX Runtime CPU",
            "batch_size": 1,
            "avg_latency_ms": 0.8,
            "p95_latency_ms": 0.9,
            "throughput": 1250.0,
            "speedup": 3.125,
        },
    ]

    report = generate_markdown_report(
        results=mock_results,
        batch_sizes=[1],
        seq_len=30,
        warmup=5,
        iterations=30,
        available_providers=["CPUExecutionProvider"],
    )

    assert "ONNX Runtime CPU" in report
    assert "PyTorch CPU" in report
    assert "3.12x" in report
    assert "Avg Latency (ms/sample)" in report
