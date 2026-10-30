"""
test_reproducibility.py - Verification of deterministic pipeline execution under fixed random seeds.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from src.pipeline.orchestrator import run_pipeline, PipelineConfig


def test_pipeline_deterministic_repeatability():
    """Verify that two independent pipeline executions with identical seeds yield identical metrics."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        cfg1 = PipelineConfig(seed=42, n_splits=3, output_dir=d1, data_dir="data/processed")
        cfg2 = PipelineConfig(seed=42, n_splits=3, output_dir=d2, data_dir="data/processed")

        res1 = run_pipeline(cfg1)
        res2 = run_pipeline(cfg2)

        for set_name in ["set_a_16s", "set_b_its", "set_c_integrated"]:
            assert pytest.approx(res1["mean_balanced_accuracy"][set_name], 1e-6) == res2["mean_balanced_accuracy"][set_name]

        assert res1["hypothesis_test_h0"]["conclusion"] == res2["hypothesis_test_h0"]["conclusion"]
