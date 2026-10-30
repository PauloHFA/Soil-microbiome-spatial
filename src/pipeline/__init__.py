"""
Pipeline package for end-to-end execution and CLI orchestration.
"""

from .orchestrator import run_pipeline, PipelineConfig
from .cli import build_parser

__all__ = ["run_pipeline", "PipelineConfig", "build_parser"]
