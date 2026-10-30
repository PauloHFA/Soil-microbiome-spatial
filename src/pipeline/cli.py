"""
cli.py - Subcommand CLI parser for soil microbiome pipeline.
"""

import argparse
from pathlib import Path
from .orchestrator import run_pipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    """Build command line interface parser."""
    parser = argparse.ArgumentParser(
        description="Soil Microbiome Environmental Discrimination Pipeline"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Pipeline subcommands")

    # Command: run-all
    run_all = subparsers.add_parser("run-all", help="Execute complete end-to-end pipeline")
    run_all.add_argument("--config", type=str, default="config/pipeline_config.yaml", help="Path to YAML config")
    run_all.add_argument("--classifier", type=str, default="rf", choices=["rf", "gbdt", "elasticnet", "svm"])
    run_all.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    run_all.add_argument("--output-dir", type=str, default="results", help="Directory for exported artifacts")

    # Command: mock
    mock = subparsers.add_parser("mock", help="Generate synthetic test dataset")
    mock.add_argument("--samples", type=int, default=60, help="Number of paired samples")
    mock.add_argument("--output-dir", type=str, default="data/processed")
    mock.add_argument("--seed", type=int, default=42)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "run-all" or args.subcommand is None:
        cfg = PipelineConfig(
            seed=getattr(args, "seed", 42),
            classifier=getattr(args, "classifier", "rf"),
            output_dir=getattr(args, "output_dir", "results")
        )
        print("Executing Soil Microbiome Pipeline (revised post-doc specification)...")
        results = run_pipeline(cfg)
        print("\n=== EXECUTION COMPLETE (PROJETO PÓS-DOC REVISADO) ===")
        print(f"Mean Balanced Accuracy (16S Bacteria):     {results['mean_balanced_accuracy']['set_a_16s']:.4f}")
        print(f"Mean Balanced Accuracy (ITS Fungi):        {results['mean_balanced_accuracy']['set_b_its']:.4f}")
        print(f"Mean Balanced Accuracy (Integrated 16S+ITS): {results['mean_balanced_accuracy']['set_c_integrated']:.4f}")
        h = results['hypothesis_test_h0']
        print(f"Null Hypothesis (H0 - Integrated not superior): {h['conclusion']}")
        if "p_value" in h:
            print(f"p-value: {h['p_value']:.4f}")
        print(f"\nResults saved to: {cfg.output_dir}/")
        print("Files generated: spatial_cv_metrics.json, confusion_matrices.json, alpha_diversity.csv, hypothesis_summary.txt")


if __name__ == "__main__":
    main()
