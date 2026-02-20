"""Evaluation script for comparing multiple runs."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import wandb
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def fetch_run_from_wandb(
    entity: str,
    project: str,
    run_id: str,
) -> Dict[str, Any]:
    """Fetch run data from WandB.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name
        
    Returns:
        Dict with run history, summary, and config
    """
    api = wandb.Api()
    
    # Find run by display name
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": run_id},
        order="-created_at"
    )
    
    if not runs:
        raise ValueError(f"No run found with name: {run_id}")
    
    run = runs[0]  # Most recent run with that name
    
    # Fetch data
    history = run.history()
    summary = dict(run.summary)
    config = dict(run.config)
    
    return {
        "history": history,
        "summary": summary,
        "config": config,
        "url": run.url,
    }


def export_run_metrics(
    run_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Export per-run metrics to JSON.
    
    Args:
        run_data: Run data from WandB
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "summary": run_data["summary"],
        "config": run_data["config"],
        "url": run_data["url"],
    }
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics: {metrics_file}")


def create_run_figures(
    run_id: str,
    run_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Create per-run figures.
    
    Args:
        run_id: Run identifier
        run_data: Run data from WandB
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = run_data["history"]
    
    if history.empty:
        print(f"No history data for {run_id}, skipping figures")
        return
    
    # Plot accuracy over time
    if "accuracy" in history.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(history["accuracy"], marker='o')
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy over Time - {run_id}")
        plt.grid(True, alpha=0.3)
        
        output_file = output_dir / f"{run_id}_accuracy.pdf"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"Created figure: {output_file}")


def export_comparison_metrics(
    all_run_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Export aggregated comparison metrics.
    
    Args:
        all_run_data: Map from run_id to run data
        output_dir: Comparison output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Primary metric is accuracy
    primary_metric = "accuracy"
    
    # Collect metrics by run_id
    metrics_by_run = {}
    for run_id, run_data in all_run_data.items():
        summary = run_data["summary"]
        metrics_by_run[run_id] = {
            "accuracy": summary.get("accuracy", 0.0),
            "num_correct": summary.get("num_correct", 0),
            "num_total": summary.get("num_total", 0),
        }
    
    # Identify proposed and baseline
    proposed_runs = [rid for rid in metrics_by_run.keys() if rid.startswith("proposed")]
    baseline_runs = [rid for rid in metrics_by_run.keys() if rid.startswith("comparative")]
    
    best_proposed = None
    best_baseline = None
    
    if proposed_runs:
        best_proposed = max(proposed_runs, key=lambda rid: metrics_by_run[rid][primary_metric])
    
    if baseline_runs:
        best_baseline = max(baseline_runs, key=lambda rid: metrics_by_run[rid][primary_metric])
    
    # Calculate gap
    gap = None
    if best_proposed and best_baseline:
        gap = metrics_by_run[best_proposed][primary_metric] - metrics_by_run[best_baseline][primary_metric]
    
    # Create aggregated metrics
    aggregated = {
        "primary_metric": primary_metric,
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }
    
    output_file = output_dir / "aggregated_metrics.json"
    with open(output_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Exported aggregated metrics: {output_file}")


def create_comparison_figures(
    all_run_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Create comparison figures across all runs.
    
    Args:
        all_run_data: Map from run_id to run data
        output_dir: Comparison output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Accuracy comparison bar chart
    plt.figure(figsize=(10, 6))
    
    run_ids = []
    accuracies = []
    
    for run_id, run_data in all_run_data.items():
        summary = run_data["summary"]
        run_ids.append(run_id)
        accuracies.append(summary.get("accuracy", 0.0))
    
    # Sort by accuracy
    sorted_pairs = sorted(zip(run_ids, accuracies), key=lambda x: x[1], reverse=True)
    run_ids, accuracies = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    colors = ['#2E86AB' if rid.startswith('proposed') else '#A23B72' for rid in run_ids]
    
    plt.bar(range(len(run_ids)), accuracies, color=colors)
    plt.xlabel("Run ID")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Methods")
    plt.xticks(range(len(run_ids)), run_ids, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='Proposed'),
        Patch(facecolor='#A23B72', label='Baseline')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    output_file = output_dir / "comparison_accuracy.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Created comparison figure: {output_file}")
    
    # 2. Accuracy over time (overlay)
    plt.figure(figsize=(12, 6))
    
    for run_id, run_data in all_run_data.items():
        history = run_data["history"]
        if not history.empty and "accuracy" in history.columns:
            label = run_id
            linestyle = '-' if run_id.startswith('proposed') else '--'
            plt.plot(history["accuracy"], label=label, linestyle=linestyle, marker='o', markersize=4)
    
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Time - All Methods")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    output_file = output_dir / "comparison_accuracy_over_time.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Created comparison figure: {output_file}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project")
    
    args = parser.parse_args()
    
    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating runs: {run_ids}")
    
    # Get WandB config (from args or env)
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY", "airas")
    project = args.wandb_project or os.environ.get("WANDB_PROJECT", "2026-0220-1631")
    
    print(f"WandB entity: {entity}")
    print(f"WandB project: {project}")
    
    # Fetch all run data
    all_run_data = {}
    for run_id in run_ids:
        print(f"\nFetching data for: {run_id}")
        try:
            run_data = fetch_run_from_wandb(entity, project, run_id)
            all_run_data[run_id] = run_data
            print(f"  URL: {run_data['url']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if not all_run_data:
        print("\nERROR: No run data fetched successfully")
        return
    
    # Create output directories
    results_dir = Path(args.results_dir)
    comparison_dir = results_dir / "comparison"
    
    # Export per-run metrics and figures
    print("\n" + "=" * 80)
    print("Exporting per-run metrics and figures...")
    print("=" * 80)
    
    for run_id, run_data in all_run_data.items():
        run_output_dir = results_dir / run_id
        export_run_metrics(run_data, run_output_dir)
        create_run_figures(run_id, run_data, run_output_dir)
    
    # Export comparison metrics and figures
    print("\n" + "=" * 80)
    print("Creating comparison metrics and figures...")
    print("=" * 80)
    
    export_comparison_metrics(all_run_data, comparison_dir)
    create_comparison_figures(all_run_data, comparison_dir)
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
