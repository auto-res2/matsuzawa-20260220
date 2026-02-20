"""Main orchestrator for running experiments."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for experiment execution.
    
    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print(f"Experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.name}")
    print("=" * 80)
    
    # Print full config for debugging
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # This is an inference-only task, so we always run inference
    run_inference(cfg)
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
