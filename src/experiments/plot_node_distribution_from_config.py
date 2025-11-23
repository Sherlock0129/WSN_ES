#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a static node distribution figure strictly following parameters
defined in src/config/simulation_config.py (ConfigManager).

Default output: paper/figures/node_distribution.png

Usage:
  # Use dataclass defaults in simulation_config.py
  python -m src.experiments.plot_node_distribution_from_config

  # Or use a JSON config file to override defaults
  python -m src.experiments.plot_node_distribution_from_config --config path/to/config.json \
    --output-dir paper/figures --filename node_distribution.png
"""
from __future__ import annotations

import argparse
import os

from src.config.simulation_config import ConfigManager
from src.viz.plotter import plot_node_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot node distribution using parameters from simulation_config.py"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file to override dataclass defaults.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper/figures",
        help="Directory to save the figure (will also be used as session_dir for stable filename).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="node_distribution.png",
        help="Output filename (stable for LaTeX inclusion).",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="If set, show mobile node trajectories when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration (defaults or from JSON file)
    cfg = ConfigManager(config_file=args.config) if args.config else ConfigManager()

    # Build network strictly from configuration
    network = cfg.create_network()

    # Ensure output directory exists and use it as session_dir for a stable path
    os.makedirs(args.output_dir, exist_ok=True)
    session_dir = args.output_dir

    # Plot and save to a stable path (no timestamped subdir)
    # Internally, the function saves to f"{session_dir}/node_distribution.png"
    # so we temporarily ensure the default name matches requested filename by
    # writing to session_dir and then renaming if needed.
    plot_node_distribution(
        nodes=network.nodes,
        output_dir=args.output_dir,
        show_paths=args.show_paths,
        path_len=None,
        session_dir=session_dir,
        title_font_size=20,
        axis_label_font_size=18,
        tick_font_size=16,
        legend_font_size=16,
    )

    # If a custom filename is requested, rename/copy the saved file
    default_path = os.path.join(session_dir, "node_distribution.png")
    target_path = os.path.join(session_dir, args.filename)
    if os.path.exists(default_path) and default_path != target_path:
        try:
            # Overwrite target if exists
            if os.path.exists(target_path):
                os.remove(target_path)
            os.replace(default_path, target_path)
        except Exception:
            # Fallback to copy if replace fails on some platforms
            import shutil
            shutil.copyfile(default_path, target_path)

    print(f"Node distribution figure saved to: {os.path.abspath(target_path)}")


if __name__ == "__main__":
    main()


