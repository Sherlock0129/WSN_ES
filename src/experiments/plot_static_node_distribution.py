#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a static node distribution figure for paper use.

Features:
- Layouts: uniform-random with optional min distance, or grid.
- Optional CSV input with columns: x,y[,is_solar]
- Distinguish solar vs non-solar nodes visually
- Equal axis scaling; optional node ID annotations

Default output: paper/figures/node_distribution.png

Examples:
  # Uniform random in a 100x100 area with 50 nodes
  python -m src.experiments.plot_static_node_distribution --layout uniform \
    --num-nodes 50 --width 100 --height 100 --solar-ratio 0.5 --seed 42

  # Grid layout 8x8 in 80x80
  python -m src.experiments.plot_static_node_distribution --layout grid \
    --grid-rows 8 --grid-cols 8 --width 80 --height 80 --solar-ratio 0.25

  # From CSV (x,y[,is_solar])
  python -m src.experiments.plot_static_node_distribution --csv positions.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


@dataclass
class NodePoint:
    node_id: int
    x: float
    y: float
    is_solar: bool


def generate_uniform_points(
    num_nodes: int,
    width: float,
    height: float,
    seed: Optional[int] = None,
    min_distance: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Generate points uniformly at random within [0,width]x[0,height].
    Enforce a simple minimum distance via rejection sampling.
    """
    rng = np.random.default_rng(seed)
    points: List[Tuple[float, float]] = []
    if min_distance <= 0.0:
        xs = rng.uniform(0.0, width, size=num_nodes)
        ys = rng.uniform(0.0, height, size=num_nodes)
        return list(zip(xs.tolist(), ys.tolist()))

    max_trials = num_nodes * 500
    trials = 0
    while len(points) < num_nodes and trials < max_trials:
        trials += 1
        x = float(rng.uniform(0.0, width))
        y = float(rng.uniform(0.0, height))
        if all((x - px) ** 2 + (y - py) ** 2 >= min_distance ** 2 for px, py in points):
            points.append((x, y))
    if len(points) < num_nodes:
        print(f"[warn] Could only place {len(points)}/{num_nodes} points with min_distance={min_distance}.")
    return points


def generate_grid_points(rows: int, cols: int, width: float, height: float) -> List[Tuple[float, float]]:
    """
    Generate evenly spaced grid points within [0,width]x[0,height].
    """
    if rows <= 0 or cols <= 0:
        return []
    xs = np.linspace(0.0, width, num=cols, endpoint=True)
    ys = np.linspace(0.0, height, num=rows, endpoint=True)
    pts: List[Tuple[float, float]] = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            pts.append((float(x), float(y)))
    return pts


def assign_solar_flags(num_nodes: int, solar_ratio: float, seed: Optional[int]) -> List[bool]:
    rng = np.random.default_rng(seed)
    num_solar = int(round(max(0.0, min(1.0, solar_ratio)) * num_nodes))
    flags = [False] * num_nodes
    if num_solar > 0:
        solar_indices = rng.choice(num_nodes, size=num_solar, replace=False)
        for idx in solar_indices:
            flags[int(idx)] = True
    return flags


def load_points_from_csv(csv_path: str) -> List[NodePoint]:
    """
    CSV columns: x,y[,is_solar]
    - x,y: float
    - is_solar: 1/0 or true/false (optional; default False)
    """
    points: List[NodePoint] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
                is_solar = False
                if len(row) >= 3:
                    val = row[2].strip().lower()
                    is_solar = val in ("1", "true", "t", "yes", "y")
                points.append(NodePoint(node_id=i + 1, x=x, y=y, is_solar=is_solar))
            except Exception:
                continue
    return points


def build_nodes(
    positions: List[Tuple[float, float]],
    solar_flags: List[bool],
) -> List[NodePoint]:
    nodes: List[NodePoint] = []
    for i, (x, y) in enumerate(positions):
        is_solar = solar_flags[i] if i < len(solar_flags) else False
        nodes.append(NodePoint(node_id=i + 1, x=x, y=y, is_solar=is_solar))
    return nodes


def plot_node_distribution_static(
    nodes: List[NodePoint],
    width: float,
    height: float,
    annotate_ids: bool,
    output_dir: str,
    filename: str,
    dpi: int = 300,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=dpi)

    solar_x = [n.x for n in nodes if n.is_solar]
    solar_y = [n.y for n in nodes if n.is_solar]
    non_x = [n.x for n in nodes if not n.is_solar]
    non_y = [n.y for n in nodes if not n.is_solar]

    ax.scatter(solar_x, solar_y, c="#f5b301", s=45, marker="o", label="Solar nodes", zorder=2)
    ax.scatter(non_x, non_y, c="#666666", s=45, marker="^", label="Non-solar nodes", zorder=2)

    if annotate_ids:
        for n in nodes:
            ax.text(n.x, n.y, str(n.node_id), fontsize=12, ha="left", va="bottom")

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=18)
    ax.set_ylabel("Y (m)", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_title("Node distribution in 2D space", fontsize=20)
    ax.legend(frameon=False, fontsize=16, loc="upper left")

    save_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved node distribution figure to: {os.path.abspath(save_path)}")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static node distribution figure.")

    parser.add_argument("--layout", type=str, default="uniform", choices=["uniform", "grid"], help="Node layout type.")
    parser.add_argument("--csv", type=str, default=None, help="CSV path with columns: x,y[,is_solar].")

    # Uniform/random params
    parser.add_argument("--num-nodes", type=int, default=36, help="Number of nodes (uniform layout).")
    parser.add_argument("--min-distance", type=float, default=0.0, help="Minimum distance between nodes (uniform).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")

    # Grid params
    parser.add_argument("--grid-rows", type=int, default=6, help="Grid rows (grid layout).")
    parser.add_argument("--grid-cols", type=int, default=6, help="Grid cols (grid layout).")

    # Area
    parser.add_argument("--width", type=float, default=100.0, help="Area width.")
    parser.add_argument("--height", type=float, default=100.0, help="Area height.")

    # Solar flags
    parser.add_argument("--solar-ratio", type=float, default=0.5, help="Ratio of solar-enabled nodes [0,1].")

    # Output
    parser.add_argument("--output-dir", type=str, default="paper/figures", help="Output directory.")
    parser.add_argument("--filename", type=str, default="node_distribution.png", help="Output filename.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--annotate-ids", action="store_true", help="Annotate node IDs near points.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        nodes = load_points_from_csv(args.csv)
        if not nodes:
            raise SystemExit(f"No valid points found in CSV: {args.csv}")
        width = args.width
        height = args.height
    else:
        if args.layout == "uniform":
            positions = generate_uniform_points(
                num_nodes=args.num_nodes,
                width=args.width,
                height=args.height,
                seed=args.seed,
                min_distance=args.min_distance,
            )
        else:
            positions = generate_grid_points(
                rows=args.grid_rows, cols=args.grid_cols, width=args.width, height=args.height
            )
        solar_flags = assign_solar_flags(num_nodes=len(positions), solar_ratio=args.solar_ratio, seed=args.seed)
        nodes = build_nodes(positions, solar_flags)
        width = args.width
        height = args.height

    plot_node_distribution_static(
        nodes=nodes,
        width=width,
        height=height,
        annotate_ids=args.annotate_ids,
        output_dir=args.output_dir,
        filename=args.filename,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()


