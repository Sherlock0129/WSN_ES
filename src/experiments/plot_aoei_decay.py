#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate the exponential decay curve of information value vs AOEI.

Default output: paper/figures/aoei_decay.png

Usage:
  python -m src.experiments.plot_aoei_decay \
    --lambdas 0.005 0.01 0.02 \
    --V0 1.0 \
    --aoei-max 240 \
    --step 1 \
    --output-dir paper/figures \
    --filename aoei_decay.png
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, List

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


def plot_aoei_decay(
    lambdas: Iterable[float] = (0.005, 0.01, 0.02),
    V0: float = 1.0,
    aoei_max: int = 240,
    step: int = 1,
    output_dir: str = "paper/figures",
    filename: str = "aoei_decay.png",
) -> str:
    """
    Plot V_info = V0 * exp(-lambda * AOEI) for multiple lambda values.

    Returns the saved file path.
    """
    aoei_values = np.arange(0, aoei_max + step, step, dtype=float)

    plt.figure(figsize=(8, 5), dpi=300)
    for lambda_value in lambdas:
        info_values = V0 * np.exp(-lambda_value * aoei_values)
        plt.plot(
            aoei_values,
            info_values,
            linewidth=2.0,
            label=f"λ = {lambda_value:g}",
        )

    plt.xlabel("AOEI (time)", fontsize=18)
    plt.ylabel("Information value V_info", fontsize=18)
    plt.title("Exponential decay of information value with AOEI", fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16, frameon=False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the exponential decay curve of information value vs AOEI."
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02],
        help="Lambda values to plot.",
    )
    parser.add_argument("--V0", type=float, default=1.0, help="Initial information value V0.")
    parser.add_argument(
        "--aoei-max",
        type=int,
        default=240,
        help="Maximum AOEI on the x-axis.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for AOEI axis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper/figures",
        help="Directory to save the figure.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="aoei_decay.png",
        help="Output filename.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    save_path = plot_aoei_decay(
        lambdas=args.lambdas,
        V0=args.V0,
        aoei_max=args.aoei_max,
        step=args.step,
        output_dir=args.output_dir,
        filename=args.filename,
    )
    print(f"Saved AOEI decay figure to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()


