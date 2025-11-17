#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch experiment runner aligned with the paper plan in paper/sections/05_experiments.tex.

This script automates the execution of all scenarios, scales, seeds and experiment
variants (E1–E4 and optional ablations) described in the writing.  It standardises
the configuration knobs (energy budget, thresholds, routing limits, K-adaptation,
etc.), iterates over the requested combinations, runs the simulation, and
post-processes the output directory so that every run is tagged with a unique,
human-readable slug (e.g. YYYYMMDD_HHMMSS_S2_N30_seed42_exp1_baseline).

Usage (examples):
    # Run the default core experiment suite (E1–E4) for all scenarios, scales, seeds
    python -m src.experiments.run_paper_experiments

    # Dry-run to inspect the execution plan without launching simulations
    python -m src.experiments.run_paper_experiments --dry-run

    # Only run the random-topology scenario for N=30 with two seeds
    python -m src.experiments.run_paper_experiments \
        --scenario random --sizes 30 --seeds 42,123

    # Execute both the core suite and the ablation suite
    python -m src.experiments.run_paper_experiments --suite core --suite ablation

Key features:
    * Applies the unified experimental assumptions (energy budget, thresholds,
      cooldown, routing range, etc.) before each run.
    * Keeps topology comparable across variants by resetting seeds per scenario.
    * Stores results under src/experiments/data/<timestamp>_<slug>.
    * Writes run_metadata.json inside every folder for quick provenance lookup.
    * Optional --skip-visuals for faster sweeps when plots are not needed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import string
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure project root is on sys.path (script resides in src/experiments/)
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.simulation_config import ConfigManager  # noqa: E402
from routing.energy_transfer_routing import set_eetor_config  # noqa: E402
from sim.refactored_main import create_scheduler  # noqa: E402
from utils.output_manager import OutputManager  # noqa: E402


# -----------------------------------------------------------------------------
# Data classes and constants
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT = Path("src") / "experiments" / "data"
DEFAULT_SEEDS: Sequence[int] = (
    42,
    123,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
DEFAULT_SIZES: Sequence[int] = (15, 30, 60, 100)


@dataclass(frozen=True)
class ScenarioDefinition:
    key: str
    label: str
    distribution_mode: str
    enable_energy_hole: bool = False
    extra_network_params: Dict[str, float] = field(default_factory=dict)


SCENARIOS: Dict[str, ScenarioDefinition] = {
    "grid": ScenarioDefinition(
        key="S1",
        label="grid",
        distribution_mode="uniform",
        enable_energy_hole=False,
    ),
    "random": ScenarioDefinition(
        key="S2",
        label="random",
        distribution_mode="random",
        enable_energy_hole=False,
    ),
    "energy_hole": ScenarioDefinition(
        key="S3",
        label="energy_hole",
        distribution_mode="random",
        enable_energy_hole=True,
        extra_network_params={
            "energy_hole_center_mode": "center",
            "energy_distribution_mode": "center_decreasing",
            "solar_node_ratio": 0.4,  # slightly fewer harvesting nodes in the hole
        },
    ),
}


@dataclass(frozen=True)
class ExperimentVariant:
    suite: str
    key: str
    description: str
    apply: Callable[[ConfigManager], None]


def _baseline_safety(cfg: ConfigManager) -> None:
    """Reset toggles that variants expect to manipulate."""
    cfg.simulation_config.enable_adcr_link_layer = False
    cfg.path_collector_config.enable_path_collector = True
    cfg.path_collector_config.enable_opportunistic_info_forwarding = True
    cfg.path_collector_config.enable_delayed_reporting = True
    cfg.scheduler_config.scheduler_type = "AdaptiveDurationAwareLyapunovScheduler"
    cfg.scheduler_config.duration_w_info = 0.1
    cfg.eetor_config.enable_info_aware_routing = True


VARIANTS: Sequence[ExperimentVariant] = (
    # Core suite (E1–E4)
    ExperimentVariant(
        suite="core",
        key="exp1_baseline_passive",
        description="E1 baseline: price-triggered smart passive scheduling",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.simulation_config, "passive_mode", True),
        ),
    ),
    ExperimentVariant(
        suite="core",
        key="exp1_active_60min",
        description="E1 counterfactual: fixed 60-minute active scheduling",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.simulation_config, "passive_mode", False),
        ),
    ),
    ExperimentVariant(
        suite="core",
        key="exp2_info_enabled",
        description="E2 baseline: InfoNode cache + value-weighted reporting",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.scheduler_config, "duration_w_info", 0.1),
            setattr(cfg.eetor_config, "enable_info_aware_routing", True),
        ),
    ),
    ExperimentVariant(
        suite="core",
        key="exp2_no_info_reward",
        description="E2 counterfactual: remove scheduler information reward",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.scheduler_config, "duration_w_info", 0.0),
        ),
    ),
    ExperimentVariant(
        suite="core",
        key="exp3_opportunistic",
        description="E3 baseline: opportunistic reporting without ADCR",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.simulation_config, "enable_adcr_link_layer", False),
            setattr(cfg.path_collector_config, "enable_opportunistic_info_forwarding", True),
            setattr(cfg.path_collector_config, "enable_delayed_reporting", True),
        ),
    ),
    ExperimentVariant(
        suite="core",
        key="exp3_adcr",
        description="E3 counterfactual: ADCR point-to-point aggregation",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.simulation_config, "enable_adcr_link_layer", True),
            setattr(cfg.path_collector_config, "enable_path_collector", False),
        ),
    ),
    ExperimentVariant(
        suite="core",
        key="exp3_direct_report",
        description="E3 counterfactual: immediate direct reporting",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.simulation_config, "enable_adcr_link_layer", False),
            setattr(cfg.path_collector_config, "enable_path_collector", True),
            setattr(cfg.path_collector_config, "enable_opportunistic_info_forwarding", False),
            setattr(cfg.path_collector_config, "enable_delayed_reporting", False),
        ),
    ),
    ExperimentVariant(
        suite="core",
        key="exp4_adaptive_duration",
        description="E4 baseline: AdaptiveDurationAwareLyapunovScheduler",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.scheduler_config, "scheduler_type", "AdaptiveDurationAwareLyapunovScheduler"),
        ),
    ),
    ExperimentVariant(
        suite="core",
        key="exp4_traditional_lyapunov",
        description="E4 counterfactual: traditional Lyapunov scheduler",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.scheduler_config, "scheduler_type", "LyapunovScheduler"),
            setattr(cfg.scheduler_config, "duration_w_info", 0.0),
        ),
    ),
    # Ablation suite
    ExperimentVariant(
        suite="ablation",
        key="ablation_baseline",
        description="Ablation baseline (identical to paper's proposal)",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.scheduler_config, "duration_w_info", 0.1),
        ),
    ),
    ExperimentVariant(
        suite="ablation",
        key="ablation_no_aoei",
        description="Ablation-1: disable smart passive trigger (fixed active)",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.simulation_config, "passive_mode", False),
        ),
    ),
    ExperimentVariant(
        suite="ablation",
        key="ablation_no_infonode",
        description="Ablation-2: disable InfoNode cache/path collector",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.path_collector_config, "enable_path_collector", False),
        ),
    ),
    ExperimentVariant(
        suite="ablation",
        key="ablation_no_dedup",
        description="Ablation-3: disable dedup/adaptive waiting policies",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.path_collector_config, "enable_info_volume_accumulation", False),
            setattr(cfg.path_collector_config, "enable_delayed_reporting", False),
            setattr(cfg.path_collector_config, "enable_adaptive_wait_time", False),
        ),
    ),
    ExperimentVariant(
        suite="ablation",
        key="ablation_no_eetor_guard",
        description="Ablation-4: disable EETOR efficiency guardrails",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.eetor_config, "min_efficiency", 0.0),
            setattr(cfg.network_config, "max_hops", 100),
        ),
    ),
    ExperimentVariant(
        suite="ablation",
        key="ablation_fixed_k",
        description="Ablation-5: disable adaptive K (fixed concurrency)",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.simulation_config, "enable_k_adaptation", False),
            setattr(cfg.simulation_config, "fixed_k", 3),
        ),
    ),
    ExperimentVariant(
        suite="ablation",
        key="ablation_no_fairness",
        description="Ablation-6: remove fairness critical ratio",
        apply=lambda cfg: (
            _baseline_safety(cfg),
            setattr(cfg.simulation_config, "critical_ratio", 0.0),
        ),
    ),
)

SUITE_NAMES = sorted({variant.suite for variant in VARIANTS})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def apply_plan_defaults(cfg: ConfigManager, output_root: Path) -> None:
    """Apply the unified experimental assumptions before any variant tweaks."""
    # Node-level assumptions
    cfg.node_config.initial_energy = 20000.0
    cfg.node_config.low_threshold = 0.30
    cfg.node_config.high_threshold = 0.80
    cfg.node_config.capacity = 3.5
    cfg.node_config.voltage = 3.7
    cfg.node_config.enable_energy_harvesting = True
    cfg.node_config.solar_efficiency = 0.20
    cfg.node_config.solar_area = 0.10
    cfg.node_config.max_solar_irradiance = 1500.0
    cfg.node_config.env_correction_factor = 1.0
    cfg.node_config.energy_char = 1000.0
    cfg.node_config.energy_decay_rate = 5.0

    # Network-level assumptions
    cfg.network_config.max_hops = 5
    cfg.network_config.network_area_width = 100.0
    cfg.network_config.network_area_height = 100.0
    cfg.network_config.min_distance = 1.0
    cfg.network_config.enable_physical_center = True
    cfg.network_config.center_initial_energy_multiplier = 10.0

    # Simulation controller
    cfg.simulation_config.time_steps = 10080  # 7 days
    cfg.simulation_config.passive_mode = True
    cfg.simulation_config.check_interval = 10
    cfg.simulation_config.cooldown_period = 30
    cfg.simulation_config.energy_variance_threshold = 0.3
    cfg.simulation_config.predictive_window = 60
    cfg.simulation_config.enable_energy_sharing = True
    cfg.simulation_config.enable_k_adaptation = True
    cfg.simulation_config.initial_K = 2
    cfg.simulation_config.K_max = 5
    cfg.simulation_config.hysteresis = 0.05
    cfg.simulation_config.output_dir = str(output_root)
    cfg.simulation_config.enable_adcr_link_layer = False

    # Scheduler defaults
    cfg.scheduler_config.scheduler_type = "AdaptiveDurationAwareLyapunovScheduler"
    cfg.scheduler_config.adaptive_lyapunov_v = 0.5
    cfg.scheduler_config.adaptive_lyapunov_k = 3
    cfg.scheduler_config.duration_min = 1
    cfg.scheduler_config.duration_max = 6
    cfg.scheduler_config.duration_w_aoi = 0.02
    cfg.scheduler_config.duration_w_info = 0.1
    cfg.scheduler_config.duration_info_rate = 10000.0
    cfg.scheduler_config.adaptive_window_size = 10
    cfg.scheduler_config.adaptive_v_min = 0.1
    cfg.scheduler_config.adaptive_v_max = 2.0
    cfg.scheduler_config.adaptive_adjust_rate = 0.1
    cfg.scheduler_config.adaptive_sensitivity = 2.0

    # Path collector defaults
    cfg.path_collector_config.enable_path_collector = True
    cfg.path_collector_config.energy_mode = "full"
    cfg.path_collector_config.enable_opportunistic_info_forwarding = True
    cfg.path_collector_config.enable_delayed_reporting = True
    cfg.path_collector_config.enable_adaptive_wait_time = True
    cfg.path_collector_config.max_wait_time = 500
    cfg.path_collector_config.min_info_volume_threshold = 1
    cfg.path_collector_config.enable_info_volume_accumulation = True
    cfg.path_collector_config.info_value_decay_rate = 0.02

    # Routing defaults
    cfg.eetor_config.max_range = 30.0
    cfg.eetor_config.min_efficiency = 0.05
    cfg.eetor_config.enable_info_aware_routing = True
    cfg.eetor_config.low_energy_threshold = 0.2
    cfg.eetor_config.medium_energy_threshold = 0.5


def configure_scenario(cfg: ConfigManager, scenario: ScenarioDefinition, num_nodes: int, seed: int) -> None:
    """Adjust configuration to reflect topology family, size and seed."""
    cfg.network_config.num_nodes = num_nodes
    cfg.network_config.random_seed = seed
    cfg.network_config.distribution_mode = scenario.distribution_mode
    cfg.network_config.enable_energy_hole = scenario.enable_energy_hole
    for key, value in scenario.extra_network_params.items():
        setattr(cfg.network_config, key, value)


def seed_random_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def slugify(parts: Iterable[str]) -> str:
    allowed = set(string.ascii_letters + string.digits + "_-")
    joined = "_".join(str(p) for p in parts if p is not None and str(p).strip())
    sanitized = "".join(ch if ch in allowed else "-" for ch in joined)
    return sanitized.strip("_-") or "run"


def run_simulation_with_config(
    cfg: ConfigManager,
    skip_visuals: bool,
) -> Path:
    """Execute a single simulation and return the absolute session directory."""
    from info_collection.physical_center import VirtualCenter  # lazy import
    from viz.plotter import (  # lazy import to avoid heavy deps during dry-run
        plot_center_node_energy,
        plot_energy_over_time,
        plot_node_distribution,
    )

    network = cfg.create_network()

    if cfg.simulation_config.enable_adcr_link_layer:
        network.adcr_link = cfg.create_adcr_link_layer(network)
    else:
        network.adcr_link = None

    if cfg.path_collector_config.enable_path_collector:
        physical_center = network.get_physical_center()
        if physical_center:
            initial_pos = tuple(physical_center.position)
        else:
            nodes = network.get_regular_nodes() if hasattr(network, "get_regular_nodes") else network.nodes
            initial_pos = (
                sum(n.position[0] for n in nodes) / len(nodes),
                sum(n.position[1] for n in nodes) / len(nodes),
            )
        virtual_center = VirtualCenter(initial_position=initial_pos, enable_logging=True)
        virtual_center.initialize_node_info(network.nodes, initial_time=0)
        network.path_info_collector = cfg.create_path_collector(virtual_center, physical_center)
    else:
        network.path_info_collector = None

    scheduler = create_scheduler(cfg, network)
    if hasattr(network, "path_info_collector") and network.path_info_collector is not None:
        setattr(scheduler, "path_collector", network.path_info_collector)

    set_eetor_config(cfg.eetor_config)

    simulation = cfg.create_energy_simulation(network, scheduler)
    session_dir = Path(simulation.session_dir)

    if hasattr(network, "path_info_collector") and network.path_info_collector is not None:
        archive_path = session_dir / "virtual_center_node_info.csv"
        network.path_info_collector.vc.archive_path = str(archive_path)

    simulation.simulate()

    if not skip_visuals:
        plot_node_distribution(network.nodes, session_dir=str(session_dir))
        plot_energy_over_time(network.nodes, simulation.result_manager.get_results(), session_dir=str(session_dir))
        plot_center_node_energy(network.nodes, simulation.result_manager.get_results(), session_dir=str(session_dir))
    simulation.plot_K_history()

    return session_dir


def rename_session_directory(session_dir: Path, slug: str) -> Path:
    parent = session_dir.parent
    target = parent / f"{session_dir.name}_{slug}"
    counter = 1
    while target.exists():
        counter += 1
        target = parent / f"{session_dir.name}_{slug}_{counter:02d}"
    session_dir.rename(target)
    return target


def write_metadata(target_dir: Path, metadata: Dict[str, object]) -> None:
    metadata_path = target_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the paper experiments (E1–E4 and optional ablations).",
    )
    parser.add_argument(
        "--suite",
        choices=SUITE_NAMES,
        action="append",
        help="Experiment suite to run (default: core). Can be repeated.",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        action="append",
        help="Restrict to one or more topology families (default: all).",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        help="Comma-separated list of node counts (default: 15,30,60,100).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds (default: 10 seeds used in the paper).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for simulation outputs (default: src/experiments/data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running simulations.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only execute the first N runs (after ordering).",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip plotting utilities to shorten long batches.",
    )
    return parser.parse_args(argv)


def expand_int_list(raw: Optional[str], default: Sequence[int]) -> List[int]:
    if raw is None:
        return list(default)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"Invalid integer list: {raw}") from exc


def select_variants(requested_suites: Optional[Sequence[str]]) -> List[ExperimentVariant]:
    suites = list(requested_suites) if requested_suites else ["core"]
    selected = [variant for variant in VARIANTS if variant.suite in suites]
    if not selected:
        raise SystemExit("No experiment variants selected. Check --suite arguments.")
    return selected


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    output_root = args.output_dir.resolve()
    OutputManager.ensure_dir_exists(str(output_root))

    variants = select_variants(args.suite)
    scenario_keys = args.scenario or list(SCENARIOS.keys())
    sizes = expand_int_list(args.sizes, DEFAULT_SIZES)
    seeds = expand_int_list(args.seeds, DEFAULT_SEEDS)

    plan: List[Tuple[str, int, int, ExperimentVariant]] = []
    for scenario_key in scenario_keys:
        for size in sizes:
            for seed in seeds:
                for variant in variants:
                    plan.append((scenario_key, size, seed, variant))

    if args.limit is not None:
        plan = plan[: args.limit]

    print(
        f"Planned runs: {len(plan)} "
        f"(scenarios={scenario_keys}, sizes={sizes}, seeds={len(seeds)}, suites={[v.suite for v in variants]})"
    )

    if args.dry_run:
        for scenario_key, size, seed, variant in plan:
            scenario = SCENARIOS[scenario_key]
            slug = slugify(
                [
                    scenario.key,
                    scenario.label,
                    f"N{size}",
                    f"seed{seed}",
                    variant.key,
                ]
            )
            print(f"[DRY-RUN] {slug}: {variant.description}")
        return

    successes: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for idx, (scenario_key, size, seed, variant) in enumerate(plan, start=1):
        scenario = SCENARIOS[scenario_key]
        slug = slugify([scenario.key, scenario.label, f"N{size}", f"seed{seed}", variant.key])
        print(f"\n[{idx}/{len(plan)}] Running {slug} → {variant.description}")

        start_time = time.time()
        cfg = ConfigManager()
        apply_plan_defaults(cfg, output_root)
        configure_scenario(cfg, scenario, size, seed)
        variant.apply(cfg)
        seed_random_generators(seed)

        try:
            session_dir = run_simulation_with_config(cfg, skip_visuals=args.skip_visuals)
            renamed_dir = rename_session_directory(session_dir, slug)
            duration = time.time() - start_time

            metadata = {
                "slug": slug,
                "variant": variant.key,
                "variant_description": variant.description,
                "suite": variant.suite,
                "scenario_key": scenario.key,
                "scenario_label": scenario.label,
                "distribution_mode": scenario.distribution_mode,
                "enable_energy_hole": scenario.enable_energy_hole,
                "num_nodes": size,
                "seed": seed,
                "started_at": datetime.fromtimestamp(start_time).isoformat(),
                "duration_seconds": round(duration, 3),
                "output_dir": str(renamed_dir.resolve()),
                "config_snapshot": {
                    "node": cfg.node_config.__dict__,
                    "network": cfg.network_config.__dict__,
                    "simulation": cfg.simulation_config.__dict__,
                    "scheduler": cfg.scheduler_config.__dict__,
                    "path_collector": cfg.path_collector_config.__dict__,
                    "eetor": cfg.eetor_config.__dict__,
                },
            }
            write_metadata(renamed_dir, metadata)
            successes.append(metadata)
            print(f"[OK] {slug} finished in {duration/60:.2f} minutes. Results at {renamed_dir}")
        except Exception as exc:
            duration = time.time() - start_time
            failure_info = {
                "slug": slug,
                "variant": variant.key,
                "suite": variant.suite,
                "scenario_key": scenario.key,
                "num_nodes": size,
                "seed": seed,
                "error": repr(exc),
                "duration_seconds": round(duration, 3),
            }
            failures.append(failure_info)
            print(f"[FAILED] {slug}: {exc!r}")

    print("\n=== Batch summary ===")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    if failures:
        for failure in failures:
            print(
                f"  - {failure['slug']} (scenario={failure['scenario_key']}, "
                f"N={failure['num_nodes']}, seed={failure['seed']}) → {failure['error']}"
            )


if __name__ == "__main__":
    main()


