"""
A/B experiment runner.

Iterates the grid defined in configs/experiments.yaml and runs ensure_ready()
for each configuration, recording metrics.

Usage (Python):
    from src.eval.run_experiments import run_experiments
    from src.app.settings import Settings
    results = run_experiments(Settings())

Optional thin CLI wrapper:
    python -m src.eval.run_experiments
"""

from __future__ import annotations

import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_EXPERIMENTS_CONFIG = Path("configs/experiments.yaml")


def load_grid() -> dict[str, list]:
    """Load the experiment grid from configs/experiments.yaml."""
    if not _EXPERIMENTS_CONFIG.exists():
        logger.warning("experiments.yaml not found; returning empty grid")
        return {}
    with _EXPERIMENTS_CONFIG.open() as f:
        data = yaml.safe_load(f) or {}
    return data.get("grid", {})


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a grid dict into a list of all combinations."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_experiments(
    base_settings: Any,
    grid: dict[str, list] | None = None,
) -> list[dict[str, Any]]:
    """
    Run bootstrap for each setting combination in the grid.

    Args:
        base_settings: Base Settings instance (values overridden per experiment).
        grid: Dict of {setting_name: [values]}. Defaults to configs/experiments.yaml.

    Returns:
        List of result dicts with config + metrics.
    """
    from src.app.bootstrap import ensure_ready
    from src.app.settings import Settings

    if grid is None:
        grid = load_grid()

    combos = expand_grid(grid)
    logger.info("Running %d experiment combinations", len(combos))

    results = []
    for i, combo in enumerate(combos):
        logger.info("Experiment %d/%d: %s", i + 1, len(combos), combo)
        t0 = datetime.utcnow()
        status = "ok"
        error = ""

        # Build settings for this combination
        settings_dict = base_settings.model_dump()
        settings_dict.update(combo)

        try:
            exp_settings = Settings(**settings_dict)
            ctx = ensure_ready(exp_settings)
        except Exception as e:
            logger.error("Experiment %d failed: %s", i + 1, e)
            status = "error"
            error = str(e)

        t1 = datetime.utcnow()
        results.append({
            **combo,
            "status": status,
            "error": error,
            "duration_ms": int((t1 - t0).total_seconds() * 1000),
            "run_at": t0.isoformat(),
        })

    # Write results
    out_path = Path("data/processed") / "experiment_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Experiment results written to %s", out_path)

    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    from src.app.settings import Settings
    run_experiments(Settings())
