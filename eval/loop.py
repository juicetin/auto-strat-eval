#!/usr/bin/env python3
"""
Inner loop automation — mechanical keep/discard cycle.

Runs a list of prompt variants against the dataset, compares to the current
best, and keeps or discards each. Logs all results to strategy/results.tsv.

Automatically snapshots the metric before each run if it has changed since
the last snapshot (guarantees regression check has complete history).

Usage:
    # Test prompts against current best, keep winner
    python -m eval.loop --config /path/to/project.yaml \
        --strategy S1 \
        --prompts v6-new.json v7-experiment.json

    # Just run baseline (no comparison)
    python -m eval.loop --config /path/to/project.yaml \
        --strategy S1 \
        --prompts v5-few-shot.json \
        --baseline
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.runner import load_config, load_dataset, create_backend, load_prompt, run_prompt, load_metric_module, print_results


# ---------------------------------------------------------------------------
# Metric snapshotting
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    """SHA256 hash of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def snapshot_metric_if_changed(cfg: dict) -> str | None:
    """
    Check if the current metric module has changed since the last snapshot.
    If so, copy it to strategy/metrics/ with a versioned name.
    Returns the snapshot filename if created, None if unchanged.
    """
    metric_path = Path(cfg["metric"]["module"])
    metrics_dir = Path(cfg["strategy_dir"]) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    current_hash = _file_hash(metric_path)

    # Check against existing snapshots
    existing = sorted(metrics_dir.glob("*.py"))
    if existing:
        last_hash = _file_hash(existing[-1])
        if last_hash == current_hash:
            return None  # Unchanged

    # Determine version number
    version = len(existing) + 1
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"v{version}_{timestamp}.py"
    snapshot_path = metrics_dir / snapshot_name

    shutil.copy2(metric_path, snapshot_path)
    print(f"  Metric snapshot: {snapshot_name} (hash: {current_hash})")
    return snapshot_name


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def append_result(cfg: dict, strategy: str, result: dict, status: str, description: str):
    """Append a result row to strategy/results.tsv."""
    results_path = Path(cfg["strategy_dir"]) / "results.tsv"

    # Ensure header exists
    if not results_path.exists() or results_path.stat().st_size == 0:
        results_path.write_text(
            "timestamp\tstrategy\tprompt\tcomposite\trecall\tprecision\t"
            "weight_mae\tparse_rate\tlatency\tstatus\tdescription\n"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = (
        f"{timestamp}\t"
        f"{strategy}\t"
        f"{result.get('version_file', '?')}\t"
        f"{result.get('composite', 0):.3f}\t"
        f"{result.get('ingredient_recall', result.get('ingredient_f2_recall', 0)):.3f}\t"
        f"{result.get('ingredient_precision', result.get('ingredient_f2_precision', 0)):.3f}\t"
        f"{result.get('weight_mae_score', 0):.3f}\t"
        f"{result.get('json_parse_rate', 0):.3f}\t"
        f"{result.get('latency_per_image', 0):.1f}\t"
        f"{status}\t"
        f"{description}\n"
    )

    with open(results_path, "a") as f:
        f.write(row)


# ---------------------------------------------------------------------------
# Keep/discard logic
# ---------------------------------------------------------------------------

def run_loop(cfg: dict, strategy: str, prompt_paths: list[str],
             baseline: bool = False, description_prefix: str = ""):
    """
    Run the inner loop: evaluate prompts, compare to best, keep or discard.

    Args:
        cfg: Loaded project config
        strategy: Strategy label (e.g., "S1", "S3")
        prompt_paths: List of prompt JSON filenames to evaluate
        baseline: If True, just run and log without keep/discard comparison
        description_prefix: Optional prefix for result descriptions
    """
    # Snapshot metric before running (automatic versioning)
    snapshot = snapshot_metric_if_changed(cfg)
    if snapshot:
        print(f"  New metric version saved: {snapshot}")

    # Load dataset, backend, metric
    examples = load_dataset(cfg)
    if not examples:
        print("No labeled examples found")
        return None

    print(f"Loaded {len(examples)} labeled examples")

    lm = create_backend(cfg)
    metric_mod = load_metric_module(cfg["metric"]["module"])

    prompts = [load_prompt(p, cfg) for p in prompt_paths]
    all_results = []

    for prompt in prompts:
        print(f"\n  Running: {prompt['name']}...")
        start = time.time()
        result = run_prompt(lm, prompt, examples, metric_mod.score_detailed)
        elapsed = time.time() - start
        result["latency_per_image"] = round(elapsed / max(result["count"], 1), 1)
        all_results.append(result)
        print(f"  Done in {elapsed:.1f}s — composite: {result.get('composite', 0):.3f}")

    if baseline:
        # Just log everything as baseline
        for result in all_results:
            desc = f"{description_prefix}baseline" if description_prefix else "baseline"
            append_result(cfg, strategy, result, "baseline", desc)
        print_results(all_results)
        if hasattr(lm, "close"):
            lm.close()
        return all_results

    # Keep/discard: find the best and compare to current best in results.tsv
    results_path = Path(cfg["strategy_dir"]) / "results.tsv"
    current_best_score = 0.0

    if results_path.exists():
        lines = results_path.read_text().strip().split("\n")
        for line in lines[1:]:  # Skip header
            parts = line.split("\t")
            if len(parts) >= 10 and parts[9] in ("keep", "baseline"):
                try:
                    score = float(parts[3])
                    current_best_score = max(current_best_score, score)
                except ValueError:
                    pass

    print(f"\n  Current best composite: {current_best_score:.3f}")

    # Evaluate each result
    best_new = None
    for result in all_results:
        composite = result.get("composite", 0)
        if composite > current_best_score:
            status = "keep"
            current_best_score = composite
            best_new = result
            print(f"  KEEP: {result['name']} ({composite:.3f} > previous best)")
        else:
            status = "discard"
            print(f"  DISCARD: {result['name']} ({composite:.3f} <= {current_best_score:.3f})")

        desc = f"{description_prefix}{result['name']}" if description_prefix else result["name"]
        append_result(cfg, strategy, result, status, desc)

    print_results(all_results)

    if hasattr(lm, "close"):
        lm.close()

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="auto-strat-eval inner loop")
    parser.add_argument("--config", required=True, help="Path to project.yaml")
    parser.add_argument("--strategy", required=True, help="Strategy label (e.g., S1)")
    parser.add_argument("--prompts", nargs="+", required=True, help="Prompt JSON files to test")
    parser.add_argument("--baseline", action="store_true", help="Log as baseline (no keep/discard)")
    parser.add_argument("--desc", default="", help="Description prefix for results")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"=== Inner Loop: Strategy {args.strategy} ===")
    print(f"  Prompts: {args.prompts}")
    print(f"  Baseline: {args.baseline}")

    run_loop(
        cfg=cfg,
        strategy=args.strategy,
        prompt_paths=args.prompts,
        baseline=args.baseline,
        description_prefix=args.desc,
    )


if __name__ == "__main__":
    main()
