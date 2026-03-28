#!/usr/bin/env python3
"""
Evaluation runner — run prompts against labeled dataset and return scores.

This is the mechanical core of auto-strat-eval. Both execution modes
(Claude Code session and API driver) call this for evaluation.

Usage:
    # Single prompt evaluation
    python runner.py --prompt prompts/v5-few-shot.json --backend chrome

    # Compare multiple prompts
    python runner.py --prompt prompts/v5.json prompts/v6.json --backend chrome

    # Regression check (best prompt against all historical metrics)
    python runner.py --prompt prompts/v7.json --regression --backend chrome

    # Use adb backend instead
    python runner.py --prompt prompts/v5.json --backend adb --serial 48181FDAP00A1U
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
PROMPTS_DIR = REPO_ROOT / "eval" / "prompts"
METRICS_DIR = REPO_ROOT / "strategy" / "metrics"


def load_dataset() -> list[dict]:
    """Load labeled examples from dataset/labels/."""
    labels_dir = DATASET_DIR / "labels"
    examples = []
    for lf in sorted(labels_dir.glob("*.json")):
        data = json.loads(lf.read_text())
        img = DATASET_DIR / "images" / data["image"]
        if not img.exists():
            print(f"  SKIP: {lf.name} — image not found: {img}")
            continue
        examples.append({
            "label_file": lf.name,
            "image": str(img),
            "dishes": data["dishes"],
        })
    return examples


def load_prompt(path: str | Path) -> dict:
    """Load a versioned prompt JSON file."""
    path = Path(path)
    if not path.is_absolute():
        path = PROMPTS_DIR / path
    data = json.loads(path.read_text())
    return {
        "name": f"{data['version']}-{data['name']}",
        "prompt": data["food_prompt"],
        "version_file": path.name,
        "metadata": data,
    }


def create_backend(backend: str, **kwargs):
    """Create an LM backend for the model being evaluated."""
    if backend == "chrome":
        from backends.chrome import GeminiNanoChromeLM
        lm = GeminiNanoChromeLM()
        if not lm.ensure_model_ready():
            print("Chrome Gemini Nano model not ready")
            sys.exit(1)
        return lm
    elif backend == "adb":
        from backends.adb import GeminiNanoLM
        return GeminiNanoLM(
            serial=kwargs.get("serial", "48181FDAP00A1U"),
            cooldown=5.0,
            retry_backoff=60.0,
            max_retries=2,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_prompt(lm, prompt: dict, examples: list[dict], metric_fn) -> dict:
    """Run a single prompt against all examples and return scored results."""
    results = []

    for ex in examples:
        img_bytes = Path(ex["image"]).read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        ext = Path(ex["image"]).suffix
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt["prompt"]},
                ],
            }
        ]

        try:
            output = lm(messages=messages)
            raw = output[0] if output else ""
        except Exception as e:
            raw = f"ERROR:{e}"

        detail = metric_fn(raw, ex["dishes"])
        detail["label_file"] = ex["label_file"]
        results.append(detail)

    n = len(results)
    avg = lambda k: sum(r[k] for r in results) / n if n else 0

    return {
        "name": prompt["name"],
        "version_file": prompt["version_file"],
        "composite": round(avg("composite"), 3),
        "dish_name_f1": round(avg("dish_name_f1"), 3),
        "ingredient_recall": round(avg("ingredient_recall"), 3),
        "ingredient_precision": round(avg("ingredient_precision"), 3),
        "weight_mae_score": round(avg("weight_mae_score"), 3),
        "json_parse_rate": round(sum(1 for r in results if r["json_parsed"]) / max(n, 1), 3),
        "count": n,
        "per_example": results,
    }


def load_metric(metric_path: str | None = None):
    """Load a metric function. Defaults to eval/metric.py:score_detailed."""
    if metric_path:
        # Load versioned metric from strategy/metrics/
        import importlib.util
        spec = importlib.util.spec_from_file_location("metric", metric_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.score_detailed
    else:
        from metric import score_detailed
        return score_detailed


def print_results(results: list[dict]):
    """Print a summary table."""
    print(f"\n{'='*80}")
    print(f"{'Prompt':<30} {'Comp':>6} {'Name':>6} {'Recall':>7} {'Prec':>6} {'Weight':>7} {'Parse':>6}")
    print(f"{'='*80}")

    for r in sorted(results, key=lambda x: -x["composite"]):
        print(
            f"{r['name']:<30} "
            f"{r['composite']:>6.3f} "
            f"{r['dish_name_f1']:>6.3f} "
            f"{r['ingredient_recall']:>7.3f} "
            f"{r['ingredient_precision']:>6.3f} "
            f"{r['weight_mae_score']:>7.3f} "
            f"{r['json_parse_rate']:>6.1%}"
        )


def main():
    parser = argparse.ArgumentParser(description="auto-strat-eval runner")
    parser.add_argument("--prompt", nargs="+", required=True, help="Prompt JSON file(s)")
    parser.add_argument("--backend", choices=["chrome", "adb"], default="chrome")
    parser.add_argument("--serial", default="48181FDAP00A1U", help="ADB device serial")
    parser.add_argument("--metric", help="Path to metric .py file (default: eval/metric.py)")
    parser.add_argument("--regression", action="store_true", help="Run against all historical metrics")
    parser.add_argument("--json-out", help="Write full results to JSON file")
    args = parser.parse_args()

    examples = load_dataset()
    if not examples:
        print(f"No labeled examples in {DATASET_DIR}/")
        sys.exit(1)

    print(f"Loaded {len(examples)} labeled examples")

    lm = create_backend(args.backend, serial=args.serial)
    prompts = [load_prompt(p) for p in args.prompt]

    if args.regression:
        # Run the prompt(s) against all historical metrics
        metric_files = sorted(METRICS_DIR.glob("*.py"))
        if not metric_files:
            print("No historical metrics found in strategy/metrics/")
            sys.exit(1)

        for prompt in prompts:
            print(f"\n=== Regression check: {prompt['name']} ===")
            regression_results = []
            for mf in metric_files:
                metric_fn = load_metric(str(mf))
                start = time.time()
                result = run_prompt(lm, prompt, examples, metric_fn)
                result["metric_version"] = mf.stem
                regression_results.append(result)
                elapsed = time.time() - start
                print(f"  {mf.stem}: composite={result['composite']:.3f} ({elapsed:.1f}s)")

    else:
        # Standard evaluation
        metric_fn = load_metric(args.metric)
        all_results = []

        for prompt in prompts:
            print(f"\nRunning: {prompt['name']}...")
            start = time.time()
            result = run_prompt(lm, prompt, examples, metric_fn)
            elapsed = time.time() - start
            result["latency_per_image"] = round(elapsed / max(result["count"], 1), 1)
            all_results.append(result)
            print(f"  Done in {elapsed:.1f}s — composite: {result['composite']:.3f}")

        print_results(all_results)

        if args.json_out:
            Path(args.json_out).write_text(json.dumps(all_results, indent=2, default=str))
            print(f"\nFull results saved to {args.json_out}")

    if hasattr(lm, "close"):
        lm.close()


if __name__ == "__main__":
    main()
