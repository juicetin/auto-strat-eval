#!/usr/bin/env python3
"""
Evaluation runner — run prompts against a labeled dataset and return scores.

This is the mechanical core of auto-strat-eval. Both execution modes
(Claude Code session and API driver) call this for evaluation.

All configuration comes from a project.yaml file — no domain-specific
logic lives here.

Usage:
    # Single prompt evaluation
    python -m eval.runner --config /path/to/project.yaml --prompt prompts/v5.json

    # Compare multiple prompts
    python -m eval.runner --config project.yaml --prompt v5.json v6.json

    # Regression check (best prompt against all historical metrics)
    python -m eval.runner --config project.yaml --prompt v7.json --regression
"""

import argparse
import base64
import importlib.util
import json
import sys
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load project.yaml and resolve relative paths."""
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def resolve(p):
        p = Path(p).expanduser()
        return p if p.is_absolute() else (config_dir / p).resolve()

    cfg["_config_dir"] = config_dir
    cfg["dataset"]["labels_dir"] = resolve(cfg["dataset"]["labels_dir"])
    cfg["dataset"]["image_root"] = resolve(cfg["dataset"]["image_root"])
    cfg["metric"]["module"] = resolve(cfg["metric"]["module"])
    cfg["prompts"]["dir"] = resolve(cfg["prompts"]["dir"])
    cfg["strategy_dir"] = resolve(cfg.get("strategy_dir", "strategy"))
    if "goal" in cfg:
        cfg["goal"] = resolve(cfg["goal"])

    return cfg


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(cfg: dict) -> list[dict]:
    """Load labeled examples using config paths."""
    labels_dir = Path(cfg["dataset"]["labels_dir"])
    image_root = Path(cfg["dataset"]["image_root"])
    image_key = cfg["dataset"].get("image_key", "image")
    gt_key = cfg["dataset"].get("ground_truth_key", "ground_truth")

    examples = []
    for lf in sorted(labels_dir.glob("*.json")):
        if lf.name.startswith("EXAMPLE"):
            continue
        data = json.loads(lf.read_text())
        img_name = data.get(image_key, "")
        # Strip any relative path prefix — just use the filename against image_root
        img_path = image_root / Path(img_name).name
        if not img_path.exists():
            # Try the raw path as-is (might be relative to labels_dir)
            img_path = labels_dir / img_name
        if not img_path.exists():
            print(f"  SKIP: {lf.name} — image not found: {img_name}")
            continue
        examples.append({
            "label_file": lf.name,
            "image": str(img_path),
            "ground_truth": data.get(gt_key, data),
        })
    return examples


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt(path: str, cfg: dict) -> dict:
    """Load a versioned prompt JSON file."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(cfg["prompts"]["dir"]) / path
    data = json.loads(path.read_text())
    prompt_key = cfg["prompts"].get("prompt_key", "prompt")
    return {
        "name": f"{data.get('version', '?')}-{data.get('name', path.stem)}",
        "prompt": data[prompt_key],
        "version_file": path.name,
        "path": str(path),
        "metadata": data,
    }


# ---------------------------------------------------------------------------
# Metric loading
# ---------------------------------------------------------------------------

def load_metric_module(module_path: str | Path):
    """Dynamically load a metric module. Must define score_detailed()."""
    module_path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location("domain_metric", str(module_path))
    mod = importlib.util.module_from_spec(spec)

    # Ensure the module's directory is in sys.path for its own imports
    mod_dir = str(module_path.parent)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)

    spec.loader.exec_module(mod)

    if not hasattr(mod, "score_detailed"):
        raise AttributeError(f"Metric module {module_path} must define score_detailed()")

    return mod


# ---------------------------------------------------------------------------
# Backend creation
# ---------------------------------------------------------------------------

def create_backend(cfg: dict):
    """Create an LM backend from config."""
    backend_cfg = cfg["backend"]
    backend_type = backend_cfg["type"]

    # Add backends directory to path
    backends_dir = str(Path(__file__).parent / "backends")
    if backends_dir not in sys.path:
        sys.path.insert(0, backends_dir)

    if backend_type == "chrome":
        from chrome import GeminiNanoChromeLM
        kwargs = {}
        if "chrome_bin" in backend_cfg:
            kwargs["chrome_bin"] = backend_cfg["chrome_bin"]
        if "profile_dir" in backend_cfg:
            kwargs["profile_dir"] = str(Path(backend_cfg["profile_dir"]).expanduser())
        lm = GeminiNanoChromeLM(**kwargs)
        if not lm.ensure_model_ready():
            print("Chrome Gemini Nano model not ready")
            sys.exit(1)
        return lm
    elif backend_type == "adb":
        from adb import GeminiNanoLM
        return GeminiNanoLM(
            serial=backend_cfg.get("serial", "48181FDAP00A1U"),
            cooldown=backend_cfg.get("cooldown", 5.0),
            retry_backoff=backend_cfg.get("retry_backoff", 60.0),
            max_retries=backend_cfg.get("max_retries", 2),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_prompt(lm, prompt: dict, examples: list[dict], score_fn) -> dict:
    """Run a single prompt against all examples and return scored results."""
    results = []

    for ex in examples:
        img_path = Path(ex["image"])
        img_bytes = img_path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        ext = img_path.suffix.lower()
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

        detail = score_fn(raw, ex["ground_truth"])
        detail["label_file"] = ex["label_file"]
        results.append(detail)

    # Aggregate — collect all numeric keys from the first result
    n = len(results)
    aggregated = {
        "name": prompt["name"],
        "version_file": prompt["version_file"],
        "count": n,
    }

    if n > 0:
        numeric_keys = [k for k, v in results[0].items()
                        if isinstance(v, (int, float)) and k != "label_file"]
        for k in numeric_keys:
            aggregated[k] = round(sum(r.get(k, 0) for r in results) / n, 3)

        # Parse rate as a special aggregate
        if "json_parsed" in results[0]:
            aggregated["json_parse_rate"] = round(
                sum(1 for r in results if r.get("json_parsed")) / n, 3
            )

    aggregated["per_example"] = results
    return aggregated


def print_results(results: list[dict]):
    """Print a summary table. Adapts columns to available metrics."""
    if not results:
        return

    # Determine which metric columns exist
    skip_keys = {"name", "version_file", "count", "per_example", "json_parse_rate",
                 "json_parsed", "latency_per_image", "metric_version"}
    first = results[0]
    metric_cols = [k for k in first if k not in skip_keys
                   and isinstance(first[k], (int, float))]

    # Header
    header = f"{'Prompt':<30}"
    for col in metric_cols:
        label = col[:8]
        header += f" {label:>8}"
    if "json_parse_rate" in first:
        header += f" {'Parse':>6}"
    print(f"\n{'='*(len(header)+2)}")
    print(header)
    print(f"{'='*(len(header)+2)}")

    for r in sorted(results, key=lambda x: -x.get("composite", 0)):
        line = f"{r['name']:<30}"
        for col in metric_cols:
            line += f" {r.get(col, 0):>8.3f}"
        if "json_parse_rate" in r:
            line += f" {r['json_parse_rate']:>6.1%}"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="auto-strat-eval runner")
    parser.add_argument("--config", required=True, help="Path to project.yaml")
    parser.add_argument("--prompt", nargs="+", required=True, help="Prompt JSON file(s)")
    parser.add_argument("--metric", help="Override metric module path")
    parser.add_argument("--regression", action="store_true",
                        help="Run against all historical metrics in strategy/metrics/")
    parser.add_argument("--json-out", help="Write full results to JSON file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    examples = load_dataset(cfg)
    if not examples:
        print("No labeled examples found")
        sys.exit(1)
    print(f"Loaded {len(examples)} labeled examples")

    lm = create_backend(cfg)
    prompts = [load_prompt(p, cfg) for p in args.prompt]

    metric_path = args.metric or cfg["metric"]["module"]

    if args.regression:
        metrics_dir = Path(cfg["strategy_dir"]) / "metrics"
        metric_files = sorted(metrics_dir.glob("*.py"))
        if not metric_files:
            print(f"No historical metrics in {metrics_dir}/")
            sys.exit(1)

        for prompt in prompts:
            print(f"\n=== Regression check: {prompt['name']} ===")
            for mf in metric_files:
                mod = load_metric_module(mf)
                start = time.time()
                result = run_prompt(lm, prompt, examples, mod.score_detailed)
                elapsed = time.time() - start
                print(f"  {mf.stem}: composite={result.get('composite', 0):.3f} ({elapsed:.1f}s)")
    else:
        mod = load_metric_module(metric_path)
        all_results = []

        for prompt in prompts:
            print(f"\nRunning: {prompt['name']}...")
            start = time.time()
            result = run_prompt(lm, prompt, examples, mod.score_detailed)
            elapsed = time.time() - start
            result["latency_per_image"] = round(elapsed / max(result["count"], 1), 1)
            all_results.append(result)
            print(f"  Done in {elapsed:.1f}s — composite: {result.get('composite', 0):.3f}")

        print_results(all_results)

        if args.json_out:
            Path(args.json_out).write_text(json.dumps(all_results, indent=2, default=str))
            print(f"\nFull results saved to {args.json_out}")

    if hasattr(lm, "close"):
        lm.close()


if __name__ == "__main__":
    main()
