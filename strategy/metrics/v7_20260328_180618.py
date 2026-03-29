"""
Bridge: wires food-specific scoring functions into auto-strat-eval's MetricBuilder.

This file is referenced by project.yaml and loaded dynamically by the runner.
It must define:
  - parse_output(text: str) -> Any | None
  - score_detailed(prediction: Any, ground_truth: Any) -> dict
  - build_metric() -> MetricBuilder (optional, for programmatic use)
"""

import sys
from pathlib import Path

# Add auto-strat-eval to path for imports
_auto_strat_eval = Path(__file__).parent.parent / "auto-strat-eval"
if str(_auto_strat_eval) not in sys.path:
    sys.path.insert(0, str(_auto_strat_eval))

from eval.metric import (
    MetricBuilder,
    parse_json_output,
    normalize,
    to_number,
    fuzzy_match,
    best_match,
    fbeta_score,
)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output(text: str):
    """Parse Gemini Nano's raw text into a list of dish dicts."""
    return parse_json_output(text, root_key="dishes")


# ---------------------------------------------------------------------------
# Food-specific sub-metrics
# ---------------------------------------------------------------------------

def dish_name_f1(pred_dishes, gt_dishes) -> float:
    """F1 score for dish name matching (fuzzy)."""
    if not gt_dishes:
        return 1.0 if not pred_dishes else 0.0
    if not pred_dishes:
        return 0.0

    pred_names = [d.get("name", "") for d in pred_dishes]
    gt_names = [d.get("name", "") for d in gt_dishes]

    matched_gt = set()
    tp = 0
    for pn in pred_names:
        for i, gn in enumerate(gt_names):
            if i not in matched_gt and fuzzy_match(pn, gn, threshold=0.5):
                tp += 1
                matched_gt.add(i)
                break

    precision = tp / len(pred_names) if pred_names else 0
    recall = tp / len(gt_names) if gt_names else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _flatten_ingredients(dishes, key="name"):
    """Extract normalized ingredient names or weights from a list of dishes."""
    items = []
    for d in (dishes or []):
        for ing in d.get("ingredients", []):
            if isinstance(ing, dict):
                items.append(ing)
            else:
                items.append({"name": str(ing)})
    return items


def ingredient_recall(pred_dishes, gt_dishes) -> float:
    """What fraction of ground truth ingredients were found?"""
    gt_ings = [normalize(i.get("name", "")) for i in _flatten_ingredients(gt_dishes)]
    if not gt_ings:
        return 1.0

    pred_ings = [normalize(i.get("name", "")) for i in _flatten_ingredients(pred_dishes)]

    found = 0
    used = set()
    for gt_name in gt_ings:
        for i, pred_name in enumerate(pred_ings):
            if i not in used and fuzzy_match(gt_name, pred_name, threshold=0.55):
                found += 1
                used.add(i)
                break

    return found / len(gt_ings)


def ingredient_precision(pred_dishes, gt_dishes) -> float:
    """What fraction of predicted ingredients are real (not hallucinated)?"""
    pred_ings = [normalize(i.get("name", "")) for i in _flatten_ingredients(pred_dishes)]
    if not pred_ings:
        return 1.0 if not gt_dishes else 0.0

    gt_ings = [normalize(i.get("name", "")) for i in _flatten_ingredients(gt_dishes)]
    if not gt_ings:
        return 0.0

    correct = 0
    used = set()
    for pred_name in pred_ings:
        for i, gt_name in enumerate(gt_ings):
            if i not in used and fuzzy_match(pred_name, gt_name, threshold=0.55):
                correct += 1
                used.add(i)
                break

    return correct / len(pred_ings)


def weight_mae_score(pred_dishes, gt_dishes) -> float:
    """Score based on mean absolute error of weight estimates (1.0 = perfect)."""
    gt_weights = {}
    for ing in _flatten_ingredients(gt_dishes):
        if "amount_g" in ing:
            gt_weights[normalize(ing["name"])] = to_number(ing["amount_g"])

    if not gt_weights:
        return 1.0

    pred_weights = {}
    for ing in _flatten_ingredients(pred_dishes):
        if "amount_g" in ing:
            pred_weights[normalize(ing["name"])] = to_number(ing["amount_g"])

    if not pred_weights:
        return 0.0

    errors = []
    for gt_name, gt_g in gt_weights.items():
        match = best_match(gt_name, list(pred_weights.keys()))
        if match is not None:
            pred_g = pred_weights[match]
            rel_error = min(abs(pred_g - gt_g) / max(gt_g, 1), 1.0)
            errors.append(rel_error)
        else:
            errors.append(1.0)

    return max(0.0, 1.0 - sum(errors) / len(errors))


def weight_hallucination_penalty(pred_dishes, gt_dishes) -> float:
    """Non-linear penalty for wildly wrong weights (>3x off → 0.3, >5x → 0.1)."""
    gt_weights = {}
    for ing in _flatten_ingredients(gt_dishes):
        if "amount_g" in ing:
            gt_weights[normalize(ing["name"])] = to_number(ing["amount_g"])

    if not gt_weights:
        return 1.0

    pred_weights = {}
    for ing in _flatten_ingredients(pred_dishes):
        if "amount_g" in ing:
            pred_weights[normalize(ing["name"])] = to_number(ing["amount_g"])

    if not pred_weights:
        return 1.0

    worst_ratio = 1.0
    for gt_name, gt_g in gt_weights.items():
        if gt_g <= 0:
            continue
        match = best_match(gt_name, list(pred_weights.keys()))
        if match is not None:
            pred_g = pred_weights[match]
            if pred_g <= 0:
                worst_ratio = max(worst_ratio, 10.0)
                continue
            ratio = max(pred_g / gt_g, gt_g / pred_g)
            worst_ratio = max(worst_ratio, ratio)

    if worst_ratio > 5:
        return 0.3
    if worst_ratio > 3:
        return 0.6
    return 1.0


# ---------------------------------------------------------------------------
# Metric builder
# ---------------------------------------------------------------------------

def build_metric() -> MetricBuilder:
    """Compose the food identification metric using auto-strat-eval framework."""
    return (
        MetricBuilder()
        .add("dish_name_f1", dish_name_f1, weight=0.10)
        .add("ingredient_recall", ingredient_recall, weight=0.15)
        .add_fbeta("ingredient_f2", ingredient_precision, ingredient_recall, beta=2, weight=0.35)
        .add("weight_mae_score", weight_mae_score, weight=0.20)
        .add("weight_hallucination_penalty", weight_hallucination_penalty, weight=0.10)
        .set_parse_bonus(0.1, fail_score=0.0)
    )


_metric = build_metric()


# ---------------------------------------------------------------------------
# Required interface for auto-strat-eval runner
# ---------------------------------------------------------------------------

def score_detailed(raw_output: str, ground_truth) -> dict:
    """Score a single prediction against ground truth. Called by runner.py."""
    pred_dishes = parse_output(raw_output)

    if pred_dishes is None:
        return {
            "composite": 0.0,
            "json_parsed": False,
            "dish_name_f1": 0.0,
            "ingredient_recall": 0.0,
            "ingredient_precision": 0.0,
            "ingredient_f2": 0.0,
            "weight_mae_score": 0.0,
            "weight_hallucination_penalty": 1.0,
            "pred_dishes": None,
            "raw_output": raw_output[:500],
        }

    # Normalize: ensure pred_dishes is a list of dicts with expected keys
    if isinstance(pred_dishes, list):
        normalized = []
        for d in pred_dishes:
            if isinstance(d, dict):
                normalized.append(d)
            elif isinstance(d, str):
                # Model returned a list of strings (dish names only)
                normalized.append({"name": d, "ingredients": []})
        pred_dishes = normalized
    elif isinstance(pred_dishes, dict):
        # Single dish returned as object
        pred_dishes = [pred_dishes]

    detail = _metric.score_detailed(pred_dishes, ground_truth)
    detail["json_parsed"] = True
    detail["pred_dishes"] = pred_dishes
    detail["raw_output"] = raw_output[:500]
    return detail
