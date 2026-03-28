"""
Generic metric framework for auto-strat-eval.

Provides building blocks for composing multi-dimensional evaluation metrics:
  - JSON output parsing with truncation salvage
  - Fuzzy string matching utilities
  - F-beta score composition (configurable recall/precision weighting)
  - Reward shaping (non-linear penalties for extreme errors)
  - Composite score builder with named sub-metrics

Domain-specific sub-metrics (e.g., food ingredient recall, code correctness)
are defined by the user and plugged into this framework.

Usage:
    from eval.metric import MetricBuilder, parse_json_output, fuzzy_match

    builder = MetricBuilder()
    builder.add("name_accuracy", my_name_scorer, weight=0.2)
    builder.add("recall", my_recall_scorer, weight=0.35)
    builder.add_fbeta("ingredient_f2", my_precision_fn, my_recall_fn, beta=2, weight=0.25)
    builder.add_reward_shaping(my_penalty_fn)
    builder.add_parse_bonus(0.1)

    score = builder.score(prediction, ground_truth)
    detail = builder.score_detailed(prediction, ground_truth)
"""

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable


# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------

def normalize(s: Any) -> str:
    """Lowercase, strip, collapse whitespace. Handles non-string inputs."""
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", " ", s.lower().strip())


def to_number(val: Any) -> float:
    """Coerce a value to float. Handles strings like '100', '100g', '$5.50'."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = re.sub(r"[^\d.]", "", val)
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    return 0.0


def fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Check if two strings are fuzzy-equal after normalization."""
    a, b = normalize(a), normalize(b)
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def best_match(name: str, candidates: list[str], threshold: float = 0.5) -> str | None:
    """Find the best fuzzy match for name in candidates."""
    name_n = normalize(name)
    best, best_score = None, 0.0
    for c in candidates:
        score = SequenceMatcher(None, name_n, normalize(c)).ratio()
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= threshold else None


# ---------------------------------------------------------------------------
# JSON parsing with truncation salvage
# ---------------------------------------------------------------------------

def parse_json_output(text: str, root_key: str | None = None) -> Any | None:
    """
    Parse LLM text output into structured data.

    Handles:
    - Markdown code fences (```json ... ```)
    - Truncated JSON (attempts to close brackets/braces)
    - Root key extraction (e.g., root_key="dishes" extracts from {"dishes": [...]})

    Returns None if parsing fails entirely.
    """
    if not text or text.startswith("ERROR:"):
        return None

    s = text.strip()

    # Strip markdown fences
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
    if s.endswith("```"):
        s = s[:-3].strip()

    # Try direct parse
    try:
        parsed = json.loads(s)
        if root_key and isinstance(parsed, dict) and root_key in parsed:
            return parsed[root_key]
        return parsed
    except json.JSONDecodeError:
        pass

    # Salvage truncated JSON — find last complete object
    for i in range(len(s) - 1, 0, -1):
        if s[i] == "}":
            try:
                candidate = s[: i + 1]
                opens = candidate.count("[") - candidate.count("]")
                candidate += "]" * max(opens, 0)
                opens = candidate.count("{") - candidate.count("}")
                candidate += "}" * max(opens, 0)
                parsed = json.loads(candidate)
                if root_key and isinstance(parsed, dict) and root_key in parsed:
                    return parsed[root_key]
                return parsed
            except json.JSONDecodeError:
                continue

    return None


# ---------------------------------------------------------------------------
# F-beta score
# ---------------------------------------------------------------------------

def fbeta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """
    Compute F-beta score.

    beta=1 → standard F1 (equal weight)
    beta=2 → F2 (recall 4x more important than precision)
    beta=0.5 → F0.5 (precision 4x more important than recall)
    """
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


# ---------------------------------------------------------------------------
# Metric builder
# ---------------------------------------------------------------------------

SubMetricFn = Callable[[Any, Any], float]
PenaltyFn = Callable[[Any, Any], float]


@dataclass
class SubMetric:
    name: str
    fn: SubMetricFn
    weight: float


@dataclass
class FBetaMetric:
    name: str
    precision_fn: SubMetricFn
    recall_fn: SubMetricFn
    beta: float
    weight: float


@dataclass
class MetricBuilder:
    """
    Composable metric builder for multi-dimensional evaluation.

    Assemble sub-metrics, F-beta scores, reward shaping penalties,
    and parse bonuses into a single composite score function.
    """
    sub_metrics: list[SubMetric] = field(default_factory=list)
    fbeta_metrics: list[FBetaMetric] = field(default_factory=list)
    penalty_fns: list[PenaltyFn] = field(default_factory=list)
    parse_bonus: float = 0.0
    parse_fail_score: float = 0.0

    def add(self, name: str, fn: SubMetricFn, weight: float = 1.0) -> "MetricBuilder":
        """Add a sub-metric with a weight."""
        self.sub_metrics.append(SubMetric(name=name, fn=fn, weight=weight))
        return self

    def add_fbeta(
        self,
        name: str,
        precision_fn: SubMetricFn,
        recall_fn: SubMetricFn,
        beta: float = 2.0,
        weight: float = 1.0,
    ) -> "MetricBuilder":
        """Add an F-beta score computed from precision and recall functions."""
        self.fbeta_metrics.append(FBetaMetric(
            name=name, precision_fn=precision_fn, recall_fn=recall_fn,
            beta=beta, weight=weight,
        ))
        return self

    def add_reward_shaping(self, penalty_fn: PenaltyFn) -> "MetricBuilder":
        """Add a reward shaping penalty (multiplier in (0, 1])."""
        self.penalty_fns.append(penalty_fn)
        return self

    def set_parse_bonus(self, bonus: float, fail_score: float = 0.0) -> "MetricBuilder":
        """Set bonus for successful parse, and score for parse failure."""
        self.parse_bonus = bonus
        self.parse_fail_score = fail_score
        return self

    def score(self, prediction: Any, ground_truth: Any) -> float:
        """Compute the composite score."""
        total = self.parse_bonus

        for sm in self.sub_metrics:
            total += sm.fn(prediction, ground_truth) * sm.weight

        for fb in self.fbeta_metrics:
            p = fb.precision_fn(prediction, ground_truth)
            r = fb.recall_fn(prediction, ground_truth)
            total += fbeta_score(p, r, fb.beta) * fb.weight

        for penalty_fn in self.penalty_fns:
            total *= penalty_fn(prediction, ground_truth)

        return min(total, 1.0)

    def score_detailed(self, prediction: Any, ground_truth: Any) -> dict:
        """Compute all sub-scores and return a detailed breakdown."""
        detail = {}

        for sm in self.sub_metrics:
            detail[sm.name] = sm.fn(prediction, ground_truth)

        for fb in self.fbeta_metrics:
            p = fb.precision_fn(prediction, ground_truth)
            r = fb.recall_fn(prediction, ground_truth)
            detail[f"{fb.name}_precision"] = p
            detail[f"{fb.name}_recall"] = r
            detail[fb.name] = fbeta_score(p, r, fb.beta)

        for i, penalty_fn in enumerate(self.penalty_fns):
            name = getattr(penalty_fn, "__name__", f"penalty_{i}")
            detail[name] = penalty_fn(prediction, ground_truth)

        detail["composite"] = self.score(prediction, ground_truth)
        return detail
