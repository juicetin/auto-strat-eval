# auto-strat-eval: Autonomous Prompt & Strategy Optimization for On-Device VLMs

## Overview

auto-strat-eval is an autonomous evaluation and optimization system for on-device vision-language models (VLMs). It combines ideas from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) (autonomous experiment loops with keep/discard discipline) and [DSPy](https://github.com/stanfordnlp/dspy) (declarative prompt optimization) into a three-layer architecture that can evolve not just prompts, but the optimization strategy itself.

The system runs overnight, executing 50-100+ experiments autonomously, and produces a reviewable morning briefing that captures every strategic decision and its reasoning.

### Origin

This design emerged from hands-on prompt optimization work on a food tracking app (Tastimate) that uses Gemini Nano on-device for food identification. During a single session, manual iteration through strategy changes yielded a 48% improvement in composite score — gains that neither DSPy (constrained to prompt text) nor autoresearch (no strategic reasoning) would have discovered independently. The key insight was that the biggest wins came not from rewording prompts, but from structural decisions: adding few-shot examples, changing the metric to F2 weighting, adding reward shaping for weight hallucinations, and switching evaluation infrastructure from adb to Chrome Built-in AI.

---

## 1. Declarative Goal

The system is anchored by a human-authored declarative goal that rarely changes. This replaces the single numeric metric (like autoresearch's `val_bpb`) with a natural language description of intent that the agent must interpret and operationalize.

### Example Goal (Food Tracking)

```
Provide the closest results possible in a human-in-the-loop flow capturing
dish names, ingredients, and weights of all food present in a photo, to allow
a user to accurately track their macro and micro nutrients throughout the day
based on photos of their food.

Key constraints:
- This is HITL: the user can delete incorrect ingredients but may not notice
  missing ones. False negatives are costlier than false positives.
- Calorically dense ingredients (proteins, fats, starches) matter more than
  low-calorie ones (herbs, spices, leafy garnishes), because errors on
  high-calorie items cause larger tracking inaccuracies.
- The model has a 256-token output limit. Multi-pass strategies are acceptable
  if they improve accuracy.
- Latency matters for UX but is secondary to accuracy.
```

### Why Declarative > Numeric

A single metric (composite score, F1, val_bpb) can be gamed. During our manual optimization session, the "nutritionist-role" prompt scored highest under equal-weighted metrics (0.594) but was actually hallucinating weights — it dropped to last place (0.313) when we added reward shaping. The prompt didn't change; our definition of "good" was wrong.

A declarative goal allows the agent to reason about what metrics to derive, rather than blindly optimizing a potentially flawed number. The agent should be able to independently conclude that:
- Recall needs a floor (from the HITL constraint)
- Weight accuracy on rice matters more than on garnish (from the caloric density constraint)
- Parse rate should be 100% (from the structured output requirement)

---

## 2. Three-Layer Architecture

### Layer 1: Declarative Goal (Human-Authored)

- Written in natural language with explicit constraints
- Stored as a versioned file (e.g., `goal.md`)
- Rarely modified — only when the product requirements change
- The agent references this as its north star for all strategic decisions

### Layer 2: Outer Loop — Strategy Evolution (Agent-Driven)

The outer loop is what distinguishes auto-strat-eval from both autoresearch and DSPy. The agent can modify:

- **Metric design**: Which sub-metrics to use, how to weight them, what penalties to apply
- **Evaluation approach**: Single-pass vs multi-pass, which images to test on, how to handle edge cases
- **Prompt architecture**: Few-shot example selection, chain-of-thought vs direct extraction, schema design
- **Multi-pass structure**: Discovery → detail passes, streaming strategies, how to decompose complex scenes

Each strategy change is logged with:
1. **Trigger**: What observation prompted the change ("nutritionist prompt scores high but weight estimates look unrealistic")
2. **Reasoning**: Why the agent believes this change aligns with the declarative goal ("HITL users can't easily verify gram weights, so we need the metric to penalize hallucinated weights more aggressively")
3. **Before/after results**: Concrete measurements showing impact
4. **Decision**: Keep or revert

Expected cadence: ~5-10 strategy changes per overnight run.

### Layer 3: Inner Loop — Experiment Execution (Mechanical)

Within each strategy, the agent runs experiments in a tight loop:

1. Generate or modify a prompt variant
2. Run it against the labeled dataset (12+ images, ~60s per full evaluation)
3. Score against the current strategy's metric
4. Keep if improved, discard if not
5. Log results
6. Repeat

This is where DSPy's optimizers (BootstrapFewShot, COPRO) can plug in directly. The inner loop is mechanical — no strategic reasoning, just try/measure/keep.

Expected cadence: ~10-15 experiments per strategy, ~60-100 total experiments per night.

---

## 3. Infrastructure

### Evaluation Backend: Chrome Built-in AI

The system uses Gemini Nano via Chrome's Built-in AI LanguageModel API, controlled through Playwright. This replaces the original adb-to-Android bridge approach.

Advantages:
- No physical device required
- No rate limiting (ML Kit's ErrorCode 9)
- No app crashes from repeated inference
- 100% JSON parse rate (vs 83% via adb)
- ~5s/image latency (vs 9.6s + rate limit backoffs)
- Runs headlessly for overnight automation

Requirements:
- Chrome Canary with flags set in Local State:
  - `optimization-guide-on-device-model@1`
  - `prompt-api-for-gemini-nano@1`
  - `prompt-api-for-gemini-nano-multimodal-input@1`
- Persistent Chrome profile with downloaded model (~4.3GB)
- Sufficient `/tmp` space for initial model download
- API: `LanguageModel` global (not `window.ai` — deprecated in Chrome 141+)
- Multimodal messages use `{role: "user", content: [{type: "image", value: imageBitmap}, {type: "text", value: promptText}]}`

Implementation: `gemini_nano_chrome_lm.py` — DSPy-compatible LM provider.

### Labeled Dataset

Ground truth images with dish names, ingredients, and gram weights in `dataset/*.json`. Currently 12 images covering diverse food types. The dataset should grow over time, ideally incorporating HITL corrections from production usage.

### Versioned Prompts

All prompts are stored as versioned JSON files in `prompts/`:

```
prompts/
├── v1-initial.json           # Basic, no weights
├── v2-weights.json           # Added gram estimates
├── v3-multipass.json         # Multi-pass for truncation
├── v4-step-by-step.json      # Chain-of-thought
├── v5-few-shot.json          # Few-shot examples (current production)
└── v6-few-shot-compact.json  # Compact output instructions (worse)
```

Each file contains:
- Version, name, date, commit hash
- Description of what changed and why
- The full prompt text (food_prompt, discovery_prompt, dish_detail_prompt)
- Grid search scores (when available)
- `deployed: true/false` flag

---

## 4. The Overnight Run

### Initialization

```
1. Agent reads goal.md (declarative goal)
2. Agent reads current best prompt (the deployed v*.json)
3. Agent reads historical strategy changelog
4. Agent establishes baseline: runs current best against labeled dataset
5. Agent begins outer loop
```

### Outer Loop Cycle

```
FOR each strategy iteration:
  1. OBSERVE: Review inner loop results from current strategy
  2. REASON: Identify what the current metric/approach is missing
     relative to the declarative goal
  3. PROPOSE: Design a new metric, evaluation approach, or prompt
     architecture change
  4. LOG: Record trigger, reasoning, and proposed change
  5. APPLY: Update the evaluation criteria
  6. RUN INNER LOOP: Execute 10-15 experiments under new strategy
  7. EVALUATE: Did results improve relative to the declarative goal?
  8. DECIDE: Keep strategy change or revert
```

### Inner Loop Cycle (per strategy)

```
FOR each experiment:
  1. Modify prompt (text, few-shot examples, structure)
  2. Run against labeled dataset via Chrome LM (~60s)
  3. Score against current strategy's metrics
  4. If improved: keep, commit to version history
  5. If not: discard
  6. Log result
```

### End-of-Night Artifact

The agent produces a morning briefing with three sections:

#### A. Strategy Evolution Narrative

A human-readable changelog showing the arc of reasoning:

```
Strategy 1: Equal-weighted metrics (baseline)
  Trigger: Starting point
  Ran 12 experiments. Best: nutritionist-role (0.594)
  Observation: High recall but weight estimates unrealistic on several images

Strategy 2: F2-weighted recall (recall 4x > precision)
  Trigger: HITL constraint — missing ingredients worse than extras
  Reasoning: F_beta with beta=2 encodes this directly
  Ran 10 experiments. Best: production prompt (0.615)
  Observation: Still rewarding prompts with hallucinated weights

Strategy 3: Reward shaping (>3x weight error = 0.1 penalty)
  Trigger: Strategy 2 let nutritionist prompt score high despite
           predicting 500g butter in a stir-fry
  Reasoning: Non-linear penalty tanks individual bad examples without
             diluting recall signal across the dataset
  Ran 8 experiments. Best: step-by-step (0.437)
  Observation: Nutritionist prompt crashed to last — was being gamed

Strategy 4: Few-shot examples (3 diverse food types)
  Trigger: Model defaulting to 100g for everything (training prior
           anchored to nutrition-label "per 100g" format)
  Reasoning: Show-don't-tell for 3B models. Examples anchor weight
             distribution better than instructions.
  Ran 8 experiments. Best: v5-few-shot (0.612)
  Observation: +48% over production. 2x faster. Best on all dimensions.
```

#### B. Regression Check

The final winning prompt evaluated against every historical metric version:

```
Prompt: v5-few-shot

Metric Version          | Score | Notes
------------------------|-------|------
Strategy 1 (equal-wt)   | 0.587 | Would have placed 2nd (vs 0.594 winner)
Strategy 2 (F2)          | 0.612 | Would have placed 1st
Strategy 3 (F2+reward)   | 0.612 | Placed 1st
Strategy 4 (few-shot)    | 0.612 | Placed 1st (same as S3, this is where it was found)
```

This surfaces regressions: if the final prompt scores poorly under an earlier metric, something may have been sacrificed that the evolved metric stopped measuring.

#### C. Recommendations

Agent's proposed next steps, with reasoning tied to the declarative goal:

```
1. DEPLOY: v5-few-shot as production prompt (high confidence)
2. INVESTIGATE: Weight accuracy on high-calorie ingredients specifically
   — current metric treats all ingredients equally but goal says
   caloric density matters
3. EXPERIMENT NEXT: Calorie-weighted recall metric, where missing 250g
   rice (325 cal) is penalized 7x more than missing 5g sesame oil (45 cal)
4. DATASET: Add more multi-dish images — current dataset is mostly
   single-dish, but production sees 2-3 dish photos frequently
```

---

## 5. Key Design Principles

### Show, Don't Tell (for Small Models)

A 3B parameter model like Gemini Nano doesn't reason — it does next-token prediction via pattern matching. Showing it examples of correct output is fundamentally more effective than giving it rules about what to avoid. This was validated empirically: v5 (few-shot examples) beat v6 (same examples + explicit formatting instructions) because the extra instructions added noise the model had to pattern-match around.

### Metric as Code, Not Config

The evaluation metric should be code that the outer loop agent can modify, not a fixed configuration. The agent needs to be able to:
- Add new sub-metrics (e.g., calorie-weighted recall)
- Change weights between sub-metrics
- Add non-linear penalties (reward shaping)
- Remove sub-metrics that aren't contributing signal

Each metric version is logged and reproducible.

### Goodhart's Law Mitigation

"When a measure becomes a target, it ceases to be a good measure."

Mitigations:
1. **Declarative goal as north star**: The agent checks strategic changes against natural language intent, not just a number
2. **Regression check**: Final prompt evaluated against all historical metrics to catch sacrificed dimensions
3. **Human morning review**: The strategy changelog is the primary review artifact — humans validate reasoning, not just scores
4. **Multiple tracked dimensions**: Even when optimizing a composite, individual sub-scores (recall, precision, weight MAE, parse rate, latency) are always logged

### Version Everything

- Prompts: `prompts/v*.json` with full metadata
- Strategies: Changelog in the morning briefing
- Metrics: Each metric version is a function that can be re-invoked
- Results: Full per-example breakdowns, not just aggregates
- Reasoning: Why each change was made, not just what changed

---

## 6. Comparison with Existing Systems

| Dimension | Autoresearch | DSPy | auto-strat-eval |
|-----------|-------------|------|----------|
| Search space | Code (architecture, optimizer, hyperparams) | Prompt text, few-shot examples | Prompts + strategy + metrics + architecture |
| Metric | Fixed (val_bpb) | Fixed (user-defined) | Evolvable (derived from declarative goal) |
| Autonomy | Full (inner loop only) | Supervised (run once) | Full (both loops), human reviews after |
| Strategic reasoning | None — try/measure/keep | None — optimize within constraints | Agent reasons about what to optimize |
| Logging | results.tsv (what happened) | Optimized program JSON | Strategy changelog (what, why, and whether reasoning held) |
| Human role | Review results morning | Define metric, run optimizer | Author goal, review strategy evolution |
| Guard rails | Single metric, keep/discard | Fixed metric function | Declarative goal + regression check + reasoning audit |
| Best for | Model training hyperparams | Prompt wording within fixed strategy | Multi-dimensional optimization where the strategy itself needs to evolve |

---

## 7. Applicability Beyond Food Tracking

The architecture is domain-agnostic. Any task with these properties is a good fit:

- **On-device or constrained model**: Small models benefit most from prompt engineering and few-shot calibration
- **Multi-dimensional quality**: Success isn't captured by a single number
- **Human-in-the-loop**: The cost of different error types is asymmetric
- **Structured output**: JSON/schema compliance adds a hard constraint dimension
- **Iterative improvement**: A labeled dataset exists or can be built from HITL corrections

Examples:
- Medical image triage (sensitivity vs specificity tradeoffs)
- Document extraction (field recall vs hallucination)
- Voice command parsing (intent accuracy vs slot filling)
- Code generation (correctness vs style vs efficiency)

---

## 8. Open Questions

1. **Outer loop LLM**: The agent doing strategy reasoning needs to be more capable than the model being optimized. For Gemini Nano optimization, Claude or GPT-4 class models are the natural choice. What's the cost/benefit of using a frontier model for ~5-10 strategic reasoning calls per night?

2. **Dataset contamination**: If few-shot examples come from the evaluation dataset (as our v5 examples did), the metric is partially measuring memorization. Should the few-shot selection pool be separate from the evaluation set?

3. **Convergence detection**: When should the agent stop iterating on a strategy and move to the next? Fixed experiment count? Diminishing returns threshold? Or let the agent decide based on the declarative goal?

4. **Multi-model generalization**: If we optimize prompts for Gemini Nano via Chrome, do the improvements transfer to Gemini Nano via ML Kit on Android? The model is the same but the runtime may differ. Should the regression check include cross-platform evaluation?

5. **Continuous learning flywheel**: Production HITL corrections (user deletes/adds ingredients) are ground truth signals. How should these feed back into the labeled dataset and trigger re-optimization?

6. **Strategy branching**: Should the agent be allowed to explore multiple strategy branches in parallel (like git branches), or must it proceed linearly? Parallel branches could discover more but make the morning review harder.

---

## 9. Repository Strategy

### New Repo, Not a Fork

Neither autoresearch nor DSPy are suitable fork targets.

**Autoresearch** is 3 files designed for one specific task (GPU training loops). The experiment loop lives as prose in `program.md`, not reusable code. The only transferable ideas are the keep/discard pattern and structured results logging — both trivial to reimplement.

**DSPy** is a large framework (~50K+ lines) with deep abstractions around signatures, modules, and teleprompters. We'd use ~5% of it (BootstrapFewShot, COPRO) and fight the rest. More fundamentally, DSPy assumes the metric is fixed — the opposite of our core innovation (evolvable metrics via the outer loop).

**The approach:**
- **DSPy as a library dependency** (`pip install dspy`): Used by the inner loop when the outer loop decides "optimize prompt wording within this fixed strategy." No fork needed.
- **Autoresearch as a design reference**: The loop discipline (try/measure/keep/discard), the `results.tsv` logging pattern, and the `program.md` concept of agent instructions as a lightweight skill file. Philosophy, not code.
- **Existing foodtracker work as foundation**: `food_metric.py`, `gemini_nano_chrome_lm.py`, the versioned prompts system, and the grid search runner port directly as Phase 1.

**The repo's novel contribution** is the outer loop (strategy evolution with declarative goal alignment) and the morning briefing (reviewable reasoning chain). Neither exists in either parent project.

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Exists)
- [x] Chrome Built-in AI evaluation backend (`gemini_nano_chrome_lm.py`)
- [x] Labeled dataset (12 images with ground truth)
- [x] Scoring metrics with F2 + reward shaping (`food_metric.py`)
- [x] Grid search runner (`grid_search.py`)
- [x] Versioned prompt system (`prompts/v*.json`)

### Phase 2: Inner Loop Automation
- [ ] Autonomous experiment loop with keep/discard logic
- [ ] Automatic prompt variant generation (beyond manual grid search)
- [ ] DSPy integration for BootstrapFewShot/COPRO within a fixed strategy
- [ ] Results logging to structured TSV/JSON with per-experiment detail

### Phase 3: Outer Loop
- [ ] Declarative goal file (`goal.md`)
- [ ] Strategy evolution agent (reads goal, proposes metric/approach changes)
- [ ] Strategy changelog format and versioning
- [ ] Metric-as-code system (metric versions are callable functions)
- [ ] Regression check: evaluate final prompt against all historical metrics

### Phase 4: Morning Briefing
- [ ] Narrative changelog generator
- [ ] Regression check table
- [ ] Recommendations with reasoning tied to goal
- [ ] Concise format reviewable in 2 minutes

### Phase 5: Continuous Improvement
- [ ] HITL correction ingestion (production → dataset)
- [ ] Automated re-optimization triggers
- [ ] Cross-platform regression (Chrome vs ML Kit)
- [ ] Dataset diversity analysis and gap detection
