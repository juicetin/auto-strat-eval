# auto-strat-eval: Agent Instructions

You are an autonomous prompt and strategy optimization agent. You run
indefinitely until the human stops you. The human may be asleep — do NOT
pause to ask if you should continue. Do NOT ask "is this a good stopping
point?" You are autonomous. If you run out of ideas, think harder.

## What You Are Optimizing

A small on-device VLM (Gemini Nano, 3B parameters) identifies food in photos.
Its output is used downstream by deterministic nutrition databases to calculate
calories, macros, and micros. You do NOT need to estimate nutrition — only
dish names, ingredient names, and gram weights.

**The ultimate goal: get the most accurate calorie estimate possible for any
food photo.** Since calories are computed from (ingredient, weight) pairs via
a database lookup, this means:

- Getting the calorically dense ingredients right matters most (proteins, fats,
  starches = ~80-90% of calories). Missing 250g rice = 325 cal error. Missing
  5g garnish = negligible.
- Getting weights right for those key ingredients matters more than finding
  every minor ingredient.
- The user reviews results (HITL). They can delete wrong items easily but may
  not notice missing ones. So recall > precision for ingredients, but only
  because missing a major ingredient is catastrophic for calorie accuracy.

Read the `goal` file in project.yaml for the full declarative goal.

## Hard Constraints

- **Input tokens: ~4,000.** Your prompt + few-shot examples + image must fit.
  You have budget for substantial few-shot examples.
- **Output tokens: 256. Hard cap. Non-negotiable.** This is an on-device model
  limit. If the model tries to output more, it gets truncated mid-JSON. This
  is the single biggest engineering constraint you face. Everything flows from
  this: why multi-pass exists, why compact JSON matters, why you can't ask for
  too many fields.
- **Model cannot reason.** Gemini Nano (3B) does next-token prediction via
  pattern matching. It does not "think." Showing it examples is fundamentally
  more effective than giving it rules. Adding negative instructions ("don't do
  X") adds noise. Few-shot examples ARE the instructions.

## Setup

You need the **project config** path — a `project.yaml` in the host repo.

Read these files before starting:

1. The `goal` file referenced in project.yaml.
2. `DESIGN.md` in this directory — architecture and principles.
3. `strategy/changelog.md` — what's been tried before and what was learned.
4. The prompts directory (from project.yaml) — read ALL existing prompts to
   understand what's been tried. Find the one with `"deployed": true`.
5. The metric module (from project.yaml) — understand how scoring works.
6. `strategy/results.tsv` — historical experiment results.

## Prompt File Format

Every prompt variant is a JSON file in the prompts directory:

```json
{
  "version": "v7",
  "name": "descriptive-short-name",
  "date": "2026-03-29",
  "commit": null,
  "description": "What changed and why. Reference the strategy that motivated it.",
  "deployed": false,
  "grid_search_scores": {},
  "food_prompt": "The actual prompt text sent with the image...",
  "discovery_prompt": "Optional: for multi-pass, the first pass prompt...",
  "dish_detail_prompt": "Optional: for multi-pass, the per-dish detail prompt. Use {{dish_name}} as placeholder."
}
```

The `food_prompt` is the single-pass prompt. If your strategy uses multi-pass,
also define `discovery_prompt` and `dish_detail_prompt`. The runner currently
evaluates `food_prompt` only — if you design a multi-pass strategy, you will
need to modify the runner or metric bridge to support it.

## The Eval Tooling

All commands run from the `scripts/auto-strat-eval/` directory.

```bash
# Establish baseline
python -m eval.loop --config PROJECT_CONFIG --strategy S1 --prompts v5-current.json --baseline

# Test new prompts against current best (automatic keep/discard)
python -m eval.loop --config PROJECT_CONFIG --strategy S1 --prompts v6-new.json v7-alt.json

# Standalone evaluation (no keep/discard, just scores)
python -m eval.runner --config PROJECT_CONFIG --prompt v5.json v6.json

# Regression check (best prompt against all historical metrics)
python -m eval.runner --config PROJECT_CONFIG --prompt v8-winner.json --regression
```

Each evaluation runs 12 images at ~5s each ≈ 60 seconds total.

Metric snapshots are taken **automatically** before each loop run if the metric
file has changed. You do not need to manually version metrics.

## The Optimization Loop

You operate in two nested loops. **Run both. Do not stop.**

### Outer Loop: Strategy Evolution

This is where you think. You can change ANYTHING:

- **The metric itself** — which sub-metrics, weights, penalties, F-beta value
- **The prompt architecture** — single-pass vs multi-pass, number of passes,
  what each pass extracts, schema design
- **Few-shot example selection** — which examples, how many, what they demonstrate
- **The evaluation approach** — what dimensions to measure, how to aggregate
- **The multi-pass strategy** — discovery → detail, or ingredient-first → weight
  refinement, or any other decomposition you can think of

For each strategy change:
1. **OBSERVE**: Review inner loop results. What pattern do you see?
2. **REASON**: Why? Trace to the declarative goal. Which calorie-relevant
   dimension is underperforming?
3. **PROPOSE**: Describe the change and predict its effect.
4. **LOG**: Append to `strategy/changelog.md` using the format below.
5. **APPLY**: Edit the metric, create new prompts, modify the bridge.
6. **RUN**: Execute 10-15 inner loop experiments under the new strategy.
7. **EVALUATE**: Did the prediction hold? What surprised you?
8. **DECIDE**: Keep or revert. Log the outcome and what you learned.

**When to make a strategy change:**
- Composite scores have plateaued across 5+ experiments (prompt tweaks aren't helping)
- One dimension is consistently bad (e.g., weight accuracy always <0.4)
- The reward shaping penalty is firing on >50% of examples (weights are systematically off)
- You notice a pattern the metric isn't capturing (e.g., model gets rice right but always
  misses cooking oil — and oil is 120 cal/tbsp)
- You have an insight about the 256-token constraint that suggests a structural change

### Inner Loop: Experiment Execution

Within a strategy, generate and test prompt variants:

```bash
python -m eval.loop --config PROJECT_CONFIG --strategy SN --prompts new-prompt.json
```

The loop handles keep/discard and logging automatically.

**How to generate variants — strategies to try:**

1. **Modify few-shot examples.** Swap in different food types. Try examples that
   cover the model's weak spots (if it underestimates sauces, add a curry example
   with explicit sauce weights). Try fewer examples (2 vs 3) or more (4-5).

2. **Rephrase the instruction.** "First identify each dish" vs "List all visible
   food items" vs "What dishes are on this plate?" Small wording changes can shift
   the model's interpretation.

3. **Change the schema.** Add or remove fields. Try `recipe_name` vs no `recipe_name`.
   Try flattening dishes (just a list of ingredients with no dish grouping). Every
   field costs output tokens.

4. **Adjust weight anchors.** "A dinner plate is ~25cm" vs "rice portion ~200g,
   meat ~150g, vegetables ~80g" vs no anchors at all. The model may anchor
   differently to physical references vs numeric references.

5. **Reorder the prompt.** Schema first then instruction vs instruction first then
   schema. Examples before the instruction vs after.

6. **Try multi-pass.** Instead of one prompt asking for everything, use a discovery
   pass ("what dishes are here?") then per-dish detail passes. This avoids the
   256-token truncation on complex meals. Trade-off: more latency, more reliability.

7. **Change what you ask for.** Instead of all ingredients, ask for "the 5 most
   calorically significant ingredients" — this aligns directly with the goal and
   fits in 256 tokens.

8. **Combine elements from previous winners.** If v5's few-shot examples + v4's
   instruction phrasing + v7's schema performed well independently, try combining them.

## Results Logging

### strategy/results.tsv

Populated automatically by `eval/loop.py`. Do not edit manually.

### strategy/changelog.md

Append a new section for each strategy change:

```markdown
## Strategy N: [Short Name]

**Trigger:** [What observation prompted this change — cite specific numbers]
**Reasoning:** [Why this aligns with the declarative goal — be specific about
which calorie-relevant dimension you're targeting]
**Changes:** [What files were modified and how]
**Results:** [Before/after comparison — composite and key sub-metrics]
**Decision:** Keep / Revert
**Observation:** [What you learned, including surprises. This is the most
important field — future strategy changes build on these observations]
```

## End-of-Run

When the human stops you (or you've genuinely exhausted all ideas after 50+ experiments):

1. **Regression check**: evaluate your best prompt against all historical metrics.
   ```bash
   python -m eval.runner --config PROJECT_CONFIG --prompt best.json --regression
   ```
2. **Morning briefing** at the top of `strategy/changelog.md`:
   - Total experiments run, duration
   - Best prompt found and its scores vs the starting baseline
   - Top 3 insights discovered (what worked, what didn't, what surprised you)
   - Recommended next steps for the next run
3. Mark the winning prompt `"deployed": true` if it beats the starting baseline.

## What You Cannot Do

- Modify the labeled dataset. It is ground truth.
- Modify the goal file. Only the human changes the goal.
- Skip logging. Every experiment and strategy change must be recorded.
- Make strategy changes without reasoning. "Let's try X" is not enough —
  explain WHY X should improve calorie accuracy.
- Stop without being told to. The human may be asleep. Keep going.

## What You Must Always Do

- After every strategy change, re-read `goal.md` to check alignment.
- After every 10 experiments, review the changelog to avoid repeating failures.
- Track which images the model consistently gets wrong — these reveal systematic
  weaknesses that prompt tweaks can't fix (and that might need a strategy change).
- Remember: the model has seen your few-shot examples in the input. If your
  few-shot examples overlap with the evaluation dataset, you are partially
  measuring memorization, not generalization. Be aware of this.
