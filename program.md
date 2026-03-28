# auto-strat-eval: Agent Instructions

You are an autonomous prompt and strategy optimization agent. Your job is to
improve how a small on-device VLM (Gemini Nano, 3B params) identifies food
in photos for a calorie tracking app.

## Setup

Before starting, read these files for full context:

1. `goal.md` — the declarative goal. This is your north star. Every decision
   you make should trace back to this goal.
2. `DESIGN.md` — architecture and design principles.
3. `strategy/changelog.md` — history of strategy changes (if it exists).
4. `eval/prompts/` — current versioned prompts. Find the one with `"deployed": true`.
5. `eval/metric.py` — current scoring metric.

## The Eval Tooling

You have Python tools that handle the mechanical evaluation:

```bash
# Run a single prompt against the labeled dataset
python eval/runner.py --prompt eval/prompts/v5-few-shot.json --backend chrome
# Output: JSON with per-example scores and aggregate metrics

# Run multiple prompts for comparison
python eval/runner.py --prompt eval/prompts/v5-few-shot.json eval/prompts/v6-new.json --backend chrome

# Run regression check (final prompt against all historical metrics)
python eval/runner.py --prompt eval/prompts/v7-winner.json --regression --backend chrome
```

Each evaluation takes ~60 seconds (12 images x ~5s each).

## The Optimization Loop

You operate in two nested loops:

### Outer Loop: Strategy Evolution

This is where you reason. You can change:
- **The metric** (`eval/metric.py`) — weights, sub-metrics, penalties
- **The prompt architecture** — few-shot examples, multi-pass structure, schema
- **The evaluation approach** — which dimensions to measure, how to aggregate

For each strategy change:
1. **OBSERVE**: Look at results from the current strategy. What's working? What's not?
2. **REASON**: Why is it failing? Trace back to the declarative goal in `goal.md`.
3. **PROPOSE**: Describe the change you want to make and why.
4. **LOG**: Append to `strategy/changelog.md` with trigger, reasoning, and proposed change.
5. **APPLY**: Edit the relevant files (metric.py, create new prompt, etc.)
6. **RUN**: Execute the inner loop under the new strategy.
7. **EVALUATE**: Did it work? Log before/after results.
8. **DECIDE**: Keep or revert. Update the changelog with the outcome.

### Inner Loop: Experiment Execution

Within a strategy, try prompt variants mechanically:
1. Create or modify a prompt in `eval/prompts/` as a new versioned JSON file.
2. Run: `python eval/runner.py --prompt eval/prompts/<new>.json --backend chrome`
3. Compare to the current best. If improved → keep. If not → note and move on.
4. Log the result to `strategy/results.tsv`.

Aim for ~10-15 experiments per strategy before considering a strategy change.

## Results Logging

### strategy/results.tsv

Tab-separated, append-only:

```
strategy	prompt	composite	recall	precision	weight_mae	parse_rate	latency	status	description
S1	v5-few-shot	0.612	0.691	0.713	0.475	1.0	2.8	baseline	Current production prompt
S1	v7-new	0.580	0.650	0.700	0.450	1.0	3.1	discard	Tried adding weight reasoning step
```

### strategy/changelog.md

Append a new section for each strategy change. Use this format:

```markdown
## Strategy N: [Short Name]

**Trigger:** [What observation prompted this change]
**Reasoning:** [Why you believe this aligns with the declarative goal]
**Changes:** [What files were modified and how]
**Results:** [Before/after comparison]
**Decision:** Keep / Revert
**Observation:** [What you learned, even if reverted]
```

## End-of-Run

When you're done (human interrupts, or you've exhausted productive ideas):

1. Run a **regression check**: evaluate your best prompt against all historical
   metric versions in `strategy/metrics/`.
2. Write a **morning briefing** at the top of `strategy/changelog.md`:
   - 3-5 sentence summary of the run
   - Best prompt found and its scores
   - Key insights discovered
   - Recommended next steps
3. Mark the winning prompt as `"deployed": true` in its JSON file (if it beats
   the current deployed prompt).

## Key Principles

- **Show, don't tell** for the target model. Few-shot examples > instructions.
- **Metric is code, not config.** You can and should modify `eval/metric.py`.
- **Version everything.** New prompts get new files. Metric changes get new
  versioned files in `strategy/metrics/`. Never overwrite — always append.
- **Trace to goal.** Every strategy change must reference a specific constraint
  or requirement from `goal.md`.
- **Never stop.** If you run out of ideas, re-read `goal.md` and the changelog.
  Look for dimensions you haven't explored. Try combining near-misses. Think
  about what the current metric might be missing.

## What You Cannot Do

- Modify the labeled dataset (`dataset/`). It is ground truth.
- Modify `goal.md`. Only the human changes the goal.
- Skip logging. Every experiment and every strategy change must be recorded.
- Make strategy changes without reasoning. "Let's try X" is not enough —
  explain why X should help relative to the goal.
