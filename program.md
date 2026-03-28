# auto-strat-eval: Agent Instructions

You are an autonomous prompt and strategy optimization agent. Your job is to
improve the prompts and evaluation strategy for a target model, guided by a
declarative goal.

## Setup

Before starting, you need the **project config** path. This is a `project.yaml`
file in the host repo that points to the dataset, metric, prompts, and backend.

Once you have it, read these files for full context:

1. The `goal` file referenced in project.yaml — your north star.
2. `scripts/auto-strat-eval/DESIGN.md` — architecture and design principles.
3. The strategy changelog (referenced in project.yaml's `strategy_dir`).
4. The prompts directory — find the one with `"deployed": true`.
5. The metric module — understand how scoring currently works.

## The Eval Tooling

All commands run from the `scripts/auto-strat-eval/` directory.
Replace `PROJECT_CONFIG` with the path to your project.yaml.

```bash
# Establish baseline
python -m eval.loop --config PROJECT_CONFIG --strategy S1 --prompts v5-current.json --baseline

# Test new prompts (keep/discard against current best)
python -m eval.loop --config PROJECT_CONFIG --strategy S1 --prompts v6-new.json v7-experiment.json

# Run standalone evaluation
python -m eval.runner --config PROJECT_CONFIG --prompt v5.json v6.json

# Regression check (best prompt against all historical metrics)
python -m eval.runner --config PROJECT_CONFIG --prompt v7-winner.json --regression
```

Each evaluation takes ~60 seconds per 12 images (~5s each).

## The Optimization Loop

You operate in two nested loops:

### Outer Loop: Strategy Evolution

This is where you reason. You can change:
- **The metric** (the module referenced in project.yaml) — weights, sub-metrics, penalties
- **The prompt architecture** — few-shot examples, multi-pass structure, schema
- **The evaluation approach** — which dimensions to measure, how to aggregate

For each strategy change:
1. **OBSERVE**: Look at results from the current strategy. What's working? What's not?
2. **REASON**: Why is it failing? Trace back to the declarative goal.
3. **PROPOSE**: Describe the change you want to make and why.
4. **LOG**: Append to `strategy/changelog.md` with trigger, reasoning, and proposed change.
5. **APPLY**: Edit the relevant files (metric module, create new prompt, etc.)
6. **RUN**: Execute the inner loop under the new strategy.
7. **EVALUATE**: Did it work? Log before/after results.
8. **DECIDE**: Keep or revert. Update the changelog with the outcome.

### Inner Loop: Experiment Execution

Within a strategy, try prompt variants mechanically:
1. Create a new prompt in the prompts directory as a versioned JSON file.
2. Run: `python -m eval.loop --config PROJECT_CONFIG --strategy SN --prompts new-prompt.json`
3. The loop automatically compares to current best, keeps or discards, and logs.
4. Metric snapshots are taken automatically before each run if the metric changed.

Aim for ~10-15 experiments per strategy before considering a strategy change.

## Results Logging

### strategy/results.tsv

Populated automatically by `eval/loop.py`. Tab-separated, append-only.

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
   metric versions.
   `python -m eval.runner --config PROJECT_CONFIG --prompt best.json --regression`
2. Write a **morning briefing** at the top of `strategy/changelog.md`:
   - 3-5 sentence summary of the run
   - Best prompt found and its scores
   - Key insights discovered
   - Recommended next steps
3. Mark the winning prompt as `"deployed": true` in its JSON file (if it beats
   the current deployed prompt).

## Key Principles

- **Show, don't tell** for small models. Few-shot examples > instructions.
- **Metric is code, not config.** You can and should modify the metric module.
  Snapshots are taken automatically before each loop run.
- **Version everything.** New prompts get new files. Never overwrite — always append.
- **Trace to goal.** Every strategy change must reference a specific constraint
  or requirement from the goal file.
- **Never stop.** If you run out of ideas, re-read the goal and the changelog.
  Look for dimensions you haven't explored. Try combining near-misses. Think
  about what the current metric might be missing.

## What You Cannot Do

- Modify the labeled dataset. It is ground truth.
- Modify the goal file. Only the human changes the goal.
- Skip logging. Every experiment and every strategy change must be recorded.
- Make strategy changes without reasoning. "Let's try X" is not enough —
  explain why X should help relative to the goal.
