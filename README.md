# auto-strat-eval

Autonomous prompt **and strategy** optimization for on-device VLMs.

Combines ideas from [autoresearch](https://github.com/karpathy/autoresearch) (autonomous experiment loops) and [DSPy](https://github.com/stanfordnlp/dspy) (declarative prompt optimization) with a novel **outer loop** that evolves the optimization strategy itself — not just prompt text.

## What's Different

| System | What it optimizes | Metric |
|--------|------------------|--------|
| autoresearch | Code (architecture, hyperparams) | Fixed (val_bpb) |
| DSPy | Prompt text, few-shot examples | Fixed (user-defined) |
| **auto-strat-eval** | Prompts + strategy + metrics + architecture | **Evolvable** (derived from declarative goal) |

The key insight: the biggest gains in prompt optimization come from **structural decisions** (adding few-shot examples, changing multi-pass strategy, redesigning the metric), not from rewording instructions. DSPy can't discover these. auto-strat-eval can.

## Quick Start

### Mode A: Claude Code Session (No API Required)

```bash
# Open this repo in Claude Code, Cursor, or any AI coding agent
# Then say:
"Read program.md and start an optimization run"
```

The AI agent IS the optimizer. It reads `program.md` for instructions, `goal.md` for the objective, and uses the Python eval tooling to run experiments. No API keys needed.

### Mode B: API-Driven (Standalone)

```bash
# Coming soon — requires Anthropic/OpenAI API key
python drivers/api_driver.py --goal goal.md --hours 8
```

## How It Works

1. **Declarative Goal** (`goal.md`) — Natural language description of what success looks like. The agent derives metrics from this, not the other way around.

2. **Outer Loop** — Agent reasons about strategy: what to measure, how to evaluate, what prompt architecture to use. Can modify the metric itself.

3. **Inner Loop** — Mechanical prompt optimization within a fixed strategy. Try variants, measure, keep or discard.

4. **Morning Briefing** — Reviewable changelog of every strategic decision with reasoning, results, and regression checks.

See [DESIGN.md](DESIGN.md) for the full architecture.

## Repo Structure

```
auto-strat-eval/
├── program.md              # Agent instructions (Mode A)
├── goal.md                 # Declarative goal
├── DESIGN.md               # Architecture & design doc
│
├── eval/                   # Mechanical eval tooling
│   ├── runner.py           # Run prompts against dataset
│   ├── metric.py           # Scoring functions
│   ├── backends/           # LM backends (Chrome, adb)
│   └── prompts/            # Versioned prompt files
│
├── strategy/               # Strategy tracking
│   ├── changelog.md        # Strategy evolution narrative
│   ├── results.tsv         # Experiment results log
│   └── metrics/            # Versioned metric definitions
│
├── dataset/                # Labeled evaluation data
│   ├── images/
│   └── labels/
│
└── drivers/                # Execution mode drivers
    └── api_driver.py       # Mode B: standalone with API calls
```

## Status

- [x] Design doc
- [x] Declarative goal
- [x] Agent instructions (program.md)
- [x] Eval runner scaffold
- [ ] Port eval tooling from foodtracker
- [ ] Port dataset from foodtracker
- [ ] Inner loop automation
- [ ] Outer loop (strategy evolution)
- [ ] API driver (Mode B)
- [ ] Morning briefing generator
