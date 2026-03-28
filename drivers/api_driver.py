#!/usr/bin/env python3
"""
API-driven execution mode for auto-strat-eval.

Runs the full outer + inner optimization loop as a standalone process,
using LLM API calls (Anthropic/OpenAI) for strategic reasoning.

Usage:
    python api_driver.py --goal goal.md --hours 8
    python api_driver.py --goal goal.md --hours 8 --model claude-sonnet-4-20250514

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.
"""

# TODO: Implement in Phase 3
# This is the Mode B driver that makes LLM API calls for outer loop reasoning.
# For now, use Mode A (Claude Code session with program.md).

raise NotImplementedError(
    "API driver is not yet implemented. "
    "Use Mode A instead: open this repo in Claude Code and say "
    "'Read program.md and start an optimization run'"
)
