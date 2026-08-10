# Model Router Evaluation Harness (v2 — generalized, multi-provider, BYOK)

Generalized from a Freeside-specific script into a tool anyone can point at
their own use cases and any combination of models — not just Gemini vs
Claude. Runs entirely with the person's own API keys (BYOK): this tool never
holds, proxies, or bills for anyone's API usage, which is also the answer to
"how do you handle the cost of hosting this for other people" — you don't,
by design.

## What changed from v1

| | v1 (Freeside-specific) | v2 (generalized) |
|---|---|---|
| Models | Hardcoded Gemini + Claude | Any provider implementing `ModelProvider` — Anthropic, Gemini, OpenAI ship out of the box |
| Use cases | Hardcoded Python dataclasses | User-authored YAML, no code changes needed |
| Pricing | Hardcoded dict | `configs/models.yaml`, editable without touching Python |
| Trial count | Fixed `n_trials=8` always | Adaptive: stops early once a clear winner emerges, runs more only when tied |
| Access | Local script only | CLI (`python -m router_eval.cli run ...`), still local, still BYOK |

## Architecture

`ModelProvider` (`providers/base.py`) is the one interface everything else
depends on. `harness.py`, `judge.py`, and `decision_engine.py` never import
an SDK directly — they only call `provider.generate(...)`. Adding a new
model vendor means writing one new file that implements three methods; nothing
elsewhere changes. `providers/openai_provider.py` exists specifically to
prove this — it was the third vendor added, and it took about 30 lines.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...

# See the cost estimate before spending anything
python -m router_eval.cli run \
  --use-cases configs/use_cases.example.yaml \
  --models claude-sonnet-5,gemini-flash \
  --dry-run

# Run for real, capped at $2
python -m router_eval.cli run \
  --use-cases configs/use_cases.example.yaml \
  --models claude-sonnet-5,gemini-flash \
  --max-spend 2.00
```

Add a third model to the comparison by just extending the list:
`--models claude-sonnet-5,gemini-flash,gpt-4o-mini`

## Why BYOK instead of a hosted platform

A hosted version (connect your project, click a button, we run the eval) was
the original ask. The honest trade-off: hosting means the platform pays for
every run, which means metering, billing, and abuse prevention before it's
safe to ship to strangers — a different, much larger project. BYOK sidesteps
the cost problem entirely instead of trying to manage it, which is also
simply the correct engineering call given the actual goal (a working,
defensible tool), not a workaround.

## Honest limitations

- The decision engine reasons pairwise (top two candidates by quality). With
  3+ models in one run, it still recommends correctly, but the "reasoning"
  trace focuses on the top two — a documented simplification, not a bug.
- Adaptive stopping uses a simplified sequential test (stdev-gap heuristic),
  not a formal statistical procedure like a sequential probability ratio
  test. It's good enough to cut spend on clear-cut cases; treat its
  early stops as directional the same way v1's fixed-trial confidence was.
- Token counts for `--dry-run` come from each provider's own tokenizer where
  available; output length is still a rough placeholder since it can't be
  known before the call happens.
