# Model Router Evaluation Harness (v2 — generalized, multi-provider, BYOK)

Generalized from a Freeside-specific script into a tool anyone can point at
their own use cases and any combination of models — not just Gemini vs
Claude. Runs entirely with the person's own API keys (BYOK): this tool never
holds, proxies, or bills for anyone's API usage, which is also the answer to
"how do you handle the cost of hosting this for other people" — you don't,
by design.

## Findings (from a real run)

On the `task_triage` use case, `groq-gpt-oss-120b` (OpenAI's open-weight
GPT-OSS-120B, served on Groq's LPU hardware) was recommended over
`gemini-flash-lite` — but calling it a "win" undersells what actually
decided it. `gemini-flash-lite` was **disqualified**: its
`classification_accuracy` scored 3.78 against this use case's 4.0 quality
floor, while `groq-gpt-oss-120b` cleared it at 4.89 (mean quality across all
criteria: 4.96 vs 4.59). The decision engine's own reasoning trace says it
directly: `groq-gpt-oss-120b` was "selected by elimination," not because it
swept every metric.

This run originally targeted three models across three vendors, and that
scope shrank twice for reasons worth stating plainly rather than glossing
over. `claude-haiku-4.5` never ran — the Anthropic API key in `.env`
returned a 401 authentication error, an infrastructure problem, not a
quality signal, so Claude is absent from this comparison entirely. Separately,
the originally planned Groq model (`llama-3.3-70b-versatile`) turned out to
already be dead — Groq's own previously-documented deprecation warning had
taken effect — so the run used `openai/gpt-oss-120b` on Groq instead, the
vendor's own recommended replacement, confirmed live before spending anything
on it. The honest limitation: this comparison rests on 3 trials per model,
the harness's adaptive-stopping minimum, and only two vendors instead of the
intended three — a real, reportable disqualification, not a null result, but
a thin sample that shouldn't be read as a durable ranking.

Groq was added as a third provider (`router_eval/providers/groq_provider.py`)
specifically to prove the `ModelProvider` interface generalizes to an
open-weight model served by a third vendor, not just Anthropic and Google —
and it ran this entire comparison at $0 additional cost, since both Groq
calls stayed inside its free tier's rate limits.

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

Set your keys either as real shell env vars, or by copying `.env.example` to
`.env` and filling it in — `cli.py` loads `.env` automatically on startup
(via `python-dotenv`) if present; actual shell env vars still take precedence.

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

## Development

```bash
pip install -r requirements-dev.txt

pytest                    # tests/ covers decision_engine.py and judge.py
ruff check .              # lint
black --check .           # formatting
mypy router_eval          # type-check the package (not tests/, which uses
                           # duck-typed fakes on purpose — see tests/conftest.py)
```

Test coverage is intentionally scoped to `decision_engine.py` and `judge.py`
for now — the two modules with the most non-obvious logic (quality-floor
gating, tie-breaking, JSON-parse retries). `harness.py`, `cli.py`, and the
provider files aren't covered yet.

By Nini Merabishvili