# ErrorNodeBench

A benchmark for the **failure modes of LLM-driven memory consolidation** — what
goes wrong when an agent distills its past trajectories into a reusable textual
memory and then keeps consolidating new experience into that same memory over
time.

Most "agent memory" / experiential-learning systems (Reflexion, ExpeL, and
friends) assume that folding more experience into memory monotonically helps.
ErrorNodeBench is built to measure the cases where it *doesn't*: where streaming
consolidation **over-generalizes** lessons past their applicability conditions,
**misgroups** unrelated tasks, or **collapses** — aggressively merging entries
until most of the input experience has been dropped on the floor.

> This repository is the standalone code + experimental data for the benchmark.
> It was extracted from a Stanford CS321m course project. The writeup/paper is
> not included here.

---

## What it measures

The benchmark runs an **Interference** scenario: a stream of agent tasks drawn
from several *families* that share surface features but require **different**
strategies. A good memory keeps each family's lesson bounded to its own
applicability conditions; a bad consolidation rule blurs them together.

**Task families** (15 tasks, 3 per family) — all set in a simple simulated
kitchen so the "right strategy" is unambiguous and rubric-checkable:

| Family | What separates it from its neighbors |
| --- | --- |
| `heat-melt` | melt a solid (butter, chocolate, cheese) — low/indirect heat |
| `heat-boil` | bring a liquid to a boil — direct high heat |
| `heat-cook` | cook a raw item (egg, rice, steak) — technique matters |
| `cool` | freeze / chill — the opposite direction of heat |
| `mix` | combine ingredients — no thermal step at all |

The same trajectories are replayed through **five consolidation arms**, and only
the *consolidation rule* changes:

| Arm | Origin | Rule |
| --- | --- | --- |
| `fresh` | this benchmark | consolidate each trajectory in isolation (no shared memory) |
| `static_group` | this benchmark | batch by family, one consolidate per family |
| `cumulative` | this benchmark | sequential running memory — each task folds into the growing memory |
| `reflexion` | Shinn et al. | per-trajectory self-reflection |
| `expel` | Zhao et al. | per-task insights + a cross-task generalization pass |

### Two complementary metrics

1. **Per-entry quality.** An LLM judge labels every surviving memory entry
   against the family rubrics as one of:
   `useful` · `over_generalized` · `over_specialized` · `garbage`.
   The interference signal is `over_generalized + garbage` per run.

2. **Coverage.** Per-entry quality only scores entries that *exist*. It misses
   the catastrophic-collapse mode where the consolidator merges so aggressively
   that whole families vanish from memory — surviving entries each look fine
   while family coverage drops to 1/5. **Coverage** is the complement: of the
   input sequence, how many *families* and *tasks* actually made it into the
   final memory.

Ordering ablations (`--sequence`) isolate whether collapse is driven by the
interleaved switch schedule or by something order-independent:
`default` (interleaved), `reversed`, and `family-blocked`.

---

## Install

Requires Python ≥ 3.10. Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
# or, plain pip:
pip install -e .
```

This installs an `errornodebench` console script (see `pyproject.toml`).

## Configure providers

ErrorNodeBench routes all model calls through
[litellm](https://github.com/BerriAI/litellm), so any provider litellm supports
works. Copy the template and fill in whatever you'll actually use:

```bash
cp .env.example .env
```

Two ways to point it at models:

- **Any litellm model string** (the path most people want). Pass a
  provider-prefixed name and set that provider's key in `.env`:

  ```bash
  export OPENAI_API_KEY=sk-...          # or ANTHROPIC_API_KEY / GEMINI_API_KEY
  errornodebench interference \
      --solver openai/gpt-4o \
      --consolidator openai/gpt-4o \
      --judge openai/gpt-4o \
      --seeds 3 --save runs/my_run.json
  ```

- **A local OpenAI-compatible proxy** (how the bundled `runs/` were produced).
  Bare slot names like `gpt-5.5` are auto-routed to `MGPT_BASE_URL`
  (default `http://localhost:8080/v1`). The judge can safely share the
  consolidator model — each judge call is a fresh request with no shared context.

> The judge is also a model. Cross-check it by routing `--judge` through a
> *different* backbone than `--consolidator` (e.g. judge with
> `anthropic/claude-opus-4-7` while consolidating with an OpenAI model).

## Run

```bash
# Headline run: all five arms, 3 seeds, save full result JSON
errornodebench interference --seeds 3 --save runs/example.json

# Ordering ablation
errornodebench interference --sequence reversed   --save runs/rev.json
errornodebench interference --sequence family-blocked --save runs/blocked.json

errornodebench interference --help
```

The CLI prints two tables (per-entry quality and coverage) plus the headline
interference and collapse signals. `--save` writes the complete
`BenchmarkResult` (every trajectory, memory entry, and judge verdict) as JSON.

---

## Repository layout

```
errornodebench/
├── errornodebench/            # the package
│   ├── cli.py                 # `errornodebench interference ...`
│   ├── models.py              # Task / Trajectory / Memory / Coverage / verdicts
│   ├── runner.py              # orchestrates the 5 arms end-to-end
│   ├── trajectory.py          # solver: task -> trajectory
│   ├── consolidator.py        # Fresh / Static-Group / Cumulative consolidation
│   ├── judge.py               # rubric-based per-entry judging
│   ├── baselines/             # reflexion.py, expel.py (external baselines)
│   ├── scenarios/
│   │   └── interference.py    # task families + the three orderings
│   ├── llm.py                 # litellm routing (+ local-proxy slot resolution)
│   ├── claude_cli.py          # optional: route Anthropic models via the Claude CLI
│   ├── vllm_backend.py        # optional: route open-weights models via a vLLM server
│   └── concurrency.py
├── runs/                      # experimental results behind the project (JSON + logs)
└── pyproject.toml
```

## `runs/` — bundled experimental data

The `runs/*.json` files are full `BenchmarkResult` dumps from cross-backbone
sweeps; the matching `*.log` files are the run consoles. Backbones covered
include `gpt-5.5`, Claude (`opus-4.7` / `sonnet-4.6` / `haiku-4.5`), and
open-weights models (`Llama-3.1-8B`, `Qwen2.5-32B`) served via vLLM. Reload any
of them with:

```python
import json
from errornodebench.models import BenchmarkResult

result = BenchmarkResult.model_validate(json.load(open("runs/v4.json")))
print(result.summary())
```

---

## Notes

- All API keys are read from the environment — nothing is hardcoded. The real
  `.env` is git-ignored; only `.env.example` ships.
- `claude_cli.py` and `vllm_backend.py` are optional backends for reproducing
  the Anthropic / open-weights rows and are not required for a basic run.
