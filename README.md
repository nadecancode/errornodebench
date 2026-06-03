# ErrorNodeBench

A benchmark for the **failure modes of LLM-driven memory consolidation** — what
goes wrong when an agent distills its past trajectories into a reusable textual
memory and then keeps consolidating new experience into that same memory over
time.

Most "agent memory" / experiential-learning systems (Reflexion, ExpeL, and
friends) assume that folding more experience into memory monotonically helps.
ErrorNodeBench is built to measure the cases where it *doesn't*: where streaming
("Cumulative") consolidation **over-generalizes** lessons past their
applicability conditions, **misgroups** unrelated tasks, or **collapses** —
aggressively merging entries until most of the input experience has been
dropped on the floor and only one task family survives.

> This repository is the standalone code + experimental data for the benchmark,
> extracted from a Stanford CS321m course project. The paper itself is not
> included; this README is self-contained.

---

## TL;DR — reproduce the paper's numbers in one command (no API key, no GPU)

The expensive half of the benchmark (calling six model backbones to produce
`runs/*.json`) is already done and committed. Regenerating every table and
figure in the paper from that data is pure Python and takes about a second:

```bash
pip install -r requirements.txt      # 5 packages; or: pip install -e .
bash reproduce.sh                     # regenerates Tables 1/3/4 + Figure 1, self-checks vs paper
```

Expected tail of the output:

```
=== CHECK: recomputed Q vs paper Table 1 (tolerance 0.01) ===
  PASS: all 30 (backbone, arm) Q values reproduce the paper.
```

---

## What it measures

The benchmark runs an **Interference** scenario: a stream of agent tasks drawn
from several *families* that share surface features but require **different**
strategies. A good memory keeps each family's lesson bounded to its own
applicability conditions; a bad consolidation rule blurs them together or
discards them.

**Task families** (15 tasks, 3 per family) — all set in a simple simulated
kitchen so the "right strategy" is unambiguous and rubric-checkable:

| Family | What separates it from its neighbours |
| --- | --- |
| `heat-melt` | melt a solid (butter, chocolate, cheese) — low/indirect heat |
| `heat-boil` | bring a liquid to a boil — direct high heat |
| `heat-cook` | cook a raw item (egg, rice, steak) — technique matters |
| `cool` | freeze / chill — the opposite direction of heat |
| `mix` | combine ingredients — no thermal step at all |

The same trajectories are replayed through **five consolidation arms**; only the
*consolidation rule* changes:

| Arm | Origin | Rule |
| --- | --- | --- |
| `fresh` | this benchmark | consolidate each trajectory in isolation (no shared memory) |
| `static_group` | this benchmark | batch by family, one consolidate per family |
| `cumulative` | this benchmark | sequential running memory — each task folds into the growing memory |
| `reflexion` | Shinn et al. 2023 | per-trajectory self-reflection (baseline) |
| `expel` | Zhao et al. 2024 | per-task insights + a cross-task generalization pass (baseline) |

### Two complementary metrics

1. **Per-entry quality.** An LLM judge labels every surviving memory entry
   against the family rubrics as one of
   `useful` · `over_generalized` · `over_specialized` · `garbage`.
   `useful_rate = useful / total`, `bad_rate = (over_generalized + garbage) / total`.

2. **Coverage.** Per-entry quality only scores entries that *exist*. It misses
   the catastrophic-collapse mode where the consolidator merges so aggressively
   that whole families vanish — surviving entries each look fine while family
   coverage drops to 1/5. **Coverage** is the complement: of the input
   sequence, how many *families* and *tasks* are still cited in the final memory.

The headline composite score combines them: **`Q = useful_rate × task_coverage`**.

Ordering ablations (`--sequence default|reversed|family-blocked`) isolate
whether collapse is driven by the interleaved switch schedule or by something
order-independent.

---

## Repository structure

```
errornodebench/
├── README.md
├── LICENSE                     # MIT
├── requirements.txt            # exact pinned versions (Python >= 3.10)
├── pyproject.toml              # installs the `errornodebench` console script
├── reproduce.sh                # offline reproduction + documented live-sweep commands
├── modal_app.py                # optional: Modal deployment for the open-weights vLLM endpoints
├── .env.example                # provider-routing template (copy to .env)
├── errornodebench/             # the package
│   ├── cli.py                  # `errornodebench interference ...`
│   ├── models.py               # Task / Trajectory / Memory / Coverage / verdicts (+ result schema)
│   ├── runner.py               # orchestrates the 5 arms end-to-end (data → consolidation → judging)
│   ├── trajectory.py           # solver: task -> trajectory          (model call; temperature 0.3)
│   ├── consolidator.py         # Fresh / Static-Group / Cumulative consolidation (temperature 0)
│   ├── judge.py                # rubric-based per-entry judging       (temperature 0)
│   ├── baselines/              # reflexion.py, expel.py (external baselines, cited in-file)
│   ├── scenarios/interference.py  # task families + the three orderings
│   ├── llm.py                  # litellm routing (+ local-proxy slot resolution, retry/repair)
│   ├── claude_cli.py           # route Anthropic models via the Claude Code CLI (no API key)
│   ├── vllm_backend.py         # route open-weights models via Modal-hosted vLLM
│   └── concurrency.py          # thread-pool fan-out helper
├── scripts/
│   └── aggregate_results.py    # regenerates every paper table/figure from runs/  (pure stdlib)
├── tests/
│   └── test_smoke.py           # offline tests: taxonomy, coverage, run-file round-trip
└── runs/                       # committed experimental results (JSON + console logs)
```

**Separation of concerns:** data generation (`trajectory.py`), the
arm-orchestration (`runner.py`), the model under test (`consolidator.py` +
`baselines/`), evaluation (`judge.py`, `models.py:compute_coverage`), provider
routing (`llm.py`, `claude_cli.py`, `vllm_backend.py`), and analysis/reporting
(`scripts/aggregate_results.py`) are each in their own module.

---

## Environment setup

Requires **Python ≥ 3.10** (uses PEP 604 unions / builtin generics). Developed
and tested on **CPython 3.12.13**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # litellm, pydantic, python-dotenv, click, rich (exact pins)
pip install -e .                   # optional: installs the `errornodebench` CLI entry point
```

`scripts/aggregate_results.py` and `tests/test_smoke.py` use **only the standard
library**, so the offline reproduction works even without the runtime
dependencies installed — you only need `pydantic` (for loading the run JSON) and
the rest for actually running the benchmark against live models.

---

## Reproducing the paper's results

### Which script produces which artifact

Every numeric result traces to a committed run file and is regenerated by one
script. Run `python scripts/aggregate_results.py` (or `bash reproduce.sh`):

| Paper artifact | How it is regenerated | Source run files |
| --- | --- | --- |
| **Table 1** — composite `Q` per (backbone, arm) | `aggregate_results.py` → "SUMMARY" | `v4`, `v5_*`, `v7_*` |
| **Table 3** — Bradley–Terry strengths (pooled + per-backbone) | `aggregate_results.py` → "BRADLEY-TERRY" | same |
| **Table 4** — full per-cell counts / coverage / `Q` | `aggregate_results.py` → "FULL PER-CELL" | same |
| **Figure 1** — panels (a) family coverage, (b) bad-rate, (c) `Q` | `aggregate_results.py` → "FIGURE 1 PANEL DATA" | same |
| **Table 2** — five collapse *mechanisms* (qualitative) | inspect `source_task_ids` in each backbone's `cumulative` arm | per-backbone `runs/*.json` |

`--check` asserts the recomputed `Q` matches the paper's Table 1 within 0.01
(all 30 cells pass). `--csv out/` also dumps `out/per_cell.csv`.

### Run-file → experiment mapping

| Run file | Backbone | Seeds | Sequence |
| --- | --- | --- | --- |
| `runs/v4.json` | `gpt-5.5` (local litellm proxy) | 3 | default |
| `runs/v5_sonnet.json` | `claude-sonnet-4.6` | 1 | default |
| `runs/v5_haiku.json` | `claude-haiku-4.5` | 1 | default |
| `runs/v5_opus.json` | `claude-opus-4.7` | 1 | default |
| `runs/v7_llama8b.json` | `Llama-3.1-8B-Instruct` (vLLM) | 3 | default |
| `runs/v7_qwen32b.json` | Qwen open-weights, slot `vllm-qwen3.5-27b` (vLLM) | 3 | default |
| `runs/v6_haiku_reversed.json` | `claude-haiku-4.5` | 1 | reversed (ablation) |
| `runs/v1–v3.json` | `gpt-5.5` | 1–3 | early development runs |
| `runs/v7_deepseek.log`, `v7_qwen27b.log`, `v8_qwen35.log` | failed backbone-substitution attempts | — | logs only (no usable JSON) |

The paper labels the Qwen row "Qwen2.5-32B"; the released `v7_qwen32b.json` was
produced with the `vllm-qwen3.5-27b` slot. This substitution (and the
DeepSeek/Qwen attempts that produced only logs) is the backbone-substitution
story documented in the paper's appendix; the analysis uses whatever model id is
recorded inside each file.

### Re-running the benchmark live (the expensive half)

`reproduce.sh` documents the exact command behind each run file. The general form:

```bash
errornodebench interference \
    --solver MODEL --consolidator MODEL --judge MODEL \
    --seeds N [--sequence default|reversed|family-blocked] \
    --save runs/out.json
```

Provider routing (see `.env.example`, copy to `.env`):

- **Any litellm model string** — set the provider key and pass a prefixed name,
  e.g. `--consolidator openai/gpt-4o` (`OPENAI_API_KEY`) or
  `anthropic/claude-...` (`ANTHROPIC_API_KEY`).
- **Claude via the Claude Code CLI** — slots `claude-opus` / `claude-sonnet` /
  `claude-haiku` shell out to `claude -p` and reuse your CLI session (no API
  key). Produced the `v5_*` / `v6_*` rows.
- **Open-weights via Modal vLLM** — slots `vllm-llama-3.1-8b`,
  `vllm-qwen3.5-27b` (see `errornodebench/vllm_backend.py`). Deploy the
  endpoints with `modal deploy modal_app.py`, or point `VLLM_*_URL` at any
  OpenAI-compatible vLLM server. Produced the `v7_*` rows.
- **Local proxy** — bare slots like `gpt-5.5` route to `MGPT_BASE_URL`
  (default `http://localhost:8080/v1`). Produced the `v4.json` row; substitute
  any litellm model if you don't have that proxy.

A cheap end-to-end live smoke run against a real provider:

```bash
OPENAI_API_KEY=sk-... bash reproduce.sh live-quickstart   # 1 seed, openai/gpt-4o-mini
```

### Data — generation, not download

There is **no external dataset to download**. The 15 tasks (goals,
environments, rubrics) are defined in code in
`errornodebench/scenarios/interference.py`. The agent *trajectories* are
generated at run time by the solver model (`trajectory.py`) and then consolidated
and judged. The committed `runs/*.json` files are the full generated artifacts
(every trajectory, memory entry, and judge verdict) for the paper's backbones.

---

## Expected runtime and computational requirements

- **Offline reproduction** (`bash reproduce.sh`, `python tests/test_smoke.py`):
  pure Python, **no GPU, no API key, ~1 second**, a few tens of MB of RAM.
- **One live cell** (one backbone, one seed, all five arms): on the order of
  **180–230 model calls** — 15 solver + ~65 consolidation (the `cumulative` arm
  is 15 sequential calls; the rest fan out over a 4-worker thread pool) + ~100
  judge calls. On a fast hosted API expect roughly **5–15 minutes per seed**;
  cost is dominated by the judge and `cumulative` passes.
- **Closed backbones** need API access (or a Claude Code CLI session). **Open-weights
  backbones** need a vLLM GPU deployment: `Llama-3.1-8B-Instruct` fits on a
  single ~24 GB GPU; the larger Qwen model needs more. `modal_app.py` provisions
  these on Modal.
- The full 12-cell paper sweep is the sum of the above across six backbones; the
  committed `runs/` let you skip it entirely.

---

## Reproducibility notes

- **Seeds.** Every model call takes a `seed` and runs at `temperature=0.0`
  except the solver (`temperature=0.3`, so distinct seeds regenerate a varied
  trajectory pool). The seed is threaded through `runner → trajectory /
  consolidator / judge / baselines → llm.structured_call → litellm`. LLM
  determinism is best-effort (providers do not guarantee bit-exact outputs at
  fixed seed/temperature), which is exactly why the **pre-computed `runs/` are
  committed** and the analysis is deterministic and offline.
- **Pinned dependencies** in `requirements.txt`.
- **End-to-end without modification:** `bash reproduce.sh` and
  `python tests/test_smoke.py` run as-is from a fresh clone.

## Tests

```bash
python tests/test_smoke.py        # 5 offline tests, no deps beyond pydantic
# or, if pytest is installed:
python -m pytest tests/
```

They check the task taxonomy shape, the three orderings, `get_task`, the
coverage metric (full vs. collapsed), and that a released run file round-trips
through the Pydantic models and reproduces the 20% Cumulative family coverage.

---

## Code reuse and attribution

- **Reflexion** (`errornodebench/baselines/reflexion.py`) adapts the method of
  Shinn, Cassano, Gopinath, Narasimhan & Yao (2023), *"Reflexion: Language
  Agents with Verbal Reinforcement Learning,"* NeurIPS 2023 (arXiv:2303.11366).
- **ExpeL** (`errornodebench/baselines/expel.py`) adapts the method of Zhao,
  Huang, Xu, Lin, Liu & Huang (2024), *"ExpeL: LLM Agents Are Experiential
  Learners,"* AAAI 2024 (arXiv:2308.10144).

Both baselines are **original re-implementations** of the published methods
(consolidation/write step only, adapted to this benchmark's `MemoryEntry`
schema) — no code from the original releases is used. Each file's module
docstring states the reference and the modifications. All other code is original.

Routing uses [litellm](https://github.com/BerriAI/litellm); see
`requirements.txt` for the full dependency list and their licenses.

## License

MIT — see [LICENSE](LICENSE). (Copyright line is a placeholder; edit it to your
name.)
