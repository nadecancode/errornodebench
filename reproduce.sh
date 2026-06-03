#!/usr/bin/env bash
# Reproduce ErrorNodeBench-Interference results.
#
# Two paths:
#   1. OFFLINE (default) — regenerate every table/figure number in the paper
#      from the committed runs/*.json. No API keys, no GPUs, ~1 second.
#      This is the quickstart and the main-result reproduction in one.
#   2. LIVE  — re-run the benchmark against real model backbones. Expensive
#      (API/GPU + minutes per backbone). Commands are documented below.
#
# Usage:
#   bash reproduce.sh                # offline reproduction + self-check
#   bash reproduce.sh live-quickstart   # one tiny live run (needs a provider)
set -euo pipefail

cd "$(dirname "$0")"
PY="${PYTHON:-python}"

offline() {
  echo "==> Offline reproduction: regenerating paper tables/figures from runs/*.json"
  echo "    (no API keys required)"
  echo
  "$PY" scripts/aggregate_results.py --check --csv out
  echo
  echo "==> Done. Tables above correspond to:"
  echo "      SUMMARY            -> paper Table 1 (tab:summary)"
  echo "      FIGURE 1 PANEL DATA-> paper Figure 1 (fig:multipanel)"
  echo "      FULL PER-CELL      -> paper Table 4 (tab:fullmain)"
  echo "      BRADLEY-TERRY      -> paper Table 3 (tab:bt)"
  echo "    Machine-readable copy written to out/per_cell.csv"
}

# A single, cheap LIVE run to confirm the harness works end-to-end against a
# real provider. Requires OPENAI_API_KEY (or another litellm provider key) and
# a model your key can serve. Override MODEL to taste.
live_quickstart() {
  MODEL="${MODEL:-openai/gpt-4o-mini}"
  echo "==> Live quickstart: 1 seed, all 5 arms, backbone=$MODEL"
  echo "    (this makes real API calls and costs a small amount)"
  "$PY" -m errornodebench.cli interference \
      --solver "$MODEL" --consolidator "$MODEL" --judge "$MODEL" \
      --seeds 1 --save runs/quickstart.json
}

case "${1:-offline}" in
  offline)         offline ;;
  live-quickstart) live_quickstart ;;
  *) echo "unknown mode: $1 (use 'offline' or 'live-quickstart')"; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# FULL LIVE SWEEP — the exact commands that produced the released runs/*.json.
# These are documented, not executed (each needs the corresponding provider).
# Model display names in the paper may differ from the internal slot name; the
# Qwen substitution story is in Appendix C of the paper.
#
#   # gpt-5.5, 3 seeds  -> runs/v4.json   (gpt-5.5 is the default backbone;
#   #                                      served via a local litellm proxy,
#   #                                      MGPT_BASE_URL, see .env.example)
#   errornodebench interference --seeds 3 --save runs/v4.json
#
#   # Claude backbones, 1 seed each (via the Claude Code CLI: claude -p)
#   errornodebench interference --solver claude-opus   --consolidator claude-opus   --judge claude-opus   --seeds 1 --save runs/v5_opus.json
#   errornodebench interference --solver claude-sonnet --consolidator claude-sonnet --judge claude-sonnet --seeds 1 --save runs/v5_sonnet.json
#   errornodebench interference --solver claude-haiku  --consolidator claude-haiku  --judge claude-haiku  --seeds 1 --save runs/v5_haiku.json
#
#   # Ordering ablation (reversed schedule), Claude Haiku, 1 seed -> runs/v6_haiku_reversed.json
#   errornodebench interference --solver claude-haiku --consolidator claude-haiku --judge claude-haiku --seeds 1 --sequence reversed --save runs/v6_haiku_reversed.json
#
#   # Open-weights via Modal-hosted vLLM, 3 seeds each (set the VLLM_*_URL env
#   # vars to your deployed endpoints; see errornodebench/vllm_backend.py)
#   errornodebench interference --solver vllm-llama-3.1-8b --consolidator vllm-llama-3.1-8b --judge vllm-llama-3.1-8b --seeds 3 --save runs/v7_llama8b.json
#   errornodebench interference --solver vllm-qwen3.5-27b  --consolidator vllm-qwen3.5-27b  --judge vllm-qwen3.5-27b  --seeds 3 --save runs/v7_qwen32b.json
# ---------------------------------------------------------------------------
