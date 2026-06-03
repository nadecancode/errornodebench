#!/usr/bin/env python3
"""Regenerate every numeric result in the ErrorNodeBench-Interference paper
from the released per-cell JSON run artifacts in ``runs/``.

This is the analysis half of the benchmark. The expensive half (calling the
six backbones to produce ``runs/*.json``) is already done and committed, so
this script reproduces the paper's tables and figure **deterministically and
offline** — no API keys, no GPUs, standard library only.

What it regenerates (paper artifact -> output section):

  * Table 1  (``tab:summary``)   -> "SUMMARY: composite Q per (backbone, arm)"
  * Table 4  (``tab:fullmain``)  -> "FULL PER-CELL RESULTS"
  * Figure 1 (``fig:multipanel``)-> "FIGURE 1 PANEL DATA" (a/b/c series)
  * Table 3  (``tab:bt``)        -> "BRADLEY-TERRY" (pooled + per-backbone)

Metric definitions (paper, Section "Metrics"); all rates are computed
per (backbone, seed) cell and then averaged across seeds:

    useful_rate = useful / total_entries
    bad_rate    = (over_generalized + garbage) / total_entries
    Q           = useful_rate * task_coverage

Usage::

    python scripts/aggregate_results.py            # tables to stdout
    python scripts/aggregate_results.py --check     # also assert Q matches paper
    python scripts/aggregate_results.py --csv out/  # also dump machine-readable CSV

Run from the repository root with the project venv active (so that the
``errornodebench`` package is importable).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

# Make the package importable when run as `python scripts/aggregate_results.py`
# from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from errornodebench.models import BenchmarkResult, VerdictLabel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Canonical paper lineup: the 12 (backbone, seed) cells behind the main tables.
# Display name -> released run file. This mirrors the paper's own mapping
# (Setup section: "runs/v4.json (gpt-5.5), v5_{opus,sonnet,haiku}.json,
# v7_{llama8b,qwen32b}.json").
# ---------------------------------------------------------------------------
BACKBONES: list[tuple[str, str]] = [
    ("gpt-5.5", "runs/v4.json"),
    ("claude-sonnet-4.6", "runs/v5_sonnet.json"),
    ("claude-haiku-4.5", "runs/v5_haiku.json"),
    ("claude-opus-4.7", "runs/v5_opus.json"),
    ("Llama-3.1-8B", "runs/v7_llama8b.json"),
    ("Qwen2.5-32B", "runs/v7_qwen32b.json"),
]

ARMS = ["fresh", "static_group", "cumulative", "reflexion", "expel"]
ARM_LABEL = {
    "fresh": "Fresh",
    "static_group": "Static-Group",
    "cumulative": "Cumulative",
    "reflexion": "Reflexion",
    "expel": "ExpeL",
}

# Published composite Q from Table 1 / Figure 1c, used by --check to confirm
# this script reproduces the paper within rounding tolerance.
PAPER_Q = {
    "gpt-5.5": {"fresh": 0.47, "static_group": 0.75, "cumulative": 0.02, "reflexion": 0.80, "expel": 0.71},
    "claude-sonnet-4.6": {"fresh": 0.86, "static_group": 0.85, "cumulative": 0.06, "reflexion": 0.93, "expel": 0.78},
    "claude-haiku-4.5": {"fresh": 0.65, "static_group": 0.77, "cumulative": 0.05, "reflexion": 0.53, "expel": 0.73},
    "claude-opus-4.7": {"fresh": 0.65, "static_group": 0.89, "cumulative": 0.19, "reflexion": 0.80, "expel": 0.72},
    "Llama-3.1-8B": {"fresh": 0.51, "static_group": 0.42, "cumulative": 0.03, "reflexion": 0.71, "expel": 0.74},
    "Qwen2.5-32B": {"fresh": 0.76, "static_group": 0.76, "cumulative": 0.05, "reflexion": 0.78, "expel": 0.80},
}


# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------
def _stdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def cell_metrics(arm_result) -> dict:
    """Compute the per-seed metrics for one arm's :class:`ScenarioResult`."""
    counts = arm_result.label_counts
    total = arm_result.total_entries
    useful = counts[VerdictLabel.USEFUL.value]
    over_gen = counts[VerdictLabel.OVER_GENERALIZED.value]
    over_spec = counts[VerdictLabel.OVER_SPECIALIZED.value]
    garbage = counts[VerdictLabel.GARBAGE.value]
    task_cov = arm_result.coverage.task_coverage
    fam_cov = arm_result.coverage.family_coverage
    useful_rate = useful / total if total else 0.0
    bad_rate = (over_gen + garbage) / total if total else 0.0
    return {
        "entries": total,
        "useful": useful,
        "over_generalized": over_gen,
        "over_specialized": over_spec,
        "garbage": garbage,
        "useful_rate": useful_rate,
        "bad_rate": bad_rate,
        "family_coverage": fam_cov,
        "task_coverage": task_cov,
        "Q": useful_rate * task_cov,
    }


def aggregate_arm(result: BenchmarkResult, arm: str) -> dict:
    """Mean/std of every metric across the seeds for one arm.

    Counts (entries, useful, ...) and Q are reported as (mean, std) across
    seeds, matching the paper's Table 1/4. The displayed *rates* (useful_rate,
    bad_rate) are additionally provided pooled — Sum(bad)/Sum(entries) across
    seeds — which is what the paper's Figure 1b uses; this is identical to the
    per-seed mean on single-seed backbones and differs by <=2pp on the
    multi-seed ones (more under Llama's high-variance Static-Group cell).
    """
    per_seed = [cell_metrics(getattr(s, arm)) for s in result.seeds]
    keys = per_seed[0].keys()
    agg = {}
    for k in keys:
        vals = [c[k] for c in per_seed]
        agg[k] = (statistics.fmean(vals), _stdev(vals))

    tot_entries = sum(c["entries"] for c in per_seed)
    tot_useful = sum(c["useful"] for c in per_seed)
    tot_bad = sum(c["over_generalized"] + c["garbage"] for c in per_seed)
    agg["useful_rate_pooled"] = (tot_useful / tot_entries if tot_entries else 0.0, 0.0)
    agg["bad_rate_pooled"] = (tot_bad / tot_entries if tot_entries else 0.0, 0.0)

    agg["_per_seed"] = per_seed
    agg["n_seeds"] = len(per_seed)
    return agg


def load_all() -> dict[str, BenchmarkResult]:
    """Load and validate every canonical run file through the package models."""
    out: dict[str, BenchmarkResult] = {}
    for name, rel in BACKBONES:
        path = REPO_ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing run file: {rel} (run from the repo root)")
        out[name] = BenchmarkResult.model_validate(json.loads(path.read_text()))
    return out


# ---------------------------------------------------------------------------
# Bradley-Terry (matches the paper: per-cell pairwise wins on Q with
# family-coverage tiebreak, +0.5 smoothing, 500 MM iterations, Fresh = 1).
# ---------------------------------------------------------------------------
def _cell_q_cov(result: BenchmarkResult) -> list[dict[str, tuple[float, float]]]:
    """For each seed, return {arm: (Q, family_coverage)} for that (backbone, seed) cell."""
    cells = []
    for s in result.seeds:
        cell = {}
        for arm in ARMS:
            m = cell_metrics(getattr(s, arm))
            cell[arm] = (m["Q"], m["family_coverage"])
        cells.append(cell)
    return cells


def bradley_terry(cells: list[dict[str, tuple[float, float]]],
                  smoothing: float = 0.5, iters: int = 500) -> tuple[dict[str, float], dict[str, float]]:
    """Fit a Bradley-Terry model over arm-vs-arm wins via MM iterations.

    Within each cell every pair of arms is compared on Q (higher wins), with
    family coverage as the tiebreak; an exact tie splits the win 0.5/0.5. Wins
    are pooled across ``cells`` into a matrix, ``smoothing`` phantom wins are
    added to every ordered pair (so a winless arm still gets positive
    strength), and the standard Bradley-Terry MM update is run ``iters`` times.

    Returns ``(strengths, wins)`` where strengths are normalised to Fresh = 1.
    """
    n = len(ARMS)
    idx = {a: i for i, a in enumerate(ARMS)}
    W = [[0.0] * n for _ in range(n)]  # W[i][j] = wins of i over j
    for cell in cells:
        for a in range(n):
            for b in range(a + 1, n):
                qa, ca = cell[ARMS[a]]
                qb, cb = cell[ARMS[b]]
                if (qa, ca) > (qb, cb):
                    W[a][b] += 1.0
                elif (qb, cb) > (qa, ca):
                    W[b][a] += 1.0
                else:
                    W[a][b] += 0.5
                    W[b][a] += 0.5

    raw_wins = {ARMS[i]: sum(W[i]) for i in range(n)}

    # +0.5 smoothing on every ordered pair.
    Ws = [[W[i][j] + (smoothing if i != j else 0.0) for j in range(n)] for i in range(n)]
    w = [sum(Ws[i]) for i in range(n)]
    nij = [[Ws[i][j] + Ws[j][i] for j in range(n)] for i in range(n)]

    pi = [1.0] * n
    for _ in range(iters):
        new = [0.0] * n
        for i in range(n):
            denom = sum(nij[i][j] / (pi[i] + pi[j]) for j in range(n) if j != i)
            new[i] = w[i] / denom if denom else pi[i]
        s = statistics.fmean(new)
        pi = [x / s for x in new]  # keep numbers from drifting

    ref = pi[idx["fresh"]]
    strengths = {ARMS[i]: pi[i] / ref for i in range(n)}
    return strengths, raw_wins


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _hr(width: int = 96) -> None:
    print("-" * width)


def print_summary(data: dict[str, dict]) -> None:
    print("\n=== SUMMARY: composite Q per (backbone, arm)  [paper Table 1, tab:summary] ===")
    header = f"{'Backbone (seeds)':<26}" + "".join(f"{ARM_LABEL[a]:>14}" for a in ARMS)
    print(header)
    _hr(len(header))
    for name, _ in BACKBONES:
        agg = data[name]
        row = f"{name + ' (' + str(agg['fresh']['n_seeds']) + ')':<26}"
        for a in ARMS:
            row += f"{agg[a]['Q'][0]:>14.2f}"
        print(row)


def print_figure(data: dict[str, dict]) -> None:
    print("\n=== FIGURE 1 PANEL DATA  [paper Figure 1, fig:multipanel] ===")
    panels = [
        ("(a) Family coverage", "family_coverage", 1.0),
        ("(b) Bad-entry rate (%)", "bad_rate_pooled", 100.0),
        ("(c) Composite Q", "Q", 1.0),
    ]
    for title, key, scale in panels:
        print(f"\n  {title}")
        header = f"    {'Backbone':<20}" + "".join(f"{ARM_LABEL[a][:5]:>9}" for a in ARMS)
        print(header)
        for name, _ in BACKBONES:
            row = f"    {name:<20}"
            for a in ARMS:
                row += f"{data[name][a][key][0] * scale:>9.1f}" if scale == 100.0 \
                    else f"{data[name][a][key][0] * scale:>9.3f}"
            print(row)


def print_full(data: dict[str, dict]) -> None:
    print("\n=== FULL PER-CELL RESULTS  [paper Table 4, tab:fullmain] ===")
    cols = ["entries", "useful", "over_generalized", "over_specialized", "garbage"]
    head = (f"{'Backbone':<18}{'Arm':<14}{'Entries':>12}{'Useful':>12}"
            f"{'Over-gen':>12}{'Over-spec':>12}{'Garbage':>10}{'Fam%':>7}{'Task%':>7}{'Q':>7}")
    print(head)
    _hr(len(head))
    for name, _ in BACKBONES:
        agg = data[name]
        for ai, a in enumerate(ARMS):
            label = name if ai == 0 else ""
            m = agg[a]
            cellstr = ""
            for c in cols:
                mean, std = m[c]
                cellstr += f"{mean:>7.1f}±{std:<4.1f}" if agg["fresh"]["n_seeds"] > 1 else f"{mean:>11.0f} "
            print(f"{label:<18}{ARM_LABEL[a]:<14}{cellstr}"
                  f"{m['family_coverage'][0]*100:>6.0f}%{m['task_coverage'][0]*100:>6.0f}%{m['Q'][0]:>7.2f}")
        _hr(head and 112)


def print_bt(results: dict[str, BenchmarkResult]) -> None:
    print("\n=== BRADLEY-TERRY strengths on Q  [paper Table 3, tab:bt] ===")
    # Pooled across all 12 cells.
    pooled_cells = []
    per_backbone_cells: dict[str, list] = {}
    for name, _ in BACKBONES:
        cells = _cell_q_cov(results[name])
        per_backbone_cells[name] = cells
        pooled_cells.extend(cells)

    strengths, wins = bradley_terry(pooled_cells)
    total_cells = len(pooled_cells)
    comps = 4 * total_cells  # each arm vs 4 opponents per cell
    print(f"\n  Pooled ({total_cells} cells; wins out of {comps} matched comparisons):")
    print(f"    {'Arm':<14}{'wins':>8}{'pi (Fresh=1)':>16}")
    for a in sorted(ARMS, key=lambda x: -strengths[x]):
        print(f"    {ARM_LABEL[a]:<14}{wins[a]:>8.0f}{strengths[a]:>16.2f}")

    print("\n  Per-backbone pi (Fresh=1):")
    header = f"    {'Arm':<14}" + "".join(f"{n[:12]:>13}" for n, _ in BACKBONES)
    print(header)
    pb_strengths = {name: bradley_terry(per_backbone_cells[name])[0] for name, _ in BACKBONES}
    for a in ARMS:
        row = f"    {ARM_LABEL[a]:<14}"
        for name, _ in BACKBONES:
            row += f"{pb_strengths[name][a]:>13.2f}"
        print(row)


def run_check(data: dict[str, dict]) -> int:
    """Assert recomputed Q matches the paper's published Table 1 within 0.01."""
    print("\n=== CHECK: recomputed Q vs paper Table 1 (tolerance 0.01) ===")
    fails = 0
    for name, _ in BACKBONES:
        for a in ARMS:
            got = round(data[name][a]["Q"][0], 2)
            exp = PAPER_Q[name][a]
            ok = abs(got - exp) <= 0.01 + 1e-9
            if not ok:
                fails += 1
                print(f"  FAIL {name:<20} {ARM_LABEL[a]:<13} got={got:.2f} paper={exp:.2f}")
    if fails == 0:
        print("  PASS: all 30 (backbone, arm) Q values reproduce the paper.")
    else:
        print(f"  {fails} mismatch(es).")
    return fails


def dump_csv(data: dict[str, dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "per_cell.csv"
    with path.open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["backbone", "n_seeds", "arm", "entries_mean", "entries_std",
                      "useful", "over_generalized", "over_specialized", "garbage",
                      "useful_rate", "bad_rate", "family_coverage", "task_coverage", "Q"])
        for name, _ in BACKBONES:
            agg = data[name]
            for a in ARMS:
                m = agg[a]
                wri.writerow([name, m["n_seeds"], a,
                              f"{m['entries'][0]:.3f}", f"{m['entries'][1]:.3f}",
                              f"{m['useful'][0]:.3f}", f"{m['over_generalized'][0]:.3f}",
                              f"{m['over_specialized'][0]:.3f}", f"{m['garbage'][0]:.3f}",
                              f"{m['useful_rate'][0]:.4f}", f"{m['bad_rate'][0]:.4f}",
                              f"{m['family_coverage'][0]:.4f}", f"{m['task_coverage'][0]:.4f}",
                              f"{m['Q'][0]:.4f}"])
    print(f"\nWrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="assert recomputed Q matches the paper's Table 1")
    ap.add_argument("--csv", metavar="DIR", default=None,
                    help="also write machine-readable CSV to DIR")
    args = ap.parse_args()

    results = load_all()
    data = {name: {a: aggregate_arm(results[name], a) for a in ARMS}
            for name, _ in BACKBONES}

    print_summary(data)
    print_figure(data)
    print_full(data)
    print_bt(results)

    rc = 0
    if args.check:
        rc = 1 if run_check(data) else 0
    if args.csv:
        dump_csv(data, Path(args.csv))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
