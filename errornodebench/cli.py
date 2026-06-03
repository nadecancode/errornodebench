"""CLI entry point: `errornodebench interference [...]`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from errornodebench.models import BenchmarkResult, VerdictLabel
from errornodebench.runner import ModelConfig, run_interference

console = Console()


def _print_result(result: BenchmarkResult) -> None:
    arm_stats = {arm: result.aggregate(arm) for arm in result.arm_names}

    # --- Table 1: per-entry quality ---
    quality = Table(
        title=(
            f"ErrorNodeBench — {result.scenario} (quality)  "
            f"consolidator={result.consolidator_model}, "
            f"judge={result.judge_model}, seeds={len(result.seeds)}"
        )
    )
    quality.add_column("Arm")
    quality.add_column("Entries", justify="right")
    for label in VerdictLabel:
        quality.add_column(label.value, justify="right")
    for arm_name in result.arm_names:
        stats = arm_stats[arm_name]
        row = [
            arm_name,
            f"{stats.mean_total_entries:.1f}±{stats.std_total_entries:.1f}",
        ]
        for label in VerdictLabel:
            m = stats.label_means[label.value]
            s = stats.label_stds[label.value]
            row.append(f"{m:.1f}±{s:.1f}")
        quality.add_row(*row)
    console.print(quality)

    # --- Table 2: coverage ---
    coverage = Table(
        title=(
            "ErrorNodeBench — coverage (what survived in the final memory)"
        )
    )
    coverage.add_column("Arm")
    coverage.add_column("Family coverage", justify="right")
    coverage.add_column("Task coverage", justify="right")
    coverage.add_column("Missing families (seed 0)")
    for arm_name in result.arm_names:
        stats = arm_stats[arm_name]
        first = getattr(result.seeds[0], arm_name).coverage
        coverage.add_row(
            arm_name,
            f"{stats.mean_family_coverage:.0%}±{stats.std_family_coverage:.0%}"
            f"  ({first.n_families_covered}/{first.n_families_total})",
            f"{stats.mean_task_coverage:.0%}±{stats.std_task_coverage:.0%}"
            f"  ({first.n_tasks_covered}/{first.n_tasks_total})",
            ", ".join(first.missing_families) or "—",
        )
    console.print(coverage)

    # --- Headline ---
    bad_by_arm = {
        arm_name: (
            arm_stats[arm_name].label_means[VerdictLabel.OVER_GENERALIZED.value]
            + arm_stats[arm_name].label_means[VerdictLabel.GARBAGE.value]
        )
        for arm_name in result.arm_names
    }
    console.print(
        "\n[bold]Interference signal (over_generalized + garbage, "
        "mean entries per run):[/bold]"
    )
    for arm_name, v in bad_by_arm.items():
        console.print(f"  {arm_name:>13}: {v:.2f}")
    if bad_by_arm["fresh"] > 0 or bad_by_arm["cumulative"] > 0:
        delta = bad_by_arm["cumulative"] - bad_by_arm["fresh"]
        console.print(
            f"\n  cumulative − fresh delta = [bold]{delta:+.2f}[/bold] "
            f"(paper predicts > 0)"
        )

    console.print(
        "\n[bold]Collapse signal (family coverage; paper predicts "
        "static_group ≥ fresh > cumulative):[/bold]"
    )
    for arm_name in result.arm_names:
        cov = arm_stats[arm_name].mean_family_coverage
        console.print(f"  {arm_name:>13}: {cov:.0%}")


@click.group()
def main() -> None:
    """ErrorNodeBench: benchmark LLM memory consolidation failure modes."""
    load_dotenv()


@main.command()
@click.option(
    "--solver",
    default="gpt-5.5",
    show_default=True,
    help="Model used to produce trajectories.",
)
@click.option(
    "--consolidator",
    default="gpt-5.5",
    show_default=True,
    help="Model under test — produces the memory.",
)
@click.option(
    "--judge",
    default="gpt-5.5",
    show_default=True,
    help=(
        "Model used to score entries. Safe to be the same as --consolidator: "
        "each judge call is a fresh request with no shared context."
    ),
)
@click.option(
    "--seeds",
    default=1,
    show_default=True,
    type=int,
    help="Number of independent passes; results aggregated as mean±std.",
)
@click.option(
    "--mgpt-base-url",
    default=None,
    help=(
        "Override MGPT_BASE_URL (default: from env, typically "
        "http://localhost:8080/v1)."
    ),
)
@click.option(
    "--save",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write full BenchmarkResult JSON here.",
)
@click.option(
    "--sequence",
    type=click.Choice(["default", "reversed", "family-blocked"]),
    default="default",
    show_default=True,
    help="Task ordering. `default` is the interleaved switch schedule. "
    "`reversed` and `family-blocked` are ablations for the collapse finding.",
)
def interference(
    solver: str,
    consolidator: str,
    judge: str,
    seeds: int,
    mgpt_base_url: str | None,
    save: Path | None,
    sequence: str,
) -> None:
    """Run the Interference scenario across five arms (Fresh / Static-Group / Cumulative / Reflexion / ExpeL)."""
    if mgpt_base_url:
        os.environ["MGPT_BASE_URL"] = mgpt_base_url
    from errornodebench.scenarios.interference import SEQUENCES

    models = ModelConfig(solver=solver, consolidator=consolidator, judge=judge)
    result = run_interference(
        models=models,
        seeds=seeds,
        tasks=SEQUENCES[sequence],
        progress=lambda msg: console.log(msg),
    )
    _print_result(result)
    if save:
        save.write_text(json.dumps(result.model_dump(mode="json"), indent=2))
        console.print(f"\n[green]Saved full result to {save}[/green]")


if __name__ == "__main__":
    main()
