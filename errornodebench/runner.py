"""Orchestrate the Interference benchmark end-to-end.

For each seed we:
  1. Generate one trajectory per task in the sequence (solver, with seed).
  2. Replay those trajectories through THREE arms:
       fresh         consolidate each in isolation
       static_group  group by family, one batched consolidate per group
       cumulative    sequential running-memory consolidation
  3. Judge every entry in each arm's final memory.

The same trajectories are reused across arms within a seed, matching the
paper's "same trajectories, different schedule" methodology — only the
consolidation rule varies. Across seeds, trajectories are regenerated to
expose solver-variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from errornodebench.baselines.expel import consolidate_expel
from errornodebench.baselines.reflexion import reflect
from errornodebench.concurrency import parallel_map
from errornodebench.consolidator import consolidate, consolidate_batch
from errornodebench.judge import judge_memory
from errornodebench.models import (
    BenchmarkResult,
    Memory,
    ScenarioResult,
    SeedRun,
    Task,
    Trajectory,
    compute_coverage,
)
from errornodebench.scenarios.interference import ALL_FAMILIES, default_sequence
from errornodebench.trajectory import generate_trajectory


@dataclass
class ModelConfig:
    # Defaults route through the local mgpt proxy. See .env.example.
    # Bare mgpt slot names (gpt-5.5 / gpt-5.4 / gpt-5.3-codex /
    # gpt-5.3-codex-spark) are auto-resolved to openai/<name> with
    # api_base=MGPT_BASE_URL by errornodebench.llm.LLMConfig.resolve().
    #
    # Judge uses the same slot as the consolidator: the judge call is a
    # fresh request with its own system prompt and no shared context.
    solver: str = "gpt-5.5"
    consolidator: str = "gpt-5.5"
    judge: str = "gpt-5.5"


ProgressFn = Callable[[str], None]


def _noop(_: str) -> None: ...


def _group_by_family(trajectories: list[Trajectory]) -> dict[str, list[Trajectory]]:
    groups: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        groups.setdefault(t.task.family, []).append(t)
    return groups


def _run_one_seed(
    *,
    seed: int,
    tasks: list[Task],
    families: dict[str, list[Task]],
    models: ModelConfig,
    progress: ProgressFn,
    max_workers: int = 4,
) -> SeedRun:
    # --- Step 1: generate trajectories (parallel — tasks are independent) ---
    progress(f"[seed {seed}][solver] {len(tasks)} tasks in parallel")

    def _solve(t: Task) -> Trajectory:
        traj = generate_trajectory(t, model=models.solver, seed=seed)
        progress(f"[seed {seed}][solver] done {t.task_id}")
        return traj

    trajectories: list[Trajectory] = parallel_map(
        _solve, tasks, max_workers=max_workers
    )

    # --- Arm A: Fresh — each task consolidated in isolation (parallel) ---
    progress(f"[seed {seed}][fresh] {len(trajectories)} consolidations in parallel")

    def _fresh(traj: Trajectory) -> list:
        m = consolidate(
            prior_memory=Memory(),
            new_trajectory=traj,
            model=models.consolidator,
            seed=seed,
        )
        progress(f"[seed {seed}][fresh] done {traj.task.task_id}")
        return m.entries

    fresh_entries: list = []
    for entries in parallel_map(_fresh, trajectories, max_workers=max_workers):
        fresh_entries.extend(entries)
    fresh_memory = Memory(entries=fresh_entries)

    # --- Arm B: Static-Group — batched per family, empty prior memory ---
    groups = list(_group_by_family(trajectories).items())
    progress(
        f"[seed {seed}][static-group] {len(groups)} family batches in parallel"
    )

    def _batch(item: tuple[str, list[Trajectory]]) -> list:
        family_name, family_trajs = item
        m = consolidate_batch(
            prior_memory=Memory(),
            trajectories=family_trajs,
            model=models.consolidator,
            seed=seed,
        )
        progress(
            f"[seed {seed}][static-group] done {family_name} "
            f"({len(family_trajs)} trajectories)"
        )
        return m.entries

    static_entries: list = []
    for entries in parallel_map(_batch, groups, max_workers=max_workers):
        static_entries.extend(entries)
    static_memory = Memory(entries=static_entries)

    # --- Arm C: Cumulative — same trajectories, running memory (sequential) ---
    cumulative_memory = Memory()
    for traj in trajectories:
        progress(f"[seed {seed}][cumulative] {traj.task.task_id}")
        cumulative_memory = consolidate(
            prior_memory=cumulative_memory,
            new_trajectory=traj,
            model=models.consolidator,
            seed=seed,
        )

    # --- Arm D: Reflexion (Shinn et al.) — per-trajectory self-reflection ---
    progress(
        f"[seed {seed}][reflexion] {len(trajectories)} reflections in parallel"
    )

    def _refl(traj: Trajectory):
        e = reflect(trajectory=traj, model=models.consolidator, seed=seed)
        progress(f"[seed {seed}][reflexion] done {traj.task.task_id}")
        return e

    reflexion_memory = Memory(
        entries=parallel_map(_refl, trajectories, max_workers=max_workers)
    )

    # --- Arm E: EXPEL (Zhao et al.) — per-task insights + cross-task pass ---
    progress(
        f"[seed {seed}][expel] phase1 ({len(trajectories)} parallel), then phase2 (1 call)"
    )
    expel_memory = consolidate_expel(
        trajectories=trajectories,
        model=models.consolidator,
        seed=seed,
        max_workers=max_workers,
    )

    sequence_ids = [t.task_id for t in tasks]

    def _judge_arm(arm: str, mem: Memory) -> ScenarioResult:
        progress(f"[seed {seed}][judge:{arm}] {len(mem.entries)} entries")
        return ScenarioResult(
            arm=arm,
            consolidator_model=models.consolidator,
            judge_model=models.judge,
            final_memory=mem,
            verdicts=judge_memory(
                memory=mem,
                families=families,
                model=models.judge,
                seed=seed,
            ),
            task_sequence=sequence_ids,
            coverage=compute_coverage(
                memory=mem,
                task_sequence=sequence_ids,
                families=families,
            ),
        )

    return SeedRun(
        seed=seed,
        fresh=_judge_arm("fresh", fresh_memory),
        static_group=_judge_arm("static_group", static_memory),
        cumulative=_judge_arm("cumulative", cumulative_memory),
        reflexion=_judge_arm("reflexion", reflexion_memory),
        expel=_judge_arm("expel", expel_memory),
    )


def run_interference(
    *,
    tasks: Optional[list[Task]] = None,
    families: Optional[dict[str, list[Task]]] = None,
    models: Optional[ModelConfig] = None,
    seeds: int = 1,
    progress: ProgressFn = _noop,
) -> BenchmarkResult:
    tasks = tasks or default_sequence()
    families = families or ALL_FAMILIES
    models = models or ModelConfig()

    seed_runs: list[SeedRun] = []
    for s in range(seeds):
        progress(f"=== seed {s + 1}/{seeds} ===")
        seed_runs.append(
            _run_one_seed(
                seed=s,
                tasks=tasks,
                families=families,
                models=models,
                progress=progress,
            )
        )

    return BenchmarkResult(
        scenario="interference",
        consolidator_model=models.consolidator,
        judge_model=models.judge,
        solver_model=models.solver,
        seeds=seed_runs,
    )
