"""Core data models used across the benchmark."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single task instance the solver tries to complete.

    `family` groups tasks that share applicability conditions — the unit that
    matters for the Interference scenario. The `correct_strategy` and
    `applicability_conditions` are the rubric anchors the judge uses to decide
    whether a consolidated memory entry kept its bounds or over-generalized.
    """

    task_id: str
    family: str
    goal: str
    environment: str
    correct_strategy: str
    applicability_conditions: str


class TrajectoryStep(BaseModel):
    observation: str
    thought: str
    action: str
    result: str


class Trajectory(BaseModel):
    task: Task
    steps: list[TrajectoryStep]
    final_outcome: str
    success: bool


class MemoryEntry(BaseModel):
    """One atomic lesson in the agent's textual memory."""

    when_to_use: str = Field(
        description="Applicability conditions — when this lesson applies."
    )
    strategy: str = Field(description="What to do.")
    source_task_ids: list[str] = Field(
        default_factory=list,
        description="Trajectories this entry was distilled from.",
    )


class Memory(BaseModel):
    entries: list[MemoryEntry] = Field(default_factory=list)

    def render(self) -> str:
        if not self.entries:
            return "(empty)"
        chunks = []
        for i, e in enumerate(self.entries, 1):
            chunks.append(
                f"[Entry {i}]\nWhen to use: {e.when_to_use}\nStrategy: {e.strategy}"
            )
        return "\n\n".join(chunks)


class VerdictLabel(str, Enum):
    USEFUL = "useful"
    OVER_GENERALIZED = "over_generalized"
    OVER_SPECIALIZED = "over_specialized"
    GARBAGE = "garbage"


class JudgeVerdict(BaseModel):
    entry_index: int
    entry: MemoryEntry
    label: VerdictLabel
    reasoning: str
    affected_families: list[str] = Field(
        default_factory=list,
        description="Task families this entry would mislead, if any.",
    )


class Coverage(BaseModel):
    """How much of the input survives in the final memory.

    Per-entry rubric (useful/over_generalized/...) only measures the
    quality of entries that EXIST. It misses the catastrophic-collapse
    failure mode where the consolidator merges aggressively and drops
    entries on the floor — the surviving entries can each look fine
    while the memory's family coverage drops to 1/5. Coverage is the
    complementary metric: of the input sequence, how much made it.
    """

    n_families_total: int
    n_families_covered: int
    n_tasks_total: int
    n_tasks_covered: int
    missing_families: list[str]
    missing_task_ids: list[str]

    @property
    def family_coverage(self) -> float:
        return (
            self.n_families_covered / self.n_families_total
            if self.n_families_total
            else 0.0
        )

    @property
    def task_coverage(self) -> float:
        return (
            self.n_tasks_covered / self.n_tasks_total
            if self.n_tasks_total
            else 0.0
        )


def compute_coverage(
    *,
    memory: Memory,
    task_sequence: list[str],
    families: dict[str, list[Task]],
) -> Coverage:
    # Map task_id -> family using the canonical taxonomy.
    task_to_family: dict[str, str] = {}
    for fam, tasks in families.items():
        for t in tasks:
            task_to_family[t.task_id] = fam

    # Which task_ids and families appear in any entry's source_task_ids?
    covered_task_ids: set[str] = set()
    covered_families: set[str] = set()
    for entry in memory.entries:
        for tid in entry.source_task_ids:
            covered_task_ids.add(tid)
            fam = task_to_family.get(tid)
            if fam is not None:
                covered_families.add(fam)

    sequence_families = {
        task_to_family[tid] for tid in task_sequence if tid in task_to_family
    }
    return Coverage(
        n_families_total=len(sequence_families),
        n_families_covered=len(covered_families & sequence_families),
        n_tasks_total=len(task_sequence),
        n_tasks_covered=len(covered_task_ids & set(task_sequence)),
        missing_families=sorted(sequence_families - covered_families),
        missing_task_ids=sorted(set(task_sequence) - covered_task_ids),
    )


class ScenarioResult(BaseModel):
    """Result of running one consolidation arm (Fresh / Static-Group / Cumulative)."""

    arm: str
    consolidator_model: str
    judge_model: str
    final_memory: Memory
    verdicts: list[JudgeVerdict]
    task_sequence: list[str]  # task_ids in order
    coverage: Coverage

    @property
    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {label.value: 0 for label in VerdictLabel}
        for v in self.verdicts:
            counts[v.label.value] += 1
        return counts

    @property
    def total_entries(self) -> int:
        return len(self.verdicts)


class SeedRun(BaseModel):
    """One end-to-end pass at a given seed: all five arms + verdicts.

    Arm taxonomy:
      fresh         own arm — consolidate each trajectory in isolation
      static_group  own arm — batched-by-family
      cumulative    own arm — sequential running memory
      reflexion     baseline from arXiv 2605.20616 (Shinn et al.)
      expel         baseline from arXiv 2605.20616 (Zhao et al.)
    """

    seed: int
    fresh: ScenarioResult
    static_group: ScenarioResult
    cumulative: ScenarioResult
    reflexion: ScenarioResult
    expel: ScenarioResult


class ArmStats(BaseModel):
    """Aggregated stats for one arm across N seeds."""

    arm: str
    n_seeds: int
    # mean and std of entry counts per label, across seeds
    label_means: dict[str, float]
    label_stds: dict[str, float]
    # mean total entries across seeds
    mean_total_entries: float
    std_total_entries: float
    # coverage aggregates
    mean_family_coverage: float
    std_family_coverage: float
    mean_task_coverage: float
    std_task_coverage: float


class BenchmarkResult(BaseModel):
    scenario: str  # "interference"
    consolidator_model: str
    judge_model: str
    solver_model: str
    seeds: list[SeedRun]
    notes: Optional[str] = None

    def _arm_runs(self, arm_name: str) -> list[ScenarioResult]:
        return [getattr(s, arm_name) for s in self.seeds]

    def aggregate(self, arm_name: str) -> ArmStats:
        import statistics

        runs = self._arm_runs(arm_name)
        n = len(runs)
        per_seed_counts: list[dict[str, int]] = [r.label_counts for r in runs]
        per_seed_totals: list[int] = [r.total_entries for r in runs]
        per_seed_fam_cov = [r.coverage.family_coverage for r in runs]
        per_seed_task_cov = [r.coverage.task_coverage for r in runs]

        labels = [label.value for label in VerdictLabel]
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for label in labels:
            vals = [c[label] for c in per_seed_counts]
            means[label] = statistics.fmean(vals) if vals else 0.0
            stds[label] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

        def _stdev(xs):
            return statistics.pstdev(xs) if len(xs) > 1 else 0.0

        return ArmStats(
            arm=arm_name,
            n_seeds=n,
            label_means=means,
            label_stds=stds,
            mean_total_entries=(
                statistics.fmean(per_seed_totals) if per_seed_totals else 0.0
            ),
            std_total_entries=_stdev(per_seed_totals),
            mean_family_coverage=(
                statistics.fmean(per_seed_fam_cov) if per_seed_fam_cov else 0.0
            ),
            std_family_coverage=_stdev(per_seed_fam_cov),
            mean_task_coverage=(
                statistics.fmean(per_seed_task_cov)
                if per_seed_task_cov
                else 0.0
            ),
            std_task_coverage=_stdev(per_seed_task_cov),
        )

    @property
    def arm_names(self) -> list[str]:
        return ["fresh", "static_group", "cumulative", "reflexion", "expel"]

    def summary(self) -> str:
        lines = [
            f"Scenario: {self.scenario} "
            f"(consolidator={self.consolidator_model}, "
            f"judge={self.judge_model}, seeds={len(self.seeds)})"
        ]
        for arm_name in self.arm_names:
            stats = self.aggregate(arm_name)
            parts = []
            for label in VerdictLabel:
                m = stats.label_means[label.value]
                s = stats.label_stds[label.value]
                parts.append(f"{label.value}={m:.1f}±{s:.1f}")
            lines.append(
                f"  {arm_name:>13}: total={stats.mean_total_entries:.1f}±"
                f"{stats.std_total_entries:.1f}  " + ", ".join(parts)
            )
        return "\n".join(lines)
