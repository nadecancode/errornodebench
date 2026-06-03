"""Memory consolidator: distill trajectories into a textual memory.

Three arms drive the same primitive `consolidate()` / `consolidate_batch()`
calls in different orders:

  fresh        Start from an empty Memory each time and consolidate one
               trajectory. Per-task lessons; no cross-task pressure.
  static-group Group all trajectories by family, consolidate each group in
               ONE call against an empty memory. Paper's best-case offline
               schedule — the consolidator sees a clean batch per family.
  cumulative   Take the running Memory built from prior trajectories and
               merge a new trajectory in one at a time. Paper's worst-case
               online schedule — interference grows turn by turn.

The prompt deliberately does NOT tell the consolidator about the rubric or
the family taxonomy. The whole point of the benchmark is to see how often a
generic "extract lessons from this trajectory" instruction strips
applicability conditions when faced with different schedules.
"""

from __future__ import annotations

from pydantic import BaseModel

from errornodebench.llm import structured_call
from errornodebench.models import Memory, MemoryEntry, Trajectory


CONSOLIDATOR_SYSTEM = """You maintain a textual notebook of reusable lessons
distilled from past trajectories of an agent operating in an environment.

A good lesson is concrete, actionable, and clearly bounded — it states
*when* it applies, not just *what* to do. When updating the notebook, you
may add new entries, edit existing entries to incorporate new evidence, or
merge entries that capture the same underlying pattern. Do not duplicate.

CRITICAL: every entry must list its `source_task_ids` — the task_id of
every trajectory that contributed to it. If you merge two entries derived
from tasks A and B into one new entry, the merged entry's source_task_ids
must include both A and B. Never return an entry with an empty
source_task_ids list — if you cannot attribute it, do not include it.

Return the FULL updated notebook (not a diff). Each entry has:
  - when_to_use:      applicability conditions
  - strategy:         what to do
  - source_task_ids:  REQUIRED non-empty list of contributing task_ids
"""


class _ConsolidatorResponse(BaseModel):
    entries: list[MemoryEntry]


def _render_trajectory(traj: Trajectory) -> str:
    lines = [
        f"Task ID: {traj.task.task_id}",
        f"Goal: {traj.task.goal}",
        f"Environment: {traj.task.environment}",
        "Steps:",
    ]
    for i, s in enumerate(traj.steps, 1):
        lines.append(
            f"  {i}. observation={s.observation!r} thought={s.thought!r} "
            f"action={s.action!r} result={s.result!r}"
        )
    lines.append(f"Final outcome: {traj.final_outcome}")
    lines.append(f"Success: {traj.success}")
    return "\n".join(lines)


def consolidate(
    *,
    prior_memory: Memory,
    new_trajectory: Trajectory,
    model: str,
    seed: int | None = None,
) -> Memory:
    """Merge one trajectory into ``prior_memory`` and return the new memory.

    The single-trajectory primitive behind the Fresh arm (empty prior) and the
    Cumulative arm (running prior). The model is asked to return the full
    updated notebook, not a diff.
    """
    user = (
        "Current notebook:\n"
        f"{prior_memory.render()}\n\n"
        "New trajectory to incorporate:\n"
        f"{_render_trajectory(new_trajectory)}\n\n"
        "Return the updated notebook as JSON matching the schema. Remember: "
        f"every entry must list source_task_ids including '{new_trajectory.task.task_id}' "
        "if the entry was informed by this trajectory."
    )
    resp = structured_call(
        model=model,
        system=CONSOLIDATOR_SYSTEM,
        user=user,
        response_model=_ConsolidatorResponse,
        temperature=0.0,
        max_tokens=6000,
        seed=seed,
    )
    return Memory(entries=resp.entries)


def consolidate_batch(
    *,
    prior_memory: Memory,
    trajectories: list[Trajectory],
    model: str,
    seed: int | None = None,
) -> Memory:
    """Consolidate multiple trajectories in a single call.

    Used by the Static-Group arm: a clean batch of one task family per call
    is the best-case offline schedule from the paper.
    """
    rendered = "\n\n---\n\n".join(_render_trajectory(t) for t in trajectories)
    ids = [t.task.task_id for t in trajectories]
    user = (
        "Current notebook:\n"
        f"{prior_memory.render()}\n\n"
        f"New trajectories to incorporate ({len(trajectories)} of them):\n"
        f"{rendered}\n\n"
        "Return the updated notebook as JSON matching the schema. Remember: "
        f"every entry's source_task_ids must include the task_ids of the "
        f"trajectories that contributed to it (available IDs: {ids})."
    )
    resp = structured_call(
        model=model,
        system=CONSOLIDATOR_SYSTEM,
        user=user,
        response_model=_ConsolidatorResponse,
        temperature=0.0,
        max_tokens=8000,
        seed=seed,
    )
    return Memory(entries=resp.entries)
