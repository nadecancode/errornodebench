"""Reflexion baseline.

Originally: Shinn et al. 2023. The agent reflects on a trajectory and
produces a short self-critique that biases its next attempt at a similar
task. The distinctive property for our benchmark:

  * No cross-task merging. Each reflection is appended verbatim.
  * Memory grows linearly with the number of trajectories.
  * Per-entry quality depends on the model's introspection on one trajectory
    at a time — no opportunity for interference, but also no opportunity for
    generalization.

So Reflexion should look a lot like Fresh on coverage (full) and like Fresh
or better on the bad-entry signal (no merging → no over-generalization).
The main reason to include it is as a calibration point: it's what you get
when you simply refuse to merge.
"""

from __future__ import annotations

from pydantic import BaseModel

from errornodebench.consolidator import _render_trajectory
from errornodebench.llm import structured_call
from errornodebench.models import Memory, MemoryEntry, Trajectory


REFLEXION_SYSTEM = """You analyze the trajectory of an agent attempting a task
and produce a short SELF-REFLECTION that will help the agent on similar
future tasks.

Your reflection should:
- Identify the key insight from this attempt (what worked, what didn't, what
  to try differently).
- Be stated as actionable guidance for future similar tasks.
- Include the precise applicability conditions (when this reflection
  applies) so it doesn't fire on unrelated tasks.

Return exactly ONE memory entry per call. The entry must have:
  - when_to_use:     applicability conditions for similar future tasks
  - strategy:        the actionable insight
  - source_task_ids: a single-element list with the trajectory's task_id
"""


class _ReflexionResponse(BaseModel):
    entry: MemoryEntry


def reflect(*, trajectory: Trajectory, model: str, seed: int | None = None) -> MemoryEntry:
    user = (
        "Trajectory to reflect on:\n"
        f"{_render_trajectory(trajectory)}\n\n"
        "Produce one self-reflection entry as JSON matching the schema. "
        f"`source_task_ids` must be exactly ['{trajectory.task.task_id}']."
    )
    resp = structured_call(
        model=model,
        system=REFLEXION_SYSTEM,
        user=user,
        response_model=_ReflexionResponse,
        temperature=0.0,
        max_tokens=2000,
        seed=seed,
    )
    return resp.entry


def consolidate_reflexion(
    *,
    trajectories: list[Trajectory],
    model: str,
    seed: int | None = None,
) -> Memory:
    """Process trajectories independently; concatenate reflections.

    Caller is expected to parallelize across trajectories (Reflexion has no
    cross-trajectory dependency).
    """
    entries = [
        reflect(trajectory=t, model=model, seed=seed) for t in trajectories
    ]
    return Memory(entries=entries)
