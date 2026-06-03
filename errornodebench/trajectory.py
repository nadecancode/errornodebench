"""Solver LLM: turn a Task spec into a multi-step Trajectory.

The solver does NOT see the rubric (correct_strategy /
applicability_conditions). It only sees the goal and environment, the way an
agent in the wild would. That keeps the trajectory honest — it can succeed,
partially succeed, or fail, and the consolidator has to do its own
distillation from what actually happened.
"""

from __future__ import annotations

from pydantic import BaseModel

from errornodebench.llm import structured_call
from errornodebench.models import Task, Trajectory, TrajectoryStep


SOLVER_SYSTEM = """You are an agent operating in a simulated environment.
You produce a trajectory of (observation, thought, action, result) tuples
showing how you attempt to accomplish a goal.

Rules:
- Be concrete. Reference specific objects from the environment.
- Each step is one atomic action (e.g., "pick up butter", "turn stove to low").
- 3 to 8 steps is normal. Stop when the goal is achieved or clearly blocked.
- The `result` field reports what actually happens after the action — be
  realistic. If you applied an inappropriate strategy, the result should
  reflect the failure (e.g., putting soda in the freezer ruptures the bottle).
- Conclude with whether the goal was achieved.
"""


class _SolverResponse(BaseModel):
    steps: list[TrajectoryStep]
    final_outcome: str
    success: bool


def generate_trajectory(
    task: Task, *, model: str, seed: int | None = None
) -> Trajectory:
    """Run the solver on one task and return its multi-step trajectory.

    Uses ``temperature=0.3`` (the only non-zero-temperature call in the
    pipeline) so that distinct seeds regenerate a varied trajectory pool;
    consolidation and judging downstream are deterministic at temperature 0.
    """
    user = (
        f"Goal: {task.goal}\n"
        f"Environment: {task.environment}\n\n"
        "Produce a trajectory that attempts this goal in the environment. "
        "Return JSON matching the schema."
    )
    resp = structured_call(
        model=model,
        system=SOLVER_SYSTEM,
        user=user,
        response_model=_SolverResponse,
        temperature=0.3,
        seed=seed,
    )
    return Trajectory(
        task=task,
        steps=resp.steps,
        final_outcome=resp.final_outcome,
        success=resp.success,
    )
