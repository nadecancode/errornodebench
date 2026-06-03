"""EXPEL baseline.

Originally: Zhao et al. 2024. EXPEL extracts insights from successful
trajectories in two phases:

  Phase 1 (per-task): for each trajectory, prompt the model to extract a
                      small list of operational insights — generalizable
                      lessons distilled from that one trajectory.

  Phase 2 (cross-task): collect insights across all trajectories and prompt
                        the model to refine them — merging duplicates,
                        sharpening applicability conditions, and producing a
                        consolidated insight set.

Phase 2 is the interesting one for ErrorNodeBench. It is exactly the kind
of bulk-rewrite step the Faulty Memory paper identifies as the source of
interference: the model is asked to look at lessons from heterogeneous tasks
side-by-side and is implicitly invited to over-generalize across families.

Reference (method adapted, no code copied):
    Zhao, Huang, Xu, Lin, Liu & Huang (2024), "ExpeL: LLM Agents Are
    Experiential Learners," AAAI 2024. arXiv:2308.10144.

Adapted for ErrorNodeBench: we re-implement the two-phase insight extraction
and cross-task refinement only (not ExpeL's task-solving or retrieval loop),
emit MemoryEntry objects in this benchmark's schema, and run it offline over a
fixed trajectory pool at temperature 0 with a fixed seed.
"""

from __future__ import annotations

from pydantic import BaseModel

from errornodebench.consolidator import _render_trajectory
from errornodebench.llm import structured_call
from errornodebench.models import Memory, MemoryEntry, Trajectory


PHASE1_SYSTEM = """You extract operational insights from an agent trajectory.

An insight is a concrete, reusable lesson. Each insight should:
- State applicability conditions (when to use it).
- State the action or strategy (what to do).
- Be specific enough to be actionable, but general enough to apply to close
  variants of this task.

Return a SHORT list of insights (typically 1-3). Each entry's
source_task_ids must contain ONLY this trajectory's task_id.
"""


PHASE2_SYSTEM = """You are refining a draft insight list distilled from
multiple agent trajectories.

Your job:
- Merge insights that capture the same underlying pattern into one entry.
  When you merge, combine their source_task_ids into one non-empty list.
- Sharpen applicability conditions when an insight is too broad.
- Drop redundant or contradictory entries.
- Do NOT invent new insights that aren't supported by the draft list.

Every returned entry must have:
  - when_to_use:     bounded applicability conditions
  - strategy:        what to do
  - source_task_ids: non-empty list of task_ids the entry was derived from
"""


class _InsightListResponse(BaseModel):
    entries: list[MemoryEntry]


def _phase1_one(
    *, trajectory: Trajectory, model: str, seed: int | None
) -> list[MemoryEntry]:
    user = (
        "Trajectory:\n"
        f"{_render_trajectory(trajectory)}\n\n"
        "Extract 1-3 insights as JSON matching the schema. Each entry's "
        f"source_task_ids must be exactly ['{trajectory.task.task_id}']."
    )
    resp = structured_call(
        model=model,
        system=PHASE1_SYSTEM,
        user=user,
        response_model=_InsightListResponse,
        temperature=0.0,
        max_tokens=3000,
        seed=seed,
    )
    return resp.entries


def _render_draft(entries: list[MemoryEntry]) -> str:
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(
            f"[{i}] when_to_use: {e.when_to_use}\n"
            f"    strategy: {e.strategy}\n"
            f"    source_task_ids: {e.source_task_ids}"
        )
    return "\n".join(lines)


def _phase2_generalize(
    *,
    draft: list[MemoryEntry],
    model: str,
    seed: int | None,
) -> list[MemoryEntry]:
    user = (
        f"Draft insight list ({len(draft)} entries):\n"
        f"{_render_draft(draft)}\n\n"
        "Refine the list. Return the FINAL insights as JSON matching the "
        "schema. Preserve source_task_ids — every returned entry's "
        "source_task_ids must be the union of the contributing draft "
        "entries' source_task_ids."
    )
    resp = structured_call(
        model=model,
        system=PHASE2_SYSTEM,
        user=user,
        response_model=_InsightListResponse,
        temperature=0.0,
        max_tokens=6000,
        seed=seed,
    )
    return resp.entries


def consolidate_expel(
    *,
    trajectories: list[Trajectory],
    model: str,
    seed: int | None = None,
    max_workers: int = 4,
) -> Memory:
    """Two-phase EXPEL consolidation.

    Phase 1 is parallelizable per trajectory (independent extractions).
    Phase 2 is a single batched call over the full draft.
    """
    from errornodebench.concurrency import parallel_map

    def _do(t: Trajectory) -> list[MemoryEntry]:
        return _phase1_one(trajectory=t, model=model, seed=seed)

    drafts: list[list[MemoryEntry]] = parallel_map(
        _do, trajectories, max_workers=max_workers
    )
    flat_draft: list[MemoryEntry] = [e for sub in drafts for e in sub]
    final = _phase2_generalize(draft=flat_draft, model=model, seed=seed)
    return Memory(entries=final)
