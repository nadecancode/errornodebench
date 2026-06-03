"""Judge LLM: label each memory entry against the task-family rubrics.

The judge sees:
  - the full set of task families with their `correct_strategy` and
    `applicability_conditions` (the ground truth for what bounds *should*
    look like)
  - one memory entry at a time

It returns a per-entry verdict:
  useful            : strategy correct AND applicability conditions properly
                      bounded for the family it actually targets
  over_generalized  : strategy reasonable for ONE family but applicability
                      reads as broader, so the entry would mislead the agent
                      on a different family (e.g., "apply heat to change
                      state" being retrieved for a freezing task)
  over_specialized  : entry only fires for the exact instances seen — fails
                      on close variants in the same family
  garbage           : incorrect, contradictory, or referencing things not in
                      any family
"""

from __future__ import annotations

from pydantic import BaseModel

from errornodebench.llm import structured_call
from errornodebench.models import (
    JudgeVerdict,
    Memory,
    MemoryEntry,
    Task,
    VerdictLabel,
)


JUDGE_SYSTEM = """You are an evaluator scoring entries in an agent's memory
notebook against a known taxonomy of task families.

For each entry, decide whether it is:

  useful           - strategy is correct AND applicability conditions are
                     properly bounded for ONE specific family. The entry
                     would help on tasks in that family and would NOT
                     misfire on tasks in other families.

  over_generalized - the strategy is correct for a specific family, but the
                     applicability conditions are too broad. The entry would
                     be retrieved for tasks in a DIFFERENT family and lead
                     the agent astray (e.g., a 'use a heat source for state
                     changes' lesson firing on a freezing task).

  over_specialized - the entry only fires for the exact task instance(s)
                     seen, not for plausible close variants in the same
                     family.

  garbage          - incorrect, contradictory, or unrelated to any family.

Be strict about over_generalized: if an entry's `when_to_use` would plausibly
trigger on a task from a DIFFERENT family in the taxonomy, label it
over_generalized and list the affected family names.
"""


def _render_taxonomy(families: dict[str, list[Task]]) -> str:
    chunks = []
    for fam_name, tasks in families.items():
        # Use the first task's rubric as the canonical family rubric. Tasks
        # within a family share the same correct_strategy/applicability
        # conditions by construction.
        canon = tasks[0]
        examples = "; ".join(t.goal for t in tasks)
        chunks.append(
            f"FAMILY: {fam_name}\n"
            f"  Example goals: {examples}\n"
            f"  Correct strategy: {canon.correct_strategy}\n"
            f"  Applicability conditions: {canon.applicability_conditions}"
        )
    return "\n\n".join(chunks)


class _VerdictResponse(BaseModel):
    label: VerdictLabel
    reasoning: str
    affected_families: list[str] = []


def judge_entry(
    *,
    entry: MemoryEntry,
    entry_index: int,
    families: dict[str, list[Task]],
    model: str,
    seed: int | None = None,
) -> JudgeVerdict:
    """Score a single memory entry against the family taxonomy.

    Returns a :class:`JudgeVerdict` with one of the four labels (useful /
    over_generalized / over_specialized / garbage) and the families the entry
    would mislead on, if any.
    """
    user = (
        "Task family taxonomy:\n"
        f"{_render_taxonomy(families)}\n\n"
        "Memory entry to evaluate:\n"
        f"  when_to_use: {entry.when_to_use}\n"
        f"  strategy:    {entry.strategy}\n"
        f"  source_task_ids: {entry.source_task_ids}\n\n"
        "Return JSON: label, reasoning, affected_families (family names this "
        "entry would mislead on — empty list if none)."
    )
    resp = structured_call(
        model=model,
        system=JUDGE_SYSTEM,
        user=user,
        response_model=_VerdictResponse,
        temperature=0.0,
        seed=seed,
    )
    return JudgeVerdict(
        entry_index=entry_index,
        entry=entry,
        label=resp.label,
        reasoning=resp.reasoning,
        affected_families=resp.affected_families,
    )


def judge_memory(
    *,
    memory: Memory,
    families: dict[str, list[Task]],
    model: str,
    seed: int | None = None,
    max_workers: int = 4,
) -> list[JudgeVerdict]:
    """Score every entry in the memory.

    Judge calls are independent and read-only, so we fan them out across a
    thread pool. Result order matches `memory.entries`.
    """
    from errornodebench.concurrency import parallel_map

    indexed = list(enumerate(memory.entries))

    def _score(pair: tuple[int, MemoryEntry]) -> JudgeVerdict:
        i, e = pair
        return judge_entry(
            entry=e,
            entry_index=i,
            families=families,
            model=model,
            seed=seed,
        )

    return parallel_map(_score, indexed, max_workers=max_workers)
