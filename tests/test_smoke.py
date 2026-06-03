"""Offline smoke tests for the deterministic core of ErrorNodeBench.

These exercise everything that does NOT require a model backbone: scenario
construction, the coverage metric, task lookup, and round-tripping a released
``runs/*.json`` through the Pydantic models. They make no API calls.

Run either way (no pytest dependency required)::

    python tests/test_smoke.py        # plain asserts, prints PASS/FAIL
    python -m pytest tests/           # if pytest is installed
"""

from __future__ import annotations

import json
from pathlib import Path

from errornodebench.models import (
    BenchmarkResult,
    Memory,
    MemoryEntry,
    compute_coverage,
)
from errornodebench.scenarios.interference import (
    ALL_FAMILIES,
    SEQUENCES,
    default_sequence,
    get_task,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_taxonomy_shape():
    """5 families x 3 tasks = a 15-task interference sequence."""
    assert len(ALL_FAMILIES) == 5
    assert all(len(tasks) == 3 for tasks in ALL_FAMILIES.values())
    assert len(default_sequence()) == 15


def test_sequences():
    """All three orderings cover the same 15 tasks; reversed is the mirror."""
    assert set(SEQUENCES) == {"default", "reversed", "family-blocked"}
    base_ids = [t.task_id for t in SEQUENCES["default"]]
    for name, seq in SEQUENCES.items():
        assert sorted(t.task_id for t in seq) == sorted(base_ids), name
    assert [t.task_id for t in SEQUENCES["reversed"]] == list(reversed(base_ids))


def test_get_task():
    """Lookup returns the right Task and raises KeyError on a bad id."""
    a_task = default_sequence()[0]
    assert get_task(a_task.task_id).task_id == a_task.task_id
    try:
        get_task("does-not-exist")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown task_id")


def test_coverage_full_vs_collapsed():
    """Coverage is 100% when every task is cited and collapses with attribution loss."""
    seq_ids = [t.task_id for t in default_sequence()]

    full = Memory(entries=[
        MemoryEntry(when_to_use="x", strategy="y", source_task_ids=[tid])
        for tid in seq_ids
    ])
    cov_full = compute_coverage(memory=full, task_sequence=seq_ids, families=ALL_FAMILIES)
    assert cov_full.family_coverage == 1.0
    assert cov_full.task_coverage == 1.0

    # One surviving family, one cited task -> the Cumulative-collapse signature.
    one_id = seq_ids[0]
    collapsed = Memory(entries=[
        MemoryEntry(when_to_use="x", strategy="y", source_task_ids=[one_id])
    ])
    cov_collapsed = compute_coverage(memory=collapsed, task_sequence=seq_ids, families=ALL_FAMILIES)
    assert abs(cov_collapsed.family_coverage - 0.2) < 1e-9   # 1 of 5 families
    assert abs(cov_collapsed.task_coverage - 1 / 15) < 1e-9  # 1 of 15 tasks


def test_run_json_roundtrips():
    """A released run file validates against the models and reproduces the
    headline collapse (Cumulative family coverage = 20%)."""
    result = BenchmarkResult.model_validate(
        json.loads((REPO_ROOT / "runs" / "v4.json").read_text())
    )
    assert result.scenario == "interference"
    assert result.arm_names == ["fresh", "static_group", "cumulative", "reflexion", "expel"]
    assert abs(result.aggregate("cumulative").mean_family_coverage - 0.2) < 1e-9
    assert abs(result.aggregate("fresh").mean_family_coverage - 1.0) < 1e-9


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
