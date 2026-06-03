"""ThreadPool helper for fanning out independent LLM calls.

litellm + the underlying HTTP client are thread-safe, and mgpt's account
pool handles concurrent requests up to the number of registered accounts.
We default to a small pool (4 workers) to be polite to mgpt and to avoid
saturating its upstream rate limits.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


DEFAULT_MAX_WORKERS = 4


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[R]:
    """Like list(map(fn, items)) but uses a thread pool.

    Preserves input order. Falls back to sequential when there's nothing to
    gain from threading (single item or max_workers <= 1).
    """
    items_list = list(items)
    if not items_list:
        return []
    workers = min(max_workers, len(items_list))
    if workers <= 1:
        return [fn(x) for x in items_list]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items_list))
