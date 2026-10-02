"""Trusted, turn-scoped Kanban ownership propagated through native gateway wakes."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class KanbanRunClaim:
    """A claim reserved by the gateway before it starts a native wake turn."""

    board: Optional[str]
    task_id: str
    run_id: int


_current_claim: ContextVar[Optional[KanbanRunClaim]] = ContextVar(
    "hermes_kanban_run_claim", default=None
)


def current_kanban_run_claim() -> Optional[KanbanRunClaim]:
    """Return the trusted claim for the current gateway turn, if any."""
    return _current_claim.get()


@contextmanager
def kanban_run_claim(claim: Optional[KanbanRunClaim]) -> Iterator[None]:
    """Bind a claim only for one wake turn and restore the prior context."""
    token = _current_claim.set(claim)
    try:
        yield
    finally:
        _current_claim.reset(token)
