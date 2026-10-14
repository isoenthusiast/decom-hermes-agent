"""Trusted, turn-scoped Kanban ownership propagated through native gateway wakes."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional


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


def kanban_run_claim_payload(claim: Optional[KanbanRunClaim]) -> Optional[dict[str, Any]]:
    """Return the JSON-safe form used only by trusted internal continuation events."""
    if claim is None:
        return None
    return {"board": claim.board, "task_id": claim.task_id, "run_id": claim.run_id}


def kanban_run_claim_from_payload(payload: Any) -> Optional[KanbanRunClaim]:
    """Decode a trusted internal continuation claim, failing closed on malformed state."""
    if not isinstance(payload, dict):
        return None
    board = payload.get("board")
    task_id = payload.get("task_id")
    run_id = payload.get("run_id")
    if board is not None and not isinstance(board, str):
        return None
    if not isinstance(task_id, str) or not task_id.strip():
        return None
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        return None
    return KanbanRunClaim(board=board, task_id=task_id, run_id=run_id)


@contextmanager
def kanban_run_claim(claim: Optional[KanbanRunClaim]) -> Iterator[None]:
    """Bind a claim only for one wake turn and restore the prior context."""
    token = _current_claim.set(claim)
    try:
        yield
    finally:
        _current_claim.reset(token)
