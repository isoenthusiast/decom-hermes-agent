"""Regression tests: pending-drain + finally-cleanup races must not spawn
duplicate agents OR silently drop messages that arrived during cleanup.

Two related races in gateway/platforms/base.py:_process_message_background:

1. Pending-drain path (previous line 1931):
   ``del self._active_sessions[session_key]`` opened a window where a
   concurrent inbound message could pass the Level-1 guard, spawn its
   own _process_message_background, and run simultaneously with the
   recursive drain.  Two agents on one session_key = duplicate responses.

2. Finally-cleanup path (previous line 1990-1991):
   Between the awaits in finally (typing_task, stop_typing) and the
   ``del self._active_sessions[session_key]``, a new message could
   land in _pending_messages.  The del ran anyway, and the message was
   silently dropped — user never got a reply.

Fix: keep the _active_sessions entry live across the turn chain and
clear the Event instead of deleting; in finally, drain any
late-arrival pending message by spawning a task instead of
dropping it.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.kanban_claim import KanbanRunClaim, current_kanban_run_claim
from gateway.run_turn import GatewayTurnMixin
from gateway.turn_context import TurnContext
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools import kanban_tools


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter._send_with_retry = AsyncMock(return_value=None)
    return adapter


def _make_event(text="hi", chat_id="42"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"
        ),
    )


def _sk(chat_id="42"):
    return build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    )


@pytest.mark.asyncio
async def test_pending_drain_keeps_active_session_guard_live():
    """Fix for R5: during pending-drain cleanup, _active_sessions must stay
    populated so concurrent inbound messages can't spawn a duplicate
    _process_message_background.  We only CLEAR the Event, never delete."""
    adapter = _make_adapter()
    sk = _sk()

    # Register a slow handler so the agent is "mid-processing" when the
    # pending message arrives.
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_processed = asyncio.Event()
    handoff_entered = asyncio.Event()
    release_handoff = asyncio.Event()

    async def handler(event):
        first_started.set()
        await release_first.wait()
        if event.text == "M2":
            second_processed.set()
        return "done"

    adapter._message_handler = handler

    original_stop_refresh = adapter._stop_typing_refresh

    async def stop_typing_during_handoff(*args, **kwargs):
        handoff_entered.set()
        await release_handoff.wait()
        return await original_stop_refresh(*args, **kwargs)

    adapter._stop_typing_refresh = stop_typing_during_handoff

    # Spawn M1 through handle_message.
    await adapter.handle_message(_make_event(text="M1"))

    # Wait until M1 is actively running inside the handler.
    await asyncio.wait_for(first_started.wait(), timeout=5.0)

    # Assert: session is active.
    assert sk in adapter._active_sessions
    active_event = adapter._active_sessions[sk]

    # Simulate pending message (M2) queued while M1 runs.
    adapter._pending_messages[sk] = _make_event(text="M2")

    # Release M1 — pending-drain block now runs.  During its cleanup
    # awaits, _active_sessions[sk] must remain populated (same object
    # reference) so any M3 arriving in that window hits the busy-handler.
    release_first.set()

    try:
        # Pause inside the handoff's typing cleanup. Production has already
        # cleared the guard and has not yet transferred task ownership.
        await asyncio.wait_for(handoff_entered.wait(), timeout=5.0)

        # Across the drain transition, the Event object must be the SAME
        # reference (not replaced, not deleted).
        assert sk in adapter._active_sessions, (
            "_active_sessions[session_key] was deleted during pending-drain — "
            "opens a window for duplicate-agent spawn"
        )
        assert adapter._active_sessions[sk] is active_event, (
            "_active_sessions[session_key] was replaced during pending-drain — "
            "the old Event may have waiters that now won't be signaled"
        )

        # Finish drain without relying on scheduler speed.
        release_handoff.set()
        await asyncio.wait_for(second_processed.wait(), timeout=5.0)
    finally:
        release_handoff.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_finally_cleanup_drains_late_arrival_pending():
    """Fix for R6: if a message lands in _pending_messages during the
    finally-block cleanup awaits, the finally must spawn a drain task
    instead of deleting _active_sessions and dropping the message."""
    adapter = _make_adapter()
    sk = _sk()

    processed = []
    late_processed = asyncio.Event()

    async def handler(event):
        processed.append(event.text)
        if event.text == "LATE":
            late_processed.set()
        return "ok"

    adapter._message_handler = handler

    # Instrument stop_typing to inject a late-arrival pending message
    # during the finally-block await window.  This exactly simulates the
    # R6 race: the message arrives after the response has been sent but
    # before _active_sessions is deleted.
    original_stop = adapter.stop_typing if hasattr(adapter, "stop_typing") else None

    injected = {"done": False}

    async def stop_typing_injects_pending(*args, **kwargs):
        # Yield so the injection happens mid-await.
        await asyncio.sleep(0)
        if not injected["done"]:
            adapter._pending_messages[sk] = _make_event(text="LATE")
            injected["done"] = True
        if original_stop:
            return await original_stop(*args, **kwargs)
        return None

    adapter.stop_typing = stop_typing_injects_pending

    # Send M1.
    await adapter.handle_message(_make_event(text="M1"))

    # Drain: wait for the late-drain task itself to process LATE.
    await asyncio.wait_for(late_processed.wait(), timeout=5.0)

    await adapter.cancel_background_tasks()

    assert "M1" in processed, "M1 was not processed"
    assert "LATE" in processed, (
        "Late-arrival pending message was silently dropped — finally "
        "cleanup should have spawned a drain task"
    )


@pytest.mark.asyncio
async def test_no_pending_cleans_up_normally():
    """Regression guard: when no pending message exists, the finally
    block must still delete _active_sessions as before (no leak)."""
    adapter = _make_adapter()
    sk = _sk()

    async def handler(event):
        return "ok"

    adapter._message_handler = handler

    await adapter.handle_message(_make_event(text="solo"))

    # Await the task that owns this session rather than sampling cleanup after
    # an arbitrary wall-clock delay.
    owner_task = adapter._session_tasks[sk]
    await asyncio.wait_for(asyncio.shield(owner_task), timeout=5.0)

    assert sk not in adapter._active_sessions, (
        "_active_sessions was not cleaned up after a normal turn with no pending"
    )
    assert sk not in adapter._pending_messages

    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_native_wake_claim_survives_busy_queue_and_binds_at_consumer():
    """A claim admitted while busy must authorize the later queued consumer, not admission only."""
    adapter = _make_adapter()
    sk = _sk()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_processed = asyncio.Event()
    seen: list[tuple[str, int] | None] = []

    async def handler(event):
        claim = current_kanban_run_claim()
        seen.append((claim.task_id, claim.run_id) if claim is not None else None)
        if event.text == "first":
            first_started.set()
            await release_first.wait()
        else:
            second_processed.set()
        return None

    adapter._message_handler = handler
    first = _make_event(text="first")
    first._kanban_claim = KanbanRunClaim("default", "task-first", 101)
    second = _make_event(text="second")
    second._kanban_claim = KanbanRunClaim("default", "task-second", 202)

    await adapter.handle_message(first)
    await asyncio.wait_for(first_started.wait(), timeout=5.0)
    await adapter.handle_message(second)
    assert adapter._pending_messages[sk] is second

    release_first.set()
    await asyncio.wait_for(second_processed.wait(), timeout=5.0)
    await adapter.cancel_background_tasks()

    assert seen == [("task-first", 101), ("task-second", 202)]
    assert current_kanban_run_claim() is None


@pytest.mark.asyncio
async def test_runner_followup_binds_event_claim_and_clears_prior_claim():
    """The real runner recursion must bind the dequeued event claim at turn execution."""

    class RunnerHarness(GatewayTurnMixin):
        _MAX_INTERRUPT_DEPTH = 3

        def __init__(self):
            self.seen = []

        async def _run_agent_deliver_first_response(self, *args, **kwargs):
            return None

        def _adapter_for_source(self, source):
            return None

        async def _refresh_agent_cache_message_count(self, *args):
            return None

        def _session_key_for_source(self, source):
            return "telegram:dm:42"

        async def _prepare_profile_scoped_inbound_message_text(self, *, event, **kwargs):
            return event.text

        def _reply_anchor_for_event(self, event):
            return None

        def _is_goal_continuation_event(self, event):
            return False

        def _goal_still_active_for_session(self, session_id):
            return True

        async def _run_agent(self, **kwargs):
            claim = current_kanban_run_claim()
            self.seen.append(None if claim is None else (claim.task_id, claim.run_id))
            return {"messages": kwargs["history"]}

    runner = RunnerHarness()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    ctx = TurnContext(
        source=source, session_id="session", session_key="telegram:dm:42", history=[],
    )
    event = _make_event(text="review")
    event._kanban_claim = KanbanRunClaim("default", "task-review", 123)

    await runner._run_agent_queued_followup(
        ctx, adapter=None, pending=event.text, pending_event=event,
        response="done", result={"messages": []}, stream_task=None,
    )
    assert runner.seen == [("task-review", 123)]
    assert current_kanban_run_claim() is None

    second = _make_event(text="review-again")
    second._kanban_claim = KanbanRunClaim("default", "task-second", 456)
    await runner._run_agent_queued_followup(
        ctx, adapter=None, pending=second.text, pending_event=second,
        response="done", result={"messages": []}, stream_task=None,
    )
    assert runner.seen == [("task-review", 123), ("task-second", 456)]
    assert current_kanban_run_claim() is None

    from gateway.kanban_claim import kanban_run_claim
    with kanban_run_claim(KanbanRunClaim("default", "stale", 999)):
        await runner._run_agent_queued_followup(
            ctx, adapter=None, pending="ordinary", pending_event=None,
            response="done", result={"messages": []}, stream_task=None,
        )
    assert runner.seen == [("task-review", 123), ("task-second", 456), None]


def _make_review_run(conn, title):
    task_id = kb.create_task(conn, title=title, assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:test")
    assert implementation is not None
    assert kb.request_review(
        conn, task_id, reviewer="reviewer", summary="ready",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:test")
    assert review is not None
    return task_id, review.current_run_id


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["complete", "request_changes"])
@pytest.mark.parametrize("claim_kind", ["valid", "stale", "foreign", "cross_board", "none"])
async def test_real_busy_queue_runner_followup_reaches_lifecycle_handlers(
    tmp_path, monkeypatch, action, claim_kind
):
    """Exercise adapter admission -> pending dequeue -> runner recursion -> real tool handler.

    The temporary board makes the assertions about DB mutation, rather than merely
    ContextVar observation.  Invalid claims must leave the live review run intact.
    """
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    conn = kbc.connect(db_path)
    task_id, run_id = _make_review_run(conn, f"{action}-{claim_kind}")
    foreign_task, foreign_run_id = _make_review_run(conn, "foreign")
    conn.close()

    target_claim = {
        "valid": KanbanRunClaim("default", task_id, run_id),
        "stale": KanbanRunClaim("default", task_id, run_id - 1),
        "foreign": KanbanRunClaim("default", foreign_task, foreign_run_id),
        "cross_board": KanbanRunClaim("other-board", task_id, run_id),
        "none": None,
    }[claim_kind]
    adapter = _make_adapter()
    runner = None
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    session_key = build_session_key(source)
    ctx = TurnContext(source=source, session_id="session", session_key=session_key, history=[])

    class RealRunner(GatewayTurnMixin):
        _MAX_INTERRUPT_DEPTH = 3

        def __init__(self):
            self.seen = []
            self._draining = False

        async def _run_agent_deliver_first_response(self, *args, **kwargs):
            return None

        def _adapter_for_source(self, source):
            return adapter

        async def _refresh_agent_cache_message_count(self, *args):
            return None

        def _session_key_for_source(self, source):
            return session_key

        async def _prepare_profile_scoped_inbound_message_text(self, *, event, **kwargs):
            return event.text

        def _reply_anchor_for_event(self, event):
            return None

        def _is_goal_continuation_event(self, event):
            return False

        def _goal_still_active_for_session(self, session_id):
            return True

        def _promote_queued_event(self, session_key, adapter, pending_event):
            return pending_event

        def _pending_event_audio_paths(self, event):
            return []

        async def _run_agent(self, **kwargs):
            event_claim = current_kanban_run_claim()
            self.seen.append(None if event_claim is None else (event_claim.task_id, event_claim.run_id))
            args = {"task_id": task_id}
            if action == "complete":
                args["summary"] = "approved by queued runner"
                output = kanban_tools._handle_complete(args)
            else:
                args["reason"] = "queued runner found a defect"
                output = kanban_tools._handle_request_changes(args)
            return {"messages": kwargs["history"], "tool_output": json.loads(output)}

    runner = RealRunner()

    async def first_handler(event):
        first_started.set()
        await release_first.wait()
        pending_event, pending = await runner._run_agent_drain_pending(
            {"messages": []}, adapter, source, ctx.session_key,
        )
        await runner._run_agent_queued_followup(
            ctx, adapter, pending, pending_event, "done", {"messages": []}, None,
        )
        return "ok"

    adapter._message_handler = first_handler
    first = _make_event(text="first")
    second = _make_event(text=action)
    second._kanban_claim = target_claim
    await adapter.handle_message(first)
    await asyncio.wait_for(first_started.wait(), timeout=5)
    await adapter.handle_message(second)
    assert adapter._pending_messages[session_key] is second
    release_first.set()
    owner_task = adapter._session_tasks[session_key]
    await asyncio.wait_for(asyncio.shield(owner_task), timeout=5)
    await adapter.cancel_background_tasks()

    conn = kbc.connect(db_path)
    task = kb.get_task(conn, task_id)
    assert task is not None
    if claim_kind == "valid":
        assert task.status == ("done" if action == "complete" else "ready")
    else:
        assert task.status == "running"
        assert task.assignee == "reviewer"
        assert task.current_run_id == run_id
    conn.close()
