"""Crash-window recovery for durable client-turn admission."""

from __future__ import annotations

import queue

import pytest

from api import config, models, routes, streaming
from api.client_turn_ledger import ClientTurnLedger


def _session(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    session_dir = state_dir / "sessions"
    session_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(routes, "SESSION_DIR", session_dir)
    monkeypatch.setattr(streaming, "SESSION_DIR", session_dir)
    monkeypatch.setenv("HERMES_TURN_LEDGER_ENABLED", "1")
    models.SESSIONS.clear()
    routes.SESSIONS.clear()
    streaming.SESSIONS.clear()
    config.STREAMS.clear()
    config.ACTIVE_RUNS.clear()
    session = models.Session(
        session_id="crash-window-session",
        workspace=str(tmp_path),
        messages=[],
        profile="work",
    )
    session.save()
    models.SESSIONS[session.session_id] = session
    routes.SESSIONS[session.session_id] = session
    streaming.SESSIONS[session.session_id] = session
    return session


def _start_kwargs(tmp_path, client_turn_id):
    return {
        "msg": "persist this turn once",
        "attachments": [],
        "workspace": str(tmp_path),
        "model": "test-model",
        "model_provider": "test-provider",
        "client_turn_id": client_turn_id,
    }


def test_started_claim_crash_reconciles_to_recovery_required_on_retry(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "HERMES_WEBUI_CLIENT_TURN_FAILPOINT",
        "after_started_claim",
    )

    with pytest.raises(routes.ClientTurnAdmissionCrash):
        routes._start_chat_stream_for_session(
            session,
            **_start_kwargs(tmp_path, "started-crash-client"),
        )

    ledger = ClientTurnLedger(config.STATE_DIR / "client_turn_ledger.sqlite3")
    claimed = ledger.get(session.session_id, "started-crash-client")
    assert claimed["state"] == "started"
    assert ledger.get_admission_anchor(
        session.session_id, "started-crash-client"
    )["phase"] == "claimed"
    persisted = models.Session.load(session.session_id)
    assert persisted.active_stream_id is None
    assert persisted.pending_user_message is None

    monkeypatch.delenv("HERMES_WEBUI_CLIENT_TURN_FAILPOINT")
    retry = routes._start_chat_stream_for_session(
        persisted,
        **_start_kwargs(tmp_path, "started-crash-client"),
    )

    assert retry["_status"] == 409
    assert retry["code"] == "client_turn_recovery_required"
    assert retry["turn_id"] == claimed["turn_id"]
    assert ledger.get(session.session_id, "started-crash-client")["state"] == (
        "recovery_required"
    )
    assert config.STREAMS == {}


def test_queued_claim_crash_never_returns_phantom_queue_receipt(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, monkeypatch)
    session.active_stream_id = "busy-stream"
    session.pending_user_message = "active turn"
    session.pending_started_at = routes.time.time()
    session.save()
    config.STREAMS["busy-stream"] = queue.Queue()
    monkeypatch.setenv(
        "HERMES_WEBUI_CLIENT_TURN_FAILPOINT",
        "after_queued_claim",
    )

    with pytest.raises(routes.ClientTurnAdmissionCrash):
        routes._start_chat_stream_for_session(
            session,
            **_start_kwargs(tmp_path, "queued-crash-client"),
        )

    ledger = ClientTurnLedger(config.STATE_DIR / "client_turn_ledger.sqlite3")
    claimed = ledger.get(session.session_id, "queued-crash-client")
    assert claimed["state"] == "queued"
    assert ledger.get_admission_anchor(
        session.session_id, "queued-crash-client"
    )["phase"] == "claimed"
    persisted = models.Session.load(session.session_id)
    assert persisted.queued_user_turns == []

    monkeypatch.delenv("HERMES_WEBUI_CLIENT_TURN_FAILPOINT")
    retry = routes._start_chat_stream_for_session(
        persisted,
        **_start_kwargs(tmp_path, "queued-crash-client"),
    )

    assert retry["_status"] == 409
    assert retry["code"] == "client_turn_recovery_required"
    assert retry["turn_id"] == claimed["turn_id"]
    assert ledger.get(session.session_id, "queued-crash-client")["state"] == (
        "recovery_required"
    )
    assert models.Session.load(session.session_id).queued_user_turns == []


def test_recoverable_queue_anchor_keeps_original_receipt(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch)
    session.active_stream_id = "busy-stream"
    session.pending_user_message = "active turn"
    session.pending_started_at = routes.time.time()
    session.save()
    config.STREAMS["busy-stream"] = queue.Queue()
    kwargs = _start_kwargs(tmp_path, "durable-queue-client")

    accepted = routes._start_chat_stream_for_session(session, **kwargs)
    retry = routes._start_chat_stream_for_session(session, **kwargs)

    assert accepted == retry
    ledger = ClientTurnLedger(config.STATE_DIR / "client_turn_ledger.sqlite3")
    assert ledger.get_admission_anchor(
        session.session_id, "durable-queue-client"
    )["phase"] == "queue_persisted"


def test_cancel_after_thread_start_before_worker_entry_settles_durable_ledger(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, monkeypatch)
    captured_threads = []

    class DelayedThread:
        def __init__(self, *, target, args, kwargs, daemon):
            self.target = target
            self.args = args
            self.kwargs = kwargs
            self.daemon = daemon

        def start(self):
            captured_threads.append(self)

    monkeypatch.setattr(routes.threading, "Thread", DelayedThread)
    monkeypatch.setattr(routes, "set_last_workspace", lambda _workspace: None)
    result = routes._start_chat_stream_for_session(
        session,
        **_start_kwargs(tmp_path, "pre-entry-cancel-client"),
    )
    stream_id = result["stream_id"]
    assert len(captured_threads) == 1
    assert ClientTurnLedger(
        config.STATE_DIR / "client_turn_ledger.sqlite3"
    ).get_by_stream(stream_id)["state"] == "started"

    assert streaming.cancel_stream(stream_id) is True
    assert stream_id not in config.STREAMS
    assert ClientTurnLedger(
        config.STATE_DIR / "client_turn_ledger.sqlite3"
    ).get_by_stream(stream_id)["state"] == "recovery_required"
    delayed = captured_threads[0]
    delayed.target(*delayed.args, **delayed.kwargs)

    restarted = ClientTurnLedger(config.STATE_DIR / "client_turn_ledger.sqlite3")
    assert restarted.get_by_stream(stream_id)["state"] == "recovery_required"


def test_late_stop_after_success_writeback_cannot_downgrade_completed_ledger(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, monkeypatch)
    stream_id = "late-stop-after-success"
    session.active_stream_id = None
    session.pending_user_message = None
    session.messages = [
        {"role": "user", "content": "finish"},
        {"role": "assistant", "content": "finished"},
    ]
    session.save()
    config.STREAMS[stream_id] = queue.Queue()
    config.register_stream_owner(stream_id, session.session_id)
    ledger = ClientTurnLedger(config.STATE_DIR / "client_turn_ledger.sqlite3")
    ledger.claim(
        lineage_root_id=session.session_id,
        client_turn_id="late-stop-client",
        turn_id="late-stop-turn",
        current_session_id=session.session_id,
        stream_id=stream_id,
        request_sha256="e" * 64,
        receipt={
            "stream_id": stream_id,
            "session_id": session.session_id,
            "turn_id": "late-stop-turn",
        },
        state="started",
    )
    streaming._settle_client_turn_ledger(
        stream_id,
        "completed",
        current_session_id=session.session_id,
    )

    assert streaming.cancel_stream(stream_id) is True

    restarted = ClientTurnLedger(config.STATE_DIR / "client_turn_ledger.sqlite3")
    assert restarted.get_by_stream(stream_id)["state"] == "completed"
    assert models.Session.load(session.session_id).messages[-1]["content"] == "finished"
