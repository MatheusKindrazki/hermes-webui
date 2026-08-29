"""Durability contract for browser-provided ``client_turn_id`` values."""

from __future__ import annotations

import json
import inspect
import queue
import sqlite3

import pytest

from api import config, models, routes, streaming
from api.client_turn_ledger import (
    ClientTurnLedger,
    ClientTurnPayloadMismatch,
)


def _claim(
    ledger: ClientTurnLedger,
    *,
    request_sha256: str = "a" * 64,
    receipt: dict | None = None,
):
    return ledger.claim(
        lineage_root_id="lineage-root",
        client_turn_id="browser-turn-1",
        turn_id="server-turn-1",
        current_session_id="session-parent",
        stream_id=None,
        request_sha256=request_sha256,
        receipt=receipt or {
            "status": "queued",
            "turn_id": "server-turn-1",
            "session_id": "session-parent",
        },
    )


def test_client_turn_ledger_installs_v1_schema(tmp_path):
    db_path = tmp_path / "client-turn-ledger.sqlite3"
    ClientTurnLedger(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(client_turn_ledger)")
        }
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='client_turn_ledger'"
        ).fetchone()[0]

    assert columns == {
        "lineage_root_id": "TEXT",
        "client_turn_id": "TEXT",
        "turn_id": "TEXT",
        "current_session_id": "TEXT",
        "stream_id": "TEXT",
        "request_sha256": "TEXT",
        "receipt_json": "TEXT",
        "state": "TEXT",
        "created_at": "REAL",
        "updated_at": "REAL",
    }
    assert "PRIMARYKEY(lineage_root_id,client_turn_id)" in table_sql.replace(" ", "")
    assert "turn_id TEXT NOT NULL UNIQUE" in " ".join(table_sql.split())
    for state in (
        "queued",
        "started",
        "completed",
        "recovery_required",
        "failed_retryable",
    ):
        assert f"'{state}'" in table_sql


@pytest.mark.parametrize("state", ["queued", "started", "completed"])
def test_retry_returns_original_receipt_for_durable_states(tmp_path, state):
    db_path = tmp_path / "client-turn-ledger.sqlite3"
    first_store = ClientTurnLedger(db_path)
    original_receipt = {
        "status": "queued",
        "turn_id": "server-turn-1",
        "session_id": "session-parent",
    }
    record, created = _claim(first_store, receipt=original_receipt)
    assert created is True

    if state != "queued":
        record = first_store.transition(
            "lineage-root",
            "browser-turn-1",
            state=state,
        )
    assert record["state"] == state

    restarted_store = ClientTurnLedger(db_path)
    retry_record, retry_created = _claim(
        restarted_store,
        receipt={"status": "new-attempt-must-not-win"},
    )

    assert retry_created is False
    assert retry_record["state"] == state
    assert json.loads(retry_record["receipt_json"]) == original_receipt
    assert retry_record["turn_id"] == "server-turn-1"


def test_same_client_turn_id_with_different_payload_is_conflict(tmp_path):
    ledger = ClientTurnLedger(tmp_path / "client-turn-ledger.sqlite3")
    _claim(ledger)

    with pytest.raises(ClientTurnPayloadMismatch) as exc_info:
        _claim(ledger, request_sha256="b" * 64)

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "client_turn_id_payload_mismatch"


def _install_route_session(tmp_path, monkeypatch):
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
        session_id="ledger-session",
        workspace=str(tmp_path),
        messages=[],
        profile="work",
    )
    session.save()
    models.SESSIONS[session.session_id] = session
    routes.SESSIONS[session.session_id] = session
    streaming.SESSIONS[session.session_id] = session
    return session


def test_start_retry_returns_original_receipt_without_second_worker(tmp_path, monkeypatch):
    session = _install_route_session(tmp_path, monkeypatch)
    started_threads = []

    class NoopThread:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def start(self):
            started_threads.append((self.args, self.kwargs))

    monkeypatch.setattr(routes.threading, "Thread", NoopThread)
    monkeypatch.setattr(routes, "create_stream_channel", queue.Queue)
    monkeypatch.setattr(routes, "set_last_workspace", lambda _workspace: None)

    kwargs = {
        "msg": "run this once",
        "attachments": [{"name": "evidence.txt", "size": 12}],
        "workspace": str(tmp_path),
        "model": "test-model",
        "model_provider": "test-provider",
        "client_turn_id": "browser-retry-1",
    }
    first = routes._start_chat_stream_for_session(session, **kwargs)
    retry = routes._start_chat_stream_for_session(session, **kwargs)

    assert retry == first
    assert retry["stream_id"] == first["stream_id"]
    assert retry["turn_id"] == first["turn_id"]
    assert len(started_threads) == 1
    assert session.queued_user_turns == []

    mismatch = routes._start_chat_stream_for_session(
        session,
        **{**kwargs, "msg": "different payload"},
    )
    assert mismatch["_status"] == 409
    assert mismatch["code"] == "client_turn_id_payload_mismatch"


@pytest.mark.parametrize(
    "terminal_state",
    ["completed", "failed_retryable", "recovery_required"],
)
def test_stream_terminal_state_is_durable_before_cleanup(
    tmp_path, monkeypatch, terminal_state
):
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setenv("HERMES_TURN_LEDGER_ENABLED", "1")
    ledger = ClientTurnLedger(state_dir / "client_turn_ledger.sqlite3")
    receipt = {
        "stream_id": "stream-terminal",
        "session_id": "session-parent",
        "turn_id": "server-terminal",
    }
    ledger.claim(
        lineage_root_id="session-parent",
        client_turn_id="browser-terminal",
        turn_id="server-terminal",
        current_session_id="session-parent",
        stream_id="stream-terminal",
        request_sha256="c" * 64,
        receipt=receipt,
        state="started",
    )

    settled = streaming._settle_client_turn_ledger(
        "stream-terminal",
        terminal_state,
        current_session_id="session-tip",
    )

    assert settled["state"] == terminal_state
    assert settled["current_session_id"] == "session-tip"
    restarted = ClientTurnLedger(state_dir / "client_turn_ledger.sqlite3")
    retry, created = restarted.claim(
        lineage_root_id="session-parent",
        client_turn_id="browser-terminal",
        turn_id="second-worker-must-not-start",
        current_session_id="session-tip",
        stream_id="second-stream",
        request_sha256="c" * 64,
        receipt={"status": "new-attempt"},
        state="started",
    )
    assert created is False
    assert json.loads(retry["receipt_json"]) == receipt


def test_streaming_settles_ledger_before_stream_registry_cleanup():
    source = inspect.getsource(streaming._run_agent_streaming)
    settle_position = source.index("_settle_client_turn_ledger(")
    cleanup_position = source.index("STREAMS.pop(stream_id, None)")

    assert settle_position < cleanup_position
