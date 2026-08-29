"""Exactly-once ledger ownership across automatic-compression SID rotation."""

from __future__ import annotations

import inspect
import sqlite3

from api import config, streaming
from api.client_turn_ledger import ClientTurnLedger


def _claim(ledger, *, client_turn_id, turn_id, state):
    record, _created = ledger.claim(
        lineage_root_id="session-parent",
        client_turn_id=client_turn_id,
        turn_id=turn_id,
        current_session_id="session-parent",
        stream_id=f"stream-{turn_id}",
        request_sha256=(client_turn_id[0] * 64),
        receipt={
            "turn_id": turn_id,
            "session_id": "session-parent",
        },
        state=state,
    )
    return record


def test_live_intents_move_once_to_continuation_without_parent_copy(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setenv("HERMES_TURN_LEDGER_ENABLED", "1")
    ledger = ClientTurnLedger(state_dir / "client_turn_ledger.sqlite3")
    _claim(ledger, client_turn_id="queued-client", turn_id="queued-turn", state="queued")
    _claim(ledger, client_turn_id="started-client", turn_id="started-turn", state="started")
    _claim(ledger, client_turn_id="complete-client", turn_id="complete-turn", state="completed")

    moved = streaming._rotate_client_turn_ledger_session(
        "session-parent", "session-tip"
    )
    moved_again = streaming._rotate_client_turn_ledger_session(
        "session-parent", "session-tip"
    )

    assert moved == 2
    assert moved_again == 0
    restarted = ClientTurnLedger(state_dir / "client_turn_ledger.sqlite3")
    assert restarted.get("session-parent", "queued-client")["current_session_id"] == "session-tip"
    assert restarted.get("session-parent", "started-client")["current_session_id"] == "session-tip"
    assert restarted.get("session-parent", "complete-client")["current_session_id"] == "session-parent"
    with sqlite3.connect(restarted.db_path) as conn:
        replayable_parent_rows = conn.execute(
            """
            SELECT COUNT(*) FROM client_turn_ledger
            WHERE current_session_id = 'session-parent'
              AND state IN ('queued','started')
            """
        ).fetchone()[0]
    assert replayable_parent_rows == 0


def test_compression_rotation_updates_ledger_before_linking_tip():
    source = inspect.getsource(streaming._run_agent_streaming)
    session_rotation = source.index("s.session_id = new_sid")
    ledger_rotation = source.index("_rotate_client_turn_ledger_session(old_sid, new_sid)")
    parent_link = source.index("s.parent_session_id = old_sid")

    assert session_rotation < ledger_rotation < parent_link
