"""Durability contract for browser-provided ``client_turn_id`` values."""

from __future__ import annotations

import json
import sqlite3

import pytest

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
    assert "PRIMARY KEY(lineage_root_id,client_turn_id)" in table_sql.replace(" ", "")
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

