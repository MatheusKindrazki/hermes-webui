"""Durable exactly-once receipts for browser-originated chat turns.

The ledger is additive and dormant until the route integration flag is enabled.
It stores only request hashes and routing identifiers; message bodies and
attachments remain in the existing session/turn journal stores.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


LEDGER_DB_NAME = "client_turn_ledger.sqlite3"
LEDGER_STATES = frozenset(
    {
        "queued",
        "started",
        "completed",
        "recovery_required",
        "failed_retryable",
    }
)

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS client_turn_ledger(
  lineage_root_id TEXT NOT NULL,
  client_turn_id TEXT NOT NULL,
  turn_id TEXT NOT NULL UNIQUE,
  current_session_id TEXT NOT NULL,
  stream_id TEXT,
  request_sha256 TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN (
    'queued','started','completed','recovery_required','failed_retryable')),
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY(lineage_root_id,client_turn_id)
);
"""

_ADMISSION_ANCHOR_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS client_turn_admission_anchor(
  lineage_root_id TEXT NOT NULL,
  client_turn_id TEXT NOT NULL,
  phase TEXT NOT NULL CHECK(phase IN (
    'claimed','queue_persisted','stream_registered','legacy_unverified','settled')),
  updated_at REAL NOT NULL,
  PRIMARY KEY(lineage_root_id,client_turn_id),
  FOREIGN KEY(lineage_root_id,client_turn_id)
    REFERENCES client_turn_ledger(lineage_root_id,client_turn_id)
);
"""

_ADMISSION_ANCHOR_PHASES = frozenset(
    {
        "claimed",
        "queue_persisted",
        "stream_registered",
        "legacy_unverified",
        "settled",
    }
)


class ClientTurnLedgerError(RuntimeError):
    """Base typed failure for ledger operations."""


class ClientTurnPayloadMismatch(ClientTurnLedgerError):
    """A retry reused a client id for a different request payload."""

    status_code = 409
    code = "client_turn_id_payload_mismatch"


def _default_db_path() -> Path:
    from api.config import STATE_DIR

    return Path(STATE_DIR) / LEDGER_DB_NAME


def _canonical_receipt_json(receipt: dict[str, Any]) -> str:
    if not isinstance(receipt, dict):
        raise TypeError("receipt must be a dict")
    return json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class ClientTurnLedger:
    """Small SQLite store with transactionally stable turn receipts."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._install_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _install_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(_SCHEMA_V1)
            conn.execute(_ADMISSION_ANCHOR_SCHEMA_V1)
            # Existing v1 rows predate the admission anchor. Never guess that
            # their queued/started work is recoverable: the route reconciler
            # validates the persisted queue or live stream before replaying the
            # original receipt.
            conn.execute(
                """
                INSERT OR IGNORE INTO client_turn_admission_anchor(
                  lineage_root_id, client_turn_id, phase, updated_at
                )
                SELECT lineage_root_id, client_turn_id, 'legacy_unverified', updated_at
                FROM client_turn_ledger
                """
            )
            conn.commit()

    def get(self, lineage_root_id: str, client_turn_id: str) -> dict[str, Any] | None:
        lineage = _required_text(lineage_root_id, "lineage_root_id")
        client_id = _required_text(client_turn_id, "client_turn_id")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM client_turn_ledger
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (lineage, client_id),
            ).fetchone()
        return _row_dict(row)

    def get_by_stream(self, stream_id: str) -> dict[str, Any] | None:
        stream = _required_text(stream_id, "stream_id")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM client_turn_ledger
                WHERE stream_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (stream,),
            ).fetchone()
        return _row_dict(row)

    def get_admission_anchor(
        self,
        lineage_root_id: str,
        client_turn_id: str,
    ) -> dict[str, Any] | None:
        """Return the durable claim-to-work phase for one browser turn."""
        lineage = _required_text(lineage_root_id, "lineage_root_id")
        client_id = _required_text(client_turn_id, "client_turn_id")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM client_turn_admission_anchor
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (lineage, client_id),
            ).fetchone()
        return _row_dict(row)

    def claim(
        self,
        *,
        lineage_root_id: str,
        client_turn_id: str,
        turn_id: str,
        current_session_id: str,
        stream_id: str | None,
        request_sha256: str,
        receipt: dict[str, Any],
        state: str = "queued",
    ) -> tuple[dict[str, Any], bool]:
        """Create a receipt once or return its durable original on retry."""
        lineage = _required_text(lineage_root_id, "lineage_root_id")
        client_id = _required_text(client_turn_id, "client_turn_id")
        server_turn_id = _required_text(turn_id, "turn_id")
        session_id = _required_text(current_session_id, "current_session_id")
        request_hash = _required_text(request_sha256, "request_sha256")
        if state not in LEDGER_STATES:
            raise ValueError(f"invalid client turn state: {state}")
        receipt_json = _canonical_receipt_json(receipt)
        now = time.time()

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT * FROM client_turn_ledger
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (lineage, client_id),
            ).fetchone()
            if existing is not None:
                if str(existing["request_sha256"]) != request_hash:
                    conn.rollback()
                    raise ClientTurnPayloadMismatch(
                        "client_turn_id was already used for a different payload"
                    )
                conn.commit()
                return dict(existing), False

            conn.execute(
                """
                INSERT INTO client_turn_ledger(
                  lineage_root_id, client_turn_id, turn_id,
                  current_session_id, stream_id, request_sha256,
                  receipt_json, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lineage,
                    client_id,
                    server_turn_id,
                    session_id,
                    str(stream_id) if stream_id else None,
                    request_hash,
                    receipt_json,
                    state,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO client_turn_admission_anchor(
                  lineage_root_id, client_turn_id, phase, updated_at
                ) VALUES (?, ?, 'claimed', ?)
                """,
                (lineage, client_id, now),
            )
            row = conn.execute(
                """
                SELECT * FROM client_turn_ledger
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (lineage, client_id),
            ).fetchone()
            conn.commit()
        return dict(row), True

    def set_admission_anchor(
        self,
        lineage_root_id: str,
        client_turn_id: str,
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Durably mark when a claim gains queue/stream recovery evidence."""
        lineage = _required_text(lineage_root_id, "lineage_root_id")
        client_id = _required_text(client_turn_id, "client_turn_id")
        if phase not in _ADMISSION_ANCHOR_PHASES:
            raise ValueError(f"invalid client turn admission phase: {phase}")
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            exists = conn.execute(
                """
                SELECT 1 FROM client_turn_ledger
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (lineage, client_id),
            ).fetchone()
            if exists is None:
                conn.rollback()
                raise KeyError((lineage, client_id))
            conn.execute(
                """
                INSERT INTO client_turn_admission_anchor(
                  lineage_root_id, client_turn_id, phase, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(lineage_root_id,client_turn_id) DO UPDATE SET
                  phase = excluded.phase,
                  updated_at = excluded.updated_at
                """,
                (lineage, client_id, phase, now),
            )
            row = conn.execute(
                """
                SELECT * FROM client_turn_admission_anchor
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (lineage, client_id),
            ).fetchone()
            conn.commit()
        return dict(row)

    def transition(
        self,
        lineage_root_id: str,
        client_turn_id: str,
        *,
        state: str,
        stream_id: str | None = None,
        current_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Advance a claimed turn without ever replacing its receipt."""
        lineage = _required_text(lineage_root_id, "lineage_root_id")
        client_id = _required_text(client_turn_id, "client_turn_id")
        if state not in LEDGER_STATES:
            raise ValueError(f"invalid client turn state: {state}")
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT * FROM client_turn_ledger
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (lineage, client_id),
            ).fetchone()
            if existing is None:
                conn.rollback()
                raise KeyError((lineage, client_id))
            next_stream = str(stream_id) if stream_id is not None else existing["stream_id"]
            next_session = (
                _required_text(current_session_id, "current_session_id")
                if current_session_id is not None
                else existing["current_session_id"]
            )
            conn.execute(
                """
                UPDATE client_turn_ledger
                SET state = ?, stream_id = ?, current_session_id = ?, updated_at = ?
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (state, next_stream, next_session, now, lineage, client_id),
            )
            if state in {"completed", "recovery_required", "failed_retryable"}:
                conn.execute(
                    """
                    UPDATE client_turn_admission_anchor
                    SET phase = 'settled', updated_at = ?
                    WHERE lineage_root_id = ? AND client_turn_id = ?
                    """,
                    (now, lineage, client_id),
                )
            row = conn.execute(
                """
                SELECT * FROM client_turn_ledger
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (lineage, client_id),
            ).fetchone()
            conn.commit()
        return dict(row)

    def transition_by_stream(
        self,
        stream_id: str,
        *,
        state: str,
        current_session_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Settle the one durable turn currently owned by ``stream_id``."""
        stream = _required_text(stream_id, "stream_id")
        if state not in LEDGER_STATES:
            raise ValueError(f"invalid client turn state: {state}")
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT * FROM client_turn_ledger
                WHERE stream_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (stream,),
            ).fetchone()
            if existing is None:
                conn.commit()
                return None
            next_session = (
                _required_text(current_session_id, "current_session_id")
                if current_session_id is not None
                else existing["current_session_id"]
            )
            conn.execute(
                """
                UPDATE client_turn_ledger
                SET state = ?, current_session_id = ?, updated_at = ?
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (
                    state,
                    next_session,
                    now,
                    existing["lineage_root_id"],
                    existing["client_turn_id"],
                ),
            )
            if state in {"completed", "recovery_required", "failed_retryable"}:
                conn.execute(
                    """
                    UPDATE client_turn_admission_anchor
                    SET phase = 'settled', updated_at = ?
                    WHERE lineage_root_id = ? AND client_turn_id = ?
                    """,
                    (
                        now,
                        existing["lineage_root_id"],
                        existing["client_turn_id"],
                    ),
                )
            row = conn.execute(
                """
                SELECT * FROM client_turn_ledger
                WHERE lineage_root_id = ? AND client_turn_id = ?
                """,
                (existing["lineage_root_id"], existing["client_turn_id"]),
            ).fetchone()
            conn.commit()
        return dict(row)

    def rotate_live_session(self, old_session_id: str, new_session_id: str) -> int:
        """Move live intents to one continuation tip without copying rows."""
        old_sid = _required_text(old_session_id, "old_session_id")
        new_sid = _required_text(new_session_id, "new_session_id")
        if old_sid == new_sid:
            return 0
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE client_turn_ledger
                SET current_session_id = ?, updated_at = ?
                WHERE current_session_id = ?
                  AND state IN ('queued','started')
                """,
                (new_sid, now, old_sid),
            )
            moved = int(cursor.rowcount or 0)
            conn.commit()
        return moved


_DEFAULT_LEDGER_CACHE: dict[Path, ClientTurnLedger] = {}
_DEFAULT_LEDGER_CACHE_LOCK = threading.Lock()


def default_client_turn_ledger() -> ClientTurnLedger:
    """Return one store per resolved WebUI state directory."""
    path = _default_db_path().expanduser().resolve()
    with _DEFAULT_LEDGER_CACHE_LOCK:
        ledger = _DEFAULT_LEDGER_CACHE.get(path)
        if ledger is None:
            ledger = ClientTurnLedger(path)
            _DEFAULT_LEDGER_CACHE[path] = ledger
        return ledger
