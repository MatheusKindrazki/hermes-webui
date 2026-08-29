"""Regression coverage for user turns submitted while compression owns a session."""

import json
import queue

from api import config, models, routes, streaming
from api.models import Session


def _install_session(tmp_path, monkeypatch, *, sid="compression-parent"):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(routes, "SESSION_DIR", session_dir)
    monkeypatch.setattr(streaming, "SESSION_DIR", session_dir)
    models.SESSIONS.clear()
    routes.SESSIONS.clear()
    streaming.SESSIONS.clear()
    config.STREAMS.clear()
    config.ACTIVE_RUNS.clear()
    session = Session(session_id=sid, workspace=str(tmp_path), messages=[])
    session.active_stream_id = "compression-stream"
    session.pending_user_message = "compress"
    session.pending_started_at = 10.0
    session.save()
    models.SESSIONS[sid] = session
    routes.SESSIONS[sid] = session
    streaming.SESSIONS[sid] = session
    config.STREAMS[session.active_stream_id] = queue.Queue()
    return session, session_dir


def test_concurrent_chat_start_queues_second_turn_durably_once_with_attachments(
    tmp_path, monkeypatch
):
    session, session_dir = _install_session(tmp_path, monkeypatch)

    response = routes._start_chat_stream_for_session(
        session,
        msg="second turn",
        attachments=[{"name": "evidence.txt", "path": "/safe/evidence.txt"}],
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
        client_turn_id="desktop-turn-2",
    )

    assert response["_status"] == 202
    assert response["status"] == "queued"
    assert response["session_id"] == "compression-parent"
    assert response["active_stream_id"] == "compression-stream"
    persisted = json.loads(
        (session_dir / "compression-parent.json").read_text(encoding="utf-8")
    )
    assert len(persisted["queued_user_turns"]) == 1
    queued = persisted["queued_user_turns"][0]
    assert queued["message"] == "second turn"
    assert queued["attachments"] == [
        {"name": "evidence.txt", "path": "/safe/evidence.txt"}
    ]
    assert queued["source"] == "webui"
    assert queued["turn_id"] == response["turn_id"]
    assert queued["client_turn_id"] == "desktop-turn-2"
    assert session.pending_user_message == "compress"


def test_duplicate_client_turn_id_returns_same_receipt_and_drains_once(
    tmp_path, monkeypatch
):
    session, _session_dir = _install_session(tmp_path, monkeypatch)
    body = {
        "session_id": session.session_id,
        "message": "second turn",
        "attachments": [
            {
                "name": "same.bin",
                "path": "/safe/same.bin",
                "mime": "application/octet-stream",
                "size": 17,
            }
        ],
        "workspace": str(tmp_path),
        "model": "test-model",
        "model_provider": "test-provider",
        "client_turn_id": "desktop-retry-42",
    }

    monkeypatch.setattr(routes, "_get_or_materialize_session", lambda *_args, **_kwargs: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda *_args, **_kwargs: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda *_args, **_kwargs: (None, None, None))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("test-model", "test-provider", False),
    )
    monkeypatch.setattr(
        routes,
        "_start_run",
        lambda s, **kwargs: routes._start_chat_stream_for_session(
            s,
            msg=kwargs["msg"],
            attachments=kwargs["attachments"],
            workspace=kwargs["workspace"],
            model=kwargs["model"],
            model_provider=kwargs["model_provider"],
            source=kwargs["source"],
            client_turn_id=kwargs.get("client_turn_id"),
        ),
    )
    monkeypatch.setattr(
        routes,
        "j",
        lambda _handler, payload, status=200: {"http_status": status, **payload},
    )

    first = routes._handle_chat_start(None, dict(body))
    second = routes._handle_chat_start(None, dict(body))

    assert first == second
    assert first["http_status"] == 202
    assert len(session.queued_user_turns) == 1
    assert session.queued_user_turns[0]["attachments"] == body["attachments"]

    session.active_stream_id = None
    session.pending_user_message = None
    session.pending_started_at = None
    config.STREAMS.clear()
    starts = []

    def _accept(s, **start_kwargs):
        starts.append((s.session_id, start_kwargs))
        return {"stream_id": "drained-once", "session_id": s.session_id}

    monkeypatch.setattr(routes, "_start_chat_stream_for_session", _accept)
    assert routes.drain_queued_user_turns_for_session(session.session_id) is True
    assert routes.drain_queued_user_turns_for_session(session.session_id) is False
    assert len(starts) == 1
    assert starts[0][1]["attachments"] == body["attachments"]


def test_fast_active_check_revalidates_under_lock_and_starts_if_turn_became_idle(
    tmp_path, monkeypatch
):
    session, _session_dir = _install_session(tmp_path, monkeypatch)
    entered = False
    real_lock = routes._get_session_agent_lock(session.session_id)

    class FinishBeforeLock:
        def __enter__(self):
            nonlocal entered
            real_lock.acquire()
            if not entered:
                entered = True
                config.STREAMS.pop("compression-stream", None)
                session.active_stream_id = None
                session.pending_user_message = None
                session.pending_started_at = None
            return self

        def __exit__(self, exc_type, exc, tb):
            real_lock.release()
            return False

    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(routes, "_get_session_agent_lock", lambda _sid: FinishBeforeLock())
    monkeypatch.setattr(routes.threading, "Thread", NoopThread)

    response = routes._start_chat_stream_for_session(
        session,
        msg="start now",
        attachments=[{"name": "now.txt"}],
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
        client_turn_id="became-idle",
    )

    assert response.get("status") != "queued"
    assert response["stream_id"] == session.active_stream_id
    assert session.pending_user_message == "start now"
    assert session.queued_user_turns == []


def test_queued_turn_limit_is_explicit_and_does_not_drop_existing_events(
    tmp_path, monkeypatch
):
    session, _session_dir = _install_session(tmp_path, monkeypatch)
    session.queued_user_turns = [
        {"turn_id": f"queued-{idx}", "client_turn_id": f"client-{idx}"}
        for idx in range(routes.MAX_QUEUED_USER_TURNS)
    ]
    session.save()

    response = routes._start_chat_stream_for_session(
        session,
        msg="one too many",
        attachments=[{"name": "must-not-drop.txt"}],
        workspace=str(tmp_path),
        model="test-model",
        client_turn_id="overflow-client",
    )

    assert response["_status"] == 429
    assert response["code"] == "queued_turn_limit"
    assert len(session.queued_user_turns) == routes.MAX_QUEUED_USER_TURNS


def test_runtime_adapter_response_preserves_queued_receipt_shape():
    class Result:
        stream_id = "compression-stream"
        session_id = "compression-parent"
        payload = {
            "status": "queued",
            "turn_id": "queued-adapter",
            "active_stream_id": "compression-stream",
            "_status": 202,
        }

    response = routes._chat_start_response_from_run_start(Result())

    assert response["status"] == "queued"
    assert response["turn_id"] == "queued-adapter"
    assert response["_status"] == 202


def test_concurrent_chat_start_queues_when_worker_lifecycle_outlives_stream_field(
    tmp_path, monkeypatch
):
    session, _session_dir = _install_session(tmp_path, monkeypatch)
    session.active_stream_id = None
    session.pending_user_message = None
    session.pending_started_at = None
    session.save()
    config.STREAMS.clear()
    config.ACTIVE_RUNS["compression-worker"] = {
        "stream_id": "compression-worker",
        "session_id": session.session_id,
        "started_at": routes.time.time(),
        "phase": "running",
    }

    response = routes._start_chat_stream_for_session(
        session,
        msg="second turn",
        attachments=[],
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
    )

    assert response["_status"] == 202
    assert response["active_stream_id"] == "compression-worker"
    assert [row["message"] for row in session.queued_user_turns] == ["second turn"]


def test_teardown_drain_uses_rotated_sid_and_removes_event_only_after_acceptance(
    tmp_path, monkeypatch
):
    session, session_dir = _install_session(tmp_path, monkeypatch)
    queued = routes._enqueue_queued_user_turn(
        session,
        message="second turn",
        attachments=[{"name": "checkpoint.json"}],
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
        source="webui",
    )
    old_sid = session.session_id
    new_sid = "compression-continuation"
    session.session_id = new_sid
    session.parent_session_id = old_sid
    session.active_stream_id = None
    session.pending_user_message = None
    session.pending_started_at = None
    session.save()
    routes.SESSIONS.pop(old_sid, None)
    routes.SESSIONS[new_sid] = session

    starts = []

    def _accept(s, **kwargs):
        starts.append((s.session_id, kwargs))
        return {"stream_id": "second-stream", "session_id": s.session_id}

    monkeypatch.setattr(routes, "_start_chat_stream_for_session", _accept)

    assert routes.drain_queued_user_turns_for_session(new_sid) is True
    assert routes.drain_queued_user_turns_for_session(new_sid) is False
    assert len(starts) == 1
    sid, kwargs = starts[0]
    assert sid == new_sid
    assert kwargs["msg"] == "second turn"
    assert kwargs["attachments"] == [{"name": "checkpoint.json"}]
    assert kwargs["queued_turn_id"] == queued["turn_id"]
    persisted = json.loads((session_dir / f"{new_sid}.json").read_text(encoding="utf-8"))
    assert persisted["queued_user_turns"] == []


def test_compression_exhausted_handoff_keeps_one_queued_event_without_redrain_loop(
    tmp_path, monkeypatch
):
    session, session_dir = _install_session(tmp_path, monkeypatch)
    queued = routes._enqueue_queued_user_turn(
        session,
        message="recover this after the limit",
        attachments=[{"name": "handoff.md"}],
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
        source="webui",
    )
    recovery = streaming.stamp_compression_exhausted_recovery(
        session,
        message="Cannot compress further.",
    )
    session.active_stream_id = None
    session.pending_user_message = None
    session.save()

    starts = []
    monkeypatch.setattr(
        routes,
        "_start_chat_stream_for_session",
        lambda *_args, **_kwargs: starts.append(_kwargs) or {},
    )

    assert routes.drain_queued_user_turns_for_session(session.session_id) is False
    assert starts == []
    persisted = json.loads(
        (session_dir / f"{session.session_id}.json").read_text(encoding="utf-8")
    )
    assert [item["turn_id"] for item in persisted["queued_user_turns"]] == [
        queued["turn_id"]
    ]
    assert persisted["compression_recovery"] == recovery
    assert persisted["queued_user_turns"][0]["recovery_required"] is True


def test_compression_snapshot_does_not_duplicate_queued_event_across_lineage(
    tmp_path, monkeypatch
):
    session, session_dir = _install_session(tmp_path, monkeypatch)
    queued = routes._enqueue_queued_user_turn(
        session,
        message="second turn",
        attachments=[{"name": "lineage.txt"}],
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
        source="webui",
    )
    old_sid = session.session_id
    new_sid = "compression-tip"
    session.session_id = new_sid

    streaming._preserve_pre_compression_snapshot(session, old_sid)
    session.pre_compression_snapshot = False
    session.parent_session_id = old_sid
    session.save()

    old_payload = json.loads((session_dir / f"{old_sid}.json").read_text(encoding="utf-8"))
    new_payload = json.loads((session_dir / f"{new_sid}.json").read_text(encoding="utf-8"))
    assert old_payload["queued_user_turns"] == []
    assert [row["turn_id"] for row in new_payload["queued_user_turns"]] == [queued["turn_id"]]
