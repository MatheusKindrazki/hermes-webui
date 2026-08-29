"""Behavioral coverage for non-legacy client-turn admission backends."""

from __future__ import annotations

from collections import OrderedDict
import queue

import pytest

from api import config, gateway_chat, models, routes, streaming
from api.client_turn_ledger import ClientTurnLedger
from api.runtime_adapter import RunStartResult


def _gateway_turn(tmp_path, monkeypatch, *, suffix: str):
    state_dir = tmp_path / "state"
    session_dir = state_dir / "sessions"
    session_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(models, "SESSIONS", OrderedDict())
    monkeypatch.setenv("HERMES_TURN_LEDGER_ENABLED", "1")
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", "http://gateway.local")
    monkeypatch.setattr(gateway_chat, "gateway_supports_approval", lambda *_args: False)
    monkeypatch.setattr(gateway_chat, "gateway_approval_unavailable_reason", lambda *_args: None)
    monkeypatch.setattr(
        streaming,
        "_load_webui_prefill_context",
        lambda _cfg: {
            "status": "not_configured",
            "source": "none",
            "label": "",
            "message_count": 0,
            "messages": [],
        },
    )
    monkeypatch.setattr(
        streaming,
        "_prefill_messages_with_webui_context",
        lambda _ctx, _cfg: [],
    )
    config.STREAMS.clear()
    config.CANCEL_FLAGS.clear()
    config.ACTIVE_RUNS.clear()

    session = models.new_session()
    stream_id = f"gateway-ledger-{suffix}"
    session.active_stream_id = stream_id
    session.pending_user_message = "do it once"
    session.pending_attachments = []
    session.pending_started_at = 123.0
    session.save()
    config.STREAMS[stream_id] = config.create_stream_channel()

    ledger = ClientTurnLedger(state_dir / "client_turn_ledger.sqlite3")
    ledger.claim(
        lineage_root_id=session.session_id,
        client_turn_id=f"browser-{suffix}",
        turn_id=f"turn-{suffix}",
        current_session_id=session.session_id,
        stream_id=stream_id,
        request_sha256="a" * 64,
        receipt={
            "stream_id": stream_id,
            "session_id": session.session_id,
            "turn_id": f"turn-{suffix}",
        },
        state="started",
    )
    return session, stream_id, ledger


class _SuccessResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
        yield b"data: [DONE]\n\n"


@pytest.mark.parametrize(
    ("outcome", "expected_state"),
    [
        ("success", "completed"),
        ("error", "failed_retryable"),
        ("cancel", "recovery_required"),
    ],
)
def test_gateway_worker_settles_ledger_before_stream_cleanup(
    tmp_path, monkeypatch, outcome, expected_state
):
    session, stream_id, ledger = _gateway_turn(tmp_path, monkeypatch, suffix=outcome)
    settle_observations = []
    real_settle = streaming._settle_client_turn_ledger

    def observed_settle(settled_stream_id, state, **kwargs):
        settle_observations.append(
            (settled_stream_id in config.STREAMS, state, kwargs.get("current_session_id"))
        )
        return real_settle(settled_stream_id, state, **kwargs)

    monkeypatch.setattr(streaming, "_settle_client_turn_ledger", observed_settle)

    if outcome == "success":
        monkeypatch.setattr(
            gateway_chat.urllib.request,
            "urlopen",
            lambda _req, timeout=0: _SuccessResponse(),
        )
    elif outcome == "error":
        def fail_request(_req, timeout=0):
            raise RuntimeError("gateway unavailable")

        monkeypatch.setattr(gateway_chat.urllib.request, "urlopen", fail_request)
    else:
        class CancelResponse(_SuccessResponse):
            def __iter__(self):
                config.CANCEL_FLAGS[stream_id].set()
                yield b"\n"

        monkeypatch.setattr(
            gateway_chat.urllib.request,
            "urlopen",
            lambda _req, timeout=0: CancelResponse(),
        )

    gateway_chat._run_gateway_chat_streaming(
        session.session_id,
        "do it once",
        "test-model",
        str(tmp_path),
        stream_id,
        [],
    )

    assert settle_observations == [(True, expected_state, session.session_id)]
    assert stream_id not in config.STREAMS
    assert ledger.get_by_stream(stream_id)["state"] == expected_state


def test_gateway_prestart_cancel_settles_started_receipt(tmp_path, monkeypatch):
    session, stream_id, ledger = _gateway_turn(
        tmp_path, monkeypatch, suffix="prestart-cancel"
    )
    config.STREAMS.pop(stream_id)

    gateway_chat._run_gateway_chat_streaming(
        session.session_id,
        "do it once",
        "test-model",
        str(tmp_path),
        stream_id,
        [],
    )

    assert ledger.get_by_stream(stream_id)["state"] == "recovery_required"


def test_runner_local_admission_uses_stable_backend_idempotency_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WEBUI_RUNTIME_ADAPTER", "runner-local")
    captured = []
    runs_by_key = {}

    class FakeAdapter:
        def start_run(self, request):
            captured.append(request)
            key = request.metadata["idempotency_key"]
            run_id = runs_by_key.setdefault(key, f"run-{len(runs_by_key) + 1}")
            return RunStartResult(
                run_id=run_id,
                session_id=request.session_id,
                stream_id=run_id,
                payload={"stream_id": run_id, "session_id": request.session_id},
            )

    monkeypatch.setattr(routes, "build_runtime_adapter", None, raising=False)
    import api.runtime_adapter as runtime_adapter

    monkeypatch.setattr(runtime_adapter, "build_runtime_adapter", lambda **_kwargs: FakeAdapter())
    session = models.Session(
        session_id="runner-ledger-session",
        workspace=str(tmp_path),
        messages=[],
        profile="work",
    )
    kwargs = {
        "msg": "run once remotely",
        "attachments": [{"name": "proof.txt"}],
        "workspace": str(tmp_path),
        "model": "test-model",
        "model_provider": "test-provider",
        "normalized_model": False,
        "source": "webui",
        "route": "/api/chat/start",
        "client_turn_id": "browser-runner-1",
    }

    first = routes._start_run(session, **kwargs)
    retry = routes._start_run(session, **kwargs)

    assert first == retry
    assert len(captured) == 2
    assert captured[0].metadata["idempotency_key"] == captured[1].metadata["idempotency_key"]
    assert captured[0].metadata["request_sha256"] == captured[1].metadata["request_sha256"]

