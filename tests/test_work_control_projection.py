"""Contract tests for the read-only Work Control projection adapter."""
import io
import json
from urllib.parse import urlparse

from api import routes
from api.work_control_projection import ProjectionUnavailable, WorkControlProjection


def _payload(version=1):
    """Exact K12-GATE wire shape emitted by the Jarvis read authority."""
    return {
        "schema_version": "hermes-kernel-projection.v1",
        "authority": "remote",
        "generated_at": "2026-09-02T18:00:00Z",
        "ttl_seconds": 30,
        "records": [{
            "work_id": "w1", "front": "Hermes", "profile": None,
            "status": "running", "authority_version": version,
            "updated_at": "2026-09-02T18:00:00Z",
        }],
    }


def test_projection_retains_last_snapshot_on_outage_and_never_empty_success():
    responses = [_payload(), OSError("down")]
    adapter = WorkControlProjection(fetcher=lambda: responses.pop(0))
    assert adapter.get(force=True)["stale"] is False
    stale = adapter.get(force=True)
    assert stale["stale"] is True
    assert stale["records"] == _payload()["records"]


def test_projection_refuses_lower_authority_version():
    responses = [_payload(7), _payload(6)]
    adapter = WorkControlProjection(fetcher=lambda: responses.pop(0))
    assert adapter.get(force=True)["records"][0]["authority_version"] == 7
    stale = adapter.get(force=True)
    assert stale["stale"] is True
    assert stale["records"][0]["authority_version"] == 7


def test_remote_request_uses_work_read_api_key_contract(monkeypatch):
    observed = {}
    class _Response:
        status = 200
        def read(self): return json.dumps(_payload()).encode()
        def __enter__(self): return self
        def __exit__(self, *_): return None
    monkeypatch.setenv("HERMES_WORK_CONTROL_PROJECTION_URL", "http://jarvis.local/work-control/projection")
    monkeypatch.setenv("HERMES_WORK_CONTROL_READ_TOKEN", "work-read-token")
    monkeypatch.setattr("api.work_control_projection.urlopen", lambda request, timeout: observed.update(headers=dict(request.header_items()), timeout=timeout) or _Response())
    assert WorkControlProjection().get(force=True)["records"]
    assert observed["headers"]["X-api-key"] == "work-read-token"
    assert observed["timeout"] == 2.0


def test_projection_fails_closed_before_first_success():
    adapter = WorkControlProjection(fetcher=lambda: (_ for _ in ()).throw(OSError("down")))
    try:
        adapter.get(force=True)
    except ProjectionUnavailable:
        return
    raise AssertionError("first outage must not become an empty success")


class _Handler:
    headers = {}
    def __init__(self):
        self.status = None
        self.wfile = io.BytesIO()
    def send_response(self, status): self.status = status
    def send_header(self, *_): pass
    def end_headers(self): pass


def test_route_returns_503_before_first_authoritative_snapshot(monkeypatch):
    monkeypatch.setattr(routes, "_guard_request_session_visibility", lambda *_a, **_k: True)
    monkeypatch.setattr("api.work_control_projection.projection.get", lambda **_k: (_ for _ in ()).throw(ProjectionUnavailable()))
    handler = _Handler()
    routes.handle_get(handler, urlparse("/api/work-control/projection"))
    assert handler.status == 503
    assert json.loads(handler.wfile.getvalue())["stale"] is True
