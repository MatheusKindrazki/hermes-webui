"""Contract tests for the read-only Work Control projection adapter."""
import io
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from urllib.request import Request

from api import routes
from api import work_control_projection as projection_module
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
    now = [100.0]
    adapter = WorkControlProjection(fetcher=lambda: responses.pop(0), now=lambda: now[0])
    assert adapter.get(force=True)["stale"] is False
    now[0] += 6
    stale = adapter.get(force=True)
    assert stale["stale"] is True
    assert stale["records"] == _payload()["records"]


def test_projection_refuses_lower_authority_version():
    responses = [_payload(7), _payload(6)]
    now = [100.0]
    adapter = WorkControlProjection(fetcher=lambda: responses.pop(0), now=lambda: now[0])
    assert adapter.get(force=True)["records"][0]["authority_version"] == 7
    now[0] += 6
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
    class _Opener:
        def open(self, request, timeout):
            observed.update(headers=dict(request.header_items()), timeout=timeout)
            return _Response()
    monkeypatch.setenv("HERMES_WORK_CONTROL_PROJECTION_URL", "https://jarvis.local/work-control/projection")
    monkeypatch.setenv("HERMES_WORK_CONTROL_READ_TOKEN", "work-read-token")
    monkeypatch.setattr("api.work_control_projection.build_opener", lambda *_handlers: _Opener())
    assert WorkControlProjection().get(force=True)["records"]
    assert observed["headers"]["X-api-key"] == "work-read-token"
    assert observed["timeout"] == 2.0


def test_remote_request_rejects_http_before_authenticated_request(monkeypatch):
    monkeypatch.setenv("HERMES_WORK_CONTROL_PROJECTION_URL", "http://jarvis.local/work-control/projection")
    monkeypatch.setenv("HERMES_WORK_CONTROL_READ_TOKEN", "must-not-leave")
    monkeypatch.setattr(
        projection_module,
        "Request",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("request must not be built")),
    )
    try:
        WorkControlProjection()._fetch_remote()
    except ProjectionUnavailable as exc:
        assert str(exc) == "work-control authority must use https"
    else:
        raise AssertionError("http authority must fail closed")


def test_redirect_handler_refuses_before_forwarding_api_key():
    original = Request("https://jarvis.local/projection", headers={"X-API-Key": "must-not-forward"})
    try:
        projection_module._RejectRedirects().redirect_request(
            original, None, 302, "Found", {}, "https://attacker.invalid/collect"
        )
    except ProjectionUnavailable as exc:
        assert str(exc) == "work-control authority redirects are forbidden"
    else:
        raise AssertionError("redirect must fail before a new request is created")


def test_forced_refresh_is_throttled_within_minimum_interval():
    calls = 0
    now = [100.0]
    def fetcher():
        nonlocal calls
        calls += 1
        return _payload(calls)
    adapter = WorkControlProjection(fetcher=fetcher, now=lambda: now[0])
    assert adapter.get(force=True)["source"] == "remote"
    throttled = adapter.get(force=True)
    assert throttled["source"] == "refresh_throttled"
    assert throttled["stale"] is True
    assert calls == 1


def test_failed_initial_force_refresh_is_throttled_without_second_fetch():
    calls = 0
    now = [100.0]

    def fetcher():
        nonlocal calls
        calls += 1
        raise OSError("down")

    adapter = WorkControlProjection(fetcher=fetcher, now=lambda: now[0])
    for expected_message in ("work-control projection unavailable", "work-control projection refresh throttled"):
        try:
            adapter.get(force=True)
        except ProjectionUnavailable as exc:
            assert str(exc) == expected_message
        else:
            raise AssertionError("failed/throttled refresh must fail closed")
    assert calls == 1


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


def test_webui_distinguishes_first_outage_and_fences_stale_responses():
    source = (Path(__file__).parents[1] / "static" / "sessions.js").read_text()
    assert "Indisponível — sem snapshot autoritativo" in source
    assert "unavailable:true" in source
    assert "_workControlProjectionRequestGeneration" in source
    assert "generation!==_workControlProjectionRequestGeneration" in source
    assert "new URL('api/work-control/projection',document.baseURI||location.href).href" in source
    assert "fetch('/api/work-control/projection'" not in source


def test_webui_first_outage_and_response_ordering_execute_behaviorally():
    source = (Path(__file__).parents[1] / "static" / "sessions.js").read_text()
    projection_js = "// K12 Operational Twin." + source.split("// K12 Operational Twin.", 1)[1].split(
        "function _sessionListRenderSignature", 1
    )[0]
    harness = r"""
const root={innerHTML:'',classList:{toggle(){} }};
const document={baseURI:'https://hermes.local/hermes/',getElementById(id){return id==='workControlProjection'?root:null;},createElement(){throw new Error('unexpected create');}};
function $(id){return null;}
function esc(value){return String(value);}
let pending=[];
let fetchMode='outage';
async function fetch(){
  if(fetchMode==='outage') return {ok:false,json:async()=>({error:'down'})};
  return new Promise(resolve=>pending.push(resolve));
}
(async()=>{
  await refreshWorkControlProjection();
  if(!root.innerHTML.includes('Indisponível — sem snapshot autoritativo')) throw new Error('first outage mislabeled');
  fetchMode='ordered';
  const oldRequest=refreshWorkControlProjection();
  const newRequest=refreshWorkControlProjection();
  pending[1]({ok:true,json:async()=>({records:[{work_id:'newer',status:'running'}],stale:false})});
  await newRequest;
  pending[0]({ok:true,json:async()=>({records:[{work_id:'older',status:'running'}],stale:false})});
  await oldRequest;
  if(!root.innerHTML.includes('newer')||root.innerHTML.includes('older')) throw new Error('stale response won');
  process.stdout.write('ok');
})().catch(error=>{console.error(error);process.exit(1);});
"""
    result = subprocess.run(
        ["node", "-e", projection_js + harness],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout == "ok"


def test_webui_uses_theme_tokens_for_projection_borders():
    source = (Path(__file__).parents[1] / "static" / "style.css").read_text()
    assert ".work-control-projection" in source
    assert "border:1px solid var(--border)" in source
    assert ".work-control-projection.stale{border-color:var(--warning)}" in source
