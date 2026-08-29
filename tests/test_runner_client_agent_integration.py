"""Real HTTP contract proof between WebUI and Hermes Agent's run receiver."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import time

import pytest

from api.runner_client import HttpRunnerClient
from api.runtime_adapter import StartRunRequest


_SERVER_SCRIPT = r"""
import asyncio
import threading
from aiohttp import web
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

adapter = APIServerAdapter(PlatformConfig(enabled=True))

class FakeAgent:
    def __init__(self, *, progress):
        self.progress = progress
        self.stopped = threading.Event()
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0

    def run_conversation(self, **_kwargs):
        self.progress("tool.started", tool_name="terminal", preview="running")
        self.stopped.wait(timeout=30)
        return {"final_response": "stopped"}

    def interrupt(self, _message=None):
        self.stopped.set()

    def steer(self, _text):
        return True

def create_agent(**kwargs):
    return FakeAgent(progress=kwargs["tool_progress_callback"])

adapter._create_agent = create_agent

async def main():
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    print(port, flush=True)
    await asyncio.Event().wait()

asyncio.run(main())
"""


def _agent_source() -> Path:
    raw = str(os.environ.get("HERMES_AGENT_SOURCE") or "").strip()
    if not raw:
        pytest.skip("set HERMES_AGENT_SOURCE for the cross-repo contract proof")
    source = Path(raw).resolve()
    if not (source / "gateway" / "platforms" / "api_server.py").is_file():
        pytest.fail(f"invalid HERMES_AGENT_SOURCE: {source}")
    return source


@contextmanager
def _agent_server():
    source = _agent_source()
    agent_python = str(os.environ.get("HERMES_AGENT_PYTHON") or "").strip()
    if not agent_python:
        pytest.skip("set HERMES_AGENT_PYTHON to an Agent test interpreter")
    env = dict(os.environ)
    prior_pythonpath = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source), prior_pythonpath] if prior_pythonpath else [str(source)]
    )
    process = subprocess.Popen(
        [agent_python, "-c", _SERVER_SCRIPT],
        cwd=source,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        port_line = process.stdout.readline().strip()
        if not port_line:
            detail = process.stderr.read(2000)
            pytest.fail(f"Agent server did not start: {detail}")
        yield HttpRunnerClient(base_url=f"http://127.0.0.1:{int(port_line)}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_http_runner_client_real_start_observe_message_clarify_and_cancel():
    with _agent_server() as client:
        started = client.start_run(
            StartRunRequest(
                session_id="webui-real-contract",
                message="run until cancelled",
            )
        )
        run_id = started["run_id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if client.get_run(run_id)["status"] == "running":
                break
            time.sleep(0.01)
        else:
            pytest.fail("Agent run did not reach running")

        with ThreadPoolExecutor(max_workers=1) as pool:
            observed = pool.submit(client.observe_run, run_id)
            assert client.queue_message(run_id, "follow-up", mode="interrupt")["accepted"] is True
            assert client.respond_clarify(run_id, "clarify-1", "the answer")["accepted"] is True
            cancelled = client.cancel_run(run_id)
            stream = observed.result(timeout=10)

        assert cancelled["status"] == "stopping"
        names = [event.get("event") for event in stream["events"]]
        assert "tool.started" in names
        assert names.count("run.steered") == 2
        assert "run.cancelled" in names
