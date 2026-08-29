"""HTTP client boundary for a supervised Hermes WebUI runner backend.

This module intentionally contains no process-local run maps, stream queues,
cancellation registries, approval/clarify queues, or cached agent instances. It
is only a JSON-over-HTTP transport used by ``RunnerRuntimeAdapter`` when an
operator explicitly configures a runner endpoint.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


_RUNNER_BASE_URL_ENV = "HERMES_WEBUI_RUNNER_BASE_URL"
_RUNNER_API_KEY_ENV = "HERMES_WEBUI_RUNNER_API_KEY"


class RunnerClientError(RuntimeError):
    """Raised when a configured runner endpoint rejects or fails a request."""


def runner_client_configured(environ: dict[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    return bool(str(source.get(_RUNNER_BASE_URL_ENV) or "").strip())


class HttpRunnerClient:
    """Small JSON HTTP client for the external/supervised runner boundary."""

    def __init__(self, *, base_url: str, api_key: str = ""):
        self.base_url = str(base_url or "").strip().rstrip("/")
        if not self.base_url:
            raise ValueError("runner base_url is required")
        # Hardening: the runner endpoint is operator-configured, but reject any
        # non-HTTP(S) scheme so a misconfigured HERMES_WEBUI_RUNNER_BASE_URL
        # (e.g. file:///etc/passwd or ftp://) can never be handed to urlopen.
        _scheme = urllib.parse.urlsplit(self.base_url).scheme.lower()
        if _scheme not in ("http", "https"):
            raise ValueError(
                f"runner base_url must be http(s); got scheme '{_scheme or '(none)'}'"
            )
        self.api_key = str(api_key or "").strip()
        # Transport handles only: lifecycle truth remains runner-owned. Each
        # observe call pumps one frame from the same open SSE response.
        self._sse_streams: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "HttpRunnerClient":
        source = os.environ if environ is None else environ
        base_url = str(source.get(_RUNNER_BASE_URL_ENV) or "").strip()
        if not base_url:
            raise NotImplementedError("runner-local chat backend is not configured")
        return cls(base_url=base_url, api_key=str(source.get(_RUNNER_API_KEY_ENV) or ""))

    def start_run(self, request) -> dict[str, Any]:
        metadata = dict(request.metadata or {})
        idempotency_key = str(metadata.get("idempotency_key") or "").strip()
        if idempotency_key and not all(
            char.isalnum() or char in {":", ".", "_", "-"}
            for char in idempotency_key
        ):
            raise RunnerClientError("Runner idempotency key contains invalid characters")
        if len(idempotency_key) > 200:
            raise RunnerClientError("Runner idempotency key is too long")
        return self._post("/v1/runs", {
            "session_id": request.session_id,
            "message": request.message,
            "attachments": list(request.attachments or []),
            "workspace": request.workspace,
            "profile": request.profile,
            "provider": request.provider,
            "model": request.model,
            "toolsets": list(request.toolsets or []),
            "source": request.source,
            "metadata": metadata,
        }, extra_headers=(
            {"Idempotency-Key": idempotency_key}
            if idempotency_key
            else None
        ))

    def observe_run(self, run_id: str, *, cursor: str | None = None) -> dict[str, Any]:
        query = ""
        if cursor not in (None, ""):
            query = "?cursor=" + urllib.parse.quote(str(cursor), safe="")
        path = f"/v1/runs/{urllib.parse.quote(str(run_id), safe='')}/events{query}"
        headers = self._headers()
        headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(self.base_url + path, headers=headers, method="GET")
        return self._request_sse(req, run_id=str(run_id), cursor=cursor)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/v1/runs/{urllib.parse.quote(str(run_id), safe='')}")

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._post(f"/v1/runs/{urllib.parse.quote(str(run_id), safe='')}/stop", {})

    def respond_approval(self, run_id: str, approval_id: str, choice: str) -> dict[str, Any]:
        return self._post(
            f"/v1/runs/{urllib.parse.quote(str(run_id), safe='')}/approval",
            {"choice": choice, "approval_id": approval_id},
        )

    def respond_clarify(self, run_id: str, clarify_id: str, response: str) -> dict[str, Any]:
        return self._post(
            f"/v1/runs/{urllib.parse.quote(str(run_id), safe='')}/steer",
            {"message": response, "clarify_id": clarify_id},
        )

    def queue_message(self, run_id: str, message: str, *, mode: str = "queue") -> dict[str, Any]:
        return self._post(
            f"/v1/runs/{urllib.parse.quote(str(run_id), safe='')}/steer",
            {"message": message, "mode": mode},
        )

    def update_goal(self, session_id: str, action: str, text: str = "") -> dict[str, Any]:
        return self._post(
            f"/v1/sessions/{urllib.parse.quote(str(session_id), safe='')}/goal",
            {"action": action, "text": text},
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-WebUI-RunnerClient",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, headers=self._headers(), method="GET")
        return self._request_json(req)

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        headers = self._headers()
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        return self._request_json(req)

    def _opener(self) -> urllib.request.OpenerDirector:
        # Hardening: do NOT follow redirects. A misbehaving/compromised runner
        # returning 3xx Location could otherwise smuggle the Bearer token to
        # another host. Treat any redirect as an error instead.
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        return urllib.request.build_opener(_NoRedirect)

    def _request_json(self, req: urllib.request.Request) -> dict[str, Any]:
        try:
            with self._opener().open(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(2048).decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise RunnerClientError(f"Runner returned HTTP {exc.code}: {detail[:500]}") from exc
        except Exception as exc:
            raise RunnerClientError(f"Runner request failed: {exc}") from exc
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise RunnerClientError("Runner returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RunnerClientError("Runner returned a non-object JSON payload")
        return payload

    def _request_sse(
        self,
        req: urllib.request.Request,
        *,
        run_id: str,
        cursor: str | None,
    ) -> dict[str, Any]:
        """Pump one Agent SSE event without buffering the response to EOF."""
        state = self._sse_streams.get(run_id)
        if state is None:
            try:
                response = self._opener().open(req, timeout=60)
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read(2048).decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                raise RunnerClientError(
                    f"Runner returned HTTP {exc.code}: {detail[:500]}"
                ) from exc
            except Exception as exc:
                raise RunnerClientError(f"Runner request failed: {exc}") from exc
            try:
                seq = int(str(cursor or "0").rsplit(":", 1)[-1])
            except (TypeError, ValueError):
                seq = 0
            state = {
                "response": response,
                "iterator": iter(response),
                "data_lines": [],
                "wire_event": None,
                "wire_id": None,
                "seq": max(0, seq),
            }
            self._sse_streams[run_id] = state

        events: list[dict[str, Any]] = []
        terminal_names = {
            "run.cancelled",
            "run.completed",
            "run.failed",
            "stream_end",
        }

        def close_stream() -> None:
            if self._sse_streams.get(run_id) is state:
                self._sse_streams.pop(run_id, None)
            close = getattr(state["response"], "close", None)
            if callable(close):
                close()

        while not events:
            try:
                raw_line = next(state["iterator"])
            except StopIteration:
                close_stream()
                break
            except Exception as exc:
                close_stream()
                raise RunnerClientError(f"Runner SSE read failed: {exc}") from exc

            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                data_lines = state["data_lines"]
                if not data_lines:
                    state["wire_event"] = None
                    state["wire_id"] = None
                    continue
                raw = "\n".join(data_lines)
                state["data_lines"] = []
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    close_stream()
                    raise RunnerClientError("Runner returned invalid SSE JSON") from exc
                if not isinstance(payload, dict):
                    close_stream()
                    raise RunnerClientError("Runner returned a non-object SSE event")
                if state["wire_event"] and not payload.get("event"):
                    payload["event"] = state["wire_event"]
                if state["wire_id"] and not payload.get("event_id"):
                    payload["event_id"] = state["wire_id"]
                state["seq"] += 1
                payload.setdefault("seq", state["seq"])
                events.append(payload)
                state["wire_event"] = None
                state["wire_id"] = None
                if payload.get("event") in terminal_names:
                    close_stream()
            elif line.startswith(":"):
                continue
            elif line.startswith("data:"):
                state["data_lines"].append(line[5:].lstrip(" "))
            elif line.startswith("event:"):
                state["wire_event"] = line[6:].strip()
            elif line.startswith("id:"):
                state["wire_id"] = line[3:].strip()

        next_cursor = str(events[-1]["seq"]) if events else cursor
        last_event_id = events[-1].get("event_id") if events else None
        return {
            "run_id": run_id,
            "events": events,
            "cursor": next_cursor,
            "last_event_id": last_event_id,
        }
