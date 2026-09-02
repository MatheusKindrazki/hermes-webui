"""Read-only, fail-closed Work Control projection for the Hermes WebUI.

The WebUI is deliberately a consumer: it never creates, advances, completes,
or retries a Work.  A failed refresh preserves the last known remote snapshot
and labels it stale instead of presenting an empty successful board.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

SCHEMA = "hermes-kernel-projection.v1"
TIMEOUT_SECONDS = 2.0
CACHE_TTL_SECONDS = 30.0


class ProjectionUnavailable(RuntimeError):
    """The authority could not return a valid projection in the contract time."""


class WorkControlProjection:
    """Small cache around the Jarvis read-only projection endpoint."""

    def __init__(self, fetcher: Callable[[], dict[str, Any]] | None = None, *, now=time.time):
        self._fetcher = fetcher or self._fetch_remote
        self._now = now
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] | None = None
        self._fetched_at = 0.0
        self._authority_versions: dict[str, int] = {}

    def get(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if not force and self._snapshot is not None and self._now() - self._fetched_at < CACHE_TTL_SECONDS:
                return self._with_freshness(self._snapshot, stale=False, source="cache")
            try:
                candidate = self._validate(self._fetcher())
                versions = {
                    str(row["work_id"]): int(row["authority_version"])
                    for row in candidate["records"]
                }
                # Versions are per Work in the canonical Jarvis envelope. A
                # stale proxy response for any known Work must not replace the
                # coherent last snapshot with a silent partial rollback.
                if any(version < self._authority_versions.get(work_id, -1) for work_id, version in versions.items()):
                    return self._with_freshness(self._snapshot, stale=True, source="authority_version_regression")
                self._snapshot = candidate
                self._authority_versions = versions
                self._fetched_at = self._now()
                return self._with_freshness(candidate, stale=False, source="remote")
            except Exception as exc:
                if self._snapshot is not None:
                    return self._with_freshness(self._snapshot, stale=True, source="unavailable", error=str(exc))
                raise ProjectionUnavailable("work-control projection unavailable") from exc

    def _fetch_remote(self) -> dict[str, Any]:
        endpoint = os.environ.get("HERMES_WORK_CONTROL_PROJECTION_URL", "").strip()
        token = os.environ.get("HERMES_WORK_CONTROL_READ_TOKEN", "").strip()
        if not endpoint or not token:
            raise ProjectionUnavailable("work-control read authority is not configured")
        request = Request(endpoint, headers={"X-API-Key": token, "Accept": "application/json"}, method="GET")
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # nosec B310: explicit configured authority
                if getattr(response, "status", 200) != 200:
                    raise ProjectionUnavailable(f"authority returned {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, ValueError) as exc:
            raise ProjectionUnavailable("work-control authority request failed") from exc

    @staticmethod
    def _validate(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
            raise ProjectionUnavailable("invalid projection schema")
        if payload.get("authority") != "remote":
            raise ProjectionUnavailable("projection authority is not remote")
        if not payload.get("generated_at") or payload.get("ttl_seconds") != CACHE_TTL_SECONDS:
            raise ProjectionUnavailable("invalid projection freshness contract")
        records = payload.get("records")
        if not isinstance(records, list):
            raise ProjectionUnavailable("projection lacks records")
        required = {"work_id", "front", "profile", "status", "authority_version", "updated_at"}
        for record in records:
            if not isinstance(record, dict) or not required.issubset(record):
                raise ProjectionUnavailable("invalid projection record")
            if record["profile"] is not None and not isinstance(record["profile"], str):
                raise ProjectionUnavailable("invalid projection profile")
            if not isinstance(record["authority_version"], int):
                raise ProjectionUnavailable("invalid record authority_version")
        return copy.deepcopy(payload)

    @staticmethod
    def _with_freshness(payload: dict[str, Any], *, stale: bool, source: str, error: str | None = None) -> dict[str, Any]:
        result = copy.deepcopy(payload)
        result["stale"] = bool(stale)
        result["source"] = source
        if error:
            result["error"] = "work-control projection unavailable"
        return result


projection = WorkControlProjection()
