"""MCP must never reuse a name-only process registry across profile turns."""

from __future__ import annotations

from types import SimpleNamespace
import threading

from api import streaming


def test_v2_fails_closed_for_concurrent_same_name_profile_mcp_servers(tmp_path):
    barrier = threading.Barrier(2)
    global_discovery_calls = []
    fake_mcp_module = SimpleNamespace(
        _servers={},
        discover_mcp_tools=lambda: global_discovery_calls.append("global"),
    )
    failures = {}

    def execute(profile: str):
        profile_home = tmp_path / profile
        config_data = {
            "mcp_servers": {
                "shared": {
                    "command": f"run-{profile}",
                    "env": {"MCP_TOKEN": f"credential-{profile}"},
                }
            }
        }
        barrier.wait(timeout=5)
        try:
            streaming._discover_mcp_tools_for_request_context_v2(
                profile_home=profile_home,
                config_data=config_data,
                mcp_module=fake_mcp_module,
            )
        except streaming.RequestContextV2MCPIsolationError as exc:
            failures[profile] = str(exc)

    threads = [
        threading.Thread(target=execute, args=(profile,))
        for profile in ("work", "personal")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert set(failures) == {"work", "personal"}
    assert all("profile-scoped MCP registry" in value for value in failures.values())
    assert all("credential-" not in value for value in failures.values())
    assert global_discovery_calls == []
    assert fake_mcp_module._servers == {}


def test_v2_uses_explicit_profile_scoped_discovery_capability(tmp_path):
    calls = []

    def discover_for_profile(*, profile_home, servers):
        calls.append((profile_home, servers))
        return [f"tool:{profile_home}:shared"]

    fake_mcp_module = SimpleNamespace(
        PROFILE_SCOPED_MCP_REGISTRY=True,
        discover_mcp_tools_for_profile=discover_for_profile,
        discover_mcp_tools=lambda: (_ for _ in ()).throw(
            AssertionError("global discovery must not run")
        ),
    )
    profile_home = tmp_path / "work"
    config_data = {
        "mcp_servers": {
            "shared": {
                "command": "run-work",
                "env": {"MCP_TOKEN": "secret-work"},
            },
            "disabled": {"command": "never", "enabled": False},
        }
    }

    result = streaming._discover_mcp_tools_for_request_context_v2(
        profile_home=profile_home,
        config_data=config_data,
        mcp_module=fake_mcp_module,
    )

    assert result == [f"tool:{profile_home}:shared"]
    assert calls == [
        (
            str(profile_home),
            {
                "shared": {
                    "command": "run-work",
                    "env": {"MCP_TOKEN": "secret-work"},
                }
            },
        )
    ]


def test_v2_without_enabled_mcp_servers_does_not_require_registry_capability(
    tmp_path,
):
    fake_mcp_module = SimpleNamespace(
        discover_mcp_tools=lambda: (_ for _ in ()).throw(
            AssertionError("discovery must not run")
        )
    )

    assert streaming._discover_mcp_tools_for_request_context_v2(
        profile_home=tmp_path / "empty",
        config_data={"mcp_servers": {"disabled": {"enabled": False}}},
        mcp_module=fake_mcp_module,
    ) == []

