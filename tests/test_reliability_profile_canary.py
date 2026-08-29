"""Global reliability switches with isolated profile/tenant canary authority."""

from __future__ import annotations

from api import config, routes, streaming
from api.client_turn_ledger import ClientTurnLedger


class _Session:
    def __init__(self, profile: str, *, tenant: str | None = None):
        self.session_id = f"session-{profile}-{tenant or 'none'}"
        self.parent_session_id = None
        self.profile = profile
        self.tenant = tenant


def _profile_configs(monkeypatch, tmp_path, configs):
    homes = {profile: tmp_path / profile for profile in configs}
    for home in homes.values():
        home.mkdir()
    from api import config as config_module
    from api import profiles

    monkeypatch.setattr(
        profiles,
        "get_hermes_home_for_profile",
        lambda profile: homes[str(profile)],
    )
    monkeypatch.setattr(
        config_module,
        "get_config_for_profile_home",
        lambda home: configs[next(name for name, path in homes.items() if path == home)],
    )
    return homes


def test_turn_ledger_canary_is_authoritative_for_only_one_profile(
    tmp_path, monkeypatch
):
    homes = _profile_configs(
        monkeypatch,
        tmp_path,
        {
            "work": {"reliability": {"canary": {"turn_ledger": True}}},
            "personal": {"reliability": {"canary": {"turn_ledger": False}}},
        },
    )
    monkeypatch.setenv("HERMES_TURN_LEDGER_ENABLED", "canary")
    work = _Session("work")
    personal = _Session("personal")

    assert routes._client_turn_ledger_enabled(work) is True
    assert routes._client_turn_ledger_enabled(personal) is False

    monkeypatch.setenv("HERMES_TURN_LEDGER_ENABLED", "0")
    assert routes._client_turn_ledger_enabled(work) is False
    monkeypatch.setenv("HERMES_TURN_LEDGER_ENABLED", "1")
    assert routes._client_turn_ledger_enabled(personal) is True


def test_request_context_v2_canary_resolves_profile_config(tmp_path, monkeypatch):
    homes = _profile_configs(
        monkeypatch,
        tmp_path,
        {
            "work": {"reliability": {"canary": {"request_context_v2": True}}},
            "personal": {"reliability": {"canary": {"request_context_v2": False}}},
        },
    )
    monkeypatch.setenv("HERMES_REQUEST_CONTEXT_V2", "canary")

    assert streaming._request_context_v2_enabled(
        session=_Session("work"),
        profile_home=homes["work"],
    ) is True
    assert streaming._request_context_v2_enabled(
        session=_Session("personal"),
        profile_home=homes["personal"],
    ) is False

    monkeypatch.setenv("HERMES_REQUEST_CONTEXT_V2", "0")
    assert streaming._request_context_v2_enabled(
        session=_Session("work"),
        profile_home=homes["work"],
    ) is False


def test_canary_rule_can_narrow_one_profile_to_one_tenant(tmp_path, monkeypatch):
    homes = _profile_configs(
        monkeypatch,
        tmp_path,
        {
            "staff": {
                "reliability": {
                    "canary": {
                        "turn_ledger": {
                            "enabled": True,
                            "profiles": ["staff"],
                            "tenants": ["lugui"],
                        }
                    }
                }
            }
        },
    )
    monkeypatch.setenv("HERMES_TURN_LEDGER_ENABLED", "canary")

    assert routes._client_turn_ledger_enabled(
        _Session("staff", tenant="lugui")
    ) is True
    assert routes._client_turn_ledger_enabled(
        _Session("staff", tenant="moklabs")
    ) is False


def test_canary_kill_switch_change_does_not_strand_admitted_ledger_row(
    tmp_path, monkeypatch
):
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    ledger = ClientTurnLedger(state_dir / "client_turn_ledger.sqlite3")
    ledger.claim(
        lineage_root_id="lineage",
        client_turn_id="client",
        turn_id="turn",
        current_session_id="session",
        stream_id="stream",
        request_sha256="a" * 64,
        receipt={"stream_id": "stream", "session_id": "session"},
        state="started",
    )
    monkeypatch.setenv("HERMES_TURN_LEDGER_ENABLED", "0")

    streaming._settle_client_turn_ledger(
        "stream",
        "completed",
        current_session_id="session",
    )

    assert ledger.get_by_stream("stream")["state"] == "completed"
