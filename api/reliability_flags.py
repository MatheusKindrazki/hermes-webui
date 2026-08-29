"""Profile-aware authority resolution for default-off reliability features."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_ON = frozenset({"1", "true", "yes", "on", "enabled"})
_CANARY = frozenset({"canary", "profile", "tenant"})


def _identity_values(session, config_data: dict) -> tuple[str, str]:
    profile = str(getattr(session, "profile", "") or "").strip()
    tenant = str(
        getattr(session, "tenant", "")
        or getattr(session, "tenant_id", "")
        or ""
    ).strip()
    if not tenant:
        configured = config_data.get("tenant") if isinstance(config_data, dict) else None
        if isinstance(configured, dict):
            configured = configured.get("id") or configured.get("name")
        tenant = str(configured or "").strip()
    return profile, tenant


def _canary_rule_enabled(rule: Any, *, session, config_data: dict) -> bool:
    if isinstance(rule, bool):
        return rule
    if not isinstance(rule, dict) or rule.get("enabled") is not True:
        return False
    profile, tenant = _identity_values(session, config_data)
    profiles = rule.get("profiles")
    if profiles is not None:
        allowed_profiles = {str(value).strip() for value in profiles or []}
        if not profile or profile not in allowed_profiles:
            return False
    tenants = rule.get("tenants")
    if tenants is not None:
        allowed_tenants = {str(value).strip() for value in tenants or []}
        if not tenant or tenant not in allowed_tenants:
            return False
    return True


def reliability_feature_enabled(
    env_name: str,
    feature_name: str,
    *,
    session=None,
    profile_home: Path | str | None = None,
    config_data: dict | None = None,
    environ: dict[str, str] | None = None,
) -> bool:
    """Resolve off/global/canary authority without an ambient profile global.

    ``0`` (and unknown values) is the global kill switch. Truthy values keep the
    existing all-profile behavior. ``canary`` requires an explicit
    ``reliability.canary.<feature_name>`` rule from the owning profile config.
    """
    source = os.environ if environ is None else environ
    mode = str(source.get(env_name, "0") or "0").strip().lower()
    if mode in _ON:
        return True
    if mode not in _CANARY:
        return False
    if config_data is None:
        if profile_home is None and session is not None:
            try:
                from api.profiles import get_hermes_home_for_profile

                profile_home = get_hermes_home_for_profile(
                    getattr(session, "profile", None)
                )
            except Exception:
                return False
        if profile_home is None:
            return False
        try:
            from api.config import get_config_for_profile_home

            config_data = get_config_for_profile_home(Path(profile_home).expanduser())
        except Exception:
            return False
    if not isinstance(config_data, dict):
        return False
    reliability = config_data.get("reliability")
    canary = reliability.get("canary") if isinstance(reliability, dict) else None
    rule = canary.get(feature_name) if isinstance(canary, dict) else None
    return _canary_rule_enabled(rule, session=session, config_data=config_data)
