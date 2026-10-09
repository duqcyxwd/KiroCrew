"""Adopt a kiro-cli session as a dashboard chat.

Two endpoints:

* ``GET /api/kiro-cli-sessions`` — list the sessions in the kiro-cli store,
  optionally filtered by kind, resumability and a text query.
* ``POST /api/chat/slots/import-kiro-cli`` — pick one by sid and materialise it
  as a new dashboard slot, reusing the shared arrival path
  (:func:`~kiro_crew.dashboard.session_transfer._materialise_validated_bundle`).

The convert-then-import is one code path: :func:`transcript_rows
<kiro_crew.kiro_cli_sessions.transcript_rows>` emits the ``{role, content, ts}``
rows a bundle's ``messages`` carries, :func:`_read_layer_b
<kiro_crew.dashboard.session_transfer._read_layer_b>` ships the context, and
``_validate_bundle`` + ``_materialise_validated_bundle`` do everything a transfer
arrival does.  No new materialiser, and instances.md §14.5a — *one route, one set
of rules* — is a property of the code.

**Owner-only, SEL-audited.**  Both endpoints are behind
:func:`require_owner_dashboard_request`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import web

from kiro_crew.dashboard.handlers._shared import require_owner_dashboard_request
from kiro_crew.dashboard.session_transfer import (
    BUNDLE_VERSION,
    _materialise_validated_bundle,
    _read_layer_b,
    _slot_cap_refusal,
    _validate_bundle,
)
from kiro_crew.dashboard.state import DashboardState
from kiro_crew.kiro_cli_sessions import (
    KINDS,
    filter_sessions,
    list_sessions,
    transcript_rows,
)
from kiro_crew.sel import sel
from kiro_crew.validation import bounded_session_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GET /api/kiro-cli-sessions
# ---------------------------------------------------------------------------


async def api_kiro_cli_sessions(request: web.Request) -> web.Response:
    """List kiro-cli sessions the user can import.

    Query parameters (all optional):

    * ``q`` — case-insensitive substring matched against title and cwd.
    * ``kind`` — comma-separated subset of :data:`~kiro_crew.kiro_cli_sessions.KINDS`.
    * ``resumable`` — when ``"1"`` or ``"true"``, only resumable sessions.
    """
    denied = await require_owner_dashboard_request(request, "kiro_cli_sessions.list")
    if denied is not None:
        return denied

    query = request.query.get("q", "")
    raw_kinds = request.query.get("kind", "")
    resumable_only = request.query.get("resumable", "") in ("1", "true")

    kinds: frozenset[str] | None = None
    if raw_kinds:
        parsed = frozenset(k.strip() for k in raw_kinds.split(",") if k.strip())
        invalid = parsed - frozenset(KINDS)
        if invalid:
            return web.json_response(
                {"error": f"unknown kind(s): {', '.join(sorted(invalid))}", "code": "bad_kind"},
                status=400,
            )
        kinds = parsed if parsed else None

    state: DashboardState = request.app["state"]
    sessions_mgr = getattr(state, "sessions", None)

    # Resolve which sids are already mapped ON THE LOOP (the session map is an
    # unlocked dict with whole-file writes — see kiro_cli_sessions module
    # docstring on threading).
    mapped_sids: frozenset[str] = frozenset()
    if sessions_mgr is not None:
        sm = getattr(sessions_mgr, "_session_map", None)
        if sm is not None:
            mapped_sids = frozenset(sm.mapped_sids_by_key().values())

    # Blocking IO — enumerate + filter in a worker thread.
    rows = await asyncio.to_thread(_list_and_filter, query, kinds, resumable_only)

    return web.json_response(
        {
            "sessions": [r.as_wire(mapped=r.sid in mapped_sids) for r in rows],
            "total": len(rows),
        }
    )


def _list_and_filter(
    query: str,
    kinds: frozenset[str] | None,
    resumable_only: bool,
) -> list[Any]:
    """List + filter in one blocking call, suitable for ``asyncio.to_thread``."""
    return filter_sessions(
        list_sessions(),
        query=query,
        kinds=kinds,
        resumable_only=resumable_only,
    )


# ---------------------------------------------------------------------------
# POST /api/chat/slots/import-kiro-cli
# ---------------------------------------------------------------------------


async def api_kiro_cli_adopt(request: web.Request) -> web.Response:
    """Adopt a kiro-cli session as a new dashboard chat slot.

    Body: ``{"sid": "<kiro-cli session id>"}``

    The handler:

    1. Validates the sid.
    2. Converts the replay log to bundle ``messages`` via
       :func:`~kiro_cli_sessions.transcript_rows`.
    3. Reads Layer B via :func:`_read_layer_b` (byte-exact, unredacted — the
       same reasons as a peer transfer apply here, plus this is a LOCAL copy
       on the SAME machine).
    4. Assembles a v2 bundle and passes it through ``_validate_bundle`` +
       ``_materialise_validated_bundle``, the same code path every other
       arrival takes.

    Returns the same shape as ``POST /api/chat/slots/import``.
    """
    denied = await require_owner_dashboard_request(request, "kiro_cli_sessions.adopt")
    if denied is not None:
        return denied

    state: DashboardState = request.app["state"]
    caller = "dashboard"

    # --- slot cap (early, before any IO) ---
    cap_refused = _slot_cap_refusal(state, caller)
    if cap_refused is not None:
        return cap_refused

    # --- parse body ---
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": "request body must be JSON", "code": "adopt_bad_body"}, status=400
        )
    if not isinstance(body, dict):
        return web.json_response(
            {"error": "request body must be a JSON object", "code": "adopt_bad_body"},
            status=400,
        )
    raw_sid = body.get("sid", "")
    if not isinstance(raw_sid, str):
        return web.json_response(
            {"error": "sid must be a string", "code": "adopt_bad_sid"}, status=400
        )
    sid = bounded_session_id(raw_sid)
    if not sid:
        return web.json_response(
            {"error": "invalid or empty session id", "code": "adopt_bad_sid"}, status=400
        )

    # --- convert the replay log to bundle messages (blocking IO) ---
    messages = await asyncio.to_thread(transcript_rows, sid)
    if not messages:
        sel().log_api_access(
            caller=caller,
            operation="kiro_cli_sessions.adopt",
            outcome="denied",
            error="session has no displayable turns",
            resources=f"sid={sid}",
        )
        return web.json_response(
            {
                "error": "session has no displayable turns (empty or unreadable)",
                "code": "adopt_empty_session",
            },
            status=400,
        )

    # --- read Layer B (blocking IO) ---
    layer_b = await asyncio.to_thread(_read_layer_b, sid)

    # --- read the envelope title (blocking IO, best-effort) ---
    title = await asyncio.to_thread(_read_envelope_title, sid)

    # --- assemble a v2 bundle ---
    bundle_doc: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "messages": messages,
        "title": title,
        "origin": "kiro-cli",
        # Agent is intentionally blank: the session arrives unscoped and resolves
        # the target's default agent, matching the "no model carried" rule.
        "agent": "",
    }
    if layer_b is not None:
        bundle_doc["layer_b"] = layer_b

    # --- validate (structural) ---
    bundle, err = _validate_bundle(bundle_doc)
    if err is not None:
        logger.warning("kiro_cli_adopt: bundle validation failed for sid=%s", sid)
        return err

    # Carry through layer_b from the original doc (validation normalises
    # messages but does not re-derive layer_b for a bundle that was never
    # serialised — it expects the caller to have passed it through, which
    # _validate_bundle does for wire bundles).
    if layer_b is not None and "layer_b" not in bundle:
        bundle["layer_b"] = layer_b

    sel().log_api_access(
        caller=caller,
        operation="kiro_cli_sessions.adopt",
        outcome="allowed",
        resources=f"sid={sid}, messages={len(messages)}, layer_b={'yes' if layer_b else 'no'}",
    )

    # --- materialise (shared path) ---
    return await _materialise_validated_bundle(request, bundle)


def _read_envelope_title(sid: str) -> str:
    """Best-effort title from the kiro-cli envelope. **Blocking IO.**"""
    import json

    from kiro_crew.config.paths import kiro_sessions_dir
    from kiro_crew.jsonl_util import open_regular_nofollow

    try:
        path = kiro_sessions_dir() / f"{sid}.json"
        with open_regular_nofollow(path, max_bytes=8 * 1024 * 1024) as handle:
            envelope = json.loads(handle.read().decode("utf-8", errors="replace"))
        if isinstance(envelope, dict):
            t = envelope.get("title", "")
            return str(t)[:512] if isinstance(t, str) else ""
    except Exception:
        return ""
    return ""
