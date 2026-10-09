"""Tests for the kiro-cli session browser (``kiro_cli_sessions``).

The module reads kiro-cli's own replay store and converts one session to the
``{role, content, ts}`` rows a transfer bundle's ``messages`` carries. Four
properties matter and each is pinned here:

1. KIND. ``session_created_reason`` maps onto the three kinds, and an ABSENT
   reason is an interactive ``cli`` session rather than its own bucket -- a
   session written before that field existed must still land in a bucket a
   filter selects.
2. CONVERSION. ``transcript_rows`` admits ``Prompt`` -> ``user`` and an
   ``AssistantMessage``'s ``text`` blocks -> ``assistant``, and drops
   ``toolUse`` / ``ToolResults`` -- the role set a bundle may carry stops at
   two. A tool-only assistant turn produces no row.
3. TRUST. The store is agent-writable, so a malformed envelope degrades to a
   thinner row instead of failing the whole listing, and a bad ``Prompt``
   timestamp drops to ``""`` instead of aborting the conversion.
4. FILTERING. ``filter_sessions`` is a pure narrowing of a listing -- substring
   over title/cwd, kind subset, resumable-only -- with no IO.

All timestamps are written explicitly; nothing here sleeps or reads the wall
clock, so the file is order-independent and reshuffle-safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiro_crew import kiro_cli_sessions as mod


@pytest.fixture
def store(monkeypatch, tmp_path: Path) -> Path:
    """Point the module's store resolver at an empty tmp dir, cache cleared."""
    sessions = tmp_path / "cli"
    sessions.mkdir()
    monkeypatch.setattr(mod, "kiro_sessions_dir", lambda: sessions)
    mod.invalidate_cache()
    return sessions


def _write_session(
    store: Path,
    sid: str,
    *,
    envelope: dict | None = None,
    records: list[dict] | None = None,
) -> None:
    """Write one ``<sid>.json`` envelope and ``<sid>.jsonl`` replay log.

    ``encoding="utf-8"`` is explicit, not tidiness: the reader decodes bytes as
    UTF-8 and a Windows runner's locale codec (cp1252) cannot represent every
    character a title may carry, which would kill the test in its own setup.
    """
    (store / f"{sid}.json").write_text(json.dumps(envelope or {}), encoding="utf-8")
    lines = "\n".join(json.dumps(r) for r in (records or []))
    (store / f"{sid}.jsonl").write_text(lines, encoding="utf-8")


def _prompt(text: str, ts: float | None = None) -> dict:
    meta = {} if ts is None else {"meta": {"timestamp": ts}}
    return {
        "kind": "Prompt",
        "version": "v1",
        "data": {"content": [{"kind": "text", "data": text}], **meta},
    }


def _assistant(*blocks: dict) -> dict:
    return {"kind": "AssistantMessage", "version": "v1", "data": {"content": list(blocks)}}


# --------------------------------------------------------------------- KIND


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("subagent", mod.KIND_SUBAGENT),
        ("rewind", mod.KIND_REWIND),
        ("cli", mod.KIND_CLI),
        (None, mod.KIND_CLI),  # absent -> interactive chat, not an "unknown" bucket
        ("", mod.KIND_CLI),
        ("something-this-build-does-not-know", mod.KIND_CLI),
    ],
)
def test_kind_of_maps_reason(reason, expected):
    assert mod._kind_of(reason) == expected


# ---------------------------------------------------------------- CONVERSION


def test_transcript_rows_admits_user_and_assistant_text(store: Path):
    _write_session(
        store,
        "s1",
        records=[
            _prompt("hello", ts=1_700_000_000),
            _assistant({"kind": "text", "data": "hi there"}),
        ],
    )
    rows = mod.transcript_rows("s1")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["content"] == "hello"
    assert rows[1]["content"] == "hi there"
    # The user turn carries an ISO-8601 UTC ts; the model turn carries none.
    assert rows[0]["ts"].startswith("2023-11-14T")
    assert rows[1]["ts"] == ""


def test_transcript_rows_drops_tool_frames(store: Path):
    _write_session(
        store,
        "s2",
        records=[
            _prompt("do the thing"),
            # Assistant turn that ONLY calls a tool -> nothing to display.
            _assistant({"kind": "toolUse", "data": {"name": "shell"}}),
            {"kind": "ToolResults", "version": "v1", "data": {"toolUseId": "x"}},
            _assistant({"kind": "text", "data": "done"}),
        ],
    )
    rows = mod.transcript_rows("s2")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[1]["content"] == "done"


def test_transcript_rows_mixed_assistant_keeps_only_text(store: Path):
    _write_session(
        store,
        "s3",
        records=[
            _assistant(
                {"kind": "text", "data": "part one "},
                {"kind": "toolUse", "data": {"name": "x"}},
                {"kind": "text", "data": "part two"},
            ),
        ],
    )
    rows = mod.transcript_rows("s3")
    assert rows == [{"role": "assistant", "content": "part one part two", "ts": ""}]


def test_transcript_rows_bad_timestamp_drops_to_empty(store: Path):
    # datetime.fromtimestamp raises on an out-of-range number; it must not abort
    # the whole conversion over one bad record.
    _write_session(store, "s4", records=[_prompt("hi", ts=1e308)])
    rows = mod.transcript_rows("s4")
    assert rows[0]["ts"] == ""


def test_transcript_rows_unreadable_file_is_empty(store: Path):
    assert mod.transcript_rows("does-not-exist") == []


def test_transcript_rows_rejects_bad_sid(store: Path):
    assert mod.transcript_rows("../etc/passwd") == []


# --------------------------------------------------------------------- TRUST


def test_scan_degrades_on_malformed_envelope(store: Path):
    # A non-JSON envelope must still list -- the user can see it on disk -- just
    # with empty labels, rather than taking the whole listing down.
    (store / "bad.json").write_text("{not json", encoding="utf-8")
    (store / "bad.jsonl").write_text("", encoding="utf-8")
    rows = mod.list_sessions(cached=False)
    assert any(r.sid == "bad" and r.title == "" for r in rows)


def test_blocks_text_tolerates_non_list():
    assert mod._blocks_text("not a list", want="text") == ""
    assert mod._blocks_text(None, want="text") == ""


def test_prompt_ts_rejects_bool_and_non_number():
    # bool is an int subclass; a flag must not be read as an epoch.
    assert mod._prompt_ts({"timestamp": True}) == ""
    assert mod._prompt_ts({"timestamp": "nope"}) == ""
    assert mod._prompt_ts("not a dict") == ""


# ----------------------------------------------------------------- FILTERING


def _row(
    sid: str, *, title: str = "", cwd: str = "", kind: str = mod.KIND_CLI, resumable: bool = True
):
    return mod.KiroCliSession(
        sid=sid,
        title=title,
        cwd=cwd,
        created_at="",
        updated_at="",
        bytes=0,
        kind=kind,
        resumable=resumable,
    )


def test_filter_by_query_matches_title_and_cwd():
    rows = [
        _row("a", title="Fix the alarm filter"),
        _row("b", cwd="/Users/me/code/KiroCrew"),
        _row("c", title="unrelated"),
    ]
    assert {r.sid for r in mod.filter_sessions(rows, query="alarm")} == {"a"}
    assert {r.sid for r in mod.filter_sessions(rows, query="kirocrew")} == {"b"}
    # Empty query returns everything.
    assert len(mod.filter_sessions(rows, query="")) == 3


def test_filter_by_kind_and_resumable():
    rows = [
        _row("a", kind=mod.KIND_CLI, resumable=True),
        _row("b", kind=mod.KIND_SUBAGENT, resumable=False),
        _row("c", kind=mod.KIND_REWIND, resumable=True),
    ]
    assert {r.sid for r in mod.filter_sessions(rows, kinds=frozenset({mod.KIND_SUBAGENT}))} == {"b"}
    assert {r.sid for r in mod.filter_sessions(rows, resumable_only=True)} == {"a", "c"}


def test_filter_is_pure_leaves_input_untouched():
    rows = [_row("a", title="keep"), _row("b", title="drop")]
    before = list(rows)
    mod.filter_sessions(rows, query="keep")
    assert rows == before


# -------------------------------------------------------------------- WIRE


def test_as_wire_mapped_is_caller_supplied():
    row = _row("a", title="t", cwd="/c", kind=mod.KIND_CLI)
    assert row.as_wire(mapped=True)["mapped"] is True
    assert row.as_wire(mapped=False)["mapped"] is False
    # mapped is not a field of the record itself.
    assert "mapped" not in {f for f in row.__dataclass_fields__}
