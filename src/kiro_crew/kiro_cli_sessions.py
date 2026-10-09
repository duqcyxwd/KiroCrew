"""Browse kiro-cli's own session store, and read one as transcript rows.

The dashboard's session list is built from Kiro Crew's **transcripts**
(``<data home>/sessions/<stem>.jsonl``), so it can only offer a session that has
one. kiro-cli's own store is the other half — ``<kiro home>/sessions/cli/<sid>.json``
plus ``<sid>.jsonl`` — and most of what lives there has no transcript at all: a
subagent run leaves only a replay log, and a plain ``kiro-cli chat`` conversation
predates the dashboard entirely. On the machine this was written for that is
~1600 replay logs against ~60 transcripts, so "every session on this machine"
cannot be answered from the transcript side.

This module reads the replay side, for two callers:

* :func:`list_sessions` — the picker behind **+New ▸ Import a kiro-cli session**.
* :func:`transcript_rows` — the *Layer A* (display) conversion for the one
  session a user picked, in the ``{role, content, ts}`` shape a transfer
  bundle's ``messages`` already carries
  ([instances §14.1a](../../docs/system-specs/modules/instances.md)). Emitting
  that shape is what lets the import reuse the existing arrival path instead of
  growing a second materialiser: ``_validate_bundle`` screens these rows like
  any other bundle's, and ``_build_redacted_rows`` mints their ids, fills a
  missing ``ts`` monotonically and redacts the content.

**READ-ONLY.** Nothing here writes, moves or deletes a file in either store.
The import path owns every write, and what it writes is a COPY under a fresh sid
(``session_transfer._write_layer_b_files``), so a browsed session is never the
one mutated — copy-never-move (§14.2) holds for a local adopt exactly as it does
for a peer transfer.

The native record format
------------------------
Every line of ``<sid>.jsonl`` is ``{"kind": ..., "version": "v1", "data": ...}``
and only three kinds occur:

======================  ===================================================
``Prompt``              one user turn. ``data.content`` is a list of blocks;
                        a ``text`` block's ``data`` is the typed message.
                        ``data.meta.timestamp`` is unix epoch seconds — the
                        only timestamp the log carries.
``AssistantMessage``    one model turn. ``data.content`` mixes ``text`` blocks
                        with ``toolUse`` blocks.
``ToolResults``         tool output, keyed back by ``toolUseId``.
======================  ===================================================

:func:`transcript_rows` admits ``Prompt`` → ``user`` and an
``AssistantMessage``'s ``text`` blocks → ``assistant``, and drops the rest. That
is not a simplification for its own sake: ``user``/``assistant`` is exactly the
role set a bundle's ``messages`` may carry (``session_transfer._VISIBLE_ROLES``)
and the set ``context.build_session_replay`` admits, and tool frames reference
local tool state that means nothing once replayed. The model's real tool history
is not lost by this — it rides in Layer B, which travels byte-exact.

``data.meta.additionalContext`` is deliberately NOT read. It carries the context
entries the host injected around the user's message, which the user never typed;
``data.content`` is already just their text.

Threading
---------
Both public functions do **blocking IO** and must be called through
``asyncio.to_thread`` from the gateway. Neither touches the session map: the one
read of shared mutable state is :func:`list_sessions`'s ``mapped_sids``
argument, which the caller resolves on the loop and passes in — the same
contract ``session_transfer._read_layer_b`` keeps, and for the same reason
(``SessionMap`` is an unlocked dict with whole-file writes).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kiro_crew.config.paths import kiro_sessions_dir
from kiro_crew.jsonl_util import RECORD_CAP, bounded_records, open_regular_nofollow
from kiro_crew.session_map import session_files_resumable
from kiro_crew.validation import bounded_session_id

logger = logging.getLogger(__name__)

#: How a session came to exist, read from the envelope's ``session_created_reason``.
#: A session written before that field existed carries none, and every one of
#: those is an interactive ``kiro-cli chat`` — which is why absence maps to
#: :data:`KIND_CLI` rather than to an "unknown" bucket nothing would filter on.
KIND_CLI = "cli"
KIND_SUBAGENT = "subagent"
KIND_REWIND = "rewind"
KINDS: tuple[str, ...] = (KIND_CLI, KIND_SUBAGENT, KIND_REWIND)

#: Envelope bytes past which the metadata is NOT parsed. Both stores are
#: agent-writable, so the envelope is untrusted input: a crafted ``<sid>.json``
#: would otherwise hand ``json.load`` one allocation the size of the whole file.
#: Far above any real envelope (the largest on the reference machine is ~90 KB,
#: and the whole store's envelopes total 14 MB), so hitting this means a hostile
#: or broken file rather than a big conversation. Such a row still LISTS — from
#: its stat alone, with an empty title — because hiding it would make a session
#: the user can see on disk silently absent from the picker.
_ENVELOPE_MAX_BYTES = 8 * 1024 * 1024

#: Replay-log bytes past which a conversion is refused. Memory does not need
#: this — :func:`bounded_records` streams one capped record at a time — but TIME
#: does: the user is waiting on this read behind a picker click, so a file three
#: orders of magnitude larger than any real session (the largest on the
#: reference machine is 3.7 MB) is refused rather than scanned.
_REPLAY_MAX_BYTES = 256 * 1024 * 1024

#: Longest title/cwd kept per row. These are rendered in a picker and travel
#: into an audit line, and the envelope they come from is agent-writable.
_LABEL_CAP = 512

#: Rows :func:`transcript_rows` returns at most, newest kept. A replay log can
#: hold six figures of turns; the import writes every row it is given to the
#: transcript and hydrates a window from it, so an unbounded conversion would
#: turn one picker click into a multi-hundred-MB document. The cap is well above
#: any conversation a person reads back and matches the order of the import
#: path's own window ceiling.
_MAX_ROWS = 20_000

#: Characters kept per converted row. A single turn can carry a pasted file;
#: the display copy is for reading, and Layer B keeps the full text for the
#: model regardless.
_MAX_ROW_CHARS = 200_000

# ---------------------------------------------------------------- scan cache
#
# Parsing every envelope is the whole cost of a listing: 1601 files / 14 MB /
# 0.38s on the reference machine. That is cheap enough to do on an explicit
# picker open and far too expensive to repeat per keystroke, so one pass is kept
# briefly and shared. Filtering is deliberately NOT cached — it is a pure
# function of the cached rows plus the caller's query, so a narrowing search
# re-filters in memory instead of re-reading the store.
#
# Keyed on the store PATH, like ``session_storage._scan_key``: the directory is
# resolved per call (a pod overrides ``KIRO_HOME``), so a process can legitimately
# enumerate different stores over its lifetime and a key without the path would
# answer a question about one store with a pass over another.
_SCAN_CACHE_TTL = 30.0

_scan_lock = threading.Lock()
# (expires_at, store path, rows)
_scan_cache: tuple[float, str, list["KiroCliSession"]] | None = None


@dataclass(frozen=True)
class KiroCliSession:
    """One row of the kiro-cli store, as the picker shows it.

    ``resumable`` is the whole-rule predicate
    :func:`kiro_crew.session_map.session_files_resumable` — the envelope present
    and the event log holding at least one turn. A row that is not resumable is
    still listed, because it exists and the user can see it on disk, but it
    cannot be imported: ``session/load`` would have nothing to restore and the
    map entry would be pruned on the next read.
    """

    sid: str
    title: str
    cwd: str
    created_at: str
    updated_at: str
    bytes: int
    kind: str
    resumable: bool

    def as_wire(self, *, mapped: bool) -> dict[str, Any]:
        """The JSON shape the picker consumes.

        ``mapped`` is not a field of this record because it is not a fact about
        the store: it is whether THIS instance's session map already points a
        conversation at the sid, which only a caller holding the live map can
        answer.
        """
        return {
            "sid": self.sid,
            "title": self.title,
            "cwd": self.cwd,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "bytes": self.bytes,
            "kind": self.kind,
            "resumable": self.resumable,
            "mapped": mapped,
        }


def _iso(value: Any) -> str:
    """An envelope timestamp as a bounded string, or ``""``.

    kiro-cli writes these as ISO-8601 already; they are echoed rather than
    reformatted, because a value this module cannot parse is still more useful
    to a reader than one it silently drops. Bounded like every other
    envelope-sourced label.
    """
    return str(value)[:_LABEL_CAP] if isinstance(value, str) else ""


def _kind_of(reason: Any) -> str:
    """Map ``session_created_reason`` onto :data:`KINDS`."""
    if reason == KIND_SUBAGENT:
        return KIND_SUBAGENT
    if reason == KIND_REWIND:
        return KIND_REWIND
    # Absent, empty, or a value this build does not know: an interactive chat.
    # See KIND_CLI for why absence is not its own bucket.
    return KIND_CLI


def _read_envelope(path: Path, size: int) -> dict[str, Any]:
    """Parse one ``<sid>.json``, or ``{}`` when it cannot be trusted or read.

    Never raises: a listing must degrade to a thinner row rather than fail
    wholesale because one file in a 1600-file store is malformed.
    """
    if size > _ENVELOPE_MAX_BYTES:
        logger.debug("kiro-cli envelope %s is %d bytes; listing it from stat only", path.name, size)
        return {}
    try:
        # ``open_regular_nofollow`` for the reason every other read of these two
        # stores uses it: the directory is agent-writable, so a symlink planted
        # at an envelope's name would otherwise be followed out of the store. The
        # ceiling is passed as well as pre-checked above — it is charged against
        # every read, so a file that GROWS after the stat is still refused.
        with open_regular_nofollow(path, max_bytes=_ENVELOPE_MAX_BYTES) as handle:
            parsed = json.loads(handle.read().decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        logger.debug("kiro-cli envelope %s unreadable", path.name, exc_info=True)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _scan_store(sessions_dir: Path) -> list[KiroCliSession]:
    """Enumerate the store, newest first. **Blocking IO.**"""
    try:
        entries = [
            (e.name, e)
            for e in os.scandir(sessions_dir)
            if e.name.endswith(".json") and e.is_file(follow_symlinks=False)
        ]
    except OSError:
        logger.debug("kiro-cli session store unreadable", exc_info=True)
        return []

    rows: list[KiroCliSession] = []
    for name, entry in entries:
        sid = bounded_session_id(name[: -len(".json")])
        if not sid:
            continue
        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        envelope = _read_envelope(Path(entry.path), stat.st_size)
        log_path = sessions_dir / f"{sid}.jsonl"
        try:
            log_bytes = log_path.stat().st_size
        except OSError:
            log_bytes = 0
        rows.append(
            KiroCliSession(
                sid=sid,
                title=str(envelope.get("title") or "")[:_LABEL_CAP],
                cwd=str(envelope.get("cwd") or "")[:_LABEL_CAP],
                created_at=_iso(envelope.get("created_at")),
                updated_at=_iso(envelope.get("updated_at")),
                bytes=stat.st_size + log_bytes,
                kind=_kind_of(envelope.get("session_created_reason")),
                # The one place the resumability rule is consulted, so the
                # picker's badge and the import's refusal cannot disagree.
                resumable=session_files_resumable(sid),
            )
        )
    # Newest first, by the envelope's own clock where it has one. The mtime is
    # NOT used as the sort key: a session's envelope is rewritten by operations
    # that are not turns, so mtime order and conversation order disagree, and the
    # picker is a list of conversations.
    rows.sort(key=lambda r: (r.updated_at or r.created_at), reverse=True)
    return rows


def invalidate_cache() -> None:
    """Drop any cached pass. Public so a test can start from a known state: the
    cache is process-wide and keyed on the store PATH, not its contents, so a
    test that writes more sessions and re-lists inside the TTL would otherwise be
    answered from its own earlier pass."""
    global _scan_cache
    with _scan_lock:
        _scan_cache = None


def list_sessions(*, cached: bool = True) -> list[KiroCliSession]:
    """Every session in the kiro-cli store, newest first. **Blocking IO.**

    *cached* permits reuse of a recent pass (:data:`_SCAN_CACHE_TTL`). It is for
    the read path only — a listing is a display, and the import endpoint
    re-reads the one sid it was given uncached rather than trusting a row from
    here.
    """
    global _scan_cache
    sessions_dir = kiro_sessions_dir()
    key = str(sessions_dir)
    if cached:
        with _scan_lock:
            if _scan_cache is not None:
                expires, cached_key, rows = _scan_cache
                if time.monotonic() < expires and cached_key == key:
                    return rows
    rows = _scan_store(sessions_dir)
    if cached:
        with _scan_lock:
            _scan_cache = (time.monotonic() + _SCAN_CACHE_TTL, key, rows)
    return rows


def filter_sessions(
    rows: list[KiroCliSession],
    *,
    query: str = "",
    kinds: frozenset[str] | None = None,
    resumable_only: bool = False,
) -> list[KiroCliSession]:
    """Narrow a listing. Pure: no IO, so a search re-filters the cached pass.

    *query* is matched case-insensitively against the title and the cwd, which
    are the two things a person remembers about a conversation they are looking
    for. Deliberately a substring test and not a tokenizer: the titles are a
    session's first message, so a user types a phrase they remember seeing.
    """
    needle = query.strip().lower()
    out = rows
    if kinds is not None:
        out = [r for r in out if r.kind in kinds]
    if resumable_only:
        out = [r for r in out if r.resumable]
    if needle:
        out = [r for r in out if needle in r.title.lower() or needle in r.cwd.lower()]
    return out


def _blocks_text(content: Any, *, want: str) -> str:
    """Concatenate the ``want``-kind blocks of a native ``data.content`` list.

    The content of a record is a LIST of ``{kind, data}`` blocks, so a reader
    that expects a string gets nothing. Returns ``""`` for any other shape,
    which drops the record rather than raising on a malformed log.
    """
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("kind") != want:
            continue
        data = block.get("data")
        if isinstance(data, str) and data:
            parts.append(data)
    return "".join(parts)


def transcript_rows(sid: str, *, max_rows: int = _MAX_ROWS) -> list[dict[str, str]]:
    """One session's replay log as bundle ``messages``. **Blocking IO.**

    Returns ``[{"role", "content", "ts"}]`` with ``role`` in
    ``("user", "assistant")`` — the Layer A shape described in this module's
    docstring. ``ts`` is ISO-8601 UTC for a user turn (the only record that
    carries a timestamp) and ``""`` for a model turn, which the import path
    fills monotonically via ``history.monotonic_transcript_ts``.

    Never raises on content: a record that does not parse, or whose shape this
    build does not recognise, is skipped. An unreadable FILE yields ``[]``, and
    the caller refuses the import rather than creating an empty session.
    """
    safe = bounded_session_id(sid)
    if not safe:
        return []
    path = kiro_sessions_dir() / f"{safe}.jsonl"
    rows: list[dict[str, str]] = []
    try:
        with open_regular_nofollow(path, max_bytes=_REPLAY_MAX_BYTES) as handle:
            # ``bounded_records``, never ``for line in handle``: the log is
            # agent-writable, so one crafted newline-free line would otherwise
            # get a single allocation the size of the whole file. An over-cap
            # record is SKIPPED, which costs that one turn rather than the
            # import.
            for record in bounded_records(handle, path, cap=RECORD_CAP, label="kiro-cli replay"):
                record = record.strip()
                if not record:
                    continue
                try:
                    parsed = json.loads(record)
                except ValueError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                data = parsed.get("data")
                if not isinstance(data, dict):
                    continue
                kind = parsed.get("kind")
                if kind == "Prompt":
                    text = _blocks_text(data.get("content"), want="text")
                    if not text:
                        continue
                    rows.append(
                        {
                            "role": "user",
                            "content": text[:_MAX_ROW_CHARS],
                            "ts": _prompt_ts(data.get("meta")),
                        }
                    )
                elif kind == "AssistantMessage":
                    # ``text`` blocks only: a ``toolUse`` block is a local tool
                    # invocation, which is why the role set stops at two.
                    text = _blocks_text(data.get("content"), want="text")
                    if not text:
                        # A turn that only called a tool has nothing to display.
                        continue
                    rows.append({"role": "assistant", "content": text[:_MAX_ROW_CHARS], "ts": ""})
                # ToolResults: dropped. See the module docstring.
    except (OSError, ValueError):
        logger.debug("kiro-cli replay log for %s unreadable", safe, exc_info=True)
        return []
    # Newest kept, because the tail is the part of a conversation a person is
    # coming back for. Layer B still carries the whole context for the model.
    return rows[-max_rows:]


def _prompt_ts(meta: Any) -> str:
    """A ``Prompt``'s unix-epoch ``meta.timestamp`` as ISO-8601 UTC, or ``""``.

    Bounded against a nonsense value rather than trusted: the envelope is
    agent-writable, and ``datetime.fromtimestamp`` raises on an out-of-range
    number, which would abort a conversion over one bad record.
    """
    if not isinstance(meta, dict):
        return ""
    raw = meta.get("timestamp")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return ""
    try:
        return datetime.fromtimestamp(float(raw), timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""
