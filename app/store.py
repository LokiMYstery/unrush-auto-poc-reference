from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from app.config import DB_PATH, ensure_dirs
from app.schemas import ENERGY_GOALS


def _current_energy_goal(value: str | None) -> str | None:
    # Pre-two-direction deliveries remain in SQLite for history. Their removed
    # category is unavailable, never silently reinterpreted as up or down.
    return value if value in ENERGY_GOALS else None


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


TERMINAL_STATUSES = {"done", "cancelled", "failed"}
GOAL_TERMINAL_STATUSES = {"completed", "cancelled", "failed"}
ACTIVE_STATUSES = {"running", "waiting_playback"}


@dataclass(frozen=True)
class CreateRunResult:
    state: str
    run_id: str | None
    status: str | None = None


@dataclass(frozen=True)
class ReceiptClaimResult:
    state: str
    run_id: str
    status: str


@dataclass(frozen=True)
class GoalPhaseClaim:
    state: str
    attempt: int | None = None


@dataclass(frozen=True)
class GoalAdvanceResult:
    state: str
    next_phase_id: str | None = None
    cached: bool = False


@contextmanager
def connect() -> Any:
    ensure_dirs()
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    finally:
        db.close()


def init_db() -> None:
    with connect() as db:
        db.execute("""create table if not exists retrieval_events (
            id integer primary key autoincrement, run_id text not null,
            kind text not null, payload_json text not null, created_at real not null)""")
        db.execute("""create table if not exists fast_decisions (
            client_session_id text not null, client_action_seq integer not null,
            decision_version integer not null, input_scenes_json text not null,
            response_json text not null, created_at real not null,
            primary key(client_session_id,client_action_seq,decision_version))""")
        db.execute(
            """
            create table if not exists runs (
              run_id text primary key,
              status text not null,
              client_session_id text not null,
              client_action_seq integer not null,
              decision_version integer not null,
              context_json text not null,
              current_playback_json text not null,
              catalog_json text not null,
              selected_playback_unit_id text,
              selected_song_id text,
              receipt_json text,
              created_at real not null,
              updated_at real not null
            )
            """
        )
        db.execute(
            """
            create table if not exists deliveries (
              run_id text not null,
              sequence integer not null,
              phase text not null,
              text text,
              playback_unit_id text,
              voice_id text,
              status text,
              created_at real not null,
              primary key (run_id, sequence)
            )
            """
        )
        if "energy_goal" not in {row[1] for row in db.execute("pragma table_info(deliveries)")}:
            db.execute("alter table deliveries add column energy_goal text")
        db.execute(
            """
            create table if not exists voices (
              voice_id text primary key,
              run_id text,
              phase text,
              path text,
              content_type text not null,
              created_at real not null
            )
            """
        )
        db.execute(
            """
            create table if not exists session_heads (
              client_session_id text primary key,
              client_action_seq integer not null,
              decision_version integer not null,
              run_id text not null,
              updated_at real not null
            )
            """
        )
        db.execute(
            """
            create table if not exists goal_heads (
              client_session_id text primary key,
              client_action_seq integer not null,
              decision_version integer not null,
              goal_id text not null,
              version integer not null,
              updated_at real not null
            )
            """
        )
        db.execute(
            """
            create table if not exists goals (
              goal_id text primary key,
              version integer not null,
              status text not null,
              client_session_id text not null,
              client_action_seq integer not null,
              decision_version integer not null,
              input_text text not null,
              split integer not null,
              language text not null,
              context_json text not null,
              current_playback_json text not null,
              catalog_json text not null,
              title text,
              total_minutes integer,
              active_phase_id text,
              context_snapshot_json text,
              next_delivery_sequence integer not null default 1,
              control_action_seq integer not null default 0,
              created_at real not null,
              updated_at real not null
            )
            """
        )
        goal_columns = {row[1] for row in db.execute("pragma table_info(goals)")}
        if "next_delivery_sequence" not in goal_columns:
            db.execute("alter table goals add column next_delivery_sequence integer not null default 1")
        if "control_action_seq" not in goal_columns:
            db.execute("alter table goals add column control_action_seq integer not null default 0")
        db.execute(
            """
            create table if not exists goal_phases (
              phase_id text primary key,
              goal_id text not null,
              version integer not null,
              phase_index integer not null,
              title text not null,
              duration_minutes integer not null,
              icon_id text not null default 'scene-music-note',
              status text not null,
              energy_goal text,
              run_id text,
              prepared_playback_unit_id text,
              prepared_opening_text text,
              prepared_opening_voice_id text,
              elapsed_seconds real not null default 0,
              is_playing integer not null default 0,
              phase_attempt integer not null default 0,
              context_json text,
              current_playback_json text,
              created_at real not null,
              updated_at real not null
            )
            """
        )
        phase_columns = {row[1] for row in db.execute("pragma table_info(goal_phases)")}
        if "phase_attempt" not in phase_columns:
            db.execute("alter table goal_phases add column phase_attempt integer not null default 0")
        if "prepared_opening_text" not in phase_columns:
            db.execute("alter table goal_phases add column prepared_opening_text text")
        if "prepared_opening_voice_id" not in phase_columns:
            db.execute("alter table goal_phases add column prepared_opening_voice_id text")
        if "icon_id" not in phase_columns:
            db.execute("alter table goal_phases add column icon_id text not null default 'scene-music-note'")
        db.execute(
            """
            create table if not exists goal_deliveries (
              goal_id text not null,
              sequence integer not null,
              version integer not null,
              phase_id text,
              run_id text,
              phase text not null,
              text text,
              playback_unit_id text,
              voice_id text,
              status text,
              energy_goal text,
              created_at real not null,
              primary key (goal_id, sequence)
            )
            """
        )


def record_retrieval_event(run_id: str | None, kind: str, payload: dict[str, Any]) -> None:
    # Standalone tool tests have no owning run. Runtime passes a persisted run and records
    # observed calls/results and public selection rationale, never hidden reasoning items.
    if not run_id:
        return
    with connect() as db:
        row = db.execute("select client_session_id,status from runs where run_id=?", (run_id,)).fetchone()
        if row is None or row["status"] in TERMINAL_STATUSES:
            return
        head = db.execute("select run_id from session_heads where client_session_id=?", (row["client_session_id"],)).fetchone()
        if head is None or head["run_id"] != run_id:
            return
        db.execute("insert into retrieval_events(run_id,kind,payload_json,created_at) values (?,?,?,?)",
                   (run_id, kind, dumps(payload), time.time()))


def latest_retrieval_event(run_id: str, kind: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("select payload_json from retrieval_events where run_id=? and kind=? order by id desc limit 1",
                         (run_id, kind)).fetchone()
        return loads(row["payload_json"], {}) if row else None


def fast_decision(session_id: str, action_seq: int, version: int) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("select response_json from fast_decisions where client_session_id=? and client_action_seq=? and decision_version=?",
                         (session_id, action_seq, version)).fetchone()
        return loads(row["response_json"], {}) if row else None


def record_fast_decision(session_id: str, action_seq: int, version: int,
                         scenes: list[dict[str, Any]], response: dict[str, Any]) -> dict[str, Any]:
    # Store scene inputs and final ordering only: no health/context or credential payload.
    with connect() as db:
        db.execute("insert or ignore into fast_decisions values (?,?,?,?,?,?)",
                   (session_id, action_seq, version, dumps(scenes), dumps(response), time.time()))
        row = db.execute("select response_json from fast_decisions where client_session_id=? and client_action_seq=? and decision_version=?",
                         (session_id, action_seq, version)).fetchone()
        return loads(row["response_json"], {})


def create_run(
    *,
    run_id: str,
    client_session_id: str,
    client_action_seq: int,
    decision_version: int,
    context: dict[str, Any],
    current_playback: dict[str, Any],
    catalog_payload: dict[str, Any],
) -> CreateRunResult:
    now = time.time()
    ensure_dirs()
    db = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute("begin immediate")
        head = db.execute("select * from session_heads where client_session_id=?", (client_session_id,)).fetchone()
        if head is not None:
            head_seq = int(head["client_action_seq"])
            head_version = int(head["decision_version"])
            head_run_id = str(head["run_id"])
            if client_action_seq < head_seq:
                db.execute("commit")
                return CreateRunResult("stale", head_run_id)
            if client_action_seq == head_seq:
                if decision_version == head_version:
                    existing = db.execute("select status from runs where run_id=?", (head_run_id,)).fetchone()
                    db.execute("commit")
                    return CreateRunResult("duplicate", head_run_id, str(existing["status"]) if existing else None)
                db.execute("commit")
                return CreateRunResult("conflict", head_run_id)
            prior = db.execute("select status from runs where run_id=?", (head_run_id,)).fetchone()
            if prior is not None and str(prior["status"]) in ACTIVE_STATUSES:
                db.execute("update runs set status=?,updated_at=? where run_id=?", ("cancelled", now, head_run_id))
        db.execute(
            """
            insert into runs (
              run_id,status,client_session_id,client_action_seq,decision_version,
              context_json,current_playback_json,catalog_json,created_at,updated_at
            ) values (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                "running",
                client_session_id,
                client_action_seq,
                decision_version,
                dumps(context),
                dumps(current_playback),
                dumps(catalog_payload),
                now,
                now,
            ),
        )
        db.execute(
            """
            insert into session_heads (client_session_id,client_action_seq,decision_version,run_id,updated_at)
            values (?,?,?,?,?)
            on conflict(client_session_id) do update set
              client_action_seq=excluded.client_action_seq,
              decision_version=excluded.decision_version,
              run_id=excluded.run_id,
              updated_at=excluded.updated_at
            """,
            (client_session_id, client_action_seq, decision_version, run_id, now),
        )
        db.execute("commit")
        return CreateRunResult("created", run_id, "running")
    except Exception:
        db.execute("rollback")
        raise
    finally:
        db.close()


def get_session_head(client_session_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("select * from session_heads where client_session_id=?", (client_session_id,)).fetchone()
    return dict(row) if row else None


def classify_action(client_session_id: str, client_action_seq: int, decision_version: int) -> tuple[str, str | None]:
    head = get_session_head(client_session_id)
    if head is None:
        return "new", None
    head_seq = int(head["client_action_seq"])
    head_version = int(head["decision_version"])
    if client_action_seq < head_seq:
        return "stale", str(head["run_id"])
    if client_action_seq == head_seq:
        if decision_version == head_version:
            return "duplicate", str(head["run_id"])
        return "conflict", str(head["run_id"])
    return "new", str(head["run_id"])


def run_is_session_head(run: dict[str, Any]) -> bool:
    head = get_session_head(str(run["client_session_id"]))
    return bool(head and head["run_id"] == run["run_id"])


def get_run(run_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("select * from runs where run_id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def update_run(run_id: str, **fields: Any) -> bool:
    if not fields:
        return False
    fields["updated_at"] = time.time()
    assignments = ", ".join(f"{name}=?" for name in fields)
    values = list(fields.values()) + [run_id]
    with connect() as db:
        row = db.execute("select * from runs where run_id=?", (run_id,)).fetchone()
        if row is None:
            return False
        head = db.execute("select run_id from session_heads where client_session_id=?", (row["client_session_id"],)).fetchone()
        if head is not None and head["run_id"] != run_id:
            return False
        current_status = str(row["status"])
        requested_status = str(fields.get("status") or current_status)
        if current_status in TERMINAL_STATUSES and requested_status != current_status:
            return False
        if current_status == "cancelled" and fields.get("status") == "cancelled":
            return True
        db.execute(f"update runs set {assignments} where run_id=?", values)
        return db.total_changes > 0


def add_delivery(
    *,
    run_id: str,
    phase: str,
    text: str | None = None,
    playback_unit_id: str | None = None,
    voice_id: str | None = None,
    status: str | None = None,
    energy_goal: str | None = None,
) -> dict[str, Any]:
    if energy_goal is not None and energy_goal not in ENERGY_GOALS:
        raise ValueError("invalid_energy_goal")
    now = time.time()
    with connect() as db:
        row = db.execute("select * from runs where run_id=?", (run_id,)).fetchone()
        if row is None or str(row["status"]) in TERMINAL_STATUSES:
            return {}
        head = db.execute("select run_id from session_heads where client_session_id=?", (row["client_session_id"],)).fetchone()
        if head is not None and head["run_id"] != run_id:
            return {}
        sequence_row = db.execute("select coalesce(max(sequence),0)+1 as sequence from deliveries where run_id=?", (run_id,)).fetchone()
        sequence = int(sequence_row["sequence"])
        db.execute(
            """
            insert into deliveries (run_id,sequence,phase,text,playback_unit_id,voice_id,status,created_at,energy_goal)
            values (?,?,?,?,?,?,?,?,?)
            """,
            (run_id, sequence, phase, text, playback_unit_id, voice_id, status, now, energy_goal),
        )
    return {
        "sequence": sequence,
        "phase": phase,
        "text": text,
        "playbackUnitId": playback_unit_id,
        "voiceId": voice_id,
        "status": status,
        "energyGoal": energy_goal,
    }


def add_delivery_once_phase(
    *,
    run_id: str,
    phase: str,
    text: str | None = None,
    playback_unit_id: str | None = None,
    voice_id: str | None = None,
    status: str | None = None,
    energy_goal: str | None = None,
) -> dict[str, Any]:
    if energy_goal is not None and energy_goal not in ENERGY_GOALS:
        raise ValueError("invalid_energy_goal")
    now = time.time()
    with connect() as db:
        row = db.execute("select * from runs where run_id=?", (run_id,)).fetchone()
        if row is None or str(row["status"]) in TERMINAL_STATUSES:
            return {}
        head = db.execute("select run_id from session_heads where client_session_id=?", (row["client_session_id"],)).fetchone()
        if head is not None and head["run_id"] != run_id:
            return {}
        existing = db.execute("select sequence,phase,text,playback_unit_id,voice_id,status,energy_goal from deliveries where run_id=? and phase=? order by sequence asc limit 1", (run_id, phase)).fetchone()
        if existing is not None:
            return {
                "sequence": int(existing["sequence"]),
                "phase": existing["phase"],
                "text": existing["text"],
                "playbackUnitId": existing["playback_unit_id"],
                "voiceId": existing["voice_id"],
                "status": existing["status"],
                "energyGoal": _current_energy_goal(existing["energy_goal"]),
            }
        sequence_row = db.execute("select coalesce(max(sequence),0)+1 as sequence from deliveries where run_id=?", (run_id,)).fetchone()
        sequence = int(sequence_row["sequence"])
        db.execute(
            """
            insert into deliveries (run_id,sequence,phase,text,playback_unit_id,voice_id,status,created_at,energy_goal)
            values (?,?,?,?,?,?,?,?,?)
            """,
            (run_id, sequence, phase, text, playback_unit_id, voice_id, status, now, energy_goal),
        )
    return {
        "sequence": sequence,
        "phase": phase,
        "text": text,
        "playbackUnitId": playback_unit_id,
        "voiceId": voice_id,
        "status": status,
        "energyGoal": energy_goal,
    }


def deliveries_after(run_id: str, after: int) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            select sequence,phase,text,playback_unit_id,voice_id,status,energy_goal
            from deliveries where run_id=? and sequence>? order by sequence asc
            """,
            (run_id, after),
        ).fetchall()
    return [
        {
            "sequence": int(row["sequence"]),
            "phase": row["phase"],
            "text": row["text"],
            "playbackUnitId": row["playback_unit_id"],
            "voiceId": row["voice_id"],
            "status": row["status"],
            "energyGoal": _current_energy_goal(row["energy_goal"]),
        }
        for row in rows
    ]


def claim_receipt(
    *,
    run_id: str,
    client_session_id: str,
    client_action_seq: int,
    decision_version: int,
    receipt_payload: dict[str, Any],
) -> ReceiptClaimResult:
    now = time.time()
    with connect() as db:
        row = db.execute("select * from runs where run_id=?", (run_id,)).fetchone()
        if row is None:
            return ReceiptClaimResult("missing", run_id, "missing")
        status = str(row["status"])
        if not is_current(dict(row), client_session_id, client_action_seq, decision_version):
            return ReceiptClaimResult("stale", run_id, status)
        head = db.execute("select run_id from session_heads where client_session_id=?", (row["client_session_id"],)).fetchone()
        if head is not None and head["run_id"] != run_id:
            return ReceiptClaimResult("stale", run_id, status)
        existing = loads(row["receipt_json"], {})
        if existing:
            return ReceiptClaimResult("duplicate" if existing == receipt_payload else "conflict", run_id, status)
        if status == "cancelled":
            return ReceiptClaimResult("cancelled", run_id, status)
        if status != "waiting_playback":
            return ReceiptClaimResult("not_waiting", run_id, status)
        db.execute("update runs set receipt_json=?,updated_at=? where run_id=? and receipt_json is null and status=?", (dumps(receipt_payload), now, run_id, "waiting_playback"))
        if db.total_changes <= 0:
            refreshed = db.execute("select status,receipt_json from runs where run_id=?", (run_id,)).fetchone()
            if refreshed is None:
                return ReceiptClaimResult("missing", run_id, "missing")
            existing_after = loads(refreshed["receipt_json"], {})
            return ReceiptClaimResult("duplicate" if existing_after == receipt_payload else "conflict", run_id, str(refreshed["status"]))
        return ReceiptClaimResult("claimed", run_id, status)


def list_runs_for_recovery() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            select * from runs
            where status in ('running','waiting_playback')
            order by created_at asc
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delivery_exists(run_id: str, phase: str) -> bool:
    with connect() as db:
        row = db.execute("select 1 from deliveries where run_id=? and phase=? limit 1", (run_id, phase)).fetchone()
    return row is not None


def latest_delivery(run_id: str, phase: str, statuses: set[str] | None = None) -> dict[str, Any] | None:
    with connect() as db:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            rows = db.execute(
                f"""
                select sequence,phase,text,playback_unit_id,voice_id,status,energy_goal
                from deliveries where run_id=? and phase=? and status in ({placeholders})
                order by sequence desc limit 1
                """,
                (run_id, phase, *statuses),
            ).fetchall()
        else:
            rows = db.execute(
                """
                select sequence,phase,text,playback_unit_id,voice_id,status,energy_goal
                from deliveries where run_id=? and phase=?
                order by sequence desc limit 1
                """,
                (run_id, phase),
            ).fetchall()
    if not rows:
        return None
    row = rows[0]
    return {
        "sequence": int(row["sequence"]),
        "phase": row["phase"],
        "text": row["text"],
        "playbackUnitId": row["playback_unit_id"],
        "voiceId": row["voice_id"],
        "status": row["status"],
        "energyGoal": _current_energy_goal(row["energy_goal"]),
    }


def latest_delivered_environment_narration(client_session_id: str) -> dict[str, Any] | None:
    # A delivered assessment survives cancellation or later playback failure. No audio-played
    # acknowledgement exists yet, so this memory records delivery, not verified listening.
    with connect() as db:
        row = db.execute("""select r.context_json,d.text from runs r join deliveries d on d.run_id=r.run_id
            where r.client_session_id=? and d.phase='ASSESSMENT' and d.text is not null
            and d.status in ('voice_ready','text_only') order by d.created_at desc,r.client_action_seq desc,d.sequence desc limit 1""", (client_session_id,)).fetchone()
    if not row:
        return None
    context = loads(row["context_json"], {})
    return {"spokenAssessment": row["text"],
            "environmentNarrationSignature": (context.get("environmentNarration") or {}).get("signature")}


def completed_session_summaries(client_session_id: str, limit: int = 3) -> list[dict[str, Any]]:
    with connect() as db:
        runs = db.execute(
            """
            select * from runs
            where client_session_id=? and status='done'
            order by updated_at desc limit ?
            """,
            (client_session_id, limit),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in runs:
            run = dict(row)
            deliveries = db.execute(
                """
                select phase,text,playback_unit_id,status,sequence
                from deliveries where run_id=? order by sequence asc
                """,
                (run["run_id"],),
            ).fetchall()
            assessment = next((item["text"] for item in deliveries if item["phase"] == "ASSESSMENT" and item["text"]), None)
            explanations = [item["text"] for item in deliveries if item["phase"] == "EXPLANATION" and item["text"]]
            selection = next((item for item in deliveries if item["phase"] == "SELECTION"), None)
            receipt = loads(run.get("receipt_json"), {})
            current_playback = loads(run.get("current_playback_json"), {})
            context = loads(run.get("context_json"), {})
            result.append({
                "runId": run["run_id"],
                "clientActionSeq": int(run["client_action_seq"]),
                "decisionVersion": int(run["decision_version"]),
                "spokenAssessment": assessment,
                "energyGoal": _current_energy_goal(context.get("energyGoal")),
                "environmentNarrationSignature": (context.get("environmentNarration") or {}).get("signature"),
                "spokenExplanation": explanations[-1] if explanations else None,
                "selectedPlaybackUnitId": run.get("selected_playback_unit_id") or (selection["playback_unit_id"] if selection else None),
                "selectedSongId": run.get("selected_song_id"),
                "actualPlayback": {
                    "playbackUnitId": receipt.get("playbackUnitId"),
                    "songId": receipt.get("songId"),
                    "success": receipt.get("success"),
                    "changeKind": receipt.get("changeKind"),
                    "actualPlaybackState": receipt.get("actualPlaybackState"),
                },
                "userFeedback": receipt.get("userFeedback"),
                "previousCurrentPlayback": {
                    "playbackUnitId": current_playback.get("playbackUnitId"),
                    "songId": current_playback.get("songId"),
                    "sceneId": current_playback.get("sceneId"),
                },
            })
    return list(reversed(result))


def record_voice(*, voice_id: str, run_id: str | None, phase: str | None, path: str, content_type: str) -> None:
    with connect() as db:
        db.execute(
            "insert or replace into voices (voice_id,run_id,phase,path,content_type,created_at) values (?,?,?,?,?,?)",
            (voice_id, run_id, phase, path, content_type, time.time()),
        )


def get_voice(voice_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("select * from voices where voice_id=?", (voice_id,)).fetchone()
    return dict(row) if row else None


def is_current(run: dict[str, Any], client_session_id: str, client_action_seq: int, decision_version: int) -> bool:
    return (
        run["client_session_id"] == client_session_id
        and int(run["client_action_seq"]) == int(client_action_seq)
        and int(run["decision_version"]) == int(decision_version)
    )


def create_goal(
    *,
    goal_id: str,
    version: int,
    client_session_id: str,
    client_action_seq: int,
    decision_version: int,
    input_text: str,
    split: bool,
    language: str,
    context: dict[str, Any],
    current_playback: dict[str, Any],
    catalog_payload: dict[str, Any],
    context_snapshot: dict[str, Any],
) -> CreateRunResult:
    now = time.time()
    db = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute("begin immediate")
        head = db.execute("select * from goal_heads where client_session_id=?", (client_session_id,)).fetchone()
        if head is not None:
            head_seq = int(head["client_action_seq"])
            head_version = int(head["decision_version"])
            head_goal_id = str(head["goal_id"])
            if client_action_seq < head_seq:
                db.execute("commit")
                return CreateRunResult("stale", head_goal_id)
            if client_action_seq == head_seq:
                if decision_version == head_version:
                    existing = db.execute("select status from goals where goal_id=?", (head_goal_id,)).fetchone()
                    db.execute("commit")
                    return CreateRunResult("duplicate", head_goal_id, str(existing["status"]) if existing else None)
                db.execute("commit")
                return CreateRunResult("conflict", head_goal_id)
            _cancel_goal_locked(db, head_goal_id, now)
        db.execute(
            """
            insert into goals (
              goal_id,version,status,client_session_id,client_action_seq,decision_version,
              input_text,split,language,context_json,current_playback_json,catalog_json,
              context_snapshot_json,next_delivery_sequence,control_action_seq,created_at,updated_at
            ) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                goal_id,
                version,
                "planning",
                client_session_id,
                client_action_seq,
                decision_version,
                input_text,
                1 if split else 0,
                language,
                dumps(context),
                dumps(current_playback),
                dumps(catalog_payload),
                dumps(context_snapshot),
                1,
                client_action_seq,
                now,
                now,
            ),
        )
        db.execute(
            """
            insert into goal_heads (client_session_id,client_action_seq,decision_version,goal_id,version,updated_at)
            values (?,?,?,?,?,?)
            on conflict(client_session_id) do update set
              client_action_seq=excluded.client_action_seq,
              decision_version=excluded.decision_version,
              goal_id=excluded.goal_id,
              version=excluded.version,
              updated_at=excluded.updated_at
            """,
            (client_session_id, client_action_seq, decision_version, goal_id, version, now),
        )
        db.execute("commit")
        return CreateRunResult("created", goal_id, "planning")
    except Exception:
        db.execute("rollback")
        raise
    finally:
        db.close()


def _cancel_goal_locked(db: sqlite3.Connection, goal_id: str, now: float) -> None:
    goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
    if goal is None or str(goal["status"]) in GOAL_TERMINAL_STATUSES:
        return
    db.execute("update goals set status='cancelled',updated_at=? where goal_id=?", (now, goal_id))
    rows = db.execute("select run_id from goal_phases where goal_id=? and run_id is not null", (goal_id,)).fetchall()
    for row in rows:
        db.execute("update runs set status='cancelled',updated_at=? where run_id=? and status in ('running','waiting_playback')",
                   (now, row["run_id"]))
    db.execute("update goal_phases set status='cancelled',updated_at=? where goal_id=? and status not in ('done','cancelled','failed')",
               (now, goal_id))


def get_goal(goal_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
    return dict(row) if row else None


def goal_is_head(goal: dict[str, Any]) -> bool:
    with connect() as db:
        head = db.execute("select goal_id,version from goal_heads where client_session_id=?",
                          (goal["client_session_id"],)).fetchone()
    return bool(head and head["goal_id"] == goal["goal_id"] and int(head["version"]) == int(goal["version"]))


def _goal_is_head_locked(db: sqlite3.Connection, goal: sqlite3.Row | dict[str, Any]) -> bool:
    head = db.execute("select goal_id,version from goal_heads where client_session_id=?",
                      (goal["client_session_id"],)).fetchone()
    return bool(head and head["goal_id"] == goal["goal_id"] and int(head["version"]) == int(goal["version"]))


def update_goal(goal_id: str, **fields: Any) -> bool:
    if not fields:
        return False
    fields["updated_at"] = time.time()
    assignments = ", ".join(f"{name}=?" for name in fields)
    values = list(fields.values()) + [goal_id]
    with connect() as db:
        row = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        if row is None or not goal_is_head(dict(row)):
            return False
        current_status = str(row["status"])
        requested_status = str(fields.get("status") or current_status)
        if current_status in GOAL_TERMINAL_STATUSES and requested_status != current_status:
            return False
        db.execute(f"update goals set {assignments} where goal_id=?", values)
        return db.total_changes > 0


def update_goal_context_snapshot_expected(
    goal_id: str,
    *,
    version: int,
    active_phase_id: str,
    run_id: str,
    attempt: int,
    context_snapshot: dict[str, Any],
) -> bool:
    now = time.time()
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        phase = db.execute(
            "select * from goal_phases where goal_id=? and phase_id=? and run_id=? and phase_attempt=?",
            (goal_id, active_phase_id, run_id, attempt),
        ).fetchone()
        if (goal is None or phase is None or int(goal["version"]) != int(version)
                or str(goal["status"]) in GOAL_TERMINAL_STATUSES or not _goal_is_head_locked(db, goal)):
            return False
        if str(goal["active_phase_id"] or "") != str(active_phase_id):
            return False
        db.execute(
            "update goals set context_snapshot_json=?,updated_at=? where goal_id=? and version=? and active_phase_id=?",
            (dumps(context_snapshot), now, goal_id, version, active_phase_id),
        )
        return db.total_changes > 0


def claim_goal_control_action(goal_id: str, *, version: int, client_action_seq: int) -> CreateRunResult:
    now = time.time()
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        if goal is None:
            return CreateRunResult("missing", goal_id)
        if int(goal["version"]) != int(version) or not _goal_is_head_locked(db, goal):
            return CreateRunResult("stale", goal_id)
        if str(goal["status"]) in GOAL_TERMINAL_STATUSES:
            return CreateRunResult("cancelled", goal_id, str(goal["status"]))
        current_seq = int(goal["control_action_seq"] or goal["client_action_seq"] or 0)
        if int(client_action_seq) <= current_seq:
            return CreateRunResult("stale", goal_id, str(goal["status"]))
        db.execute(
            "update goals set control_action_seq=?,updated_at=? where goal_id=? and control_action_seq=?",
            (int(client_action_seq), now, goal_id, current_seq),
        )
        if db.total_changes <= 0:
            return CreateRunResult("stale", goal_id, str(goal["status"]))
        return CreateRunResult("created", goal_id, str(goal["status"]))


def set_goal_plan(goal_id: str, *, title: str, total_minutes: int,
                  phases: list[dict[str, Any]], active_phase_id: str) -> bool:
    now = time.time()
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        if goal is None or str(goal["status"]) != "planning":
            return False
        if not goal_is_head(dict(goal)):
            return False
        db.execute("delete from goal_phases where goal_id=?", (goal_id,))
        for phase in phases:
            db.execute(
                """
                insert into goal_phases (
                  phase_id,goal_id,version,phase_index,title,duration_minutes,icon_id,status,created_at,updated_at
                ) values (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    phase["phaseId"],
                    goal_id,
                    int(goal["version"]),
                    int(phase["index"]),
                    str(phase["title"]),
                    int(phase["durationMinutes"]),
                    str(phase.get("iconId") or "scene-music-note"),
                    "planned",
                    now,
                    now,
                ),
            )
        db.execute(
            "update goals set title=?,total_minutes=?,active_phase_id=?,status='active',updated_at=? where goal_id=?",
            (title, total_minutes, active_phase_id, now, goal_id),
        )
        return True


def goal_phases(goal_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("select * from goal_phases where goal_id=? order by phase_index asc", (goal_id,)).fetchall()
    return [dict(row) for row in rows]


def get_goal_phase(goal_id: str, phase_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("select * from goal_phases where goal_id=? and phase_id=?", (goal_id, phase_id)).fetchone()
    return dict(row) if row else None


def update_goal_phase(goal_id: str, phase_id: str, **fields: Any) -> bool:
    if not fields:
        return False
    fields["updated_at"] = time.time()
    assignments = ", ".join(f"{name}=?" for name in fields)
    values = list(fields.values()) + [goal_id, phase_id]
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        if goal is None or not goal_is_head(dict(goal)):
            return False
        db.execute(f"update goal_phases set {assignments} where goal_id=? and phase_id=?", values)
        return db.total_changes > 0


def update_goal_phase_progress(goal_id: str, phase_id: str, *, elapsed_seconds: float, is_playing: bool) -> bool:
    now = time.time()
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        if goal is None or str(goal["status"]) in GOAL_TERMINAL_STATUSES or not _goal_is_head_locked(db, goal):
            return False
        db.execute(
            """
            update goal_phases
            set elapsed_seconds=max(coalesce(elapsed_seconds,0),?),is_playing=?,updated_at=?
            where goal_id=? and phase_id=?
            """,
            (float(elapsed_seconds), 1 if is_playing else 0, now, goal_id, phase_id),
        )
        return db.total_changes > 0


def advance_goal_phase(goal_id: str, *, version: int, current_phase_id: str, client_action_seq: int) -> GoalAdvanceResult:
    now = time.time()
    db = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute("begin immediate")
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        current = db.execute("select * from goal_phases where goal_id=? and phase_id=?", (goal_id, current_phase_id)).fetchone()
        if goal is None or current is None:
            db.execute("commit")
            return GoalAdvanceResult("missing")
        if int(goal["version"]) != int(version) or not _goal_is_head_locked(db, goal):
            db.execute("commit")
            return GoalAdvanceResult("stale")
        if str(goal["status"]) in GOAL_TERMINAL_STATUSES:
            db.execute("commit")
            return GoalAdvanceResult("terminal")
        current_seq = int(goal["control_action_seq"] or goal["client_action_seq"] or 0)
        if int(client_action_seq) <= current_seq:
            db.execute("commit")
            return GoalAdvanceResult("stale_control")
        if str(goal["active_phase_id"] or "") != str(current_phase_id):
            db.execute("commit")
            return GoalAdvanceResult("inactive")
        if str(current["status"]) != "active":
            db.execute("commit")
            return GoalAdvanceResult("phase_not_active")
        next_phase = db.execute(
            """
            select * from goal_phases
            where goal_id=? and phase_index>?
            order by phase_index asc limit 1
            """,
            (goal_id, int(current["phase_index"])),
        ).fetchone()
        db.execute(
            "update goal_phases set status='done',is_playing=0,updated_at=? where goal_id=? and phase_id=? and status='active'",
            (now, goal_id, current_phase_id),
        )
        if db.total_changes <= 0:
            db.execute("commit")
            return GoalAdvanceResult("phase_not_active")
        if next_phase is None:
            db.execute(
                "update goals set status='completed',active_phase_id=null,control_action_seq=?,updated_at=? where goal_id=? and active_phase_id=?",
                (int(client_action_seq), now, goal_id, current_phase_id),
            )
            db.execute("commit")
            return GoalAdvanceResult("completed")
        next_phase_id = str(next_phase["phase_id"])
        db.execute(
            """
            update goals set active_phase_id=?,control_action_seq=?,updated_at=? where goal_id=? and active_phase_id=?
            """,
            (next_phase_id, int(client_action_seq), now, goal_id, current_phase_id),
        )
        if db.total_changes <= 0:
            db.execute("rollback")
            return GoalAdvanceResult("stale")
        cache_ready = (str(next_phase["status"]) == "prepared" and bool(next_phase["run_id"])
                       and bool(next_phase["prepared_playback_unit_id"])
                       and bool(next_phase["prepared_opening_text"]))
        if cache_ready:
            db.execute(
                """
                update goal_phases set status='waiting_playback',is_playing=0,elapsed_seconds=0,updated_at=?
                where goal_id=? and phase_id=? and status='prepared' and run_id is not null
                """,
                (now, goal_id, next_phase_id),
            )
            if db.total_changes <= 0:
                db.execute("rollback")
                return GoalAdvanceResult("stale")
            db.execute("commit")
            return GoalAdvanceResult("advanced", next_phase_id, True)
        db.execute(
            """
            update goal_phases set
              status='planned',run_id=null,prepared_playback_unit_id=null,prepared_opening_text=null,
              prepared_opening_voice_id=null,energy_goal=null,
              phase_attempt=coalesce(phase_attempt,0)+1,updated_at=?
            where goal_id=? and phase_id=? and status not in ('done','cancelled')
            """,
            (now, goal_id, next_phase_id),
        )
        if db.total_changes <= 0:
            db.execute("rollback")
            return GoalAdvanceResult("stale")
        db.execute("commit")
        return GoalAdvanceResult("advanced", next_phase_id, False)
    except Exception:
        db.execute("rollback")
        raise
    finally:
        db.close()


def begin_goal_phase_retry(goal_id: str, *, version: int, phase_id: str, client_action_seq: int) -> GoalAdvanceResult:
    now = time.time()
    db = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute("begin immediate")
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        phase = db.execute("select * from goal_phases where goal_id=? and phase_id=?", (goal_id, phase_id)).fetchone()
        if goal is None or phase is None:
            db.execute("commit")
            return GoalAdvanceResult("missing")
        if int(goal["version"]) != int(version) or not _goal_is_head_locked(db, goal):
            db.execute("commit")
            return GoalAdvanceResult("stale")
        if str(goal["status"]) in GOAL_TERMINAL_STATUSES:
            db.execute("commit")
            return GoalAdvanceResult("terminal")
        current_seq = int(goal["control_action_seq"] or goal["client_action_seq"] or 0)
        if int(client_action_seq) <= current_seq:
            db.execute("commit")
            return GoalAdvanceResult("stale_control")
        if str(goal["active_phase_id"] or "") != str(phase_id):
            db.execute("commit")
            return GoalAdvanceResult("inactive")
        old_run_id = phase["run_id"]
        if old_run_id:
            db.execute("update runs set status='cancelled',updated_at=? where run_id=? and status in ('running','waiting_playback')",
                       (now, old_run_id))
            db.execute("delete from goal_deliveries where goal_id=? and phase_id=? and run_id=?",
                       (goal_id, phase_id, old_run_id))
        db.execute(
            """
            update goal_phases set
              status='preparing',run_id=null,prepared_playback_unit_id=null,prepared_opening_text=null,
              prepared_opening_voice_id=null,energy_goal=null,
              phase_attempt=coalesce(phase_attempt,0)+1,updated_at=?
            where goal_id=? and phase_id=? and status not in ('done','cancelled')
            """,
            (now, goal_id, phase_id),
        )
        if db.total_changes <= 0:
            db.execute("commit")
            return GoalAdvanceResult("phase_not_active")
        db.execute("update goals set control_action_seq=?,updated_at=? where goal_id=?",
                   (int(client_action_seq), now, goal_id))
        db.execute("commit")
        return GoalAdvanceResult("advanced", phase_id)
    except Exception:
        db.execute("rollback")
        raise
    finally:
        db.close()


def claim_goal_phase_attempt(
    *,
    goal_id: str,
    phase_id: str,
    version: int,
    run_id: str,
    context: dict[str, Any],
    current_playback: dict[str, Any],
    expected_attempt: int | None = None,
) -> GoalPhaseClaim:
    now = time.time()
    db = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute("begin immediate")
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        phase = db.execute("select * from goal_phases where goal_id=? and phase_id=?", (goal_id, phase_id)).fetchone()
        if goal is None or phase is None:
            db.execute("commit")
            return GoalPhaseClaim("missing")
        if int(goal["version"]) != int(version) or not _goal_is_head_locked(db, goal):
            db.execute("commit")
            return GoalPhaseClaim("stale")
        if str(goal["status"]) in GOAL_TERMINAL_STATUSES or str(phase["status"]) in {"done", "cancelled"}:
            db.execute("commit")
            return GoalPhaseClaim("closed")
        old_run_id = phase["run_id"]
        old_status = str(phase["status"])
        current_attempt = int(phase["phase_attempt"] or 0)
        if expected_attempt is not None and current_attempt != int(expected_attempt):
            db.execute("commit")
            return GoalPhaseClaim("stale")
        attempt = current_attempt + 1
        db.execute(
            """
            update goal_phases set
              status='preparing',
              run_id=?,
              phase_attempt=?,
              prepared_playback_unit_id=null,
              prepared_opening_text=null,
              prepared_opening_voice_id=null,
              energy_goal=null,
              context_json=?,
              current_playback_json=?,
              updated_at=?
            where goal_id=? and phase_id=? and phase_attempt=?
            """,
            (run_id, attempt, dumps(context), dumps(current_playback), now,
             goal_id, phase_id, current_attempt),
        )
        if db.total_changes <= 0:
            db.execute("commit")
            return GoalPhaseClaim("stale")
        if old_run_id and old_status in {"preparing", "waiting_playback"}:
            db.execute("update runs set status='cancelled',updated_at=? where run_id=? and status in ('running','waiting_playback')",
                       (now, old_run_id))
        db.execute("commit")
        return GoalPhaseClaim("claimed", attempt)
    except Exception:
        db.execute("rollback")
        raise
    finally:
        db.close()


def update_goal_phase_expected(goal_id: str, phase_id: str, *, run_id: str, attempt: int, **fields: Any) -> bool:
    if not fields:
        return False
    fields["updated_at"] = time.time()
    assignments = ", ".join(f"{name}=?" for name in fields)
    values = list(fields.values()) + [goal_id, phase_id, run_id, attempt]
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        if goal is None or str(goal["status"]) in GOAL_TERMINAL_STATUSES or not _goal_is_head_locked(db, goal):
            return False
        db.execute(
            f"update goal_phases set {assignments} where goal_id=? and phase_id=? and run_id=? and phase_attempt=?",
            values,
        )
        return db.total_changes > 0


def goal_phase_matches(goal_id: str, phase_id: str, *, run_id: str, attempt: int) -> bool:
    with connect() as db:
        row = db.execute(
            "select 1 from goal_phases where goal_id=? and phase_id=? and run_id=? and phase_attempt=?",
            (goal_id, phase_id, run_id, attempt),
        ).fetchone()
    return row is not None


def bump_goal_phase_attempt(goal_id: str, phase_id: str) -> bool:
    now = time.time()
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        if goal is None or str(goal["status"]) in GOAL_TERMINAL_STATUSES or not _goal_is_head_locked(db, goal):
            return False
        db.execute(
            """
            update goal_phases set phase_attempt=coalesce(phase_attempt,0)+1,updated_at=?
            where goal_id=? and phase_id=?
            """,
            (now, goal_id, phase_id),
        )
        return db.total_changes > 0


def cancel_goal(goal_id: str, version: int) -> bool:
    now = time.time()
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        if goal is None or int(goal["version"]) != int(version) or str(goal["status"]) in GOAL_TERMINAL_STATUSES:
            return False
        if not goal_is_head(dict(goal)):
            return False
        _cancel_goal_locked(db, goal_id, now)
        return True


def goal_deliveries_after(goal_id: str, after: int) -> list[dict[str, Any]]:
    with connect() as db:
        goal = db.execute("select status from goals where goal_id=?", (goal_id,)).fetchone()
        if goal is not None and str(goal["status"]) == "cancelled":
            return []
        rows = db.execute(
            """
            select * from goal_deliveries where goal_id=? and sequence>? order by sequence asc
            """,
            (goal_id, after),
        ).fetchall()
    return [_goal_delivery_dict(row) for row in rows]


def delete_goal_phase_deliveries(goal_id: str, phase_id: str, run_id: str | None = None) -> None:
    with connect() as db:
        if run_id:
            db.execute("delete from goal_deliveries where goal_id=? and phase_id=? and run_id=?",
                       (goal_id, phase_id, run_id))
        else:
            db.execute("delete from goal_deliveries where goal_id=? and phase_id=?", (goal_id, phase_id))


def add_goal_delivery(
    *,
    goal_id: str,
    version: int,
    phase: str,
    phase_id: str | None = None,
    run_id: str | None = None,
    text: str | None = None,
    playback_unit_id: str | None = None,
    voice_id: str | None = None,
    status: str | None = None,
    energy_goal: str | None = None,
) -> dict[str, Any]:
    if energy_goal is not None and energy_goal not in ENERGY_GOALS:
        raise ValueError("invalid_energy_goal")
    now = time.time()
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        if goal is None or int(goal["version"]) != int(version) or str(goal["status"]) in GOAL_TERMINAL_STATUSES:
            return {}
        if not goal_is_head(dict(goal)):
            return {}
        seq = int(goal["next_delivery_sequence"] or 1)
        db.execute("update goals set next_delivery_sequence=?,updated_at=? where goal_id=? and next_delivery_sequence=?",
                   (seq + 1, now, goal_id, seq))
        if db.total_changes <= 0:
            return {}
        db.execute(
            """
            insert into goal_deliveries (
              goal_id,sequence,version,phase_id,run_id,phase,text,playback_unit_id,voice_id,status,energy_goal,created_at
            ) values (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (goal_id, seq, version, phase_id, run_id, phase, text, playback_unit_id, voice_id, status, energy_goal, now),
        )
    return {
        "sequence": seq,
        "goalId": goal_id,
        "version": version,
        "phaseId": phase_id,
        "runId": run_id,
        "phase": phase,
        "text": text,
        "playbackUnitId": playback_unit_id,
        "voiceId": voice_id,
        "status": status,
        "energyGoal": energy_goal,
    }


def add_goal_delivery_once(
    *,
    goal_id: str,
    version: int,
    phase_id: str | None,
    run_id: str | None,
    phase: str,
    text: str | None = None,
    playback_unit_id: str | None = None,
    voice_id: str | None = None,
    status: str | None = None,
    energy_goal: str | None = None,
) -> dict[str, Any]:
    with connect() as db:
        existing = db.execute(
            """
            select * from goal_deliveries
            where goal_id=? and version=? and coalesce(phase_id,'')=coalesce(?,'')
              and coalesce(run_id,'')=coalesce(?,'') and phase=? and coalesce(status,'')=coalesce(?,'')
            order by sequence asc limit 1
            """,
            (goal_id, version, phase_id, run_id, phase, status),
        ).fetchone()
    if existing:
        return _goal_delivery_dict(existing)
    return add_goal_delivery(goal_id=goal_id, version=version, phase=phase, phase_id=phase_id,
                             run_id=run_id, text=text, playback_unit_id=playback_unit_id,
                             voice_id=voice_id, status=status, energy_goal=energy_goal)


def add_goal_delivery_once_expected(
    *,
    goal_id: str,
    version: int,
    phase_id: str,
    run_id: str,
    attempt: int,
    phase: str,
    text: str | None = None,
    playback_unit_id: str | None = None,
    voice_id: str | None = None,
    status: str | None = None,
    energy_goal: str | None = None,
) -> dict[str, Any]:
    if energy_goal is not None and energy_goal not in ENERGY_GOALS:
        raise ValueError("invalid_energy_goal")
    now = time.time()
    with connect() as db:
        goal = db.execute("select * from goals where goal_id=?", (goal_id,)).fetchone()
        phase_row = db.execute(
            "select * from goal_phases where goal_id=? and phase_id=? and run_id=? and phase_attempt=?",
            (goal_id, phase_id, run_id, attempt),
        ).fetchone()
        if (goal is None or phase_row is None or int(goal["version"]) != int(version)
                or str(goal["status"]) in GOAL_TERMINAL_STATUSES or not _goal_is_head_locked(db, goal)):
            return {}
        existing = db.execute(
            """
            select * from goal_deliveries
            where goal_id=? and version=? and phase_id=? and run_id=? and phase=? and coalesce(status,'')=coalesce(?,'')
            order by sequence asc limit 1
            """,
            (goal_id, version, phase_id, run_id, phase, status),
        ).fetchone()
        if existing:
            return _goal_delivery_dict(existing)
        seq = int(goal["next_delivery_sequence"] or 1)
        db.execute("update goals set next_delivery_sequence=?,updated_at=? where goal_id=? and next_delivery_sequence=?",
                   (seq + 1, now, goal_id, seq))
        if db.total_changes <= 0:
            return {}
        db.execute(
            """
            insert into goal_deliveries (
              goal_id,sequence,version,phase_id,run_id,phase,text,playback_unit_id,voice_id,status,energy_goal,created_at
            ) values (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (goal_id, seq, version, phase_id, run_id, phase, text, playback_unit_id, voice_id, status, energy_goal, now),
        )
    return {
        "sequence": seq,
        "goalId": goal_id,
        "version": version,
        "phaseId": phase_id,
        "runId": run_id,
        "phase": phase,
        "text": text,
        "playbackUnitId": playback_unit_id,
        "voiceId": voice_id,
        "status": status,
        "energyGoal": energy_goal,
    }


def _goal_delivery_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "sequence": int(row["sequence"]),
        "goalId": row["goal_id"],
        "version": int(row["version"]),
        "phaseId": row["phase_id"],
        "runId": row["run_id"],
        "phase": row["phase"],
        "text": row["text"],
        "playbackUnitId": row["playback_unit_id"],
        "voiceId": row["voice_id"],
        "status": row["status"],
        "energyGoal": _current_energy_goal(row["energy_goal"]),
    }


def list_goals_for_recovery() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("select * from goals where status in ('planning','active') order by created_at asc").fetchall()
    return [dict(row) for row in rows]


def goal_spoken_assessment_memory(goal_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            select r.run_id,r.context_json,r.receipt_json,d.text,d.sequence,r.created_at
            from runs r
            join deliveries d on d.run_id=r.run_id
            where r.client_session_id like ? and d.phase='ASSESSMENT' and d.text is not null
              and d.status in ('voice_ready','text_only')
            order by r.created_at asc,d.sequence asc
            """,
            (f"goal-internal:{goal_id}:%",),
        ).fetchall()
    memory: list[dict[str, Any]] = []
    remembered_runs: set[str] = set()
    for row in rows:
        run_id = str(row["run_id"])
        if run_id in remembered_runs:
            continue
        receipt = loads(row["receipt_json"], {})
        if not (receipt.get("success") is True and receipt.get("actualPlaybackState") == "playing"):
            continue
        context = loads(row["context_json"], {})
        memory.append({
            "spokenAssessment": row["text"],
            "environmentNarrationSignature": (context.get("environmentNarration") or {}).get("signature"),
        })
        remembered_runs.add(run_id)
    return memory
