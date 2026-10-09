from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from app.config import DB_PATH, ensure_dirs


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
    if energy_goal not in {None, "boost", "maintain", "lower"}:
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
    if energy_goal not in {None, "boost", "maintain", "lower"}:
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
                "energyGoal": existing["energy_goal"],
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
            "energyGoal": row["energy_goal"],
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
        "energyGoal": row["energy_goal"],
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
                "energyGoal": context.get("energyGoal"),
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
