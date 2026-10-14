from __future__ import annotations

import asyncio
from typing import Any

from fastapi import BackgroundTasks, HTTPException

from app import store
from app.agent import continue_after_receipt, generate_goal_plan, new_goal_id, new_goal_phase_id, new_run_id, run_task
from app.catalog import catalog
from app.config import (
    PHASE_ASSESSMENT,
    PHASE_ERROR,
    PHASE_EXPLANATION,
    PHASE_SELECTION,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_WAITING_PLAYBACK,
)
from app.environment import context_snapshot, enrich_context, narration_context, narration_signature
from app.resource_validation import validate_playback_unit
from app.schemas import (
    ENERGY_GOALS,
    GoalControlRequest,
    GoalCreateRequest,
    GoalDelivery,
    GoalPlaybackReceipt,
    GoalProgressRequest,
    GoalResponse,
)


_GOAL_LOCKS: dict[str, asyncio.Lock] = {}

def _goal_lock(goal_id: str) -> asyncio.Lock:
    lock = _GOAL_LOCKS.get(goal_id)
    if lock is None:
        lock = asyncio.Lock()
        _GOAL_LOCKS[goal_id] = lock
    return lock

async def create_goal(request: GoalCreateRequest, background_tasks: BackgroundTasks) -> GoalResponse:
    current_playback = _canonical_current_playback(dict(request.currentPlayback))
    _validate_current_playback(current_playback)
    context = await _goal_context(_public_goal_context(request.context), request.language or "zh", current_playback)
    goal_id = new_goal_id()
    version = 1
    created = store.create_goal(
        goal_id=goal_id,
        version=version,
        client_session_id=request.clientSessionId,
        client_action_seq=request.clientActionSeq,
        decision_version=request.decisionVersion,
        input_text=request.input,
        split=request.split,
        language=request.language or "zh",
        context=context,
        current_playback=current_playback,
        catalog_payload=request.catalog,
        context_snapshot=context_snapshot(context),
    )
    if created.state == "stale":
        raise HTTPException(status_code=409, detail="stale_goal_action")
    if created.state == "conflict":
        raise HTTPException(status_code=409, detail="conflicting_goal_action")
    if created.state == "duplicate" and created.run_id:
        return response_for_goal(created.run_id, after=0)
    if created.state != "created":
        raise HTTPException(status_code=409, detail="goal_not_created")
    background_tasks.add_task(plan_goal, goal_id)
    return response_for_goal(goal_id, after=0)


async def plan_goal(goal_id: str) -> None:
    goal = store.get_goal(goal_id)
    if not goal or goal["status"] != "planning" or not store.goal_is_head(goal):
        return
    context = store.loads(goal["context_json"], {})
    current_playback = store.loads(goal["current_playback_json"], {})
    try:
        plan = await generate_goal_plan(
            user_input=str(goal["input_text"]),
            split=bool(goal["split"]),
            language=str(goal["language"] or "zh"),
            context=context,
            current_playback=current_playback,
        )
    except Exception as exc:
        store.add_goal_delivery(goal_id=goal_id, version=int(goal["version"]), phase="ERROR",
                                text=None, status=f"plan_failed:{type(exc).__name__}")
        store.update_goal(goal_id, status="failed")
        return
    phases = [
        {
            "phaseId": new_goal_phase_id(),
            "index": index,
            "title": phase.title,
            "durationMinutes": phase.duration_minutes,
            "iconId": phase.icon_id,
        }
        for index, phase in enumerate(plan.phases)
    ]
    if not phases:
        store.update_goal(goal_id, status="failed")
        return
    if plan.response_id:
        context["goalAgentPreviousResponseId"] = plan.response_id
        if plan.pending_tool_outputs:
            context["goalAgentPendingToolOutputs"] = plan.pending_tool_outputs
        context.setdefault("goalUsedServiceSongIds", [])
        context.setdefault("goalPreparedOpenings", [])
        store.update_goal(goal_id, context_json=store.dumps(context))
    if not store.set_goal_plan(goal_id, title=plan.title, total_minutes=plan.total_minutes,
                               phases=phases, active_phase_id=phases[0]["phaseId"]):
        return
    version = int(goal["version"])
    store.add_goal_delivery_once(
        goal_id=goal_id,
        version=version,
        phase_id=None,
        run_id=None,
        phase="PLAN",
        text=plan.title,
        status="ready",
    )
    asyncio.create_task(prepare_phase(goal_id, phases[0]["phaseId"], expose_deliveries=True, refresh_context=False))


GOAL_PRIVATE_CONTEXT_KEYS = {
    "goalAgentPreviousResponseId", "goalAgentPendingToolOutputs", "goalUsedServiceSongIds",
    "goalPreparedOpenings", "goalSpokenExplanations", "goalPostReceiptText", "goalMode", "goal",
    "goalPhase", "userFeedback", "energyGoal", "selectionStatus", "suppressAssessmentVoice",
}


def _public_goal_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    return {key: value for key, value in context.items() if key not in GOAL_PRIVATE_CONTEXT_KEYS}


def _merged_goal_context(stored: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    context = dict(stored)
    for key, value in _public_goal_context(override).items():
        context[key] = value
    return context


async def prepare_phase(
    goal_id: str,
    phase_id: str,
    *,
    expose_deliveries: bool,
    refresh_context: bool,
    context_override: dict[str, Any] | None = None,
    playback_override: dict[str, Any] | None = None,
) -> None:
    async with _goal_lock(goal_id):
        await _prepare_phase_unlocked(
            goal_id,
            phase_id,
            expose_deliveries=expose_deliveries,
            refresh_context=refresh_context,
            context_override=context_override,
            playback_override=playback_override,
        )


async def _prepare_phase_unlocked(
    goal_id: str,
    phase_id: str,
    *,
    expose_deliveries: bool,
    refresh_context: bool,
    context_override: dict[str, Any] | None = None,
    playback_override: dict[str, Any] | None = None,
) -> None:
    goal = store.get_goal(goal_id)
    phase = store.get_goal_phase(goal_id, phase_id)
    if not goal or not phase or goal["status"] in {"cancelled", "failed", "completed"} or not store.goal_is_head(goal):
        return
    version = int(goal["version"])
    expected_attempt = int(phase.get("phase_attempt") or 0)
    context = _merged_goal_context(store.loads(goal["context_json"], {}), context_override)
    current_playback = _canonical_current_playback(dict(playback_override or store.loads(goal["current_playback_json"], {})))
    if refresh_context:
        context = await _goal_context(context, str(goal["language"] or "zh"), current_playback,
                                      memory=store.goal_spoken_assessment_memory(goal_id))
    goal = store.get_goal(goal_id)
    phase = store.get_goal_phase(goal_id, phase_id)
    if not goal or not phase or int(goal["version"]) != version or goal["status"] in {"cancelled", "failed", "completed"} or not store.goal_is_head(goal):
        return
    if not expose_deliveries and (phase.get("run_id") or phase.get("status") != "planned"):
        return
    context["goalMode"] = True
    context["goal"] = {"goalId": goal_id, "version": version, "title": goal.get("title"), "input": goal.get("input_text")}
    context["goalPhase"] = {
        "phaseId": phase_id,
        "index": int(phase["phase_index"]),
        "title": phase["title"],
        "durationMinutes": int(phase["duration_minutes"]),
        "iconId": phase.get("icon_id") or "scene-music-note",
    }
    context["userFeedback"] = "goal:" + str(goal.get("input_text") or "")
    if not expose_deliveries:
        existing_narration = context.get("environmentNarration") if isinstance(context.get("environmentNarration"), dict) else {}
        signature = existing_narration.get("signature")
        if not isinstance(signature, dict):
            signature = narration_signature(context)
        context["environmentNarration"] = {"mode": "music_only", "changedFields": [], "signature": signature}
    context["suppressAssessmentVoice"] = False
    run_id = new_run_id()
    claim = store.claim_goal_phase_attempt(goal_id=goal_id, phase_id=phase_id, version=version,
                                           run_id=run_id, context=context, current_playback=current_playback,
                                           expected_attempt=expected_attempt)
    if claim.state != "claimed" or claim.attempt is None:
        return
    attempt = claim.attempt
    if expose_deliveries:
        store.update_goal_context_snapshot_expected(
            goal_id,
            version=version,
            active_phase_id=phase_id,
            run_id=run_id,
            attempt=attempt,
            context_snapshot=context_snapshot(context),
        )
    created = store.create_run(
        run_id=run_id,
        client_session_id=f"goal-internal:{goal_id}:{phase_id}",
        client_action_seq=attempt,
        decision_version=version,
        context=context,
        current_playback=current_playback,
        catalog_payload={},
    )
    if created.state != "created":
        store.update_goal_phase_expected(goal_id, phase_id, run_id=run_id, attempt=attempt, status="failed")
        return
    task = asyncio.create_task(run_task(run_id))
    if expose_deliveries:
        while not task.done():
            await copy_phase_assessment(goal_id, phase_id, run_id, attempt)
            await asyncio.sleep(0.1)
    await task
    await copy_phase_preparation(goal_id, phase_id, run_id, attempt, expose_deliveries=expose_deliveries)


async def copy_phase_assessment(goal_id: str, phase_id: str, run_id: str, attempt: int) -> None:
    goal = store.get_goal(goal_id)
    phase = store.get_goal_phase(goal_id, phase_id)
    if not goal or not phase or phase.get("run_id") != run_id or int(phase.get("phase_attempt") or 0) != int(attempt) or not store.goal_is_head(goal):
        return
    for item in store.deliveries_after(run_id, 0):
        if item["phase"] != PHASE_ASSESSMENT:
            continue
        store.add_goal_delivery_once_expected(goal_id=goal_id, version=int(goal["version"]), phase_id=phase_id,
                                              run_id=run_id, attempt=attempt, phase=PHASE_ASSESSMENT,
                                              text=item.get("text"), voice_id=item.get("voiceId"),
                                              status=item.get("status"), energy_goal=item.get("energyGoal"))


async def copy_phase_preparation(goal_id: str, phase_id: str, run_id: str, attempt: int, *, expose_deliveries: bool) -> None:
    goal = store.get_goal(goal_id)
    phase = store.get_goal_phase(goal_id, phase_id)
    run = store.get_run(run_id)
    if not goal or not phase or not run or not store.goal_is_head(goal):
        return
    if (phase.get("run_id") != run_id or int(phase.get("phase_attempt") or 0) != int(attempt)
            or goal["status"] in store.GOAL_TERMINAL_STATUSES):
        return
    deliveries = store.deliveries_after(run_id, 0)
    assessment = next((item for item in reversed(deliveries) if item["phase"] == PHASE_ASSESSMENT and item.get("text")), None)
    selection = next((item for item in deliveries if item["phase"] == PHASE_SELECTION and item.get("playbackUnitId")), None)
    error = next((item for item in deliveries if item["phase"] == PHASE_ERROR), None)
    cacheable_selection_statuses = {"model_selected", "model_selected_unchanged"}
    version = int(goal["version"])
    if assessment:
        store.update_goal_phase_expected(goal_id, phase_id, run_id=run_id, attempt=attempt,
                                         energy_goal=assessment.get("energyGoal"),
                                         prepared_opening_text=assessment.get("text"),
                                         prepared_opening_voice_id=assessment.get("voiceId"))
    if error and not selection:
        store.update_goal_phase_expected(goal_id, phase_id, run_id=run_id, attempt=attempt, status="failed")
        if expose_deliveries:
            store.add_goal_delivery_once_expected(goal_id=goal_id, version=version, phase_id=phase_id,
                                                  run_id=run_id, attempt=attempt, phase=PHASE_ERROR,
                                                  text=None, status=error.get("status"))
        return
    if not selection:
        return
    selection_status = str(selection.get("status") or "")
    if not expose_deliveries and selection_status not in cacheable_selection_statuses:
        store.update_goal_phase_expected(goal_id, phase_id, run_id=run_id, attempt=attempt,
                                         status="failed", prepared_playback_unit_id=None,
                                         prepared_opening_text=None, prepared_opening_voice_id=None)
        return
    selected_id = str(selection["playbackUnitId"])
    next_status = "waiting_playback" if expose_deliveries else "prepared"
    if not store.update_goal_phase_expected(goal_id, phase_id, run_id=run_id, attempt=attempt,
                                            status=next_status, prepared_playback_unit_id=selected_id):
        return
    if not expose_deliveries:
        sync_goal_chain_context_from_run(goal_id, phase, run, selected_id, assessment)
        return
    delivered_selection = False
    for item in deliveries:
        if item["phase"] == PHASE_ASSESSMENT:
            store.add_goal_delivery_once_expected(goal_id=goal_id, version=version, phase_id=phase_id,
                                                  run_id=run_id, attempt=attempt, phase=PHASE_ASSESSMENT,
                                                  text=item.get("text"), voice_id=item.get("voiceId"),
                                                  status=item.get("status"), energy_goal=item.get("energyGoal"))
        elif item["phase"] == PHASE_SELECTION:
            delivered = store.add_goal_delivery_once_expected(goal_id=goal_id, version=version, phase_id=phase_id,
                                                              run_id=run_id, attempt=attempt, phase=PHASE_SELECTION,
                                                              playback_unit_id=item.get("playbackUnitId"),
                                                              status=item.get("status"), energy_goal=item.get("energyGoal"))
            delivered_selection = bool(delivered) or delivered_selection
    if delivered_selection:
        sync_goal_chain_context_from_run(goal_id, phase, run, selected_id, assessment)
        asyncio.create_task(prepare_future_phases(goal_id, int(phase["phase_index"])))


def sync_goal_chain_context_from_run(goal_id: str, phase: dict[str, Any], run: dict[str, Any], selected_id: str, assessment: dict[str, Any] | None) -> None:
    goal = store.get_goal(goal_id)
    if not goal or goal["status"] in store.GOAL_TERMINAL_STATUSES or not store.goal_is_head(goal):
        return
    run_context = store.loads(run.get("context_json"), {})
    goal_context = store.loads(goal.get("context_json"), {})
    if isinstance(run_context.get("goalAgentPreviousResponseId"), str):
        goal_context["goalAgentPreviousResponseId"] = run_context["goalAgentPreviousResponseId"]
    if isinstance(run_context.get("goalAgentPendingToolOutputs"), list):
        goal_context["goalAgentPendingToolOutputs"] = run_context["goalAgentPendingToolOutputs"]
    used = list(goal_context.get("goalUsedServiceSongIds") if isinstance(goal_context.get("goalUsedServiceSongIds"), list) else [])
    profile = catalog().profile(selected_id)
    scene = _scene_id(store.loads(run.get("current_playback_json"), {}))
    service_id = profile.song_id if profile else None
    if service_id and service_id not in used:
        used.append(service_id)
    goal_context["goalUsedServiceSongIds"] = used
    openings = list(goal_context.get("goalPreparedOpenings") if isinstance(goal_context.get("goalPreparedOpenings"), list) else [])
    text = assessment.get("text") if assessment else None
    if isinstance(text, str) and text and text not in openings:
        openings.append(text)
    goal_context["goalPreparedOpenings"] = openings[-8:]
    store.update_goal(goal_id, context_json=store.dumps(goal_context))


def refresh_run_goal_history_from_goal(goal_id: str, run_id: str) -> None:
    goal = store.get_goal(goal_id)
    run = store.get_run(run_id)
    if not goal or not run:
        return
    goal_context = store.loads(goal.get("context_json"), {})
    run_context = store.loads(run.get("context_json"), {})
    for key in ("goalSpokenExplanations", "goalPreparedOpenings", "goalUsedServiceSongIds"):
        if key in goal_context:
            run_context[key] = goal_context[key]
    store.update_run(run_id, context_json=store.dumps(run_context))


def sync_goal_explanation_context(goal_id: str, explanation: dict[str, Any]) -> None:
    goal = store.get_goal(goal_id)
    if not goal or goal["status"] in store.GOAL_TERMINAL_STATUSES or not store.goal_is_head(goal):
        return
    text = explanation.get("text")
    status = str(explanation.get("status") or "")
    if not isinstance(text, str) or not text.strip() or status.startswith("model_error_keep") or status in {"model_unavailable_keep", "model_selection_invalid_keep"}:
        return
    goal_context = store.loads(goal.get("context_json"), {})
    spoken = list(goal_context.get("goalSpokenExplanations") if isinstance(goal_context.get("goalSpokenExplanations"), list) else [])
    if text not in spoken:
        spoken.append(text)
    goal_context["goalSpokenExplanations"] = spoken[-8:]
    store.update_goal(goal_id, context_json=store.dumps(goal_context))


async def prepare_future_phases(goal_id: str, after_index: int) -> None:
    goal = store.get_goal(goal_id)
    if not goal or goal["status"] in store.GOAL_TERMINAL_STATUSES or not store.goal_is_head(goal):
        return
    for future in store.goal_phases(goal_id):
        goal = store.get_goal(goal_id)
        if not goal or goal["status"] in store.GOAL_TERMINAL_STATUSES or not store.goal_is_head(goal):
            return
        if int(future["phase_index"]) <= int(after_index):
            continue
        refreshed = store.get_goal_phase(goal_id, str(future["phase_id"]))
        if not refreshed:
            return
        if refreshed.get("status") == "prepared":
            continue
        if refreshed.get("run_id") or refreshed.get("status") in {"preparing", "waiting_playback", "active"}:
            return
        if refreshed.get("status") != "planned":
            continue
        await prepare_phase(goal_id, str(future["phase_id"]), expose_deliveries=False, refresh_context=False)


async def expose_cached_phase(goal_id: str, phase_id: str, *, context: dict[str, Any], current_playback: dict[str, Any]) -> None:
    async with _goal_lock(goal_id):
        await _expose_cached_phase_unlocked(goal_id, phase_id, context=context, current_playback=current_playback)


async def _expose_cached_phase_unlocked(goal_id: str, phase_id: str, *, context: dict[str, Any], current_playback: dict[str, Any]) -> None:
    goal = store.get_goal(goal_id)
    phase = store.get_goal_phase(goal_id, phase_id)
    if not goal or not phase or not store.goal_is_head(goal):
        return
    run_id = str(phase.get("run_id") or "")
    attempt = int(phase.get("phase_attempt") or 0)
    if not run_id or phase.get("status") != "waiting_playback":
        return
    run = store.get_run(run_id)
    if not run:
        return
    cached_context = store.loads(run.get("context_json"), {})
    if not await validate_cached_activation_binding(goal_id, phase_id, run_id, attempt, current_playback):
        return
    authoritative_context = store.loads(goal.get("context_json"), {})
    merged_panel_seed = _merged_goal_context(authoritative_context, context)
    panel_context = await _goal_context(merged_panel_seed, str(goal["language"] or "zh"), current_playback,
                                        memory=store.goal_spoken_assessment_memory(goal_id))
    activated_context = _merged_goal_context(authoritative_context, panel_context)
    for key in ("goalMode", "goal", "goalPhase", "userFeedback", "suppressAssessmentVoice", "energyGoal", "selectionStatus", "goalPostReceiptText"):
        if key in cached_context:
            activated_context[key] = cached_context[key]
    # Preserve the authoritative continuous goal-agent chain/history. Cached run context
    # contains phase-local prompt fields, not the global Responses cursor.
    for key in ("goalAgentPreviousResponseId", "goalAgentPendingToolOutputs", "goalUsedServiceSongIds",
                "goalPreparedOpenings", "goalSpokenExplanations"):
        if key in authoritative_context:
            activated_context[key] = authoritative_context[key]
    activated_context["currentPlaybackForExplanation"] = current_playback
    if not store.update_goal_context_snapshot_expected(goal_id, version=int(goal["version"]), active_phase_id=phase_id,
                                                       run_id=run_id, attempt=attempt, context_snapshot=context_snapshot(panel_context)):
        return
    if not store.update_goal_phase_expected(goal_id, phase_id, run_id=run_id, attempt=attempt,
                                            context_json=store.dumps(activated_context), current_playback_json=store.dumps(current_playback)):
        return
    if not store.update_run(run_id, context_json=store.dumps(activated_context), current_playback_json=store.dumps(current_playback)):
        return
    await copy_cached_phase_deliveries(goal_id, phase_id, run_id, attempt)


async def validate_cached_activation_binding(goal_id: str, phase_id: str, run_id: str, attempt: int,
                                             current_playback: dict[str, Any]) -> bool:
    phase = store.get_goal_phase(goal_id, phase_id)
    run = store.get_run(run_id)
    if not phase or not run or phase.get("run_id") != run_id or int(phase.get("phase_attempt") or 0) != int(attempt):
        return False
    selected_id = str(phase.get("prepared_playback_unit_id") or run.get("selected_playback_unit_id") or "")
    profile = catalog().profile(selected_id)
    if not profile:
        store.update_goal_phase_expected(goal_id, phase_id, run_id=run_id, attempt=attempt, status="failed",
                                         prepared_playback_unit_id=None, prepared_opening_text=None,
                                         prepared_opening_voice_id=None)
        store.update_run(run_id, status=RUN_STATUS_CANCELLED)
        return False
    prepared_playback = store.loads(phase.get("current_playback_json") or run.get("current_playback_json"), {})
    prepared_scene = _scene_id(prepared_playback)
    activation_scene = _scene_id(current_playback)
    prepared_current_id = str(prepared_playback.get("playbackUnitId") or prepared_playback.get("playback_unit_id") or "")
    activation_current_id = str(current_playback.get("playbackUnitId") or current_playback.get("playback_unit_id") or "")
    binding_changed = profile.service_song_id(prepared_scene) != profile.service_song_id(activation_scene)
    current_baseline_changed = bool(prepared_current_id and activation_current_id and prepared_current_id != activation_current_id)
    if current_baseline_changed:
        store.record_retrieval_event(run_id, "candidate_chosen", {
            "playbackUnitId": profile.playback_unit_id,
            "reason": None,
            "reasonAppliesToChosenCandidate": False,
            "selectionPolicy": "cached-baseline-revalidated-v1",
            "resourceValidation": "baseline_changed",
            "preparedCurrentPlaybackUnitId": prepared_current_id,
            "activationCurrentPlaybackUnitId": activation_current_id,
            "preparedSceneId": prepared_scene,
            "activationSceneId": activation_scene,
            "comparativeReasonCleared": True,
        })
    if not binding_changed:
        return True
    if not current_baseline_changed:
        store.record_retrieval_event(run_id, "candidate_chosen", {
            "playbackUnitId": profile.playback_unit_id,
            "reason": None,
            "reasonAppliesToChosenCandidate": False,
            "selectionPolicy": "cached-binding-revalidated-v1",
            "resourceValidation": "pending_activation_binding_validation",
            "preparedSceneId": prepared_scene,
            "activationSceneId": activation_scene,
            "comparativeReasonCleared": True,
        })
    validation = await validate_playback_unit(profile, origin_scene_id=activation_scene)
    latest_goal = store.get_goal(goal_id)
    latest_phase = store.get_goal_phase(goal_id, phase_id)
    if (not latest_goal or not latest_phase or latest_goal.get("active_phase_id") != phase_id
            or latest_phase.get("run_id") != run_id or int(latest_phase.get("phase_attempt") or 0) != int(attempt)
            or latest_goal["status"] in store.GOAL_TERMINAL_STATUSES or not store.goal_is_head(latest_goal)):
        return False
    if not validation.ok:
        store.update_goal_phase_expected(goal_id, phase_id, run_id=run_id, attempt=attempt, status="failed",
                                         prepared_playback_unit_id=None, prepared_opening_text=None,
                                         prepared_opening_voice_id=None)
        store.update_run(run_id, status=RUN_STATUS_CANCELLED)
        return False
    return True


def _scene_id(playback: dict[str, Any]) -> str | None:
    value = playback.get("sceneId") or playback.get("originSceneId")
    return value if isinstance(value, str) and value else None


async def copy_cached_phase_deliveries(goal_id: str, phase_id: str, run_id: str, attempt: int) -> None:
    goal = store.get_goal(goal_id)
    phase = store.get_goal_phase(goal_id, phase_id)
    if (not goal or not phase or goal.get("active_phase_id") != phase_id or phase.get("run_id") != run_id
            or int(phase.get("phase_attempt") or 0) != int(attempt) or goal["status"] in store.GOAL_TERMINAL_STATUSES
            or not store.goal_is_head(goal)):
        return
    deliveries = store.deliveries_after(run_id, 0)
    version = int(goal["version"])
    assessment = next((item for item in reversed(deliveries)
                       if item["phase"] == PHASE_ASSESSMENT and item.get("status") in {"voice_ready", "text_only"}), None)
    selection = next((item for item in deliveries if item["phase"] == PHASE_SELECTION and item.get("playbackUnitId")), None)
    if assessment:
        store.add_goal_delivery_once_expected(goal_id=goal_id, version=version, phase_id=phase_id,
                                              run_id=run_id, attempt=attempt, phase=PHASE_ASSESSMENT,
                                              text=assessment.get("text"), voice_id=assessment.get("voiceId"),
                                              status=assessment.get("status"), energy_goal=assessment.get("energyGoal"))
    if selection:
        store.add_goal_delivery_once_expected(goal_id=goal_id, version=version, phase_id=phase_id,
                                              run_id=run_id, attempt=attempt, phase=PHASE_SELECTION,
                                              playback_unit_id=selection.get("playbackUnitId"),
                                              status=selection.get("status"), energy_goal=selection.get("energyGoal"))



async def advance_goal(goal_id: str, request: GoalControlRequest, background_tasks: BackgroundTasks) -> GoalResponse:
    goal = _current_goal_or_error(goal_id, request.version)
    current_phase_id = str(request.phaseId or "")
    if not current_phase_id or current_phase_id != goal.get("active_phase_id"):
        raise HTTPException(status_code=409, detail="inactive_phase_advance")
    phases = store.goal_phases(goal_id)
    current = next((phase for phase in phases if phase["phase_id"] == current_phase_id), None)
    if current is None:
        raise HTTPException(status_code=404, detail="phase_not_found")
    if current.get("status") != "active":
        raise HTTPException(status_code=409, detail="phase_not_active")
    result = store.advance_goal_phase(goal_id, version=int(goal["version"]), current_phase_id=current_phase_id,
                                      client_action_seq=request.clientActionSeq)
    if result.state == "missing":
        raise HTTPException(status_code=404, detail="phase_not_found")
    if result.state == "inactive":
        raise HTTPException(status_code=409, detail="inactive_phase_advance")
    if result.state == "phase_not_active":
        raise HTTPException(status_code=409, detail="phase_not_active")
    if result.state == "stale_control":
        raise HTTPException(status_code=409, detail="stale_goal_control")
    if result.state in {"stale", "terminal"}:
        raise HTTPException(status_code=409, detail="stale_goal_version")
    if result.state == "completed":
        return response_for_goal(goal_id, after=0)
    if result.next_phase_id is None:
        raise HTTPException(status_code=409, detail="advance_not_applied")
    context = dict(request.context or {})
    current_playback = _canonical_current_playback(dict(request.currentPlayback or {}))
    if result.cached:
        await expose_cached_phase(goal_id, result.next_phase_id, context=context, current_playback=current_playback)
    else:
        background_tasks.add_task(prepare_phase, goal_id, result.next_phase_id, expose_deliveries=True,
                                  refresh_context=True, context_override=context, playback_override=current_playback)
    return response_for_goal(goal_id, after=0)


async def retry_goal_phase(goal_id: str, request: GoalControlRequest, background_tasks: BackgroundTasks) -> GoalResponse:
    goal = _current_goal_or_error(goal_id, request.version)
    phase_id = str(request.phaseId or goal.get("active_phase_id") or "")
    if phase_id != goal.get("active_phase_id"):
        raise HTTPException(status_code=409, detail="inactive_phase_retry")
    if not store.get_goal_phase(goal_id, phase_id):
        raise HTTPException(status_code=404, detail="phase_not_found")
    result = store.begin_goal_phase_retry(goal_id, version=int(goal["version"]), phase_id=phase_id,
                                          client_action_seq=request.clientActionSeq)
    if result.state == "missing":
        raise HTTPException(status_code=404, detail="phase_not_found")
    if result.state == "inactive":
        raise HTTPException(status_code=409, detail="inactive_phase_retry")
    if result.state == "stale_control":
        raise HTTPException(status_code=409, detail="stale_goal_control")
    if result.state in {"stale", "terminal"}:
        raise HTTPException(status_code=409, detail="stale_goal_version")
    context = dict(request.context or store.loads(goal["context_json"], {}))
    current_playback = _canonical_current_playback(dict(request.currentPlayback or store.loads(goal["current_playback_json"], {})))
    background_tasks.add_task(prepare_phase, goal_id, phase_id, expose_deliveries=True,
                              refresh_context=True, context_override=context, playback_override=current_playback)
    return response_for_goal(goal_id, after=0)


async def goal_playback_receipt(goal_id: str, receipt: GoalPlaybackReceipt, background_tasks: BackgroundTasks) -> GoalResponse:
    goal = _current_goal_or_error(goal_id, receipt.version)
    phase = store.get_goal_phase(goal_id, receipt.phaseId)
    if phase is None:
        raise HTTPException(status_code=404, detail="phase_not_found")
    if goal.get("active_phase_id") != receipt.phaseId:
        raise HTTPException(status_code=409, detail="inactive_phase_receipt")
    if phase.get("run_id") != receipt.runId:
        raise HTTPException(status_code=409, detail="stale_phase_run")
    run = store.get_run(receipt.runId)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    payload = _validated_goal_receipt_payload(run, receipt)
    claim = store.claim_receipt(
        run_id=receipt.runId,
        client_session_id=str(run["client_session_id"]),
        client_action_seq=int(run["client_action_seq"]),
        decision_version=int(run["decision_version"]),
        receipt_payload=payload,
    )
    if claim.state == "stale":
        raise HTTPException(status_code=409, detail="stale_receipt")
    if claim.state == "not_waiting":
        raise HTTPException(status_code=409, detail="receipt_not_awaited")
    if claim.state == "conflict":
        raise HTTPException(status_code=409, detail="conflicting_receipt")
    if claim.state == "cancelled":
        return response_for_goal(goal_id, after=0)
    if claim.state == "claimed":
        selection_status = str((store.latest_delivery(receipt.runId, PHASE_SELECTION) or {}).get("status") or "")
        degraded_keep = (selection_status in {"model_unavailable_keep", "model_selection_invalid_keep"}
                         or selection_status.startswith("model_error_keep:"))
        if degraded_keep:
            phase_status = "failed"
            is_playing = 0
        elif receipt.success and receipt.actualPlaybackState == "playing":
            phase_status = "active"
            is_playing = 1
        elif receipt.success:
            phase_status = "waiting_playback"
            is_playing = 0
        else:
            phase_status = "failed"
            is_playing = 0
        store.update_goal_phase(goal_id, receipt.phaseId, status=phase_status,
                                prepared_playback_unit_id=receipt.playbackUnitId,
                                is_playing=is_playing)
        background_tasks.add_task(finish_phase_after_receipt, goal_id, receipt.phaseId, receipt.runId)
    return response_for_goal(goal_id, after=0)


async def finish_phase_after_receipt(goal_id: str, phase_id: str, run_id: str) -> None:
    goal_before = store.get_goal(goal_id)
    if not goal_before or goal_before["status"] in store.GOAL_TERMINAL_STATUSES:
        return
    phase_before = store.get_goal_phase(goal_id, phase_id)
    attempt = int(phase_before.get("phase_attempt") or 0) if phase_before and phase_before.get("run_id") == run_id else None
    if attempt is None:
        return
    refresh_run_goal_history_from_goal(goal_id, run_id)
    await continue_after_receipt(run_id)
    goal = store.get_goal(goal_id)
    phase = store.get_goal_phase(goal_id, phase_id)
    if (not goal or goal["status"] in store.GOAL_TERMINAL_STATUSES
            or not phase or not store.goal_is_head(goal) or phase.get("run_id") != run_id):
        return
    if int(phase.get("phase_attempt") or 0) != attempt:
        return
    explanation = store.latest_delivery(run_id, PHASE_EXPLANATION)
    if not explanation:
        return
    version = int(goal["version"])
    delivered = store.add_goal_delivery_once_expected(goal_id=goal_id, version=version, phase_id=phase_id,
                                                       run_id=run_id, attempt=attempt, phase=PHASE_EXPLANATION,
                                                       text=explanation.get("text"),
                                                       playback_unit_id=explanation.get("playbackUnitId"),
                                                       voice_id=explanation.get("voiceId"), status=explanation.get("status"),
                                                       energy_goal=explanation.get("energyGoal"))
    if delivered:
        sync_goal_explanation_context(goal_id, explanation)


def progress_goal(goal_id: str, request: GoalProgressRequest) -> GoalResponse:
    goal = _current_goal_or_error(goal_id, request.version)
    if int(request.clientActionSeq) != int(goal.get("control_action_seq") or goal.get("client_action_seq") or 0):
        raise HTTPException(status_code=409, detail="stale_goal_control")
    if request.phaseId != goal.get("active_phase_id"):
        raise HTTPException(status_code=409, detail="inactive_phase_progress")
    phase = store.get_goal_phase(goal_id, request.phaseId)
    if phase is None:
        raise HTTPException(status_code=404, detail="phase_not_found")
    if request.runId and phase.get("run_id") != request.runId:
        raise HTTPException(status_code=409, detail="stale_phase_run")
    store.update_goal_phase_progress(goal_id, request.phaseId,
                                     elapsed_seconds=float(request.elapsedSeconds),
                                     is_playing=request.isPlaying)
    if request.manualOverride:
        store.bump_goal_phase_attempt(goal_id, request.phaseId)
        run_id = phase.get("run_id")
        if run_id and phase.get("status") in {"preparing", "waiting_playback", "active"}:
            store.update_run(str(run_id), status=RUN_STATUS_CANCELLED)
            store.delete_goal_phase_deliveries(goal_id, request.phaseId, str(run_id))
            store.update_goal_phase(goal_id, request.phaseId, status="active" if request.isPlaying else "prepared",
                                    run_id=None, prepared_playback_unit_id=None,
                                    prepared_opening_text=None, prepared_opening_voice_id=None)
        for future in store.goal_phases(goal_id):
            if int(future["phase_index"]) > int(phase["phase_index"]):
                store.bump_goal_phase_attempt(goal_id, str(future["phase_id"]))
            if int(future["phase_index"]) > int(phase["phase_index"]) and future.get("run_id"):
                store.update_run(str(future["run_id"]), status=RUN_STATUS_CANCELLED)
                store.delete_goal_phase_deliveries(goal_id, str(future["phase_id"]), str(future["run_id"]))
                store.update_goal_phase(goal_id, str(future["phase_id"]), status="planned",
                                        run_id=None, prepared_playback_unit_id=None,
                                        prepared_opening_text=None, prepared_opening_voice_id=None)
    return response_for_goal(goal_id, after=0)


def cancel_goal(goal_id: str, request: GoalControlRequest) -> GoalResponse:
    goal = _current_goal_or_error(goal_id, request.version)
    claim = store.claim_goal_control_action(goal_id, version=int(goal["version"]), client_action_seq=request.clientActionSeq)
    if claim.state == "stale":
        raise HTTPException(status_code=409, detail="stale_goal_control")
    if claim.state != "created":
        raise HTTPException(status_code=409, detail="stale_goal_version")
    store.cancel_goal(goal_id, int(goal["version"]))
    return response_for_goal(goal_id, after=0)


def response_for_goal(goal_id: str, *, after: int) -> GoalResponse:
    goal = store.get_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="goal_not_found")
    return GoalResponse(
        goalId=goal_id,
        version=int(goal["version"]),
        clientActionSeq=int(goal.get("control_action_seq") or goal.get("client_action_seq") or 0),
        status=goal["status"],
        title=goal.get("title"),
        totalMinutes=goal.get("total_minutes"),
        activePhaseId=goal.get("active_phase_id"),
        phases=[_phase_model(phase) for phase in store.goal_phases(goal_id)],
        deliveries=[GoalDelivery(**item) for item in store.goal_deliveries_after(goal_id, after)],
        contextSnapshot=store.loads(goal.get("context_snapshot_json"), None),
    )


async def recover_goals() -> None:
    for goal in store.list_goals_for_recovery():
        if not store.goal_is_head(goal):
            continue
        if goal["status"] == "planning":
            asyncio.create_task(plan_goal(str(goal["goal_id"])))
        for phase in store.goal_phases(str(goal["goal_id"])):
            run_id = phase.get("run_id")
            if not run_id and goal.get("active_phase_id") == phase["phase_id"] and phase.get("status") in {"planned", "prepared", "preparing"}:
                asyncio.create_task(prepare_phase(str(goal["goal_id"]), str(phase["phase_id"]), expose_deliveries=True,
                                                  refresh_context=True))
                continue
            if not run_id:
                continue
            run = store.get_run(str(run_id))
            if not run:
                if goal.get("active_phase_id") == phase["phase_id"] and phase.get("status") in {"planned", "prepared", "preparing"}:
                    asyncio.create_task(prepare_phase(str(goal["goal_id"]), str(phase["phase_id"]), expose_deliveries=True,
                                                      refresh_context=True))
                continue
            attempt = int(phase.get("phase_attempt") or 0)
            receipt = store.loads(run.get("receipt_json"), {})
            if receipt or run["status"] == "done":
                asyncio.create_task(finish_phase_after_receipt(str(goal["goal_id"]), str(phase["phase_id"]), str(run_id)))
            elif run["status"] == RUN_STATUS_RUNNING:
                asyncio.create_task(_recover_phase_run(str(goal["goal_id"]), str(phase["phase_id"]), str(run_id), attempt))
            elif run["status"] == RUN_STATUS_WAITING_PLAYBACK:
                asyncio.create_task(copy_phase_preparation(str(goal["goal_id"]), str(phase["phase_id"]), str(run_id), attempt,
                                                           expose_deliveries=goal.get("active_phase_id") == phase["phase_id"]))
        schedule_future_preparations_after_delivered_selection(str(goal["goal_id"]))


def schedule_future_preparations_after_delivered_selection(goal_id: str) -> None:
    goal = store.get_goal(goal_id)
    if not goal or goal["status"] in store.GOAL_TERMINAL_STATUSES or not store.goal_is_head(goal):
        return
    deliveries = store.goal_deliveries_after(goal_id, 0)
    delivered_phase_ids = {item.get("phaseId") for item in deliveries if item.get("phase") == PHASE_SELECTION and item.get("phaseId")}
    if not delivered_phase_ids:
        return
    phases = store.goal_phases(goal_id)
    delivered_indexes = [int(phase["phase_index"]) for phase in phases if phase["phase_id"] in delivered_phase_ids]
    if not delivered_indexes:
        return
    after_index = max(delivered_indexes)
    for future in phases:
        if int(future["phase_index"]) <= after_index:
            continue
        if future.get("status") == "prepared":
            continue
        if future.get("run_id") or future.get("status") in {"preparing", "waiting_playback", "active"}:
            break
        if future.get("status") != "planned":
            continue
        asyncio.create_task(prepare_phase(goal_id, str(future["phase_id"]), expose_deliveries=False, refresh_context=False))
        break


async def _recover_phase_run(goal_id: str, phase_id: str, run_id: str, attempt: int) -> None:
    async with _goal_lock(goal_id):
        await run_task(run_id)
        goal = store.get_goal(goal_id)
        phase = store.get_goal_phase(goal_id, phase_id)
        expose = bool(goal and goal.get("active_phase_id") == phase_id)
        await copy_phase_preparation(goal_id, phase_id, run_id, attempt, expose_deliveries=expose)
        refreshed = store.get_goal_phase(goal_id, phase_id)
    if not expose and refreshed and refreshed.get("status") == "prepared":
        await prepare_future_phases(goal_id, int(refreshed["phase_index"]))


async def _goal_context(context: dict[str, Any], language: str, current_playback: dict[str, Any],
                        memory: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    enriched = await enrich_context(dict(context))
    enriched.setdefault("language", language)
    enriched.pop("energyGoal", None)
    goal_memory = memory or []
    enriched["sessionMemory"] = goal_memory
    enriched["environmentNarration"] = narration_context(enriched, goal_memory)
    enriched["currentPlaybackForExplanation"] = current_playback
    return enriched


def _current_goal_or_error(goal_id: str, version: int) -> dict[str, Any]:
    goal = store.get_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="goal_not_found")
    if int(goal["version"]) != int(version) or not store.goal_is_head(goal):
        raise HTTPException(status_code=409, detail="stale_goal_version")
    if goal["status"] in store.GOAL_TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail="goal_terminal")
    return goal


def _phase_model(phase: dict[str, Any]) -> dict[str, Any]:
    return {
        "phaseId": phase["phase_id"],
        "index": int(phase["phase_index"]),
        "title": phase["title"],
        "durationMinutes": int(phase["duration_minutes"]),
        "iconId": phase.get("icon_id") or "scene-music-note",
        "status": phase["status"],
        "energyGoal": phase["energy_goal"] if phase.get("energy_goal") in ENERGY_GOALS else None,
        "preparedPlaybackUnitId": phase.get("prepared_playback_unit_id"),
        "preparedOpeningText": phase.get("prepared_opening_text"),
        "preparedOpeningVoiceId": phase.get("prepared_opening_voice_id"),
        "runId": phase.get("run_id"),
        "elapsedSeconds": float(phase.get("elapsed_seconds") or 0),
        "isPlaying": bool(phase.get("is_playing")),
    }


def _canonical_current_playback(value: dict[str, Any]) -> dict[str, Any]:
    alias = value.pop("playback_unit_id", None)
    if alias and value.get("playbackUnitId") and alias != value["playbackUnitId"]:
        raise HTTPException(status_code=409, detail="conflicting_current_playback_ids")
    if alias:
        value["playbackUnitId"] = alias
    return value


def _validate_current_playback(current_playback: dict[str, Any]) -> None:
    current_id = str(current_playback.get("playbackUnitId") or "")
    current = catalog().profile(current_id)
    if current_id and current is None:
        raise HTTPException(status_code=422, detail="unknown_current_playback_unit")
    song_id = current_playback.get("songId")
    if current and song_id and not _receipt_song_matches(current, str(song_id)):
        raise HTTPException(status_code=409, detail="current_playback_identity_mismatch")


def _validated_goal_receipt_payload(run: dict[str, Any], receipt: GoalPlaybackReceipt) -> dict[str, Any]:
    if run["status"] != RUN_STATUS_WAITING_PLAYBACK and not store.loads(str(run.get("receipt_json") or ""), {}):
        raise HTTPException(status_code=409, detail="receipt_not_awaited")
    selected_id = str(run.get("selected_playback_unit_id") or "")
    if not selected_id:
        raise HTTPException(status_code=409, detail="selection_not_recorded")
    current_playback = store.loads(str(run.get("current_playback_json") or "{}"), {})
    current_id = str(current_playback.get("playbackUnitId") or "")
    actual = catalog().profile(receipt.playbackUnitId)
    selected = catalog().profile(selected_id)
    current = catalog().profile(current_id)
    if actual is None:
        raise HTTPException(status_code=422, detail="unknown_playback_unit")
    if selected is None:
        raise HTTPException(status_code=409, detail="selected_unit_missing")
    if receipt.songId and not _receipt_song_matches(actual, receipt.songId):
        raise HTTPException(status_code=409, detail="receipt_song_mismatch")
    if receipt.success and receipt.playbackUnitId != selected_id:
        raise HTTPException(status_code=409, detail="receipt_actual_differs_from_selected")
    expected_change = _expected_change_kind(success=receipt.success, current=current, actual=actual)
    if receipt.changeKind and receipt.changeKind != expected_change:
        raise HTTPException(status_code=409, detail="receipt_change_kind_mismatch")
    payload = receipt.model_dump()
    payload["selectionStatus"] = (store.latest_delivery(str(run["run_id"]), PHASE_SELECTION) or {}).get("status")
    payload["changeKind"] = expected_change
    payload["songId"] = receipt.songId or actual.song_id
    return payload


def _receipt_song_matches(profile: Any, song_id: str) -> bool:
    if song_id == profile.song_id or song_id == profile.service_song_id():
        return True
    for binding in profile.resource_bindings.get("bindings", []):
        if isinstance(binding, dict) and binding.get("serviceSongId") == song_id:
            return True
    return False


def _expected_change_kind(*, success: bool, current: Any, actual: Any) -> str:
    if not success:
        return "failed"
    if current and actual.playback_unit_id == current.playback_unit_id:
        return "unchanged"
    if current and actual.song_id == current.song_id:
        return "same_song"
    return "cross_song"
