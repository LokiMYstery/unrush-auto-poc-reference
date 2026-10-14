from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from app import store
from app.catalog import MusicProfile, catalog
from app.config import (
    PHASE_ASSESSMENT,
    PHASE_ERROR,
    PHASE_EXPLANATION,
    PHASE_SELECTION,
    RUN_STATUS_DONE,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_WAITING_PLAYBACK,
    agent_timeout_seconds,
    model_base_url,
    model_enabled,
    model_name,
)
from app.resource_validation import validate_playback_unit
from app.environment import normalize_scene_context, spoken_context, environment_lead
from app.tts import synthesize_voice
from app.schemas import ENERGY_GOALS


@dataclass(frozen=True)
class EnergyAssessment:
    text: str
    energy_goal: str | None
    failure_reason: str | None = None
    response_id: str | None = None


SPEECH_STYLE_RULE = (
    "Speak naturally about the target music and its help or a small optional suggestion. "
    "Never announce operation results: no successfully switched, success, failure, adjustment completed, "
    "music is ready, 成功切换, 切换成功, 调整完成, 适配成功 or 适配失败. "
    "Technical outcomes are handled by application state, not spoken narration."
)
MUSIC_VARIATION_RULE = (
    "Vary musical direction using trusted profile evidence: instrument families (piano, guitar, strings, "
    "percussion, synth), acoustic/electronic timbre, groove, rhythm, warmth, brightness or articulation. "
    "Refine can try a related but different musical character; Explore can try a contrasting character. "
    "Neither means always softer, fewer layers or more space. Compare current profile and prior narration, "
    "avoid recycling the last direction, and do not invent instruments missing from profile evidence. "
    "Keep the goal and energy intention coherent; lowering energy does not require stripping all rhythm."
)
SCENE_SEMANTICS_RULE = (
    "Scene semantics are strict: context.currentScene exists only for interactionMode=scene after an explicit "
    "user selection, and may be used as the main TTS support anchor. context.systemRecommendation is only a weak "
    "system suggestion, not evidence of the user's current activity. context.potentialIntent is notification-derived "
    "Auto context; when null it means no notification intent is connected yet. Never infer bedtime, current activity "
    "or user intention from systemRecommendation or music sleep/relaxation tags. A lunch-break scene is a short "
    "daytime rest, not bedtime. localTime/timezone facts constrain speech. Do not mention these prohibitions, "
    "bedtime, sleep or sleepy as a disclaimer; simply describe daytime-compatible music."
)


class ModelSelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateDecision:
    profile: MusicProfile
    status: str
    opening_text: str | None = None
    post_receipt_text: str | None = None
    energy_goal: str | None = None


GOAL_PHASE_ICON_IDS = frozenset({
    "scene-airplane", "scene-banana", "scene-barbecue", "scene-basketball",
    "scene-beach-umbrella", "scene-beach-wave", "scene-bed", "scene-bicycle",
    "scene-brain-head", "scene-briefcase", "scene-broom-sparkle", "scene-burger",
    "scene-cake", "scene-calendar-heart", "scene-campfire", "scene-car",
    "scene-chat-bubble", "scene-cigarette", "scene-cloud", "scene-cloud-sunrise",
    "scene-cocktail", "scene-coffee-cup", "scene-cupcake", "scene-dawn",
    "scene-earbuds", "scene-flip-flops", "scene-handbag", "scene-headphones",
    "scene-heart-support", "scene-heartbeat", "scene-hearts", "scene-hot-bath",
    "scene-ice-cream", "scene-island", "scene-laptop", "scene-lightbulb",
    "scene-lipstick", "scene-moon-night", "scene-mountains-water", "scene-music-note",
    "scene-noodles", "scene-paintbrush", "scene-palette", "scene-party-popper",
    "scene-pizza", "scene-potted-plant", "scene-rain-cloud", "scene-rocking-horse",
    "scene-round-glasses", "scene-rubber-duck", "scene-snowflake", "scene-sofa",
    "scene-sun", "scene-sunrise", "scene-sunrise-water", "scene-sunset",
    "scene-swimmer", "scene-swirl", "scene-telescope", "scene-thunder-cloud",
    "scene-toilet", "scene-ufo", "scene-umbrella", "scene-vinyl-record",
    "scene-watch", "scene-water-waves", "scene-wave", "scene-wind", "scene-zzz-sleep",
})
DEFAULT_GOAL_PHASE_ICON_ID = "scene-music-note"


@dataclass(frozen=True)
class GoalPlanPhase:
    title: str
    duration_minutes: int
    icon_id: str = DEFAULT_GOAL_PHASE_ICON_ID


@dataclass(frozen=True)
class GoalPlan:
    title: str
    total_minutes: int
    phases: list[GoalPlanPhase]
    response_id: str | None = None
    pending_tool_outputs: list[dict[str, Any]] | None = None


def _normalize_legacy_refine_feedback(context: dict[str, Any]) -> None:
    # The old generic Refine button always uploaded this phrase, even when the
    # user had not requested softer music. Normalize before assessment and search.
    if str(context.get("userFeedback") or "").casefold() == "refine: softer and fewer layers":
        context["userFeedback"] = "refine: try a different musical character for this moment"


def _normalize_context_for_agent(context: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_scene_context(context)
    context.clear()
    context.update(normalized)
    return context


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex}"


def new_goal_id() -> str:
    return f"goal_{uuid.uuid4().hex}"


def new_goal_phase_id() -> str:
    return f"gphase_{uuid.uuid4().hex}"


def _client() -> AsyncOpenAI:
    kwargs: dict[str, Any] = {"api_key": __import__("os").getenv("AGENT_MODEL_API_KEY") or __import__("os").getenv("OPENAI_API_KEY")}
    base_url = model_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


async def run_task(run_id: str) -> None:
    run = store.get_run(run_id)
    if not run or run["status"] != RUN_STATUS_RUNNING:
        return
    try:
        context = store.loads(run["context_json"], {})
        _normalize_legacy_refine_feedback(context)
        current_playback = store.loads(run["current_playback_json"], {})
        catalog_payload = store.loads(run["catalog_json"], {})
        allowed = catalog().candidate_ids_from_payload({"catalog": catalog_payload})
        if not allowed:
            allowed = set(catalog().by_playback_unit_id)
        current_id = str(current_playback.get("playbackUnitId") or current_playback.get("playback_unit_id") or "")
        language = str(context.get("language") or "zh")
        if context.get("goalMode") is True:
            await run_goal_task(run_id, context, current_playback, allowed, current_id, language)
            return
        existing_assessment = store.latest_delivery(run_id, PHASE_ASSESSMENT, {"text_ready", "voice_ready", "text_only"})
        assessment = str(existing_assessment.get("text") or "") if existing_assessment else ""
        energy_goal = existing_assessment.get("energyGoal") if existing_assessment else context.get("energyGoal")
        if not assessment or (model_enabled() and energy_goal not in ENERGY_GOALS):
            result = await generate_assessment(context, current_playback, language)
            if not assessment:
                assessment = result.text
            energy_goal = result.energy_goal
            if result.failure_reason:
                store.record_retrieval_event(run_id, "energy_goal_failure", {"reason": result.failure_reason})
        if energy_goal in ENERGY_GOALS:
            context["energyGoal"] = energy_goal
            if not store.update_run(run_id, context_json=store.dumps(context)):
                return
            store.record_retrieval_event(run_id, "energy_goal_selected", {"energyGoal": energy_goal, "source": "agent"})
        if energy_goal is None and model_enabled():
            raise ModelSelectionError("energy_goal_not_chosen")
        # An interrupted older release may have persisted a numeric text_ready
        # assessment before synthesis. Apply the same speech boundary on recovery.
        if _speaks_pressure_score(assessment) or _announces_operation_outcome(assessment):
            assessment = environment_lead(context, language) + energy_goal_copy(energy_goal, language, context)
        if _cancelled(run_id):
            return
        decision_task = asyncio.create_task(choose_with_agent_tools(
            run_id=run_id,
            assessment=assessment,
            context=context,
            current_playback=current_playback,
            allowed=allowed,
            current_id=current_id,
            language=language,
        ))
        if not existing_assessment:
            store.add_delivery(run_id=run_id, phase=PHASE_ASSESSMENT, text=assessment, status="text_ready", energy_goal=energy_goal)
        voice_done = store.latest_delivery(run_id, PHASE_ASSESSMENT, {"voice_ready", "text_only"})
        if not voice_done:
            if context.get("suppressAssessmentVoice") is True:
                store.add_delivery(run_id=run_id, phase=PHASE_ASSESSMENT, text=assessment,
                                   status="text_only", energy_goal=energy_goal)
            else:
                assessment_voice_task = asyncio.create_task(synthesize_voice(run_id, PHASE_ASSESSMENT, assessment))
                await _deliver_assessment_voice_before_selection(assessment_voice_task, run_id, assessment, energy_goal)
        if _cancelled(run_id):
            decision_task.cancel()
            return
        decision = await decision_task
        selected = decision.profile
        if _cancelled(run_id):
            return
        if selected is None:
            raise RuntimeError("no_valid_playback_unit")
        context["selectionStatus"] = decision.status
        store.update_run(
            run_id,
            context_json=store.dumps(context),
            status=RUN_STATUS_WAITING_PLAYBACK,
            selected_playback_unit_id=selected.playback_unit_id,
            selected_song_id=selected.song_id,
        )
        if _cancelled(run_id):
            return
        store.add_delivery_once_phase(
            run_id=run_id,
            phase=PHASE_SELECTION,
            playback_unit_id=selected.playback_unit_id,
            status=decision.status, energy_goal=energy_goal,
        )
    except Exception as exc:
        text = "The adjustment did not finish. You can keep listening or try again." if str(context.get("language") or "zh").startswith("en") else "这次自动适配未完成，可以继续听或重试。"
        store.add_delivery(run_id=run_id, phase=PHASE_ERROR, text=text, status=type(exc).__name__)
        store.update_run(run_id, status=RUN_STATUS_FAILED)


async def run_goal_task(
    run_id: str,
    context: dict[str, Any],
    current_playback: dict[str, Any],
    allowed: set[str],
    current_id: str,
    language: str,
) -> None:
    persisted_selected_id = str(store.get_run(run_id).get("selected_playback_unit_id") or "") if store.get_run(run_id) else ""
    selected = catalog().profile(persisted_selected_id) if persisted_selected_id else None
    if selected is not None:
        energy_goal = context.get("energyGoal") if context.get("energyGoal") in ENERGY_GOALS else None
        opening = sanitized_goal_opening_text(context.get("goalOpeningText"), energy_goal, language, context)
        post_receipt_text = sanitized_goal_post_receipt_text(context.get("goalPostReceiptText"), language, context)
        decision_status = str(context.get("selectionStatus") or (store.latest_delivery(run_id, PHASE_SELECTION) or {}).get("status") or "model_selected")
    else:
        phase = context.get("goalPhase") if isinstance(context.get("goalPhase"), dict) else {}
        phase_title = str(phase.get("title") or context.get("userFeedback") or "goal")
        decision = await choose_with_agent_tools(
            run_id=run_id,
            assessment=phase_title,
            context=context,
            current_playback=current_playback,
            allowed=allowed,
            current_id=current_id,
            language=language,
        )
        if _cancelled(run_id):
            return
        selected = decision.profile
        energy_goal = decision.energy_goal if decision.energy_goal in ENERGY_GOALS else (context.get("energyGoal") if context.get("energyGoal") in ENERGY_GOALS else None)
        opening = sanitized_goal_opening_text(decision.opening_text, energy_goal, language, context)
        post_receipt_text = sanitized_goal_post_receipt_text(decision.post_receipt_text, language, context)
        decision_status = decision.status
        context["selectionStatus"] = decision_status
        context["goalOpeningText"] = opening
        if energy_goal in ENERGY_GOALS:
            context["energyGoal"] = energy_goal
        if post_receipt_text:
            context["goalPostReceiptText"] = post_receipt_text
        if not store.update_run(
            run_id,
            context_json=store.dumps(context),
            selected_playback_unit_id=selected.playback_unit_id,
            selected_song_id=selected.song_id,
        ):
            return
    if _cancelled(run_id):
        return
    existing_assessment = store.latest_delivery(run_id, PHASE_ASSESSMENT, {"voice_ready", "text_only", "text_ready"})
    if not existing_assessment:
        if context.get("suppressAssessmentVoice") is True:
            store.add_delivery(run_id=run_id, phase=PHASE_ASSESSMENT, text=opening, status="text_only", energy_goal=energy_goal)
        else:
            voice_id = await synthesize_voice(run_id, PHASE_ASSESSMENT, opening)
            if _cancelled(run_id):
                return
            if voice_id:
                store.add_delivery(run_id=run_id, phase=PHASE_ASSESSMENT, text=opening, voice_id=voice_id, status="voice_ready", energy_goal=energy_goal)
            else:
                store.add_delivery(run_id=run_id, phase=PHASE_ASSESSMENT, text=opening, status="text_only", energy_goal=energy_goal)
    if _cancelled(run_id):
        return
    store.add_delivery_once_phase(
        run_id=run_id,
        phase=PHASE_SELECTION,
        playback_unit_id=selected.playback_unit_id,
        status=decision_status,
        energy_goal=energy_goal,
    )
    store.update_run(run_id, status=RUN_STATUS_WAITING_PLAYBACK)


async def choose_with_agent_tools(
    *,
    run_id: str,
    assessment: str,
    context: dict[str, Any],
    current_playback: dict[str, Any],
    allowed: set[str],
    current_id: str,
    language: str,
) -> CandidateDecision:
    current = catalog().profile(current_id)
    if not model_enabled():
        if current:
            return CandidateDecision(current, "model_unavailable_keep")
        raise ModelSelectionError("model_unavailable_no_current_unit")
    try:
        selected_ids = await model_select_with_retrieval_tools(
            assessment=assessment,
            context=context,
            current_playback=current_playback,
            allowed=allowed,
            current_id=current_id,
            language=language,
            run_id=run_id,
        )
    except Exception as exc:
        if current:
            return CandidateDecision(current, "model_error_keep:" + (str(exc) if isinstance(exc, ModelSelectionError) else type(exc).__name__))
        raise
    for playback_id in selected_ids:
        if _cancelled(run_id):
            raise asyncio.CancelledError()
        profile = catalog().profile(playback_id)
        if profile is None:
            continue
        origin_scene_id = current_playback.get("sceneId") or current_playback.get("originSceneId")
        result = await validate_playback_unit(profile, origin_scene_id=origin_scene_id if isinstance(origin_scene_id, str) else None)
        store.record_retrieval_event(run_id, "resource_validation", {
            "playbackUnitId": playback_id, "ok": result.ok, "reason": result.reason})
        if result.ok:
            submitted = store.latest_retrieval_event(run_id, "selection_submitted") or {}
            primary = submitted.get("reasonPlaybackUnitId")
            store.record_retrieval_event(run_id, "candidate_chosen", {
                "playbackUnitId": profile.playback_unit_id, "modelPrimaryPlaybackUnitId": primary,
                "reason": submitted.get("reason") if primary == profile.playback_unit_id else None,
                "reasonAppliesToChosenCandidate": primary == profile.playback_unit_id,
                "selectionPolicy": "internal-nine-preferred-v1", "resourceValidation": result.reason})
            opening = submitted.get("openingText") if primary == profile.playback_unit_id else None
            post_receipt = submitted.get("postReceiptText") if primary == profile.playback_unit_id else None
            selected_energy = submitted.get("energyGoal") if submitted.get("energyGoal") in ENERGY_GOALS else None
            if profile.playback_unit_id == current_id:
                return CandidateDecision(profile, "model_selected_unchanged", opening, post_receipt, selected_energy)
            return CandidateDecision(profile, "model_selected", opening, post_receipt, selected_energy)
    if current:
        return CandidateDecision(current, "model_selection_invalid_keep")
    raise ModelSelectionError("model_selection_had_no_valid_candidate")


async def continue_after_receipt(run_id: str) -> None:
    run = store.get_run(run_id)
    if not run or run["status"] not in {RUN_STATUS_WAITING_PLAYBACK, RUN_STATUS_RUNNING}:
        return
    if store.delivery_exists(run_id, PHASE_EXPLANATION):
        store.update_run(run_id, status=RUN_STATUS_DONE)
        return
    receipt = store.loads(run["receipt_json"], {})
    context = store.loads(run["context_json"], {})
    current_playback = store.loads(run["current_playback_json"], {})
    spoken_assessment = store.latest_delivery(run_id, PHASE_ASSESSMENT, {"voice_ready", "text_only", "text_ready"})
    if spoken_assessment:
        context.setdefault("alreadySpokenAssessment", spoken_assessment.get("text"))
    selection_trace = store.latest_retrieval_event(run_id, "candidate_chosen")
    if selection_trace and receipt.get("success") and selection_trace.get("playbackUnitId") == receipt.get("playbackUnitId"):
        context["selectionRationale"] = selection_trace.get("reason")
        context["selectionPolicy"] = selection_trace.get("selectionPolicy")
    selected_id = run.get("selected_playback_unit_id")
    actual_id = str(receipt.get("playbackUnitId") or current_playback.get("playbackUnitId") or "")
    actual_profile = catalog().profile(actual_id)
    selected_profile = catalog().profile(str(selected_id or ""))
    selection_delivery = store.latest_delivery(run_id, PHASE_SELECTION) or {}
    selection_status = str(selection_delivery.get("status") or "")
    receipt["selectionStatus"] = selection_status
    degraded_keep = (selection_status == "model_unavailable_keep" or selection_status == "model_selection_invalid_keep"
                     or selection_status.startswith("model_error_keep:"))
    prebuilt_goal_explanation = context.get("goalPostReceiptText") if context.get("goalMode") is True else None
    if (isinstance(prebuilt_goal_explanation, str) and prebuilt_goal_explanation.strip()
            and receipt.get("success") and receipt.get("actualPlaybackState") == "playing" and not degraded_keep
            and actual_profile is not None and selected_profile is not None
            and actual_profile.playback_unit_id == selected_profile.playback_unit_id):
        explanation = sanitized_goal_post_receipt_text(prebuilt_goal_explanation, str(context.get("language") or "zh"), context)
    else:
        explanation = await generate_explanation(
            context=context,
            selected=selected_profile,
            actual=actual_profile,
            receipt=receipt,
            language=str(context.get("language") or "zh"),
        )
    # Failure/degraded-keep copy is text only; never synthesize a spoken failure notification.
    voice_id = await synthesize_voice(run_id, PHASE_EXPLANATION, explanation) if receipt.get("success") and not degraded_keep else None
    if _cancelled(run_id):
        return
    store.add_delivery_once_phase(
        run_id=run_id,
        phase=PHASE_EXPLANATION,
        text=explanation,
        playback_unit_id=actual_profile.playback_unit_id if actual_profile else actual_id,
        voice_id=voice_id,
        status="completed" if receipt.get("success") else "playback_failed",
        energy_goal=context.get("energyGoal") if context.get("energyGoal") in ENERGY_GOALS else None,
    )
    store.update_run(run_id, status=RUN_STATUS_DONE)


async def generate_assessment(context: dict[str, Any], current_playback: dict[str, Any], language: str) -> EnergyAssessment:
    context = _normalize_context_for_agent(context)
    fallback = "I’ll look for a musical adjustment that fits your request." if language.startswith("en") else "我会结合正在听的音乐，找一个更贴合你的调整方向。"
    lead = environment_lead(context, language)
    if not model_enabled():
        return EnergyAssessment(lead + fallback, None)
    prompt = {
        "task": "Choose one musical energy direction, then describe the music you intend to select or adjust.",
        "language": language,
        "context": spoken_context(context, language),
        "environmentLead": lead,
        "currentPlayback": current_playback,
        "currentMusic": _music_for_narration(catalog().profile(str(current_playback.get("playbackUnitId") or ""))),
        "goalContinuity": {
            "previousOpenings": context.get("goalPreparedOpenings") if isinstance(context.get("goalPreparedOpenings"), list) else [],
            "usedCarrierSongIds": context.get("goalUsedServiceSongIds") if isinstance(context.get("goalUsedServiceSongIds"), list) else [],
        },
        "rules": [
            SPEECH_STYLE_RULE,
            MUSIC_VARIATION_RULE,
            "Interpret localTime in context.timezone; a UTC Z timestamp must be converted before describing the user's local time of day.",
            "Use available place type, weather, optional health references, activity/sleep and user feedback to describe a musical direction. Do not invent missing context or report a checklist of missing permissions.",
            "Do not claim health measurements when stress source is default or historical.",
            "Pressure is a computed reference. recent_measurement means fresh underlying HRV, not a direct measurement of stress. When baselineReady is false, describe the musical direction instead of claiming detected high stress.",
            "Do not repeat the fast welcome.",
            "When context.goalMode is true, make the user's goal title and current goalPhase the main anchor. The goal may be emotional care, companionship or venting; do not force productivity, completion, positivity, ongoing conversation, active listening, or therapeutic guarantees. Describe short musical support and at most one optional energy sentence.",
            "Follow context.environmentNarration: initial must briefly include available facts through the server's environmentLead, which is already part of this utterance. For ordinary Auto, do not repeat place, weather or body data after that lead; add the musical direction and, when context.currentScene exists, how the music helps that scene. For goalMode, return a natural short support description for the goal phase and optional energy direction. changed has only changedFields in that lead; music_only must not repeat weather, place, time of day, pressure, health or missing data. Focus on music and how it can help.",
            "Use health.pressure.levelLabel from StressLab; never speak a numeric pressure/stress score, rating, percent, or points. HRV labels and heart-rate estimate labels differ; retain the supplied label. Historical references must be identified as earlier data, never a current reading. When baselineReady is false or levelLabel is absent, do not assert a body-pressure state.",
            "The server will prepend environmentLead to this same first utterance. For ordinary Auto, return a concise musical direction after that lead; if context.currentScene exists, include how the music helps that scene. For goalMode, return a concise phase support sentence and optional energy direction. Do not repeat weather, place, body reference or other facts.",
            "Never speak track/stem numbers or say keep the same song. Describe musical qualities instead.",
            "Choose exactly one of two directions: boost supports gently raising energy; lower supports lowering energy and settling. Do not choose maintain or a neutral third direction. This is a musical support intention, not a measured energy level or medical effect.",
            "For goalMode, the assessment may naturally contain zero or one explicit energy sentence. It should sound like a short opening for the phase, not a task-management command. Vary wording from goalContinuity.previousOpenings.",
            SCENE_SEMANTICS_RULE,
            "For ordinary Auto, anchor the utterance to the selected energy goal and intended musical qualities. Use at most one energy sentence; spend more of the sentence on how the music helps context.currentScene when it is present. For goalMode, anchor to the phase and musical support; energy wording is optional. Do not announce the current song or current playback. Do not name an unknown next song before retrieval.",
            "Submit the set_energy_goal function with energyGoal and one concise user-language assessment.",
        ],
    }
    try:
        request_input: Any = json.dumps(prompt, ensure_ascii=False)
        previous_response_id = context.get("goalAgentPreviousResponseId") if context.get("goalMode") is True else None
        if isinstance(previous_response_id, str) and previous_response_id:
            request_input = _goal_chain_input(context, prompt)
        request_kwargs: dict[str, Any] = {
            "model": model_name(),
            "input": request_input,
            "tools": [energy_goal_tool_schema()],
            "tool_choice": {"type": "function", "name": "set_energy_goal"},
            "max_output_tokens": 2400,
        }
        if isinstance(previous_response_id, str) and previous_response_id:
            request_kwargs["previous_response_id"] = previous_response_id
        response = await asyncio.wait_for(_client().responses.create(**request_kwargs), timeout=agent_timeout_seconds())
        response_id = getattr(response, "id", None)
        if context.get("goalMode") is True and isinstance(response_id, str) and response_id:
            context["goalAgentPreviousResponseId"] = response_id
        calls = [call for call in function_calls(response) if call.name == "set_energy_goal"]
        if len(calls) != 1 or getattr(response, "status", None) == "incomplete":
            raise ModelSelectionError("energy_goal_not_submitted")
        value = json.loads(calls[0].arguments)
        goal = value.get("energyGoal")
        if goal not in ENERGY_GOALS:
            raise ModelSelectionError("invalid_energy_goal")
        utterance = " ".join(str(value.get("assessment") or "").split())
        if len(utterance) > 240:
            utterance = utterance[:240].rsplit(" ", 1)[0] if language.startswith("en") else utterance[:240]
        goal_mode = context.get("goalMode") is True
        repeated_energy = goal_mode and _explicit_energy_mentions(utterance, language) > 1
        if (not utterance or _forbidden_music_copy(utterance) or _announces_current_music(utterance)
                or _implies_ongoing_listening(utterance) or _speaks_pressure_score(utterance)
                or _announces_operation_outcome(utterance) or repeated_energy or _repeats_environment(utterance, lead)):
            utterance = energy_goal_copy(goal, language, context)
        if context.get("goalMode") is True:
            _set_goal_chain_pending_outputs(context, calls, _energy_goal_ack_payload(goal, utterance))
        return EnergyAssessment(lead + utterance, goal, response_id=response_id if isinstance(response_id, str) else None)
    except Exception as exc:
        # A failed model call does not invent a goal; only safe diagnostics are retained.
        reason = str(exc) if isinstance(exc, ModelSelectionError) else type(exc).__name__
        return EnergyAssessment(lead + fallback, None, reason)


async def generate_goal_plan(*, user_input: str, split: bool, language: str,
                             context: dict[str, Any], current_playback: dict[str, Any]) -> GoalPlan:
    if not model_enabled():
        return _fallback_goal_plan(user_input=user_input, split=split, language=language)
    prompt = {
        "task": "Turn a user's playback-page goal into an executable music journey plan.",
        "language": language,
        "userGoal": user_input,
        "splitRequested": split,
        "context": spoken_context(context, language),
        "currentPlayback": current_playback,
        "rules": [
            "Return a concise title that preserves the user's intent.",
            "Return totalMinutes as a positive integer. If the user did not give a time, estimate a practical duration.",
            "The goal may be productivity, emotional care, companionship, venting, grief, reassurance or a small life moment. Preserve that intent; do not force every goal into task completion, productivity, positivity or therapeutic promises.",
            "If splitRequested is false, return exactly one phase. Do not split emotional support or companionship unless splitRequested is true.",
            "If splitRequested is true, return at least two and at most eight ordered phases; for emotional or companionship goals, make phases gentle and optional rather than task-like milestones.",
            "Every phase durationMinutes must be positive; the phase durations must sum to totalMinutes.",
            "Each phase must choose one iconId from the enum that matches the phase intent. Use scene-heart-support, scene-hearts, scene-chat-bubble, scene-rain-cloud, scene-sofa or scene-moon-night for care/companionship when appropriate.",
            "Do not output long-term habit tracking, notifications, medical diagnosis, guaranteed healing, forced optimism, or completion claims.",
        ],
    }
    response = await asyncio.wait_for(
        _client().responses.create(
            model=model_name(),
            input=json.dumps(prompt, ensure_ascii=False),
            tools=[goal_plan_tool_schema()],
            tool_choice={"type": "function", "name": "submit_goal_plan"},
            max_output_tokens=1800,
        ),
        timeout=agent_timeout_seconds(),
    )
    calls = [call for call in function_calls(response) if call.name == "submit_goal_plan"]
    if len(calls) != 1 or getattr(response, "status", None) == "incomplete":
        raise ModelSelectionError("goal_plan_not_submitted")
    value = json.loads(calls[0].arguments)
    plan = normalize_goal_plan(value, split=split)
    response_id = getattr(response, "id", None)
    if isinstance(response_id, str) and response_id:
        plan = GoalPlan(plan.title, plan.total_minutes, plan.phases, response_id=response_id,
                        pending_tool_outputs=_tool_outputs_for_calls(calls, _goal_plan_ack_payload(plan)))
    return plan


def goal_plan_tool_schema() -> dict[str, Any]:
    phase = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "maxLength": 48},
            "durationMinutes": {"type": "integer", "minimum": 1, "maximum": 120},
            "iconId": {"type": "string", "enum": sorted(GOAL_PHASE_ICON_IDS)},
        },
        "required": ["title", "durationMinutes", "iconId"],
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "name": "submit_goal_plan",
        "description": "Submit one current playback goal plan.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": 64},
                "totalMinutes": {"type": "integer", "minimum": 1, "maximum": 960},
                "phases": {"type": "array", "items": phase, "minItems": 1, "maxItems": 8},
            },
            "required": ["title", "totalMinutes", "phases"],
            "additionalProperties": False,
        },
    }


def normalize_goal_plan(value: dict[str, Any], *, split: bool) -> GoalPlan:
    if not isinstance(value, dict):
        raise ModelSelectionError("goal_plan_invalid_payload")
    title = " ".join(str(value.get("title") or "").split())[:64]
    if not title:
        raise ModelSelectionError("goal_plan_missing_title")
    raw_phases = value.get("phases")
    if not isinstance(raw_phases, list):
        raise ModelSelectionError("goal_plan_missing_phase")
    if len(raw_phases) > 8:
        raise ModelSelectionError("goal_plan_too_many_phases")
    phases: list[GoalPlanPhase] = []
    for raw in raw_phases:
        if not isinstance(raw, dict):
            raise ModelSelectionError("goal_plan_invalid_phase")
        phase_title = " ".join(str(raw.get("title") or "").split())[:48]
        icon_id = normalize_goal_phase_icon_id(raw.get("iconId"))
        try:
            minutes = int(raw.get("durationMinutes"))
        except (TypeError, ValueError):
            raise ModelSelectionError("goal_plan_invalid_phase_duration") from None
        if not phase_title:
            raise ModelSelectionError("goal_plan_invalid_phase_title")
        if not 1 <= minutes <= 120:
            raise ModelSelectionError("goal_plan_invalid_phase_duration")
        phases.append(GoalPlanPhase(phase_title, minutes, icon_id))
    if split and len(phases) < 2:
        raise ModelSelectionError("goal_plan_needs_multiple_phases")
    if not split:
        if phases:
            phases = [phases[0]]
        else:
            raise ModelSelectionError("goal_plan_missing_phase")
    if not phases:
        raise ModelSelectionError("goal_plan_missing_phase")
    total = sum(phase.duration_minutes for phase in phases)
    try:
        declared = int(value.get("totalMinutes"))
    except (TypeError, ValueError):
        raise ModelSelectionError("goal_plan_invalid_total") from None
    if declared <= 0 or declared > 960:
        raise ModelSelectionError("goal_plan_invalid_total")
    if declared != total:
        raise ModelSelectionError("goal_plan_total_mismatch")
    return GoalPlan(title=title, total_minutes=total, phases=phases)


def normalize_goal_phase_icon_id(value: object) -> str:
    icon_id = " ".join(str(value or "").split())
    if not icon_id:
        return DEFAULT_GOAL_PHASE_ICON_ID
    if icon_id not in GOAL_PHASE_ICON_IDS:
        raise ModelSelectionError("goal_plan_invalid_phase_icon")
    return icon_id


def _fallback_goal_plan(*, user_input: str, split: bool, language: str) -> GoalPlan:
    title = " ".join(user_input.split())[:32]
    if not title:
        title = "Current goal" if language.startswith("en") else "当前目标"
    if not split:
        phase_title = "Stay with this" if language.startswith("en") else "陪你待一会儿"
        return GoalPlan(title=title, total_minutes=25, phases=[GoalPlanPhase(phase_title, 25, "scene-heart-support")])
    if language.startswith("en"):
        phases = [
            GoalPlanPhase("Arrive gently", 10, "scene-sofa"),
            GoalPlanPhase("Stay with it", 25, "scene-heart-support"),
            GoalPlanPhase("Close softly", 10, "scene-moon-night"),
        ]
    else:
        phases = [
            GoalPlanPhase("慢慢进入", 10, "scene-sofa"),
            GoalPlanPhase("陪你停留", 25, "scene-heart-support"),
            GoalPlanPhase("轻轻收住", 10, "scene-moon-night"),
        ]
    return GoalPlan(title=title, total_minutes=sum(phase.duration_minutes for phase in phases), phases=phases)


def energy_goal_tool_schema() -> dict[str, Any]:
    return {"type": "function", "name": "set_energy_goal", "description": "Set one musical energy intention, not a health measurement.",
        "parameters": {"type": "object", "properties": {
            "energyGoal": {"type": "string", "enum": ["boost", "lower"]},
            "assessment": {"type": "string", "description": "One brief context-aware musical energy direction; qualitative body stress only, no scores or current-song announcement.", "maxLength": 240}},
            "required": ["energyGoal", "assessment"], "additionalProperties": False}}


def energy_goal_copy(goal: str | None, language: str, context: dict[str, Any] | None = None) -> str:
    phase_title = _goal_phase_title_for_copy(context)
    if phase_title:
        if language.startswith("en"):
            return {"boost": f"First, I’ll prepare music for “{phase_title}”, with a gentle rhythmic lift to help raise your energy.",
                    "lower": f"First, I’ll prepare music for “{phase_title}”, with a musical character that supports lowering your energy."}.get(goal,
                        f"First, I’ll prepare music for “{phase_title}” that supports this step.")
        return {"boost": f"先为『{phase_title}』准备音乐，选择有轻盈律动的声音，帮助能量慢慢上升。",
                "lower": f"先为『{phase_title}』准备音乐，换一种音乐表达，陪能量慢慢下降。"}.get(goal,
                    f"先为『{phase_title}』准备贴合这一步的音乐。")
    if language.startswith("en"):
        return {"boost": "I’ll look for a gentle rhythmic lift to help raise your energy.",
                "lower": "I’ll look for a different musical character to support lowering your energy."}.get(goal,
                    "I’ll look for music that supports the moment.")
    return {"boost": "我会选一些有轻盈律动的音乐，帮助能量慢慢上升。",
            "lower": "我会从音乐的音色与节奏找个方向，陪能量慢慢下降。"}.get(goal, "我会选择贴合此刻的音乐。")


def _goal_phase_title_for_copy(context: dict[str, Any] | None) -> str | None:
    if not isinstance(context, dict) or context.get("goalMode") is not True:
        return None
    phase = context.get("goalPhase")
    title = phase.get("title") if isinstance(phase, dict) else None
    if not isinstance(title, str):
        return None
    compact = " ".join(title.split())[:32]
    return compact or None


def _explicit_energy_mentions(text: str, language: str) -> int:
    if language.startswith("en"):
        return len(re.findall(r"\benergy\b|raise(?:s|d|ing)? your energy|lower(?:s|ed|ing)? your energy", text, re.I))
    return text.count("能量")


def _implies_ongoing_listening(text: str) -> bool:
    return bool(re.search(r"我(?:会|在)?听你(?:说|讲)|我会继续听|继续(?:跟我|和我)说|I(?:'ll| will)? listen to you|keep talking to me|tell me more", text, re.I))


def _announces_current_music(text: str) -> bool:
    return bool(re.search(r"(?:now|currently) (?:playing|listening)|\bis (?:currently )?playing\b|当前(?:正在)?播放|现在(?:正在)?(?:播放|在播)|正在播放", text, re.I))


def _announces_operation_outcome(text: str) -> bool:
    # Operation reports must not reach speech, including persisted prebuilt text.
    # Goal/emotional language about the user's own experience remains allowed.
    return bool(re.search(
        r"(?:成功|失败)[^。！？，,;]{0,6}(?:切换|调整|适配|播放)|"
        r"(?:切换|调整|适配|播放)(?:已经|已)?(?:成功|失败|完成)|"
        r"(?:调整|适配|切换)[^。！？，,;]{0,8}(?:未完成|未成功|失败)|"
        r"(?:音乐|歌曲)(?:已经|已)?(?:准备好|就绪)|"
        r"successfully|(?:switch|adjustment|adaptation|playback|music)[^.!?,;]{0,24}"
        r"(?:succeeded|successful|failed|completed|is ready)|"
        r"(?:success|failure)\s*(?:in|of)?\s*(?:switch|adjust|adapt|play)", text, re.I))


async def generate_explanation(
    *,
    context: dict[str, Any],
    selected: MusicProfile | None,
    actual: MusicProfile | None,
    receipt: dict[str, Any],
    language: str,
) -> str:
    context = _normalize_context_for_agent(context)
    success = bool(receipt.get("success"))
    change_kind = receipt.get("changeKind") or ("failed" if not success else "unchanged")
    status = str(receipt.get("selectionStatus") or "")
    degraded_keep = status in {"model_unavailable_keep", "model_selection_invalid_keep"} or status.startswith("model_error_keep:")
    if not success or degraded_keep:
        return ""
    else:
        names = (_music_for_narration(actual) or {}).get("songName") or {}
        title = names.get("en" if language.startswith("en") else "zh") or next(iter(names.values()), "")
        if receipt.get("actualPlaybackState") == "playing":
            if context.get("goalMode") is True:
                fallback = goal_explanation_copy(title=title, language=language, context=context)
            else:
                fallback = (f"{title} can accompany this moment. " if language.startswith("en") else f"《{title}》陪着此刻的你。") + energy_goal_copy(context.get("energyGoal"), language, context)
        else:
            fallback = "You can take a moment at your own pace." if language.startswith("en") else "可以按自己的节奏，慢慢待一会儿。"
    if not model_enabled():
        return fallback
    prompt = {
        "task": "Briefly connect the verified target music to the user's goal and a helpful musical direction.",
        "language": language,
        "context": spoken_context(context, language),
        "selectedProfile": _music_for_narration(selected),
        "actualProfile": _music_for_narration(actual),
        "receipt": receipt,
        "alreadySpokenAssessment": context.get("alreadySpokenAssessment"),
        "alreadySpokenGoalExplanations": context.get("goalSpokenExplanations") if isinstance(context.get("goalSpokenExplanations"), list) else [],
        "currentPlayback": context.get("currentPlaybackForExplanation"),
        "actualPlaybackState": receipt.get("actualPlaybackState"),
        "rules": [
            SPEECH_STYLE_RULE,
            MUSIC_VARIATION_RULE,
            "Use actual receipt internally to ground the music description; never narrate operation status.",
            "When playback succeeds, briefly connect the actual music profile and recorded selection rationale to the current environment, explicit currentScene or user request; do not merely announce the song title.",
            SCENE_SEMANTICS_RULE,
            "When context.goalMode is true, explain the selected music through context.goal.title and context.goalPhase.title first. Do not mention energy/能量 here; explanation should be one short music phrase plus one concrete optional light suggestion. Avoid repeating any prior suggestion in alreadySpokenGoalExplanations. Do not imply the app is listening to the user speak or will continue a conversation.",
            "Use public-facing musical qualities. Do not mention internal resource-binding scenes, tool names, database IDs, or claim that a model rationale overrides the actual receipt.",
            "Failed/degraded outcomes remain silent. Do not announce either success or failure.",
            "When receipt.isExactPlaybackUnit is false, the actual audio is a manual mix; do not describe a solo track profile as the exact sound.",
            "Do not claim audible ongoing playback unless receipt.actualPlaybackState is playing. Missing state is unverified.",
            "Distinguish same_song, cross_song, unchanged, failed.",
            "When success is true and changeKind is unchanged with a normal selection status, describe the current music and its help. A valid unchanged choice is not a failed adaptation; do not say it failed, was not implemented, 没有落实 or 未完成. Do not claim an audible change either.",
            "For same_song, describe the musical change and its benefit only; never say keep the same song, same song, 同一首歌 or an equivalent reassurance about retaining it.",
            "Use songName when naming the music. Never mention track/stem numbers, 轨道几 or internal unit names.",
            "The assessment already covered any needed context. Do not repeat weather, place, pressure or time of day here. Focus on the actual music, its changes and intended help; do not guarantee a health effect.",
            "For ordinary Auto, use context.energyGoal (boost or lower) as a light musical direction, never a guaranteed measured energy change. If context.currentScene exists, explain how the selected music helps that scene; otherwise focus on the selected music's qualities and the light energy direction. For goalMode, do not mention energy/能量; keep the goal and music as the primary anchor.",
            "Do not announce now playing/current playback or introduce the previous song. Describe the selected/target music and how it supports the energy goal. Same-song changes describe the target musical change only.",
            "For ordinary Auto, use at most one energy sentence; energy wording may be brief and does not need the literal word energy/能量 when the scene support is clearer. For goalMode, avoid the words energy/能量 and keep the music description shorter than the suggestion.",
            "Return only one short utterance.",
        ],
    }
    utterance = await _responses_text(prompt, fallback=fallback, max_chars=140)
    # The live model called a successful unchanged choice '没有落实调整'. Such failure
    # copy contradicts the receipt and must not reach TTS as a success explanation.
    contradictory_failure = success and not degraded_keep and bool(re.search(
        r"未完成|未成功|未能|没能|失败|没有(?:落实|完成|生效|实现)|did(?:n't| not) (?:finish|complete|take effect|succeed)|failed|could(?:n't| not)", utterance, re.I))
    goal_mode = context.get("goalMode") is True
    repeated_goal_energy = goal_mode and _explicit_energy_mentions(utterance, language) > 0
    return fallback if (_forbidden_music_copy(utterance) or _announces_current_music(utterance)
                        or _implies_ongoing_listening(utterance) or _speaks_pressure_score(utterance)
                        or _announces_operation_outcome(utterance) or contradictory_failure or repeated_goal_energy) else utterance


def _invalid_goal_prebuilt_text(text: str, language: str, *, max_energy_mentions: int) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True
    return (
        _forbidden_music_copy(text)
        or _announces_operation_outcome(text)
        or _announces_current_music(text)
        or _implies_ongoing_listening(text)
        or _speaks_pressure_score(text)
        or _explicit_energy_mentions(text, language) > max_energy_mentions
    )


def sanitized_goal_opening_text(text: str | None, energy_goal: str | None, language: str, context: dict[str, Any]) -> str:
    candidate = " ".join(str(text or "").split())[:120]
    if _invalid_goal_prebuilt_text(candidate, language, max_energy_mentions=1):
        return goal_opening_copy(energy_goal, language, context)
    return candidate


def goal_post_receipt_copy(language: str, context: dict[str, Any]) -> str:
    phase_title = _goal_phase_title_for_copy(context)
    if language.startswith("en"):
        return f"The music can accompany “{phase_title}”; you can take one small note forward." if phase_title else "Let the music accompany this moment; you can take one small note forward."
    return f"这段音乐会贴着『{phase_title}』往前一点；可以先带走一个小线索。" if phase_title else "这段音乐会贴着这个目标往前一点；可以先带走一个小线索。"


def sanitized_goal_post_receipt_text(text: str | None, language: str, context: dict[str, Any]) -> str:
    candidate = " ".join(str(text or "").split())[:120]
    if _invalid_goal_prebuilt_text(candidate, language, max_energy_mentions=0):
        return goal_post_receipt_copy(language, context)
    return candidate


def goal_opening_copy(energy_goal: str | None, language: str, context: dict[str, Any]) -> str:
    phase_title = _goal_phase_title_for_copy(context)
    if language.startswith("en"):
        if phase_title:
            return f"I’ll look for a musical direction to accompany “{phase_title}”."
        return "I’ll look for a musical direction to accompany this goal."
    if phase_title:
        return f"先为『{phase_title}』找个音乐方向，陪你走过这一段。"
    return "先为这个目标找个音乐方向，陪你走过这一段。"


def goal_explanation_copy(*, title: str, language: str, context: dict[str, Any]) -> str:
    phase_title = _goal_phase_title_for_copy(context)
    if language.startswith("en"):
        music = f"{title} can accompany this goal." if title else "Let the music accompany this goal."
        suggestion = f"For “{phase_title}”, you can place one thought beside the music." if phase_title else "You can place one thought beside the music."
        return f"{music} {suggestion}"
    music = f"《{title}》会陪着这个目标往前一点。" if title else "这段音乐会陪着这个目标往前一点。"
    suggestion = f"如果愿意，可以把『{phase_title}』里的一点念头先放在音乐旁边。" if phase_title else "如果愿意，可以把此刻的一点念头先放在音乐旁边。"
    return music + suggestion


async def _responses_text(prompt: dict[str, Any], *, fallback: str, max_chars: int) -> str:
    try:
        response = await asyncio.wait_for(
            _client().responses.create(
                model=model_name(),
                input=json.dumps(prompt, ensure_ascii=False),
                max_output_tokens=768,
            ),
            timeout=agent_timeout_seconds(),
        )
        text = getattr(response, "output_text", None)
        if not text:
            parts: list[str] = []
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    value = getattr(content, "text", None)
                    if value:
                        parts.append(value)
            text = " ".join(parts)
        text = " ".join(str(text or "").strip().strip('"').split())
        if not text:
            return fallback
        if len(text) <= max_chars:
            return text
        clipped = text[:max_chars].rstrip("，。,.!！?？ ")
        if text[max_chars - 1].isascii() and text[max_chars - 1].isalpha() and text[max_chars].isalpha():
            clipped = clipped.rsplit(" ", 1)[0]
        return clipped
    except Exception:
        return fallback


async def model_select_with_retrieval_tools(
    *,
    assessment: str,
    context: dict[str, Any],
    current_playback: dict[str, Any],
    allowed: set[str],
    current_id: str,
    language: str,
    run_id: str | None = None,
) -> list[str]:
    context = _normalize_context_for_agent(context)
    current = catalog().profile(current_id)
    feedback = str(context.get("userFeedback") or "").casefold()
    supports_internal = bool(current and current.unit_kind == "touchpad_track"
                             and current_playback.get("supportsTouchpad", True) is True)
    # Refine retains the current Touchpad carrier; Explore and Goal's explicit
    # music-adjustment button request a new carrier. A plain retry is neither.
    cross_song_request = bool(context.get("goalFeedback")) or feedback.startswith("explore:") or (
        feedback.startswith("refine:") and (not supports_internal or context.get("goalMode") is True))
    refine_internal = feedback.startswith("refine:") and supports_internal and not cross_song_request
    _normalize_legacy_refine_feedback(context)
    excluded_current = {pid for pid in allowed if current and catalog().profile(pid)
                        and catalog().profile(pid).song_id == current.song_id} if cross_song_request else set()
    allowed = allowed - excluded_current
    has_internal_tool = supports_internal and not cross_song_request
    tools = retrieval_tool_schemas(include_internal=has_internal_tool)
    required_tools = {tool["name"] for tool in tools}

    internal = {item.playback_unit_id for item in catalog().same_song_candidates(current_id, allowed)}

    def current_scope(recalled: set[str]) -> set[str]:
        if refine_internal:
            return recalled & internal
        if cross_song_request and current:
            return {pid for pid in recalled if catalog().profile(pid)
                    and catalog().profile(pid).song_id != current.song_id}
        return recalled

    def selection_scope(recalled: set[str]) -> set[str]:
        recalled = current_scope(recalled)
        if context.get("goalMode") is True:
            return _exclude_used_goal_carriers(recalled, context, origin_scene_id)
        if current and not has_internal_tool and not cross_song_request:
            return {item for item in recalled if item == current_id or catalog().profile(item).song_id != current.song_id}
        return recalled
    retrieval_prompt = {
                    "task": "Retrieve candidates for one Auto music adaptation. Call all available retrieval tools before choosing. Do not invent playback IDs.",
                    "toolAvailability": {"supportsTouchpad": has_internal_tool, "requiredTools": sorted(required_tools),
                                         "userRequestedDifferentSong": cross_song_request},
                    "assessment": assessment,
                    "language": language,
                    "context": context,
                    "currentPlayback": current_playback,
                    "currentTrustedProfile": _profile_for_prompt(catalog().profile(current_id)),
                    "retrievalContract": {
                        "same_song_profile_search": "inspect and compare the full nine logical Touchpad tracks of the current carrier; optional structured criteria rank only this internal set, never another song",
                        "library_profile_search": "send donor Music Profile structured fields: nameQuery, tagPreferences, axisTargets, candidateKinds",
                        "music_description_recall": "send an explicit English queryText; server rejects non-English instead of rewriting",
                    },
                    "selectionRules": [
                        SPEECH_STYLE_RULE,
                        MUSIC_VARIATION_RULE,
                        SCENE_SEMANTICS_RULE,
                        "Only when toolAvailability.supportsTouchpad is true, prefer an internal track adjustment when it fits the target; choose another song when the internal set is not a good fit or Explore requests another song.",
                        "Ordinary Refine stays inside the current Touchpad carrier and may change instrumentation, groove or timbre across its nine profiles. With no Touchpad, Refine chooses a different carrier. Explore and the Goal music-adjustment button (goalFeedback) always choose a different carrier; the entire current carrier is excluded from retrieval and selection for those requests. Automatic adaptation without feedback retains internal Touchpad preference. Keep is handled by the client without a new selection.",
                        "Retrieval tools may run in parallel. Internal preference is a comparison/selection rule, not a ban on parallel library search.",
                        "Include one short user-facing rationale comparing the chosen candidate to current/internal options; do not return private reasoning.",
                        "In the public rationale describe musical changes and help, never track numbers or keep the same song.",
                        "Use context.energyGoal as the anchor for musical search/ranking: boost raises energy, lower lowers energy. Compare with currentTrustedProfile internally, but public speech names only target music/qualities.",
                        "Treat context.currentScene as the explicit scene anchor only when present; it may influence ranking and public rationale. Treat context.systemRecommendation as weak context for candidate diversity/ranking, not as proof of what the user is doing. Treat context.potentialIntent as a notification-derived Auto hint when present or null when notifications are not connected.",
                        "When context.goalMode is true, goal.title and goalPhase.title are the primary intent. Goal mode may choose across the full recalled library and must not apply ordinary Refine hard limits to the current carrier. Touchpad internal adjustment is useful only when it best supports the goal phase.",
                        "For goalMode, avoid repeating any carrier/service song listed in context.goalUsedServiceSongIds when enough validated alternatives exist. If resources are insufficient, repeats are allowed only with a clear public rationale.",
                        "For goalMode, submit energyGoal, openingText and postReceiptText with the selected music. openingText is the phase-opening TTS: natural care or goal support, at most one energy sentence. postReceiptText is a shorter after-playback line: one music phrase plus one concrete optional light suggestion, no energy/能量 and no repeated prior suggestion from context.goalSpokenExplanations.",
                    ],
                }
    request_input: Any = json.dumps(retrieval_prompt, ensure_ascii=False)
    previous_response_id = context.get("goalAgentPreviousResponseId") if context.get("goalMode") is True else None
    if isinstance(previous_response_id, str) and previous_response_id:
        request_input = _goal_chain_input(context, retrieval_prompt)
    first_kwargs: dict[str, Any] = {
        "model": model_name(),
        "input": request_input,
        "tools": tools,
        "parallel_tool_calls": True,
        "tool_choice": "required",
        "max_output_tokens": 1600,
    }
    if isinstance(previous_response_id, str) and previous_response_id:
        first_kwargs["previous_response_id"] = previous_response_id
    first = await asyncio.wait_for(_client().responses.create(**first_kwargs), timeout=agent_timeout_seconds())
    first_response_id = getattr(first, "id", None)
    if context.get("goalMode") is True and isinstance(first_response_id, str) and first_response_id:
        context["goalAgentPreviousResponseId"] = first_response_id
    store.record_retrieval_event(run_id, "retrieval_started", {"responseId": getattr(first, "id", None),
        "model": model_name(), "parallelToolCalls": True, "selectionPolicy": "internal-nine-preferred-v1"})
    called_tools: set[str] = set()
    # Current is already supplied as a trusted profile, including when a single-track
    # song has no internal retrieval tool. Keeping it must not require a fake nine-track call.
    recalled_ids: set[str] = {current_id} if current and not cross_song_request else set()
    response = first
    resource_validation_cache: dict[str, Any] = {}
    origin_scene_id = current_playback.get("sceneId") or current_playback.get("originSceneId")
    origin_scene_id = origin_scene_id if isinstance(origin_scene_id, str) else None
    for _ in range(4):
        tool_calls = function_calls(response)
        if not tool_calls:
            raise ModelSelectionError("model_did_not_request_retrieval_tools")
        names = {str(getattr(call, "name", "")) for call in tool_calls}
        if not names.issubset(required_tools):
            raise ModelSelectionError("model_requested_unknown_tool")
        excluded_goal_ids = (
            _excluded_goal_playback_ids(allowed, context, origin_scene_id)
            | _resource_rejected_playback_ids(allowed, origin_scene_id, resource_validation_cache)
        ) if context.get("goalMode") is True else set()
        raw_outputs = await asyncio.gather(*[
            _call_retrieval_tool(call, assessment, allowed, current_id, excluded_goal_ids)
            for call in tool_calls
        ])
        outputs = await asyncio.gather(*[
            filter_tool_output_for_playable_resources(output, origin_scene_id=origin_scene_id,
                                                      cache=resource_validation_cache, run_id=run_id)
            for output in raw_outputs
        ])
        for call, raw_output, output in zip(tool_calls, raw_outputs, outputs):
            store.record_retrieval_event(run_id, "tool_result", {
                "responseId": getattr(response, "id", None), "callId": getattr(call, "call_id", None),
                "toolName": getattr(call, "name", None), "arguments": getattr(call, "arguments", "{}"),
                "rawResult": json.loads(raw_output["output"]),
                "result": json.loads(output["output"])})
        called_tools.update(names)
        recalled_ids.update(recalled_playback_unit_ids(outputs))
        missing = required_tools - called_tools
        if missing:
            # The model may request required tools across turns. Continue the same Responses chain.
            response = await asyncio.wait_for(
                _client().responses.create(
                    model=model_name(), previous_response_id=getattr(response, "id", None), input=outputs,
                    tools=[tool for tool in tools if tool["name"] in missing],
                    parallel_tool_calls=True,
                    tool_choice="required", max_output_tokens=1600,
                ), timeout=agent_timeout_seconds(),
            )
            response_id = getattr(response, "id", None)
            if context.get("goalMode") is True and isinstance(response_id, str) and response_id:
                context["goalAgentPreviousResponseId"] = response_id
            continue
        permitted = sorted(selection_scope(recalled_ids))
        selection_input_outputs = outputs
        if not permitted and context.get("goalMode") is True and excluded_goal_ids:
            repeat_allowed, repeat_reason = _goal_repeat_allowed_after_resource_coverage(
                allowed, context, origin_scene_id, resource_validation_cache
            )
            if repeat_allowed:
                context["goalRepeatAllowedReason"] = repeat_reason
                repeat_raw_outputs = await asyncio.gather(*[
                    _call_retrieval_tool(call, assessment, allowed, current_id, set())
                    for call in tool_calls
                ])
                repeat_outputs = await asyncio.gather(*[
                    filter_tool_output_for_playable_resources(output, origin_scene_id=origin_scene_id,
                                                              cache=resource_validation_cache, run_id=run_id)
                    for output in repeat_raw_outputs
                ])
                for call, raw_output, output in zip(tool_calls, repeat_raw_outputs, repeat_outputs):
                    store.record_retrieval_event(run_id, "tool_result_repeat_fallback", {
                        "responseId": getattr(response, "id", None), "callId": getattr(call, "call_id", None),
                        "toolName": getattr(call, "name", None), "arguments": getattr(call, "arguments", "{}"),
                        "rawResult": json.loads(raw_output["output"]),
                        "result": json.loads(output["output"]),
                        "reason": repeat_reason,
                    })
                recalled_ids.update(recalled_playback_unit_ids(repeat_outputs))
                permitted = sorted(current_scope(recalled_ids))
                selection_input_outputs = repeat_outputs
            else:
                store.record_retrieval_event(run_id, "goal_repeat_not_proven", {
                    "responseId": getattr(response, "id", None),
                    "reason": repeat_reason,
                    "coveredServiceSongIds": sorted(_completed_resource_validation_service_ids(resource_validation_cache)),
                })
                response = await asyncio.wait_for(
                    _client().responses.create(
                        model=model_name(), previous_response_id=getattr(response, "id", None), input=outputs,
                        tools=tools, parallel_tool_calls=True, tool_choice="required", max_output_tokens=1600,
                    ), timeout=agent_timeout_seconds(),
                )
                response_id = getattr(response, "id", None)
                if context.get("goalMode") is True and isinstance(response_id, str) and response_id:
                    context["goalAgentPreviousResponseId"] = response_id
                continue
        if not permitted:
            raise ModelSelectionError("no_recalled_candidate_for_requested_scope")
        second = await asyncio.wait_for(
            _client().responses.create(
                model=model_name(), previous_response_id=getattr(response, "id", None), input=selection_input_outputs,
                tools=[submit_selection_tool_schema(permitted, goal_mode=context.get("goalMode") is True)],
                parallel_tool_calls=False,
                # Live GPT-6 Luna returned incomplete/max_output_tokens at 1000 before
                # submitting a second Refine choice. Budget includes reasoning, not just JSON.
                tool_choice={"type": "function", "name": "submit_selection"}, max_output_tokens=3000,
            ), timeout=agent_timeout_seconds(),
        )
        second_response_id = getattr(second, "id", None)
        if context.get("goalMode") is True and isinstance(second_response_id, str) and second_response_id:
            context["goalAgentPreviousResponseId"] = second_response_id
        store.record_retrieval_event(run_id, "selection_response", {
            "responseId": getattr(second, "id", None), "status": getattr(second, "status", None),
            "incompleteReason": getattr(getattr(second, "incomplete_details", None), "reason", None),
            "outputTypes": [getattr(item, "type", None) for item in getattr(second, "output", []) or []],
            "permittedPlaybackUnitIds": permitted})
        if getattr(second, "status", None) == "incomplete":
            reason = getattr(getattr(second, "incomplete_details", None), "reason", None) or "unknown"
            raise ModelSelectionError("model_selection_incomplete:" + str(reason))
        submit_calls = [call for call in function_calls(second) if getattr(call, "name", None) == "submit_selection"]
        if len(submit_calls) != 1:
            raise ModelSelectionError("model_selection_not_submitted_once")
        break
    else:
        raise ModelSelectionError("retrieval_step_limit_exceeded")
    selected = selected_ids_from_response(second)
    if not selected:
        raise ModelSelectionError("model_did_not_submit_selection")
    accepted = [playback_id for playback_id in selected if playback_id in set(permitted)]
    reason = selection_reason_from_response(second)
    goal_selection_fields = goal_selection_fields_from_response(second)
    if context.get("goalMode") is True:
        _set_goal_chain_pending_outputs(context, function_calls(second), _selection_ack_payload(accepted, reason, prepared=True))
    store.record_retrieval_event(run_id, "selection_submitted", {
        "responseId": getattr(second, "id", None), "submittedPlaybackUnitIds": selected,
        "acceptedPlaybackUnitIds": accepted, "reason": reason, "feedback": context.get("userFeedback"),
        "reasonPlaybackUnitId": selected[0] if selected else None,
        "selectionPolicy": "internal-nine-preferred-v1",
        "goalRepeatAllowedReason": context.get("goalRepeatAllowedReason"),
        **goal_selection_fields})
    if not accepted:
        raise ModelSelectionError("model_selection_not_in_retrieval_results")
    return accepted


async def filter_tool_output_for_playable_resources(
    output: dict[str, str],
    *,
    origin_scene_id: str | None,
    cache: dict[str, Any],
    run_id: str | None,
) -> dict[str, str]:
    try:
        payload = json.loads(output.get("output", "{}"))
    except json.JSONDecodeError:
        return output
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return output

    candidate_items: list[tuple[dict[str, Any], MusicProfile, str]] = []
    seen_playback_ids: set[str] = set()
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        playback_id = candidate.get("playbackUnitId")
        profile = catalog().profile(playback_id) if isinstance(playback_id, str) else None
        if profile is None:
            filtered.append({"playbackUnitId": playback_id, "reason": "unknown_playback_unit"})
            continue
        service_song_id = profile.service_song_id(origin_scene_id)
        if profile.playback_unit_id in seen_playback_ids:
            filtered.append({"playbackUnitId": profile.playback_unit_id, "serviceSongId": service_song_id,
                             "reason": "duplicate_playback_unit_id"})
            continue
        seen_playback_ids.add(profile.playback_unit_id)
        candidate_items.append((candidate, profile, service_song_id))

    representatives = {service_id: profile for _, profile, service_id in candidate_items}
    for service_id, profile in representatives.items():
        if service_id not in cache:
            cache[service_id] = asyncio.create_task(validate_playback_unit(profile, origin_scene_id=origin_scene_id))
    pending = [value for value in cache.values() if isinstance(value, asyncio.Task)]
    if pending:
        await asyncio.gather(*pending)
        for service_id, value in list(cache.items()):
            if isinstance(value, asyncio.Task) and value.done():
                cache[service_id] = value.result()

    playable: list[dict[str, Any]] = []
    for candidate, profile, service_song_id in candidate_items:
        result = cache[service_song_id]
        ok = bool(getattr(result, "ok", False))
        reason = str(getattr(result, "reason", "unknown"))
        if ok:
            playable.append(candidate)
        else:
            filtered.append({"playbackUnitId": profile.playback_unit_id, "serviceSongId": service_song_id,
                             "reason": reason})

    if filtered:
        store.record_retrieval_event(run_id, "resource_candidate_filtered", {
            "originSceneId": origin_scene_id,
            "toolName": payload.get("toolName"),
            "filtered": filtered,
        })
    payload["candidates"] = playable
    payload["resourceEligibility"] = {
        "originSceneId": origin_scene_id,
        "inputCandidateCount": len(candidates),
        "playableCandidateCount": len(playable),
        "filteredCandidateCount": len(filtered),
        "candidateDedupeKey": "playbackUnitId",
        "validationDedupeKey": "serviceSongId",
        "presetRequired": True,
    }
    if candidates and not playable and str(payload.get("toolStatus") or "ok") == "ok":
        payload["toolStatus"] = "empty_after_resource_validation"
    return {**output, "output": json.dumps(payload, ensure_ascii=False)}


def _goal_chain_input(context: dict[str, Any], prompt: dict[str, Any]) -> list[dict[str, Any]]:
    pending = context.pop("goalAgentPendingToolOutputs", None)
    items: list[dict[str, Any]] = []
    if isinstance(pending, list):
        for item in pending:
            if isinstance(item, dict) and item.get("type") == "function_call_output" and isinstance(item.get("call_id"), str):
                items.append({"type": "function_call_output", "call_id": item["call_id"], "output": str(item.get("output") or "{}")})
    items.append({"role": "user", "content": json.dumps(prompt, ensure_ascii=False)})
    return items


def _tool_outputs_for_calls(calls: list[Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for call in calls:
        call_id = getattr(call, "call_id", None)
        if not isinstance(call_id, str) or not call_id:
            call_id = getattr(call, "id", None)
        if isinstance(call_id, str) and call_id:
            outputs.append({"type": "function_call_output", "call_id": call_id, "output": json.dumps(payload, ensure_ascii=False)})
    return outputs


def _set_goal_chain_pending_outputs(context: dict[str, Any], calls: list[Any], payload: dict[str, Any]) -> None:
    outputs = _tool_outputs_for_calls(calls, payload)
    if outputs:
        context["goalAgentPendingToolOutputs"] = outputs


def _goal_plan_ack_payload(plan: GoalPlan) -> dict[str, Any]:
    return {
        "accepted": True,
        "status": "planned",
        "title": plan.title,
        "totalMinutes": plan.total_minutes,
        "phases": [{"title": phase.title, "durationMinutes": phase.duration_minutes, "iconId": phase.icon_id}
                   for phase in plan.phases],
    }


def _energy_goal_ack_payload(goal: str, utterance: str) -> dict[str, Any]:
    return {"accepted": True, "status": "opening_prepared", "energyGoal": goal, "openingText": utterance}


def _selection_ack_payload(playback_unit_ids: list[str], reason: str | None, *, prepared: bool) -> dict[str, Any]:
    return {
        "accepted": True,
        "status": "prepared_not_played" if prepared else "accepted",
        "playbackUnitIds": playback_unit_ids,
        "reason": reason,
        "playbackState": "not_started",
    }


def _goal_used_carrier_ids(context: dict[str, Any]) -> set[str]:
    used = context.get("goalUsedServiceSongIds")
    if not isinstance(used, list):
        return set()
    result: set[str] = set()
    music_catalog = catalog()
    for item in used:
        if not isinstance(item, str) or not item:
            continue
        result.add(item)
        for profile in music_catalog.by_song_id.get(item, []) + music_catalog.by_service_song_id.get(item, []):
            result.add(profile.song_id)
    return result


def _profile_carrier_keys(profile: MusicProfile, origin_scene_id: str | None) -> set[str]:
    return {profile.song_id, profile.service_song_id(origin_scene_id)}


def _exclude_used_goal_carriers(recalled: set[str], context: dict[str, Any], origin_scene_id: str | None) -> set[str]:
    used_ids = _goal_used_carrier_ids(context)
    if not used_ids:
        return recalled
    filtered: set[str] = set()
    for playback_id in recalled:
        profile = catalog().profile(playback_id)
        if profile and _profile_carrier_keys(profile, origin_scene_id).isdisjoint(used_ids):
            filtered.add(playback_id)
    return filtered


def _excluded_goal_playback_ids(allowed: set[str], context: dict[str, Any], origin_scene_id: str | None) -> set[str]:
    used_ids = _goal_used_carrier_ids(context)
    if not used_ids:
        return set()
    excluded: set[str] = set()
    for playback_id in allowed:
        profile = catalog().profile(playback_id)
        if profile and not _profile_carrier_keys(profile, origin_scene_id).isdisjoint(used_ids):
            excluded.add(playback_id)
    return excluded


def _completed_resource_validation_service_ids(cache: dict[str, Any]) -> set[str]:
    return {service_id for service_id, result in cache.items() if not isinstance(result, asyncio.Task)}


def _goal_unused_allowed_service_ids(allowed: set[str], context: dict[str, Any], origin_scene_id: str | None) -> set[str]:
    used_ids = _goal_used_carrier_ids(context)
    service_ids: set[str] = set()
    for playback_id in allowed:
        profile = catalog().profile(playback_id)
        if profile and _profile_carrier_keys(profile, origin_scene_id).isdisjoint(used_ids):
            service_ids.add(profile.service_song_id(origin_scene_id))
    return service_ids


def _goal_repeat_allowed_after_resource_coverage(
    allowed: set[str],
    context: dict[str, Any],
    origin_scene_id: str | None,
    cache: dict[str, Any],
) -> tuple[bool, str]:
    unused_service_ids = _goal_unused_allowed_service_ids(allowed, context, origin_scene_id)
    if not unused_service_ids:
        return True, "all_allowed_carriers_used"
    completed = _completed_resource_validation_service_ids(cache)
    if not unused_service_ids.issubset(completed):
        return False, "unused_carrier_resource_coverage_incomplete"
    for service_id in unused_service_ids:
        result = cache.get(service_id)
        if bool(getattr(result, "ok", False)):
            return False, "unused_carrier_resource_available"
    return True, "all_allowed_unused_carrier_resources_unavailable"


def _resource_rejected_playback_ids(allowed: set[str], origin_scene_id: str | None, cache: dict[str, Any]) -> set[str]:
    rejected_service_ids = {
        service_id for service_id, result in cache.items()
        if not isinstance(result, asyncio.Task) and not bool(getattr(result, "ok", False))
    }
    if not rejected_service_ids:
        return set()
    rejected: set[str] = set()
    for playback_id in allowed:
        profile = catalog().profile(playback_id)
        if profile and profile.service_song_id(origin_scene_id) in rejected_service_ids:
            rejected.add(playback_id)
    return rejected


def _goal_retrieval_retry_input(outputs: list[dict[str, str]], reason: str) -> list[dict[str, Any]]:
    return [
        *outputs,
        {
            "role": "user",
            "content": json.dumps({
                "task": "Continue retrieval before selection.",
                "reason": reason,
                "repeatPolicy": "Do not repeat an already used carrier yet. Unused catalog resource coverage is incomplete; adjust criteria or broaden retrieval and exclude resource-rejected carriers before selecting.",
            }, ensure_ascii=False),
        },
    ]


def retrieval_tool_schemas(include_internal: bool = True) -> list[dict[str, Any]]:
    axis_target = {
        "type": "object",
        "properties": {
            "target": {"type": "number", "minimum": 1, "maximum": 5},
            "tolerance": {"type": "number", "exclusiveMinimum": 0, "maximum": 4},
        },
        "required": ["target", "tolerance"],
        "additionalProperties": False,
    }
    axis_targets = {
        "type": "object",
        "properties": {name: axis_target for name in [
            "arousal", "rhythmicDrive", "attentionCapture", "globalStability", "suddenChangeRisk",
            "textureDensity", "brightness", "warmth", "spaciousness",
        ]},
        "additionalProperties": False,
    }
    candidate_kinds = {"type": "array", "items": {"type": "string", "enum": ["song", "touchpad_track"]}, "uniqueItems": True}
    profile_params = {
        "type": "object",
        "properties": {
            "nameQuery": {"type": "string"},
            "tagPreferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"tagId": {"type": "string"}, "importance": {"type": "number", "minimum": 0, "maximum": 1}},
                    "required": ["tagId", "importance"],
                    "additionalProperties": False,
                },
            },
            "axisTargets": axis_targets,
            "candidateKinds": candidate_kinds,
            "excludePlaybackUnitIds": {"type": "array", "items": {"type": "string"}},
            "topK": {"type": "integer", "minimum": 1, "maximum": 5},
            "language": {"type": "string", "enum": ["zh", "en"]},
        },
        "additionalProperties": False,
    }
    # This tool always returns the complete owning song scope, including all nine tracks.
    # Legacy topK arguments are ignored rather than silently omitting track 9.
    same_song_params = {"type": "object", "properties": {
        key: value for key, value in profile_params["properties"].items()
        if key in {"nameQuery", "tagPreferences", "axisTargets", "language"}
    }, "additionalProperties": False}
    description_params = {
        "type": "object",
        "properties": {
            "queryText": {"type": "string"},
            "textLanguage": {"type": "string", "enum": ["en"]},
            "candidateKinds": candidate_kinds,
            "excludePlaybackUnitIds": {"type": "array", "items": {"type": "string"}},
            "topK": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["queryText", "textLanguage"],
        "additionalProperties": False,
    }
    tools = [
        {"type": "function", "name": "same_song_profile_search", "description": "Search the current carrier song's complete nine logical Touchpad tracks. Optional profile criteria rank this internal set; no criteria returns current first plus all alternatives. Ordinary songs return their sole unit.", "parameters": same_song_params},
        {"type": "function", "name": "library_profile_search", "description": "Donor Music Profile structured search across the supplied valid catalog.", "parameters": profile_params},
        {"type": "function", "name": "music_description_recall", "description": "Exact donor CLAP description recall. Requires model-authored English queryText; no server rewrite or translation.", "parameters": description_params},
    ]
    return tools if include_internal else [tool for tool in tools if tool["name"] != "same_song_profile_search"]


def submit_selection_tool_schema(candidate_ids: list[str] | None = None, *, goal_mode: bool = False) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "submit_selection",
        "description": "Submit ordered playbackUnitId choices selected only from retrieval results.",
        "parameters": {
            "type": "object",
            "properties": {
                "playbackUnitIds": {
                    "type": "array",
                    "items": {"type": "string", **({"enum": candidate_ids} if candidate_ids is not None else {})},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "reason": {"type": "string", "description": "One short user-facing rationale for the FIRST submitted playbackUnitId, comparing it to current/internal options. No private reasoning.", "maxLength": 240},
                **({
                    "energyGoal": {"type": "string", "enum": sorted(ENERGY_GOALS), "description": "Goal phase energy direction."},
                    "openingText": {"type": "string", "description": "Goal phase opening TTS for the selected music. Natural, brief, at most one energy sentence.", "maxLength": 120},
                    "postReceiptText": {"type": "string", "description": "After-playback explanation for this selected candidate. Short music phrase plus one optional light suggestion. No energy/能量.", "maxLength": 120},
                } if goal_mode else {}),
            },
            "required": ["playbackUnitIds", "reason"] + (["energyGoal", "openingText", "postReceiptText"] if goal_mode else []),
            "additionalProperties": False,
        },
    }


def function_calls(response: Any) -> list[Any]:
    return [
        item for item in getattr(response, "output", []) or []
        if getattr(item, "type", None) == "function_call"
    ]


async def _call_retrieval_tool(call: Any, assessment: str, allowed: set[str], current_id: str,
                               excluded_playback_ids: set[str]) -> dict[str, str]:
    try:
        return await execute_retrieval_tool(call, assessment, allowed, current_id,
                                            excluded_playback_ids=excluded_playback_ids)
    except TypeError as exc:
        if "excluded_playback_ids" not in str(exc):
            raise
        # Backward-compatible for tests or local monkeypatches that emulate the old helper.
        return await execute_retrieval_tool(call, assessment, allowed, current_id)


async def execute_retrieval_tool(call: Any, assessment: str, allowed: set[str], current_id: str,
                                 excluded_playback_ids: set[str] | None = None) -> dict[str, str]:
    return await asyncio.to_thread(_execute_retrieval_tool_sync, call, allowed, current_id, excluded_playback_ids or set())


def _execute_retrieval_tool_sync(call: Any, allowed: set[str], current_id: str, excluded_playback_ids: set[str] | None = None) -> dict[str, str]:
    name = str(getattr(call, "name", ""))
    try:
        args = json.loads(getattr(call, "arguments", "{}") or "{}")
    except json.JSONDecodeError:
        args = {}
    if excluded_playback_ids is None:
        excluded_playback_ids = set()
    if excluded_playback_ids:
        existing_excludes = args.get("excludePlaybackUnitIds") if isinstance(args.get("excludePlaybackUnitIds"), list) else []
        args["excludePlaybackUnitIds"] = sorted({str(item) for item in existing_excludes if isinstance(item, str)} | set(excluded_playback_ids))
        # Goal mode excludes already-used/resource-rejected units before retrieval.
        # Keep the model-authored topK within the donor tool contract (1..5); coverage
        # for repeat decisions is proved by additional retrieval turns, not by widening
        # a single tool call beyond the real API schema.
        requested_top_k = args.get("topK")
        if not isinstance(requested_top_k, int) or isinstance(requested_top_k, bool) or requested_top_k < 1 or requested_top_k > 5:
            args["topK"] = 5
        allowed = {item for item in allowed if item not in excluded_playback_ids}
    payload: dict[str, Any]
    try:
        if name == "same_song_profile_search":
            result = catalog().search_same_song(args, allowed, current_id)
            profiles = catalog().profiles_from_tool_result(result, allowed)
            payload = project_tool_candidates(result, profiles)
        elif name == "music_description_recall":
            result = catalog().description_recall(args, allowed, current_id)
            profiles = catalog().profiles_from_tool_result(result, allowed)
            payload = project_tool_candidates(result, profiles)
        elif name == "library_profile_search":
            result = catalog().search_profiles(args, allowed, current_id)
            profiles = catalog().profiles_from_tool_result(result, allowed)
            payload = project_tool_candidates(result, profiles)
        else:
            payload = {"toolStatus": "rejected", "issues": [{"code": "UNKNOWN_TOOL", "message": f"Unknown retrieval tool {name}"}], "candidates": []}
    except Exception as exc:
        if hasattr(exc, "as_tool_result"):
            payload = exc.as_tool_result()
        else:
            payload = {"toolStatus": "error", "issues": [{"code": type(exc).__name__, "message": "retrieval tool failed"}], "candidates": []}
    payload.update({
        "toolName": name,
        "resolutionPath": "bundled playbackUnitId -> iOS PlaybackUnitCatalog resource binding -> /api/scenes/song/meta serviceSongId -> AR03 preset resource",
    })
    return {
        "type": "function_call_output",
        "call_id": str(getattr(call, "call_id", "")),
        "output": json.dumps(payload, ensure_ascii=False),
    }


def project_tool_candidates(result: dict[str, Any], profiles: list[MusicProfile]) -> dict[str, Any]:
    raw = {item["playbackUnitId"]: item for item in result.get("candidates", []) if isinstance(item, dict) and "playbackUnitId" in item}
    candidates = []
    for profile in profiles:
        item = dict(_profile_for_prompt(profile))
        evidence = raw.get(profile.playback_unit_id, {})
        for key in ("rank", "channelScore", "matchEvidence", "retrievalChannel", "tagSupport"):
            if key in evidence:
                item[key] = evidence[key]
        candidates.append(item)
    return {**result, "candidates": candidates}


def selection_reason_from_response(response: Any) -> str | None:
    for call in function_calls(response):
        if getattr(call, "name", None) == "submit_selection":
            try:
                reason = json.loads(getattr(call, "arguments", "{}") or "{}").get("reason")
            except (json.JSONDecodeError, AttributeError):
                return None
            return reason if isinstance(reason, str) else None
    return None


def goal_selection_fields_from_response(response: Any) -> dict[str, Any]:
    for call in function_calls(response):
        if getattr(call, "name", None) == "submit_selection":
            try:
                args = json.loads(getattr(call, "arguments", "{}") or "{}")
            except (json.JSONDecodeError, AttributeError):
                return {}
            result: dict[str, Any] = {}
            energy_goal = args.get("energyGoal")
            if energy_goal in ENERGY_GOALS:
                result["energyGoal"] = energy_goal
            for key in ("openingText", "postReceiptText"):
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    result[key] = value.strip()
            return result
    return {}


def selected_ids_from_response(response: Any) -> list[str]:
    for call in function_calls(response):
        if getattr(call, "name", None) == "submit_selection":
            try:
                args = json.loads(getattr(call, "arguments", "{}") or "{}")
            except json.JSONDecodeError:
                return []
            ids = args.get("playbackUnitIds")
            return [item for item in ids if isinstance(item, str)] if isinstance(ids, list) else []
    text = getattr(response, "output_text", None)
    if text:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
        ids = value.get("playbackUnitIds") if isinstance(value, dict) else None
        return [item for item in ids if isinstance(item, str)] if isinstance(ids, list) else []
    return []


def recalled_playback_unit_ids(outputs: list[dict[str, str]]) -> set[str]:
    recalled: set[str] = set()
    for output in outputs:
        try:
            payload = json.loads(output.get("output", "{}"))
        except json.JSONDecodeError:
            continue
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("playbackUnitId"), str):
                recalled.add(candidate["playbackUnitId"])
    return recalled


def _profile_for_prompt(profile: MusicProfile | None) -> dict[str, Any] | None:
    if not profile:
        return None
    return {
        "playbackUnitId": profile.playback_unit_id,
        "songId": profile.song_id,
        "carrierSongId": profile.song_id,
        "name": profile.names,
        "summary": profile.summary,
        "tags": profile.tags,
        "axes": profile.axes,
        "unitKind": profile.unit_kind,
        "trackOrder": profile.track_order,
        "resourceBindings": profile.resource_bindings,
    }


def _music_for_narration(profile: MusicProfile | None) -> dict[str, Any] | None:
    if not profile:
        return None
    # Published Touchpad names have a catalog suffix, e.g. 星落 · 轨道 1 / Starfall · Track 1.
    names = {language: next((parent for parent in sorted(profile.parent_names, key=len, reverse=True)
                             if name == parent or name.startswith(parent + " · ")),
                            re.sub(r"\s*·\s*(?:轨道|Track)\s*\d+\s*$", "", name, flags=re.I))
             for language, name in profile.names.items()}
    return {"songName": names, "summary": profile.summary, "tags": profile.tags, "axes": profile.axes}


def _forbidden_music_copy(text: str) -> bool:
    return bool(re.search(r"(?:track|stem)\s*\d+|轨道\s*[\d一二三四五六七八九]|(?:keep|keeping|same)\s+(?:the\s+)?same\s+song|同一首|same\s+song", text, re.I))


def _speaks_pressure_score(text: str) -> bool:
    numbers = r"\d+(?:\.\d+)?|(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"
    return bool(re.search(r"(?:stress|pressure)(?:\s+(?:level|score|reference|is|at|of|about|around|rated|estimated|as|currently|the|a|has|been|to|be|your|body)){0,8}\s*[:=]?\s*(?:" + numbers + r")\b|(?:压力|身体状态|压力参考)[^。！？，,;]{0,12}[\d零一二三四五六七八九十百]+(?:\.\d+)?\s*(?:分|点|/|％|%|(?=[。！？，,;]|$))|\d+\s*(?:points?\s*(?:of\s*)?(?:stress|pressure)|分(?:的)?压力)", text, re.I))


def _repeats_environment(text: str, lead: str) -> bool:
    # Live first assessment repeated the office/weather/history after the factual lead.
    # Reuse the agent's chosen goal but keep one context mention and a musical direction.
    if not lead:
        return False
    terms = r"写字楼|办公室|住宅区|公园|图书馆|商场|阴天|多云|晴朗|有雾|下雨|下雪|上次身体|注意压力|压力偏高|\b(?:office|park|library|overcast|cloudy|rainy|snowy|foggy|earlier body|pressure reference)\b|\d+(?:\.\d+)?°[CF]"
    return any(match.group().lower() in text.lower() for match in re.finditer(terms, lead, re.I))


async def _deliver_assessment_voice_before_selection(task: asyncio.Task[str | None], run_id: str, text: str, energy_goal: str | None = None) -> None:
    try:
        voice_id = await task
    except asyncio.CancelledError:
        return
    except Exception:
        voice_id = None
    if _cancelled(run_id):
        return
    if voice_id:
        store.add_delivery(run_id=run_id, phase=PHASE_ASSESSMENT, text=text, voice_id=voice_id, status="voice_ready", energy_goal=energy_goal)
    else:
        store.add_delivery(run_id=run_id, phase=PHASE_ASSESSMENT, text=text, status="text_only", energy_goal=energy_goal)


def _cancelled(run_id: str) -> bool:
    run = store.get_run(run_id)
    return not run or run["status"] in {"cancelled", "done", "failed"}
