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
from app.environment import spoken_context, environment_lead
from app.tts import synthesize_voice


ENERGY_GOALS = {"boost", "maintain", "lower"}


@dataclass(frozen=True)
class EnergyAssessment:
    text: str
    energy_goal: str | None
    failure_reason: str | None = None


class ModelSelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateDecision:
    profile: MusicProfile
    status: str


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex}"


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
        current_playback = store.loads(run["current_playback_json"], {})
        catalog_payload = store.loads(run["catalog_json"], {})
        allowed = catalog().candidate_ids_from_payload({"catalog": catalog_payload})
        if not allowed:
            allowed = set(catalog().by_playback_unit_id)
        current_id = str(current_playback.get("playbackUnitId") or current_playback.get("playback_unit_id") or "")
        language = str(context.get("language") or "zh")
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
        if _speaks_pressure_score(assessment):
            assessment = environment_lead(context, language) + energy_goal_copy(energy_goal, language)
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
        result = await validate_playback_unit(profile)
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
            if profile.playback_unit_id == current_id:
                return CandidateDecision(profile, "model_selected_unchanged")
            return CandidateDecision(profile, "model_selected")
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
    fallback = "I’ll look for a musical adjustment that fits your request." if language.startswith("en") else "我会结合正在听的音乐，找一个更贴合你的调整方向。"
    lead = environment_lead(context, language)
    if not model_enabled():
        return EnergyAssessment(lead + fallback, None)
    prompt = {
        "task": "Choose one musical energy goal, then describe the music you intend to select or adjust toward it.",
        "language": language,
        "context": spoken_context(context, language),
        "environmentLead": lead,
        "currentPlayback": current_playback,
        "currentMusic": _music_for_narration(catalog().profile(str(current_playback.get("playbackUnitId") or ""))),
        "rules": [
            "Interpret localTime in context.timezone; a UTC Z timestamp must be converted before describing the user's local time of day.",
            "Use available place type, weather, optional health references, activity/sleep and user feedback to describe a musical direction. Do not invent missing context or report a checklist of missing permissions.",
            "Do not claim health measurements when stress source is default or historical.",
            "Pressure is a computed reference. recent_measurement means fresh underlying HRV, not a direct measurement of stress. When baselineReady is false, describe the musical direction instead of claiming detected high stress.",
            "Do not repeat the fast welcome.",
            "Follow context.environmentNarration: initial must briefly include available facts through the server's environmentLead, which is already part of this utterance. Your assessment adds only the musical energy direction, never another description of place, weather or body data. changed has only changedFields in that lead; music_only must not repeat weather, place, time of day, pressure, health or missing data. Focus on music and how it can help.",
            "Use health.pressure.levelLabel from StressLab; never speak a numeric pressure/stress score, rating, percent, or points. HRV labels and heart-rate estimate labels differ; retain the supplied label. Historical references must be identified as earlier data, never a current reading. When baselineReady is false or levelLabel is absent, do not assert a body-pressure state.",
            "The server will prepend environmentLead to this same first utterance. Return only your concise musical energy direction after that lead; do not repeat its weather, place, body reference or other facts. Use those facts to decide the energy goal.",
            "Never speak track/stem numbers or say keep the same song. Describe musical qualities instead.",
            "Choose exactly one goal: boost supports gentle activation/energy replenishment; maintain supports stable sustainable energy; lower supports settling/rest. This is a musical support intention, not a measured energy level or medical effect.",
            "Use user feedback and recommendedScene intent alongside available context. Low-confidence/default/historical health cannot justify claiming you measured the user's energy.",
            "Anchor the utterance to the selected energy goal and intended musical qualities. Do not announce the current song or current playback. Do not name an unknown next song before retrieval.",
            "Submit the set_energy_goal function with energyGoal and one concise user-language assessment.",
        ],
    }
    try:
        response = await asyncio.wait_for(_client().responses.create(
            model=model_name(), input=json.dumps(prompt, ensure_ascii=False),
            tools=[energy_goal_tool_schema()], tool_choice={"type": "function", "name": "set_energy_goal"},
            max_output_tokens=2400), timeout=agent_timeout_seconds())
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
        missing_energy_anchor = "energy" not in utterance.lower() if language.startswith("en") else "能量" not in utterance
        if (not utterance or _forbidden_music_copy(utterance) or _announces_current_music(utterance)
                or _speaks_pressure_score(utterance) or missing_energy_anchor or _repeats_environment(utterance, lead)):
            utterance = energy_goal_copy(goal, language)
        return EnergyAssessment(lead + utterance, goal)
    except Exception as exc:
        # A failed model call does not invent a goal; only safe diagnostics are retained.
        reason = str(exc) if isinstance(exc, ModelSelectionError) else type(exc).__name__
        return EnergyAssessment(lead + fallback, None, reason)


def energy_goal_tool_schema() -> dict[str, Any]:
    return {"type": "function", "name": "set_energy_goal", "description": "Set one musical energy intention, not a health measurement.",
        "parameters": {"type": "object", "properties": {
            "energyGoal": {"type": "string", "enum": ["boost", "maintain", "lower"]},
            "assessment": {"type": "string", "description": "One brief context-aware musical energy direction; qualitative body stress only, no scores or current-song announcement.", "maxLength": 240}},
            "required": ["energyGoal", "assessment"], "additionalProperties": False}}


def energy_goal_copy(goal: str | None, language: str) -> str:
    if language.startswith("en"):
        return {"boost": "I’ll look for a little more rhythmic lift to help replenish your energy.",
                "maintain": "I’ll choose steady, spacious music to help sustain your energy.",
                "lower": "I’ll choose softer, lighter music to help your energy settle."}.get(goal,
                    "I’ll look for music that supports the moment.")
    return {"boost": "我会选一些有轻盈律动的音乐，陪你补充能量。", "maintain": "我会选一些稳定、留白更多的音乐，陪你维持能量。",
            "lower": "我会选更轻柔、层次更少的音乐，让能量慢慢降下来。"}.get(goal, "我会选择贴合此刻的音乐。")


def _announces_current_music(text: str) -> bool:
    return bool(re.search(r"(?:now|currently) (?:playing|listening)|\bis (?:currently )?playing\b|当前(?:正在)?播放|现在(?:正在)?(?:播放|在播)|正在播放", text, re.I))


async def generate_explanation(
    *,
    context: dict[str, Any],
    selected: MusicProfile | None,
    actual: MusicProfile | None,
    receipt: dict[str, Any],
    language: str,
) -> str:
    success = bool(receipt.get("success"))
    change_kind = receipt.get("changeKind") or ("failed" if not success else "unchanged")
    status = str(receipt.get("selectionStatus") or "")
    degraded_keep = status in {"model_unavailable_keep", "model_selection_invalid_keep"} or status.startswith("model_error_keep:")
    if not success or degraded_keep:
        fallback = "The adjustment did not finish. You can try again when ready." if language.startswith("en") else "这次调整未完成，准备好后可以再试一次。"
    else:
        names = (_music_for_narration(actual) or {}).get("songName") or {}
        title = names.get("en" if language.startswith("en") else "zh") or next(iter(names.values()), "")
        if receipt.get("actualPlaybackState") == "playing":
            fallback = (f"The selected music, {title}, is ready. " if language.startswith("en") else f"为你选的《{title}》已准备好。") + energy_goal_copy(context.get("energyGoal"), language)
        else:
            fallback = "The music adjustment is ready." if language.startswith("en") else "音乐调整已完成。"
    if not model_enabled():
        return fallback
    prompt = {
        "task": "Explain the actual playback outcome after Auto adaptation.",
        "language": language,
        "context": spoken_context(context, language),
        "selectedProfile": _music_for_narration(selected),
        "actualProfile": _music_for_narration(actual),
        "receipt": receipt,
        "alreadySpokenAssessment": context.get("alreadySpokenAssessment"),
        "currentPlayback": context.get("currentPlaybackForExplanation"),
        "actualPlaybackState": receipt.get("actualPlaybackState"),
        "rules": [
            "Use actual receipt outcome only.",
            "When playback succeeds, briefly connect the actual music profile and recorded selection rationale to the current environment or user request; do not merely announce the song title.",
            "Use public-facing musical qualities. Do not mention internal resource-binding scenes, tool names, database IDs, or claim that a model rationale overrides the actual receipt.",
            "Do not claim success when success is false.",
            "If selectionStatus reports a model error/unavailable/invalid keep, acknowledge that adaptation did not finish and current audio was kept; do not claim a successful comparison.",
            "When receipt.isExactPlaybackUnit is false, the actual audio is a manual mix; do not describe a solo track profile as the exact sound.",
            "Do not claim audible ongoing playback unless receipt.actualPlaybackState is playing. Missing state is unverified.",
            "Distinguish same_song, cross_song, unchanged, failed.",
            "When success is true and changeKind is unchanged with a normal selection status, describe the current music and its help. A valid unchanged choice is not a failed adaptation; do not say it failed, was not implemented, 没有落实 or 未完成. Do not claim an audible change either.",
            "For same_song, describe the musical change and its benefit only; never say keep the same song, same song, 同一首歌 or an equivalent reassurance about retaining it.",
            "Use songName when naming the music. Never mention track/stem numbers, 轨道几 or internal unit names.",
            "The assessment already covered any needed context. Do not repeat weather, place, pressure or time of day here. Focus on the actual music, its changes and intended help; do not guarantee a health effect.",
            "Anchor the explanation to context.energyGoal (boost, maintain or lower) and the selected music's qualities; use the goal as an intended support, never a guaranteed measured energy change.",
            "Do not announce now playing/current playback or introduce the previous song. Describe the selected/target music and how it supports the energy goal. Same-song changes describe the target musical change only.",
            "Explicitly mention the intended energy support using energy/能量, not only a synonym such as relaxation. Keep it brief enough to finish the thought within the length budget.",
            "Return only one short utterance.",
        ],
    }
    utterance = await _responses_text(prompt, fallback=fallback, max_chars=140)
    # The live model called a successful unchanged choice '没有落实调整'. Such failure
    # copy contradicts the receipt and must not reach TTS as a success explanation.
    contradictory_failure = success and not degraded_keep and bool(re.search(
        r"未完成|未成功|未能|没能|失败|没有(?:落实|完成|生效|实现)|did(?:n't| not) (?:finish|complete|take effect|succeed)|failed|could(?:n't| not)", utterance, re.I))
    missing_goal_anchor = context.get("energyGoal") in ENERGY_GOALS and (
        "energy" not in utterance.lower() if language.startswith("en") else "能量" not in utterance)
    return fallback if _forbidden_music_copy(utterance) or _announces_current_music(utterance) or _speaks_pressure_score(utterance) or contradictory_failure or missing_goal_anchor else utterance


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
    current = catalog().profile(current_id)
    has_internal_tool = bool(current and current.unit_kind == "touchpad_track"
                             and current_playback.get("supportsTouchpad", True) is True)
    tools = retrieval_tool_schemas(include_internal=has_internal_tool)
    required_tools = {tool["name"] for tool in tools}
    feedback = str(context.get("userFeedback") or "").casefold()
    internal = {item.playback_unit_id for item in catalog().same_song_candidates(current_id, allowed)}

    def selection_scope(recalled: set[str]) -> set[str]:
        if current and has_internal_tool and feedback.startswith("refine:"):
            return recalled & internal
        if current and (feedback.startswith("explore:") or (not has_internal_tool and feedback.startswith("refine:"))):
            return {item for item in recalled if catalog().profile(item).song_id != current.song_id}
        if current and not has_internal_tool:
            return {item for item in recalled if item == current_id or catalog().profile(item).song_id != current.song_id}
        return recalled
    first = await asyncio.wait_for(
        _client().responses.create(
            model=model_name(),
            input=json.dumps(
                {
                    "task": "Retrieve candidates for one Auto music adaptation. Call all available retrieval tools before choosing. Do not invent playback IDs.",
                    "toolAvailability": {"supportsTouchpad": has_internal_tool, "requiredTools": sorted(required_tools)},
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
                        "Only when toolAvailability.supportsTouchpad is true, prefer an internal track adjustment when it fits the target; choose another song when the internal set is not a good fit or Explore requests another song.",
                        "Refine stays in the current carrier only when internal Touchpad adjustment is available. With no Touchpad, Refine must choose a different carrier song. Explore always requests a different carrier song. Keep is handled by the client without a new selection.",
                        "Retrieval tools may run in parallel. Internal preference is a comparison/selection rule, not a ban on parallel library search.",
                        "Include one short user-facing rationale comparing the chosen candidate to current/internal options; do not return private reasoning.",
                        "In the public rationale describe musical changes and help, never track numbers or keep the same song.",
                        "Use context.energyGoal as the anchor for musical search/ranking: boost, maintain or lower. Compare with currentTrustedProfile internally, but public speech names only target music/qualities.",
                    ],
                },
                ensure_ascii=False,
            ),
            tools=tools,
            parallel_tool_calls=True,
            tool_choice="required",
            max_output_tokens=1600,
        ),
        timeout=agent_timeout_seconds(),
    )
    store.record_retrieval_event(run_id, "retrieval_started", {"responseId": getattr(first, "id", None),
        "model": model_name(), "parallelToolCalls": True, "selectionPolicy": "internal-nine-preferred-v1"})
    called_tools: set[str] = set()
    # Current is already supplied as a trusted profile, including when a single-track
    # song has no internal retrieval tool. Keeping it must not require a fake nine-track call.
    recalled_ids: set[str] = {current_id} if current else set()
    response = first
    for _ in range(4):
        tool_calls = function_calls(response)
        if not tool_calls:
            raise ModelSelectionError("model_did_not_request_retrieval_tools")
        names = {str(getattr(call, "name", "")) for call in tool_calls}
        if not names.issubset(required_tools):
            raise ModelSelectionError("model_requested_unknown_tool")
        outputs = await asyncio.gather(*[
            execute_retrieval_tool(call, assessment, allowed, current_id)
            for call in tool_calls
        ])
        for call, output in zip(tool_calls, outputs):
            store.record_retrieval_event(run_id, "tool_result", {
                "responseId": getattr(response, "id", None), "callId": getattr(call, "call_id", None),
                "toolName": getattr(call, "name", None), "arguments": getattr(call, "arguments", "{}"),
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
            continue
        permitted = sorted(selection_scope(recalled_ids))
        if not permitted:
            raise ModelSelectionError("no_recalled_candidate_for_requested_scope")
        second = await asyncio.wait_for(
            _client().responses.create(
                model=model_name(), previous_response_id=getattr(response, "id", None), input=outputs,
                tools=[submit_selection_tool_schema(permitted)],
                # Live GPT-6 Luna returned incomplete/max_output_tokens at 1000 before
                # submitting a second Refine choice. Budget includes reasoning, not just JSON.
                tool_choice={"type": "function", "name": "submit_selection"}, max_output_tokens=3000,
            ), timeout=agent_timeout_seconds(),
        )
        store.record_retrieval_event(run_id, "selection_response", {
            "responseId": getattr(second, "id", None), "status": getattr(second, "status", None),
            "incompleteReason": getattr(getattr(second, "incomplete_details", None), "reason", None),
            "outputTypes": [getattr(item, "type", None) for item in getattr(second, "output", []) or []],
            "permittedPlaybackUnitIds": permitted})
        if getattr(second, "status", None) == "incomplete":
            reason = getattr(getattr(second, "incomplete_details", None), "reason", None) or "unknown"
            raise ModelSelectionError("model_selection_incomplete:" + str(reason))
        break
    else:
        raise ModelSelectionError("retrieval_step_limit_exceeded")
    selected = selected_ids_from_response(second)
    if not selected:
        raise ModelSelectionError("model_did_not_submit_selection")
    accepted = [playback_id for playback_id in selected if playback_id in selection_scope(recalled_ids)]
    reason = selection_reason_from_response(second)
    store.record_retrieval_event(run_id, "selection_submitted", {
        "responseId": getattr(second, "id", None), "submittedPlaybackUnitIds": selected,
        "acceptedPlaybackUnitIds": accepted, "reason": reason, "feedback": context.get("userFeedback"),
        "reasonPlaybackUnitId": selected[0] if selected else None,
        "selectionPolicy": "internal-nine-preferred-v1"})
    if not accepted:
        raise ModelSelectionError("model_selection_not_in_retrieval_results")
    return accepted


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


def submit_selection_tool_schema(candidate_ids: list[str] | None = None) -> dict[str, Any]:
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
            },
            "required": ["playbackUnitIds", "reason"],
            "additionalProperties": False,
        },
    }


def function_calls(response: Any) -> list[Any]:
    return [
        item for item in getattr(response, "output", []) or []
        if getattr(item, "type", None) == "function_call"
    ]


async def execute_retrieval_tool(call: Any, assessment: str, allowed: set[str], current_id: str) -> dict[str, str]:
    return await asyncio.to_thread(_execute_retrieval_tool_sync, call, allowed, current_id)


def _execute_retrieval_tool_sync(call: Any, allowed: set[str], current_id: str) -> dict[str, str]:
    name = str(getattr(call, "name", ""))
    try:
        args = json.loads(getattr(call, "arguments", "{}") or "{}")
    except json.JSONDecodeError:
        args = {}
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
