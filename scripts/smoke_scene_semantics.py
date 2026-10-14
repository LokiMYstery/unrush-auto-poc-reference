from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        name = key.strip()
        if name in {"AGENT_MODEL_API_KEY", "AGENT_MODEL_BASE_URL", "AGENT_MODEL_NAME"}:
            os.environ[name] = value.strip().strip('"').strip("'")
        else:
            os.environ.setdefault(name, value.strip().strip('"').strip("'"))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Auto scene semantics against the configured live model.")
    parser.add_argument("--env-file", type=Path, help="Optional local env file. Values are loaded but never printed.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path for sanitized evidence.")
    args = parser.parse_args()
    if args.env_file:
        load_env(args.env_file)

    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from app import agent

    real_client = agent._client()
    raw_calls: list[dict[str, Any]] = []
    active_case: ContextVar[str] = ContextVar("active_case", default="")

    class ResponsesProxy:
        async def create(self, **kwargs: Any) -> Any:
            response = await real_client.responses.create(**kwargs)
            prompt = json.loads(kwargs["input"]) if isinstance(kwargs.get("input"), str) else kwargs.get("input")
            calls = []
            for call in agent.function_calls(response):
                try:
                    arguments = json.loads(getattr(call, "arguments", "{}") or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": getattr(call, "arguments", "")}
                calls.append({"name": getattr(call, "name", ""), "arguments": arguments})
            text = getattr(response, "output_text", None)
            if not text:
                parts: list[str] = []
                for item in getattr(response, "output", []) or []:
                    for content in getattr(item, "content", []) or []:
                        value = getattr(content, "text", None)
                        if value:
                            parts.append(value)
                text = " ".join(parts)
            raw_calls.append({
                "case": active_case.get(),
                "task": prompt.get("task") if isinstance(prompt, dict) else None,
                "promptContext": prompt.get("context") if isinstance(prompt, dict) else None,
                "calls": calls,
                "outputText": text,
            })
            return response

    class ClientProxy:
        responses = ResponsesProxy()

    agent._client = lambda: ClientProxy()  # type: ignore[assignment]

    cases = [
        {
            "name": "legacy_recommendation_sleep_at_noon",
            "language": "zh",
            "context": {
                "recommendedScene": {"sceneName": "睡眠", "reason": "系统候选"},
                "localTime": "2026-10-14T12:30:00+08:00",
                "timezone": "Asia/Shanghai",
            },
        },
        {
            "name": "explicit_lunch_break_scene",
            "language": "zh",
            "context": {
                "interactionMode": "scene",
                "currentScene": {"source": "user_selected", "sceneName": "午休"},
                "systemRecommendation": {"sceneName": "睡眠", "reason": "相邻低能量候选"},
                "localTime": "2026-10-14T12:30:00+08:00",
                "timezone": "Asia/Shanghai",
            },
        },
        {
            "name": "auto_notification_intent_empty_contract",
            "language": "en",
            "context": {
                "interactionMode": "auto",
                "systemRecommendation": {"sceneName": "Sleep", "reason": "system candidate"},
                "potentialIntent": None,
                "localTime": "2026-10-14T12:30:00-07:00",
                "timezone": "America/Los_Angeles",
            },
        },
    ]

    def check_result(result: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        combined = " ".join(str(result.get(key) or "") for key in (
            "rawAssessment", "finalText", "rawExplanation", "finalExplanation"
        )).lower()
        forbidden = ["睡前", "临睡", "入睡前", "晚上睡", "bedtime", "before bed", "night's sleep", "sleepy"]
        if any(term in combined for term in forbidden):
            issues.append("bedtime_inference_detected")
        if result["case"] == "explicit_lunch_break_scene":
            lunch_text = f"{result.get('rawAssessment') or ''} {result.get('finalText') or ''} {result.get('rawExplanation') or ''} {result.get('finalExplanation') or ''}"
            if "午休" not in lunch_text:
                issues.append("explicit_current_scene_not_used")
        if not result.get("rawAssessment") or not result.get("rawExplanation"):
            issues.append("missing_raw_model_output")
        if not _preserves_model_text(result.get("rawAssessment"), result.get("finalText")):
            issues.append("assessment_fallback_or_rewrite_detected")
        if not _preserves_model_text(result.get("rawExplanation"), result.get("finalExplanation")):
            issues.append("explanation_fallback_or_rewrite_detected")
        return issues

    def _preserves_model_text(raw: Any, final: Any) -> bool:
        raw_text = " ".join(str(raw or "").split())
        final_text = " ".join(str(final or "").split())
        if not raw_text or not final_text:
            return False
        if raw_text == final_text or raw_text in final_text:
            return True
        # _responses_text may clip long outputs to the TTS max length. That is not
        # a fallback/rewrite as long as the delivered text is still the raw prefix.
        return raw_text.startswith(final_text.rstrip("，。,.!！?？ "))

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        token = active_case.set(str(case["name"]))
        try:
            assessment = await agent.generate_assessment(dict(case["context"]), {}, str(case["language"]))
            assessment_raw = next((item for item in reversed(raw_calls)
                                   if item.get("case") == case["name"] and item.get("calls")), {})
            profile = agent.catalog().profile("pu:0145ZM20260729:o03")
            explanation = await agent.generate_explanation(
                context={**case["context"], "energyGoal": assessment.energy_goal or "lower"},
                selected=profile,
                actual=profile,
                receipt={"success": True, "actualPlaybackState": "playing", "changeKind": "same_song"},
                language=str(case["language"]),
            )
            explanation_raw = next((item for item in reversed(raw_calls)
                                    if item.get("case") == case["name"] and item.get("outputText")), {})
        finally:
            active_case.reset(token)
        call_args = (assessment_raw.get("calls") or [{}])[0].get("arguments", {})
        result = {
            "case": case["name"],
            "energyGoal": assessment.energy_goal,
            "rawAssessment": call_args.get("assessment"),
            "finalText": assessment.text,
            "assessmentPromptContext": assessment_raw.get("promptContext"),
            "rawExplanation": explanation_raw.get("outputText"),
            "finalExplanation": explanation,
            "explanationPromptContext": explanation_raw.get("promptContext"),
        }
        result["checks"] = check_result(result)
        return result

    results = await asyncio.gather(*(run_case(case) for case in cases))

    payload = {"model": os.getenv("AGENT_MODEL_NAME", "gpt-6-luna"), "results": results}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if any(result["checks"] for result in results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
