from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("AUTO_POC_DATA_DIR", APP_DIR.parent / "data"))
VOICE_DIR = DATA_DIR / "voices"
A_MAP_PLACE_KNOWLEDGE_PATH = Path(os.getenv("AUTO_AMAP_PLACE_KNOWLEDGE_PATH", APP_DIR / "assets" / "AmapPlaceKnowledge.json"))
DB_PATH = Path(os.getenv("AUTO_POC_DB_PATH", DATA_DIR / "auto_poc.sqlite3"))
MUSIC_PROFILE_PATH = Path(
    os.getenv("AUTO_POC_MUSIC_PROFILE_PATH", APP_DIR / "assets" / "music" / "v1" / "music_profile_v1.json")
)
PLAYBACK_UNIT_CATALOG_PATH = Path(
    os.getenv("AUTO_POC_PLAYBACK_UNIT_CATALOG_PATH", APP_DIR / "assets" / "music" / "v1" / "PlaybackUnitCatalog.json")
)
DESCRIPTION_EMBEDDINGS_PATH = Path(
    os.getenv("AUTO_POC_DESCRIPTION_EMBEDDINGS_PATH", APP_DIR / "assets" / "music" / "v1" / "music_description_clap_v2.npz")
)

API_PREFIX = "/agentic-player/api/v2"

RUN_STATUS_RUNNING = "running"
RUN_STATUS_WAITING_PLAYBACK = "waiting_playback"
RUN_STATUS_DONE = "done"
RUN_STATUS_CANCELLED = "cancelled"
RUN_STATUS_FAILED = "failed"

PHASE_ASSESSMENT = "ASSESSMENT"
PHASE_SELECTION = "SELECTION"
PHASE_EXPLANATION = "EXPLANATION"
PHASE_ERROR = "ERROR"


def required_token() -> str:
    token = os.getenv("AUTO_POC_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("AUTO_POC_API_TOKEN is required")
    return token


def model_enabled() -> bool:
    return bool((os.getenv("AGENT_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip())


def model_name() -> str:
    return os.getenv("AGENT_MODEL_NAME", "gpt-6-luna").strip() or "gpt-6-luna"


def model_base_url() -> str | None:
    raw = os.getenv("AGENT_MODEL_BASE_URL", "").strip().rstrip("/")
    if not raw:
        return None
    for suffix in ("/responses", "/chat/completions"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)]
    return raw


def agent_timeout_seconds() -> float:
    return float(os.getenv("AGENT_MODEL_TIMEOUT_SECONDS", "8"))


def tts_enabled() -> bool:
    return bool(os.getenv("VOLCENGINE_TTS_API_KEY") or os.getenv("VOLCENGINE_TTS_API_KEY_FILE"))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_DIR.mkdir(parents=True, exist_ok=True)


def configure_retrieval_environment() -> None:
    os.environ.setdefault("MUSIC_PROFILE_ASSET_PATH", str(MUSIC_PROFILE_PATH))
    os.environ.setdefault("MUSIC_DESCRIPTION_EMBEDDINGS_PATH", str(DESCRIPTION_EMBEDDINGS_PATH))
    local_clap = Path.home() / ".cache" / "agentic-player" / "clap" / "365dea6ef167def6676140ed93bbc43f84dabb28"
    if local_clap.is_dir():
        os.environ.setdefault("CLAP_MODEL_ID", str(local_clap))
        os.environ.setdefault("CLAP_MODEL_REVISION", "365dea6ef167def6676140ed93bbc43f84dabb28")
