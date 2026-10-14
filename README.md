# Unrush Auto PoC：后端实现参考

2026-10-14 的后端参考快照，供开发对照**提示词、Touchpad 内部检索、并行工具调用、音乐选择及回执后的 TTS**。iOS 对应基线是 `Qiyin-Tech/unrush-ios` 的 `codex/auto-poc-20261007`，固定 commit `d44650f0`。

## 阅读入口

| 要了解什么 | 入口 |
| --- | --- |
| 当前发给模型的提示词与工具 schema | [docs/prompts.md](docs/prompts.md)；[app/agent.py](app/agent.py) |
| Touchpad 内部优先、完整九轨的范围与返回字段 | [docs/touchpad-tool.md](docs/touchpad-tool.md)；[app/catalog.py](app/catalog.py) |
| 三阶段 run、并发执行、选曲限制及真实回执 | [docs/call-chain.md](docs/call-chain.md)；[app/main.py](app/main.py)、[app/agent.py](app/agent.py) |
| 分阶段交付、取消、重启恢复和会话摘要 | [app/store.py](app/store.py)、[app/schemas.py](app/schemas.py) |
| 环境去重、压力质性表达与语言路由 | [app/environment.py](app/environment.py)、[app/tts.py](app/tts.py) |
| 全库结构化查询和描述召回 | [app/music_profile_v2_2.py](app/music_profile_v2_2.py)、[app/music_recall_v2_2.py](app/music_recall_v2_2.py) |

## 内部优先的精确含义

有真实 Touchpad 能力时，提供当前歌曲的完整内部范围，让模型比较当前轨与其余轨。内部候选适合目标时优先内部调整；内部不合适或用户 Explore 时换歌曲。**内部优先是选择偏好，全库检索仍可并行。**

普通 Auto 初次/确认适配的内部优先目前由提示词表达，没有单独的程序判定器证明“内部均不合适”才放行跨曲。代码硬约束包括：只开放适用工具、选择前完成所有可用工具的调用、只选择可信的已召回 ID，以及 Refine / Explore 的候选范围。

| 行为 | 后端/客户端约束 |
| --- | --- |
| Keep | iOS 本地保留当前声音，不创建新选歌任务。 |
| Refine，有 Touchpad | 后端限定同 carrier 的内部候选。 |
| Refine，无 Touchpad | 后端限定另一首歌曲；不提供内部工具。 |
| Explore | 后端限定不同 carrier。 |
| 普通 Auto | 有 Touchpad 时模型优先内部；跨曲可选；当前可信单元可用于正常保持。 |

## 源码与运行数据的边界

`app/` 保留当前后端实现，`source-snapshot.json` 记录各源文件摘要及公开版调整。提示词、工具和选择链路保持原实现；公开版移除了部署专属的英文音色默认值，使用时配置自己的 `VOLCENGINE_TTS_VOICE_ID_EN`。

独立运行需另行提供兼容业务资产。此仓库提供源码、schema 与说明，实际音乐画像、资源映射、向量、音频、数据库、用户请求轨迹和凭据通过本地配置接入。`.gitignore` 已排除这些运行文件。

### 必需资产

| 文件或配置 | 用途 |
| --- | --- |
| `AUTO_POC_MUSIC_PROFILE_PATH` | 与当前 Profile schema 兼容的音乐画像；含启用单元与检索属性。 |
| `AUTO_POC_PLAYBACK_UNIT_CATALOG_PATH` | `playbackUnitId → songInnerId → serviceSongId` 及资源绑定。 |
| `app/assets/music/v1/FastSceneSongTiers.json` | 当前快选使用此固定路径，配置包含 18 个场景，policy 为 `shuffle-within-high-middle-low-then-concatenate`。 |
| `AUTO_AMAP_PLACE_KNOWLEDGE_PATH` | 启用 POI 时使用的规则文件。 |
| `AUTO_POC_DESCRIPTION_EMBEDDINGS_PATH`、CLAP 模型或编码端点 | 启用描述召回需要匹配的向量、模型身份与 provenance。 |

缺少必需 Catalog/Profile 或三档配置时，服务不会完成启动。示例中不存在可播放业务媒体；阅读工具合同无需启动服务。描述召回缺少模型/向量时返回结构化错误，不能用伪向量冒充真实检索。

### 配置与启动

使用 Python 3.11+。设置 `.env.example` 中的变量到进程环境后，安装现有依赖并启动：

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Azure/OpenAI-compatible 的 `AGENT_MODEL_BASE_URL` 使用 `/openai/v1` 或对应服务根地址，`AGENT_MODEL_NAME` 使用已配置 deployment。实际地址和 key 由运行环境提供。额外的 CLAP 依赖见 `requirements-retrieval.txt`，仅在需要本地编码时安装。

当前 API 前缀是 `/agentic-player/api/v2`，请求使用自行配置的 `AUTO_POC_API_TOKEN`。完整路径、阶段和回执规则见 [调用链路](docs/call-chain.md)。

## 实现边界

- 一个逻辑 run 包含 ASSESSMENT、SELECTION、等待原生回执、EXPLANATION；这些阶段可有多次模型请求，不等同于一次模型 API 调用。
- 选曲成功不等于手机播放成功。音乐解释以实际回执为准；失败和技术降级的消息不进入 TTS。
- 健康算法在 iOS，本仓库接收摘要并处理音乐与表达。通知意图、连续感知和周期自动适配尚未接入这条 Auto 链路。
- 当前 `music_profile_v2_2.music_retrieval_instruction()` 是未使用的 donor 辅助入口，引用外部 `instruction_loader_v2_2`；本仓库 Auto 的实际提示词由 `app/agent.py` 构建。
- 本次同时将 2026-10-09 历史 Auto 快照更新为当前 Auto／Goal 共用后端，并加入场景来源与 TTS 修订。场景修改使用真实模型 smoke 验证及针对性回归；没有进行手机或模拟器测试，未部署。

## 2026-10-14：场景来源与 TTS 更新

区分系统推荐与用户当前场景。推荐只作弱参考，不能推断用户正在睡前；明确当前场景时多说音乐的场景帮助，能量轻提。Auto 预留通知潜在意图，当前为空。旧客户端字段不推断用户场景，新字段接入仍需同步。数据合同与验证见 [docs/scene-intent-contract.md](docs/scene-intent-contract.md)，完整规则见 [docs/prompt-templates.md](docs/prompt-templates.md)。
