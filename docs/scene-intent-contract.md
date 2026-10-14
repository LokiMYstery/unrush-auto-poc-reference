# 场景来源与 TTS 合同 · 2026-10-14

## 相比上一版

系统推荐与用户当前场景分开。旧版把推荐场景与用户意图混在一起，并要求每段普通 Auto 语音显式说能量；本版将系统推荐降为弱参考，当前场景说明音乐帮助，能量轻提且不强制重复。午休推荐或睡眠音乐标签都不是“正在睡前”的证据。

## 输入与可信边界

| 字段 | 合同 |
| --- | --- |
| `context.interactionMode` | `auto` / `scene` / `goal`；显式说明当前交互模式。 |
| `context.systemRecommendation` | 系统建议的 sceneId / sceneCode / sceneName 等摘要；不等于当前活动。 |
| `context.currentScene` | 仅 scene 模式且 source=user_selected；来自用户主动选择，作为场景音乐支持锚点。 |
| `context.potentialIntent` | 仅 Auto；当前 null。未来承接 notification 来源的调节需求（如压力偏高需舒缓、需要日照），不只是弱背景。通知依据不等于新的实时测量。通知接线、时效和清除另行细化。 |

旧 `recommendedScene` 只规范为弱推荐，不能转为 currentScene。`currentPlayback.sceneId` 用于资源绑定，不能推断用户主动选择。当前 iOS 基线没有上传新字段，后端不伪造它们；正式启用明确当前场景需要客户端接入。Goal 仍以 goal / goalPhase 为主要意图，非 Auto 移除 potentialIntent。

```json
{
  "interactionMode": "scene",
  "currentScene": {"source": "user_selected", "sceneName": "午休"},
  "systemRecommendation": {"sceneName": "睡眠"},
  "localTime": "2026-10-14T12:30:00+08:00",
  "timezone": "Asia/Shanghai"
}
```

Auto 示例仅有 interactionMode=auto、systemRecommendation 和 potentialIntent=null，不把推荐场景复制到 currentScene。

## 表达与选曲

1. 首次简述有依据的环境和定性压力；后续未变化不重复。
2. 用户当前场景为帮助锚点，例如午休的短暂休息、专注的减少干扰；不要求反复报场景名。
3. 能量仍只 boost / lower，用作音乐支持方向，不是测量；TTS 能量最多轻提一句，不强制每段出现 energy/能量。
4. 当前时间和环境约束一切表达。推荐与音乐标签不证明当前活动；午休不能推断睡前，夜间也不自动推断用户准备睡觉。
5. Goal 标题和阶段优先；系统推荐只是弱参考；Auto 通知携带的调节需求应主动用于音乐支持与具体轻建议，仍受用户选择及当前事实约束。
6. Touchpad 内部优先、Refine/Explore 候选范围、失败静音、真实回执后解释、工具并行保持既有合同。

## 验证

真实模型 smoke 以合成 Context 检查调整前语音与回执后解释，保留原始模型输出和最终文案，避免只验证经过兜底替换的结果。使用自己的模型配置运行 scripts/smoke_scene_semantics.py；业务画像和 Catalog 仍需另行提供。结果见 examples/scene-semantics-smoke.json（不含密钥、真实健康记录或用户日志）。这验证语音表达，不替代真机播放与真实音乐检索验收。

## 2026-10-14 补充：潜在意图承接调节需求

potentialIntent 不是事件背景或另一个推荐场景，而是通知带进来的本次调节需求。压力偏高通知 → 音乐帮助舒缓 + 可选放慢呼吸；日照需求通知 → 音乐支持行动 + 可选出去走走／接触日光。不要只复述通知；用音乐帮助连接一个具体、可选的小行动。需求为空时不编造；过期通知不说成新健康测量。当前通知链路仍未接入，正常 Auto 仍为 null。本次改提示词策略，先前 smoke 是空意图用例，不作为这两类非空意图的模型实测证据。
