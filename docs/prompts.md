# 当前提示词参考 · 2026-10-14

实际规则在 app/agent.py，服务端构造 JSON input 进入 Responses API；不是客户端保存的静态 system prompt。同一 run 允许多次模型请求和并行工具往返。

本次变更：系统推荐仅弱参考，用户 currentScene 用于 TTS 场景帮助；不从音乐标签或推荐推断用户活动。删除每段强制能量用词，能量只有 boost/lower 且轻提。通知潜在意图只在 Auto 预留，当前空值。

字段、优先级、客户端接入边界和 smoke 见 [场景合同](scene-intent-contract.md)。完整源码规则提取见 [实际模板](prompt-templates.md)。
