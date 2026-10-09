# Touchpad 内部检索与优先调整

## 能力门控与范围

`model_select_with_retrieval_tools()` 同时检查可信 Catalog 的 `unit_kind == "touchpad_track"` 和实际在播快照的 `supportsTouchpad`。满足时向模型提供 `same_song_profile_search`；普通歌曲只提供适用的全库工具。

工具参数不允许模型另传当前歌曲 ID。服务端把任务的 `current_id` 注入执行，范围由 Catalog 的 carrier 关系确定。内部工具仅有可选的 `nameQuery`、`tagPreferences`、`axisTargets`、`language`，没有 `topK`、排除 ID 或跨曲参数；具体 schema 见 [自动提取的 schema](../examples/tool-schemas.json)。

## 检索过程

1. `Catalog.same_song_candidates()` 取得当前可信单元，再收集同 carrier 的其他允许单元。
2. 无查询条件时，当前单元在首位，后面列出所有内部替代单元。
3. 有结构化条件时，使用 `include_playback_unit_ids` 限定内部集合，`top_k` 等于内部范围大小，在完整范围内重新排名。
4. `project_tool_candidates()` 用可信 Catalog 补全画像，并保留原检索返回的排名、评分和匹配证据。

完整 Touchpad Catalog 通常返回当前轨 + 其余八轨。当前代码没有独立的 `len == 9` 断言；返回的是已配置、允许的内部范围。`completeInternalScope` 表示这次返回完整范围，不证明缺失的 Catalog 数据也有九条。

## 返回字段

| 字段 | 含义 |
| --- | --- |
| `toolStatus` | 工具结果状态，失败保持错误语义。 |
| `scope=current_carrier_song` | 范围只属于当前歌曲。 |
| `carrierSongId` | 当前业务 carrier 编号。 |
| `internalTrackCount` | 内部范围中的实际单元数。 |
| `completeInternalScope` | 工具返回完整内部范围。 |
| `supportsInternalAdjustment` | 当前单元属于 Touchpad 类型。 |
| `candidates` | 完整内部候选，含当前轨。 |

每个候选包含 `playbackUnitId`、`songId`、`name`、`summary`、`tags`、`axes`、`unitKind`、`trackOrder`、`resourceBindings`。存在时保留 `rank`、`channelScore`、`matchEvidence`、`retrievalChannel`、`tagSupport`。当前轨另以 `currentTrustedProfile` 提供给模型，用于比较。

## 与全库和动作的关系

`same_song_profile_search`、`library_profile_search`、`music_description_recall` 可以在一轮同时请求。内部优先发生在比较与选择阶段，不是内部失败后才进行全库检索。

初次 Auto / 普通确认：模型判断内部更适合时优先内部，跨曲更贴合时可跨曲。TP Refine 的结果被代码限定为同曲内部；无 TP 的 Refine 和所有 Explore 则限定不同 carrier。当前单元被预先加入可信选择集合，正常保持无需伪造检索调用。

`submit_selection` 仅允许来自可信召回集合、且符合本次动作范围的 ID，提交最多五个有序选择与首选的一句公开比较理由。随后资源验证按顺序确认可用候选；替代候选不会沿用首选不适用的比较理由。

## 客户端执行

内部工具选择的是逻辑播放单元，不直接发混音权重或旋钮坐标。iOS 把它映射到同曲 Touchpad 调整模式，更新实际混音和可见控制点；跨曲使用原资源/Preset/声画链路。后端只在收到原生真实回执后解释结果。

用户手动混音时 `isExactPlaybackUnit=false`，资源锚点不能被解释为纯单轨。语音描述目标音乐的变化和帮助，不读轨号，不说“keep the same song”。
