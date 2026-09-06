<!-- Before publishing, verify that every linked record and quotation is safe for this public repository. -->
<!-- Policy: docs/research-governance/policy.md; record template: docs/research-governance/record-template.md. -->

## Changes Done

<用中文说明解决的具体问题、修改后行为及范围。类型可多选：工程 / 科研 / 实验结果。>

- 权威 PR 记录：<仓库内记录链接；每个参与 agent 的条目、完整证据和反馈历史都在这里>
- 关联主仓库/子仓 PR 与准确版本：<链接及 commit；不适用说明原因>

#### Commit Message

```text
<英文 Conventional Commit 标题，不加 AI 作者或共同作者署名>
```

## Why?

| 类型 | 回答；不适用必须说明依据 |
|---|---|
| 工程 | <实现什么；是否符合良好工程规范、项目规范与用户工程偏好> |
| 科研 | <回答什么科研问题；问题来源；与用户讨论的可公开原文或非敏感引用定位；参考引用> |
| 实验结果 | <为什么跑实验或保存脚本/分析；比较是否公平、有何意义、结论是什么> |

## Testing

- <准确命令、环境、结果和证据链接；未跑的适用验证写“未证实”，不可写“不适用”>
- 既有实验影响与数值回归：<修改前后对照、覆盖范围、预先确定的数值容差>
- 新特性关闭：<基础架构性能能否复现；实际关闭行为的证据，或不适用依据>
- 新特性开启：<改善 / 劣化 / 无明确差异 / 证据不足；公平对照和解释>

## Verification

- 协议与结果：<完整记录链接；基线 / 新特性关闭 / 新特性开启三组及原始产物>
- 独立子 agent 评审：<agent、会话、实际模型或“未证实”、base/head commit、记录链接>
- 反馈及整改：<PR 评论、只追加记录、修复 commit、复评结果；未解决项不能合并>
- 最终待合并版本：<head commit 及对应评审确认；记录归档产生的最终 commit 可在 PR 评论确认>
- 未证实项及限制：<无则明确写无；有适用证据缺口时保持草稿/禁止合并>

## Additional Documents (if any)

- <项目规范、用户原文允许公开的引用、参考文献、关联实验和评审记录；其余事实引用权威位置>

---

Did you use an AI agent to develop this CR? <Y/N>
