# AutomationBench token 记账缺陷分析：每题最后一次模型调用的 usage 丢失

> 发现日期：2026-07-26 · 发现人：Zane + Kimi（在对账 cost-ledger 与 auto-bench 导出数据时）
> 影响版本：zapier/AutomationBench @ main（2026-07-26 克隆）
> 性质：token 记账缺陷（**非评分缺陷**），排行榜结论不受影响

## TL;DR

AutomationBench 的 `input_tokens` / `output_tokens` 统计会**丢掉每个任务最后一次模型调用的
usage**，因为它的 turn 级采集钩子在终局之后没有触发机会；而 `cached_input_tokens` 走
client 级钩子不受影响。后果：50/50 个任务全部出现 `cached > input` 的"不可能"数据，
`uncached_input_tokens` 被 `max(0, …)` 截断为 0，input 侧系统性低估约 6.5%。

## 发现过程

2026-07-26，我们用自建的 cost-ledger（LiteLLM proxy 层账本，记录每一次 HTTP 调用的
完整 usage）跑 AutomationBench 50 题基线（sales 域 1-50，deepseek-v4-flash，10 并发），
并与 auto-bench 自己的导出 JSON 对账：

| 指标 | cost-ledger（proxy 层） | auto-bench 导出 |
|---|---|---|
| 模型调用次数 | 1120（含 1 次健康检查 ping） | 1119（num_model_calls 汇总） |
| cache_read tokens | 21,158,400 | 21,158,400 ✅ 分毫不差 |
| input tokens | 22,286,306 | 20,827,237 ❌ 少 1,459,069 |

调用次数吻合（1119 + 1 = 1120），缓存数分毫不差，但 input 差了近 146 万——
矛盾锁定在 auto-bench 内部的 input 口径。

进一步逐题检查：导出的 50 个任务**全部**满足 `cached_input_tokens > input_tokens`，
而 `uncached_input_tokens = max(0, input − cached)` 全部被截断为 0。
（`visualizer/runs/local/deepseek-v4-flash-20260726-232810-328.json`）

## 根因：两条采集路径，看到的调用集合不同

auto-bench 内部有两个互不相关的 usage 累计器：

**路径 A：input / output tokens —— turn 级钩子（有缺陷）**

- `runner.py` 的 `_extract_usage_and_debug(state)`，在每次 `env_response`
  （工具结果返回环境）时，从 `state["trajectory"][-1]["response"]` 取该轮 response 的
  `usage.prompt_tokens / completion_tokens` 累加进 `state["_usage"]`
- **缺陷**：任务的最后一次模型调用之后没有 `env_response`（任务结束，不再产生工具
  调用），其 usage 永远等不到被采集的时机 → **每题固定丢失最后一次调用**

**路径 B：cached / reasoning tokens —— client 级钩子（无缺陷）**

- `clients.py:112-133`，在每次原生 API 调用返回时立即累计
  `prompt_tokens_details.cached_tokens`（OpenAI Chat 格式）等到 `state["_perf"]`
- client 层对每一次调用（包括最后一次、包括框架内部重试）都照记

终局调用的 prompt 恰好是全对话历史（最大的一次），且其 token 约 99% 命中缓存，
于是丢失部分几乎全部体现为缓存 token 的"多出"——表现为 `cached > input`。

**数字验证**：缺口 1,459,069 ≈ 50 题 × 每题最后一次调用约 2.9 万 prompt token，
与"每题丢一次终局调用"的假设定量吻合。

```
                        模型调用生命周期
        ┌──────────────────────────────────────────┐
        │ turn 1 → env_response → turn 2 → … → 终局 │
        └──────────────────────────────────────────┘
 路径 A（turn 级）     ✓    ✓          ✓   …   ✗ 丢！
 路径 B（client 级）   ✓    ✓          ✓   …   ✓
 cost-ledger（proxy）  ✓    ✓          ✓   …   ✓
```

## 影响评估

| 维度 | 影响 |
|---|---|
| 排行榜指标（task_completed_correctly） | **零影响**。评分基于确定性断言，与 token 统计无关 |
| 模型间相对排名 | 基本无影响。所有模型被同等口径低估 |
| 美元成本显示 | 低估约 1–2%（丢失部分几乎全是缓存价 token） |
| 以 token 数据做精细分析（如成本曲线研究） | **系统性偏差**，input 低估 ~6.5%，且 uncached 恒为 0 完全失真 |

为什么一直没人发现：

1. token 统计不在排行榜关键路径上，没人做会计恒等式校验
2. `max(0, input − cached)` 把负值异常静默截断为 0，信号被代码自己掩埋
3. 只有上报缓存细分的 provider 才会暴露（不报则 cached 恒 0，无感）
4. 相对比较不受影响，"感觉成本低了点"不足以触发调查

## 附：同代码库发现的第二处记账问题

`usage.py:68-72`：OpenAI 语义中 `completion_tokens` **已经包含** reasoning tokens，
代码却又把 `completion_tokens_details.reasoning_tokens` 加了一遍——o 系 / reasoning
模型的 output 计数会被放大近两倍。跨模型对比时系统性偏袒/冤枉某类模型。

## 修复建议（供上游参考）

1. turn 级钩子在 rollout 结束（`is_completed`）时补记最后一条 response 的 usage；
   或干脆统一改为 client 级累计（路径 B 已证明可靠）
2. 去掉 `max(0, …)` 静默截断，改为恒等式校验失败时告警：`cached ≤ input` 应恒成立
3. reasoning tokens 只拆分不重复累加

## 对自己实验基建的教训（为什么这反而验证了我们的设计）

1. **计费口径的坑多在 harness 的采集时机，不在模型侧**。引用任何第三方的 token
   数据前，先问"它的计数钩在哪一层、终局路径有没有覆盖"。
2. **账本放在 proxy 层是对的**：每次 HTTP 调用必有一条记录，与 harness 内部
   生命周期完全解耦，天然免疫此类缺陷。
3. **对账是必要仪式**：本次缺陷正是"两边独立计数、总数互验"对出来的。
   以后每接一个 harness，先跑小样本做对账，再信它的数据。

## 证据文件

- auto-bench 导出：`AutomationBench/visualizer/runs/local/deepseek-v4-flash-20260726-232810-328.json`
- proxy 账本：`cost-ledger/logs/ledger.jsonl`（1120 行，per-call 完整 usage）
- 关键源码：`automationbench/runner.py:248-269`（turn 级钩子）、
  `automationbench/clients.py:107-133`（client 级钩子）、
  `automationbench/export.py:138-156`（导出与截断）、
  `automationbench/usage.py:64-72`（reasoning 重复计数）

*备注：结论基于代码阅读 + 一轮 50 题实测的三条独立证据（调用数吻合、50/50 全异常、
缺口 ≈ 50 × 2.9 万），未写最小复现钉死具体行号；对"以 ledger 为准"的用途已充分。
如需上报上游（GitHub issue），建议先跑单题最小复现。*
