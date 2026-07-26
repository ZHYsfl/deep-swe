# cost-ledger：harness 无关的 token 成本账本

给「经验复用 → token 成本递减」研究用的基础设施。
一个本地 LiteLLM proxy 挡在所有 agent 和模型 API 之间，**每次调用的用量
（含缓存分层）原样落 JSONL**；美元在分析阶段才换算，原始数据永远不过时。

```
agent (harbor / pier / auto-bench / 你的 harness)
   │  OPENAI_BASE_URL=http://127.0.0.1:4000
   │  X-Ledger-Run / Task / Episode / Step 请求头注入归属标签
   ▼
┌─────────────────────────────────────────────┐
│ LiteLLM proxy (:4000)                       │
│   ledger/callback.py ──采集层，每次调用触发   │
└─────────────────────────────────────────────┘
   │ 一行一条 UsageRecord（只有 token 计数，没有美元）
   ▼
logs/ledger.jsonl ──────────────┐
   ▼                             ▼
ledger/analyze.py summary    ledger/analyze.py curve
（per-task 汇总+成本）        （per-episode 学习曲线 CSV+PNG）
   ▲
ledger/pricing.py（缓存三档价格表，显式可溯源）
```

## 为什么这么设计（原则）

1. **采集与定价解耦**：落盘只有 token 计数。价格变了重算就是，原始数据不朽。
2. **缓存三档分开记**：`input_fresh / input_cache_read / input_cache_write`。
   总 input 成本下降里绝大部分是 prompt 缓存的功劳（实测单题内命中率 ~100%），
   **只有 fresh input 的下降才是 memory 的净贡献**。不分开就是自欺。
3. **信 API 的 usage，不信本地估算**；reasoning token 从 output 里拆出来单独记
   （OpenAI 语义的 completion_tokens 已含 reasoning，再加一遍是 AutomationBench 的 bug）。
4. **harness 零侵入**：不改任何 harness 代码，base_url 指过来就接入。
5. **归属标签走请求头**：编排层启动每个任务时注入 `X-Ledger-Episode` 等，
   学习曲线的主键不依赖任何 harness 内部结构。

## 分层结构

| 文件 | 层 | 职责 | 依赖 |
|---|---|---|---|
| `ledger/schema.py` | 数据 | UsageRecord / TokenCounts / 各家 usage 归一化 / JSONL 读写 | 无 |
| `ledger/callback.py` | 采集 | LiteLLM CustomLogger，提取标签+usage 委托给 schema | litellm, schema |
| `ledger/pricing.py` | 定价 | 三档缓存价格表 + 成本计算 | schema |
| `ledger/analyze.py` | 分析 | per-task 汇总 / per-episode 曲线 | schema, pricing |

只有 callback 知道 litellm 的存在；只有 pricing 知道钱的存在。

## 快速开始

```bash
cd cost-ledger
set -a && source ../.env && set +a          # 提供 DEEPSEEK_API_KEY

# 1. 起 proxy（前台或 tmux 均可）
.venv/bin/litellm --config proxy.config.yaml --port 4000

# 2. 让 agent 把请求发过来（带归属标签）
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Ledger-Run: exp001-nomem" -H "X-Ledger-Task: sales.multi_hop_lookup" \
  -H "X-Ledger-Episode: 1" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}'

# 3. 分析
.venv/bin/python -m ledger.analyze summary logs/ledger.jsonl
.venv/bin/python -m ledger.analyze curve   logs/ledger.jsonl --out report/
```

## 各 harness 接入方式

**AutomationBench**（原生支持 base-url）：

```bash
uv run auto-bench --model deepseek-v4-flash \
  --base-url http://127.0.0.1:4000 --api-key dummy \
  --headers X-Ledger-Run=exp001-nomem ...
```

**mini-swe-agent / pier**（litellm 系，走环境变量）：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:4000   # 模型名写 openai/deepseek-v4-flash
export OPENAI_API_KEY=dummy
```

**harbor（容器内 agent）**：容器要访问宿主机 proxy，用 `--ae` 传环境变量，
网络侧用 `--allow-agent-host host.docker.internal`（或宿主机在 docker 网桥的 IP）放行：

```bash
harbor run ... --ae OPENAI_BASE_URL=http://host.docker.internal:4000 \
  --ae OPENAI_API_KEY=dummy --allow-agent-host host.docker.internal
```

**标签注入**：harness 不支持自定义 header 时，由编排层按时间窗口归属
（JSONL 里有每条记录的 `ts`，事后 join 任务起止时间即可），
或在 agent 侧包一层注入 header 的薄 wrapper。

## UsageRecord 字段（logs/ledger.jsonl 每行）

| 字段 | 含义 |
|---|---|
| `ts` | Unix 时间戳 |
| `model` | 请求模型名 |
| `counts.input_fresh` | 未命中缓存的输入 token（memory 净贡献看这里） |
| `counts.input_cache_read` | 命中前缀缓存的输入 token |
| `counts.input_cache_write` | 缓存写入（Anthropic 专属，DeepSeek 恒 0） |
| `counts.output` / `output_reasoning` | 输出 / 其中 reasoning 部分 |
| `run` / `task` / `episode` / `step` | 请求头注入的归属标签 |
| `latency_ms` / `finish_reason` | 延迟 / 结束原因（error 表示失败调用） |

## 定价

`ledger/pricing.py` 的 `PRICE_TABLE` 是唯一价格来源，分析时会打印用了哪个条目。
改价格 = 改表；历史数据重算 = 重跑 analyze。实验间对比务必冻结同一份价格表。

## 已知边界

- proxy 未启用鉴权（本地回路专用，不要暴露到网络）
- DeepSeek 的缓存是其服务端自动 context caching，块粒度 64 token，小 prompt 不命中
- 失败调用也会落一行（`finish_reason=error`，token 计数为实际已发生的部分）
