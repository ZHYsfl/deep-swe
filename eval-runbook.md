# 本地 Bench 评测运行记录

环境：WSL2 Ubuntu（24C / 15G / 946G），Docker Desktop 已开 WSL 集成，`docker` 在 WSL 内直接可用。
模型统一用 `deepseek/deepseek-v4-flash`，密钥在 `/home/zane/deep-swe/.env`（`DEEPSEEK_API_KEY`）。

工具安装：

```bash
uv tool install datacurve-pier   # pier 0.3.0，跑 DeepSWE no-network 任务包
uv tool install harbor           # harbor 0.20.0，跑 Harbor Hub 上的数据集
```

> 注意：Docker 权限若报 permission denied，需 `wsl --shutdown` 重开 WSL（用户 zane 已在 docker 组，重开后生效）。

---

## 1. DeepSWE 任务（Pier，no-network 任务包）

DeepSWE 任务包是 Harbor 格式但带 no-network 语义，必须用 Pier >= 0.3.0 跑
（Pier 的 agent 插件有 `network_allowlist()`，断网时仍放行 API 端点；Harbor 一刀切会把容器内 agent 的 API 调用也断掉）。

```bash
cd /home/zane/deep-swe
pier run -p tasks/abs-module-cache-flags \
  --agent mini-swe-agent \
  --model deepseek/deepseek-v4-flash \
  --env-file .env
```

结果（`jobs/2026-07-26__15-54-04/`）：

- 耗时 6m28s，成本 $0.03
- reward = 0；f2p 6/20，p2p 3/3，partial 0.39
- 14 个失败全是 agent 真实实现缺陷（其中 3 个 repl 测试因 agent 实现搞挂测试进程而 missing），报告链路自洽
- 任务镜像 `public.ecr.aws/d3j8x8q7/swe-bench-202605:kh75679ajj3b8dtd7se3h7z0a1833y6r-v1.1`（3.9GB）已缓存本地

换任务只换 `-p tasks/<task-id>`；采样多任务用 `--n-tasks N --sample-seed 0`；查看结果 `pier view jobs`。

---

## 2. Terminal-Bench 2.1（Harbor，Hub 数据集）

Terminal-Bench 2.1 在本地 `harbor/registry.json` 快照里没有（那是旧快照），但 Hub 上有：
`terminal-bench/terminal-bench-2-1`（Public，89 任务）。Harbor 是其官方 harness。

冒烟测试（前 2 个任务）：

```bash
cd /home/zane/deep-swe
harbor run -d terminal-bench/terminal-bench-2-1 \
  -a mini-swe-agent \
  -m deepseek/deepseek-v4-flash \
  --env-file .env \
  -l 2 -n 2 -y
```

结果（`jobs/2026-07-26__20-55-14/`）：

- 耗时 18m57s，成本 $0.044（206 万 input tokens，98% 命中缓存）
- 数据集 pinned 在 `sha256:7d7bdc1c…`（Hub digest 固定，可确认是 2.1）
- 两题均 0 分，均为真实能力失败，非 harness 问题：
  - `torch-tensor-parallelism`：agent 只验证了 world_size=1，隐藏测试跑 ws=2/4 真分布式，权重切分维度错，12 测挂 11
  - `write-compressor`：AgentTimeoutError，15 分钟任务超时没做完
- 链路验证通过：镜像构建 ✅ / 容器内装 agent ✅ / 容器内调通 DeepSeek API ✅ / verifier + CTRF 报告 ✅

常用参数：

- `-d <org/name>` 指定 Hub 数据集；`-l N` 限制任务数；`-i/-x` 按 glob  include/exclude 任务名
- `-n N` 并发 trial 数（默认 4）
- `--timeout-multiplier X` 统一缩放超时（ leaderboard 提交禁止改这个）
- `-a` 可选 ~40 种内置 agent（claude-code、codex、openhands、mini-swe-agent……），或 `module:Class` 自定义
- 结果查看 `harbor view jobs`；上传分享 `harbor upload jobs/<dir>`

---

## 3. abundant/swe-marathon（Harbor，已冒烟）

超长程任务集（20 题：excel-clone、s3-clone、rust-c-compiler 这类"从头克隆大型软件"）。
单题配置（以 excel-clone 为例）：agent 超时 4h，verifier 超时 2.2h，4 CPU / 16GB，
环境 `network_mode = "public"`。评分 = 0.5 × 18 个 pytest 正确性 gate + 0.5 × CUA UX 评分
（Playwright 驱动 Chromium + Claude 按 rubric 打分）。

**坑：verifier 需要 `ANTHROPIC_API_KEY`**（CUA UX 评审用 Claude Opus 4.7，且硬性规定
评审基础设施失败则整个 trial 报错不给分）。没有 Anthropic key 的 workaround：

```bash
# 1. 任务包拉到本地（hub 缓存目录 ~/.cache/harbor/tasks/packages/...），改 CUA 评审模型
cp -r ~/.cache/harbor/tasks/packages/abundant/excel-clone/<digest> tasks-local/excel-clone
# 2. 把 tests/cua_config.json 的 "model" 改为 "deepseek/deepseek-v4-flash"
# 3. 跑本地副本：dummy 值满足 ANTHROPIC_API_KEY 预检，DEEPSEEK_API_KEY 透传给 verifier
set -a && source .env && set +a
ANTHROPIC_API_KEY=dummy harbor run -p tasks-local/excel-clone \
  -a mini-swe-agent -m deepseek/deepseek-v4-flash \
  --env-file .env --ve DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  --agent-timeout-multiplier 0.25 -n 1 -y
```

注意：CUA 评审需要看图（截图），deepseek-v4-flash 若非多模态，UX 半场可能失败。
有真 Anthropic key 时无需以上修改，直接加进 `.env` 跑 `-d abundant/swe-marathon` 即可。

状态（2026-07-26）：冒烟完成（`jobs/2026-07-26__21-53-55/`），总耗时 45m46s。

- 镜像构建 ~15min ✅；agent 21 分钟就提前交卷（1h 限额只用 1/3）✅
- 18 个 pytest gate 正常跑完：**过 6/18**（api/copy_fill/csv/deps/persistence/anti_cheat），
  单测粒度 pass_rate 0.735，partial_score 0.333；correctness_reward = 0（二值需全过）
- **CUA 阶段确认 HARD FAIL**：`deepseek-v4-flash` 不支持 vision，Computer1 直接抛
  `ValueError: Model does not support vision input` → 按任务规则不写 reward 文件 →
  trial 报 `RewardFileNotFoundError`（这是设计行为，不是 harness bug）
- 结论：**swe-marathon 要拿完整分，必须给 CUA 配一个视觉模型**。没有真
  `ANTHROPIC_API_KEY`（或其他 vision 模型 key）时，只能参考 verifier 目录里的
  `metrics.json`（gates_passed / partial_score / 各 gate pytest 明细），
  correctness 半场数据是完整可信的

---

## 4. ProgramBench（独立 harness，mini-swe-agent 官方 baseline）

Meta（facebookresearch）的净室重建基准：只给编译好的二进制 + 文档，agent 重建整个程序，
隐藏行为测试评分。隔离设计：容器 `--network none` 全断网，模型调用从宿主机侧发出
（mini-swe-agent v2 的 agent 循环在宿主，只有 bash 进容器执行），因此不怕 agent 抄源码。

不是 Harbor 任务格式，用自己的 harness（pip 包 `programbench`），但有 mini-swe-agent 官方 baseline：

```bash
set -a && source .env && set +a
uvx --from mini-swe-agent --with programbench mini-extra programbench \
  -m deepseek/deepseek-v4-flash --slice 0:1 -o pb-runs/run1
```

注意点：

- 必须 `--with programbench`，否则 uvx 环境里缺包报 `ModuleNotFoundError`
- `--slice 0:1` 取第 1 个实例；`--filter <regex>` 按实例 ID 过滤；`-w N` 并行 worker
- 单实例上限 ~7h（容器 sleep 7h）；容器声明 20C/60G 是限额非预留
- 评分：跑完后 `uvx programbench eval <run_dir>` + `uvx programbench info <run_dir>`
- 测试 blob 从 HuggingFace 下载（`uvx programbench blob --help`）

状态（2026-07-26）：首实例 `abishekvashok_1776_cmatrix:task_cleanroom_v6` 已跑完（`pb-runs/run1/`）。

- agent：9m30s 交卷，41 次 API 调用，$0.024；提交 `cmatrix.c` 原创源码 + gcc/ncurses 编译脚本
- 评分：`uvx programbench eval pb-runs/run1` 约 3 分钟（含 HF 拉 16 个测试 blob 文件）
- 结果：**91 分（708/769 测试通过），无作弊警告、无判负**
- 61 个失败全是行为保真度的边角：`--help/--version` 退出码、无效颜色/缺参时的报错行为、
  lock 模式的默认消息渲染等——9 分钟实现主功能对、细节丢分，符合预期
- 评分约定：100 分不代表 solved（四舍五入），只有 ✅ 才算真过

---

## 5. 已调研未跑（候选）

**Hub 全量普查（2026-07-27，250 个数据集）**：清单存 `hub-datasets-2026-07-27.json`
（名字/可见性/任务数）。注意其中 ~90 个是 openthoughts/tasktrove 的实验变体、
~10 个是 tbench-* 单题功能测试残留，实际独立 bench 约 80 个。按研究目标筛选：

第一梯队（补现有矩阵空缺）：

- ~~openthoughts/tasktrove 全系~~（2026-07-27 决定弃用）：RL 训练集出身，有数据污染
  风险和弱 verifier 问题，绝对分无公信力；虽然同序对照下相对差仍无偏，但天花板
  效应可能压缩组间差，且对外讲实验时"用了别人的训练集"是额外的辩护负担。
- **swe-bench/swe-smith**（Hub 上 100 题样本；完整版可生成 50k）：同 repo 多任务，
  正是 §6 说的"SWE-bench repo 聚类"思路的放大版——repo 内经验复用的标准场景，
  且同 repo 共享镜像，成本低。
- **termigen/termigen-environments**（3566 题，UCSB，11 个类别）：TMax 同族终端任务，
  全部环境经过验证（arXiv 2602.07274）。类别标签适合做域内复用切片；
  终端题成本实测极低（TMax $0.002/题）。注意 intricate 题镜像构建慢，先 warm build。
- **userbench/UserBench**（620 题，含 train400 变体）：tau3 同族多轮用户模拟，
  量多一倍，可给多轮场景补统计功效。
- **abundant/swe-gen-{rust,go,java,js,cpp}**（各 ~1000 题）：swe-marathon 同门，
  按语言切片的生成式 SWE 任务。

第二梯队（特定机制验证）：

- **gorilla/bfcl**（3641 题函数调用）：短平快，测"工具 schema 记忆"，成本极低；
  但 episode 太短，测不了长程。
- **reasoning-gym/reasoning-gym-{easy,hard}**：程序化生成、算法可验证，无限同族题，
  适合当技能记忆的对照组。
- **openthoughts/openthoughts-tblite**（100 题，难度标定终端 bench）：快速校准难度用。
- **harbor-index/harbor-index-1.0**（82 题）：跨 bench 策展索引，广度抽样。

明确排除：

- xlang-ai/osworld-verified（361）、apple/mmau（1000）、mmtb/multimedia-terminalbench（105）：
  需要视觉，deepseek-v4-flash 无视觉能力（swe-marathon 的 CUA 已实测 HARD FAIL）
- gaia/gaia（165）：需要真联网检索
- openai/swe-lancer-*：单题成本美元级，且评测流程绑定 OpenAI 生态
- tasktrove 的 mix-h* / exp-rle-* 变体：别人 RL 实验的混合配方，噪声大，只取
  rpt 主干和 curriculum 系列
- openai/simpleqa、aime、gpqa-diamond、strongreject 等：QA/数学/安全类，非 agentic
  工程任务，与经验复用研究无关

单个 bench 调研笔记：

- **FrontierSWE**（Proximal，17 题 × 20h，连续分 0~1，mean@5/best@5）：Harbor 编排，
  repo: Proximal-Labs/frontier-swe。超长程 + 开放式研究难度；部分题要 GPU。memory 研究的
  现成案例库（Opus 丢进度、模型自发作弊 6/30）。跑法同 Harbor 数据集，注意成本。
- **GSO**（gso-bench.github.io，102 题性能优化，对标专家 commit 的加速比）：
  本地 `harbor/adapters/gso/` 有适配器；官方 harness 是 patch-based 干净环境重放（可信度高）。
  2026-07 起任务加了断网防抄上游 commit，用 Harbor 跑需确认 adapter 是加固后版本，
  否则分数与官网 leaderboard 不可比。迭代循环型任务，最适合测 memory/token 成本。
- **SkillsBench**（87 题，"有 skill vs 无 skill"对照实验）：直接测经验复用，
  Harbor `--skill` 参数可挂 skill 目录。
- **BrowseComp**（OpenAI，1266 题网页深度搜索）：调研后排除——需要真联网浏览 +
  搜索 API 基建，非 Harbor 格式，且只考"信息觅食"不考工程成果积累，与本研究方向弱相关。
  变种：BrowseComp-ZH（中文）、MM-BrowseComp（多模态）。
- **TheAgentCompany**（174 题模拟公司办公，本地 `harbor/adapters/theagentcompany/` 有适配器）：
  与 AutomationBench 同族但用 checkpoint 部分分；如需第二个办公自动化场景可启用。

## 6. Bench 需求与选型分析（经验复用 / token 成本递减研究）

**研究目标**：agent 越用越聪明——跨任务复用先前经验，成功率随 episode 数上升、
单任务 token 成本随 episode 数下降（学习曲线）。长不长程不是关键。

**选型四条标准**：

1. 任务间有可复用的重复模式（同 API / 同仓库 / 同流程 / 同陷阱类型）
2. 任务量够大（几十~几百 episode 才画得出曲线）
3. 单 episode 便宜（同预算曲线更密）
4. 评分确定 + token 可计量（成本下降可归因）

**候选对照表**：

| Bench | 复用形态 | 题量 | 单题成本 | 适配度 | 状态 |
|---|---|---|---|---|---|
| **AutomationBench**（Zapier） | 同 47 app API + 模板化陷阱（过时行/重名/埋政策） | 600 公开 | 低（≤50 步） | ★★★★★ | 已冒烟 ✅（§7） |
| **tau3-bench** | 同政策手册跨 375 对话复用 | 375 | 低 | ★★★★ | Hub 现成 `sierra-research/tau3-bench` |
| **SWE-bench 系按 repo 聚类** | 同仓库工程知识（测试命令、坑） | 500~1632 | 中 | ★★★★ | Hub 现成 |
| **TMax-15K** | 模板化任务族，规模化验证 | 14601 | 低 | ★★★★ | Hub 现成 |
| **SkillsBench** | 有/无 skill 对照，方法学参照 | 87 | 低 | ★★★（题少） | Hub 现成 |
| GSO | 同 codebase 内优化模式 | 102 | 中 | ★★（题少） | 本地有 adapter |
| ProgramBench | 仅"探测二进制"元技能 | ~60 | 低 | ★★ | 已冒烟 ✅（§4） |
| Terminal-Bench | 89 题各不同，复用面窄 | 89 | 中 | ★ | 已冒烟 ✅（§2） |
| swe-marathon / FrontierSWE | 题题独特、单题太贵 | 20/17 | 极高 | ✕ | marathon 已冒烟（§3） |

**结论**：主力 = AutomationBench + tau3-bench + SWE-bench repo 聚类（三种不同复用形态）；
SkillsBench 当对照实验设计参照；TMax-15K 做规模化验证。
AutomationBench 另两个加分项：确定性最终状态断言（无 judge 噪声）、partial_credit
可当稠密信号；失败模式（找错位置/不持久/假完成）全是经验可修复型。

**实验设计要点**：

- 任务排序按族聚类，分桶看 Q1→Q4 的成功率/token 成本变化
- **必须留无 memory 对照组**（同序跑）：同 API 重复出现时 prompt 缓存命中本身就在
  降成本（实测：Terminal-Bench 跨题 98%，AutomationBench 单题内 ~100%，见 §7），
  两组之差才是 memory 净贡献
- Harbor trial 相互隔离，跨任务 memory 要自己在外层包：共享 memory 目录（--mounts）
  + 顺序跑（-n 1），对比"memory 持久 vs 每题清空"

## 7. AutomationBench（Zapier，独立 harness）

不是 Harbor 格式（registry/adapters/Hub 均无），用官方 harness：`git clone zapier/AutomationBench`。
模拟 47 个 SaaS app（本地 Pydantic 状态机），agent 用 search/execute 两工具，≤50 步/题，
确定性最终状态断言（含负向断言防霰弹枪）。公开 600 题（6 域 × 100），leaderboard 跑私有集。

```bash
set -a && source ../.env && set +a
uv run auto-bench --model deepseek-v4-flash \
  --base-url https://api.deepseek.com --api-key "$DEEPSEEK_API_KEY" \
  --num-examples 5 --max-concurrent 5
```

冒烟结果（2026-07-26，sales 域前 5 题，2 分钟跑完）：

- **0/5 通过，partial credit 30%**——每题都踩中部分断言但都有缺漏
- 典型失败：task1 改对了 deal、查对了 tier，但两封该发的邮件没发对，最终消息却是
  "✅ 全部完成"的自信总结——官方说的首要失败模式（假完成）实锤
- token：5 题共 283 万（task4 烧满 50 步 = 136 万 input）。成本 N/A（deepseek-v4-flash
  不在价格库，需 `--input-cost/--output-cost` 显式指定）
- 结果落盘 `visualizer/runs/local/*.json`，per-task 字段很全：input/output/
  **cached_input_tokens**/uncached/reasoning_tokens、模型/工具调用数、模型与工具耗时
- **重要实测：单题内缓存命中率 ~100%**（DeepSeek 自动 context caching，每步重发全量
  前缀）。实证了"成本递减"研究必须扣掉缓存基线——memory 的净贡献 = 有 memory 组
  vs 无 memory 同序组的成本差

**50 题无 memory 基线（exp001-nomem，2026-07-26，sales 域 1-50，经 cost-ledger）**：

- **pass 6/50（12%），partial credit 37%**，7m23s，1120 次模型调用（≈22 次/题）
- 真实成本（ledger 价格表，DeepSeek 官方刊例价）：**$0.32**（曾用占位价算得 $2.21，
  2026-07-26 按官网 ¥1/¥0.02/¥2 每百万 token 修正）；fresh input 仅 1.13M（5%），
  cache_read 21.16M（95%）
- 对账：ledger 与 auto-bench 导出的 cache_read **完全一致**（21,158,400），
  input 口径有差异（auto-bench 的 input 字段与其 cached 字段不满足加和关系，以 ledger 为准）。
  原因已定位：auto-bench 的 input/output 走 turn 级钩子（env_response 时取 trajectory
  最后一条 response 的 usage），会丢掉每题**最后一次**模型调用（终局后无 env_response）；
  cached 走 client 级钩子不丢。50 题共丢 1.46M prompt token（≈50 × 2.9 万），
  且丢的 ~99% 是缓存部分 → 表现为 50/50 题 cached > input。proxy 层账本无此缺陷
- 导出：`visualizer/runs/local/deepseek-v4-flash-20260726-232810-328.json`

### usage/pricing 模块设计学习（automationbench/usage.py、pricing.py）

- 采集：挂在 `verifiers` 框架的 env_response 钩子上，每轮从原始 API response 的
  `usage` 字段累加进 `state["_usage"]`——信 API 不信本地估算，即时累加不事后扫轨迹
- 定价：三级数据源（CLI 覆盖 > llm-prices.com 在线拉取+24h 缓存 > 硬编码 fallback）+
  模型名归一化（剥 provider 前缀/日期后缀 + 别名表）四段匹配
- 已知缺陷（自己实现时要补）：console 不展示缓存分层（导出 JSON 里有）、
  reasoning_tokens 疑似重复计数（completion_tokens 已含）、刊例价非账单价、
  无 per-step 时间序列（_perf 的写法可照抄扩展）

## 8. cost-ledger：自建的 token 成本账本（memory 研究基础设施）

位置 `cost-ledger/`，文档见其 README.md。LiteLLM proxy（:4000）挡在 agent 与
DeepSeek 之间，每次调用落一行 JSONL（`logs/ledger.jsonl`），缓存三档分开记
（fresh/cache_read/cache_write），美元由 `ledger/pricing.py` 在分析时换算。

```bash
cd cost-ledger && set -a && source ../.env && set +a
.venv/bin/litellm --config proxy.config.yaml --port 4000   # 起账本
.venv/bin/python -m ledger.analyze summary logs/ledger.jsonl   # per-task 汇总
.venv/bin/python -m ledger.analyze curve logs/ledger.jsonl --out report/  # 学习曲线
```

- 归属标签（run/task/episode/step）走 `X-Ledger-*` 请求头，harness 零侵入
- 各 harness 接入：auto-bench 用 `--base-url`；mini-swe-agent 系用 `OPENAI_BASE_URL` +
  模型名 `openai/deepseek-v4-flash`；harbor 容器内 agent 用 `--ae` 传 env +
  `--allow-agent-host host.docker.internal`
- 已实测：DeepSeek 缓存字段（prompt_cache_hit/miss）正确归一化，640 cached vs 9 fresh
- 踩过的坑：litellm 回调里 `proxy_server_request` 藏在 `kwargs["litellm_params"]` 下
  而非顶层（callback.py 的 `_extract_tags` 两处都找，防版本漂移）
- 原则：JSONL 只存 token 计数不存美元；实验间对比冻结同一份 PRICE_TABLE

## 9. tau3-bench（Sierra，Harbor Hub 数据集）

多轮工具使用 agent bench（tau-bench 续作）：agent 扮演客服，与 **LLM 用户模拟器**多轮
对话，通过 MCP 工具操作业务系统（airline/retail/telecom 三域，共 375 题）。
verifier 用 NL assertions + 状态断言判定。

任务结构（Harbor 化后）：

- docker-compose 两个服务：`main`（agent 容器）+ `tau3-runtime`（MCP server，
  streamable-http），agent 经 MCP 调业务工具
- agent 超时 3600s，verifier 300s；judge 默认走 OpenAI——但 `task.toml` 里
  `OPENAI_BASE_URL` / `TAU2_USER_MODEL` / `TAU2_NL_ASSERTIONS_MODEL` 都是可替换环境变量，
  指到 cost-ledger proxy 即可**全程免真实 OpenAI key**，且用户模拟器和 judge 的
  token 也全部进账本

跑法（用户模拟器与 judge 也走 proxy 记账）：

```bash
set -a && source .env && set +a
export OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://host.docker.internal:4000 \
       TAU2_USER_MODEL=deepseek-v4-flash TAU2_NL_ASSERTIONS_MODEL=deepseek-v4-flash
harbor run -d sierra-research/tau3-bench -a mini-swe-agent -m openai/deepseek-v4-flash \
  -l 3 -n 3 -y --ae OPENAI_BASE_URL=$OPENAI_BASE_URL --ae OPENAI_API_KEY=dummy \
  --allow-agent-host host.docker.internal
```

冒烟结果（2026-07-26，3 题，job `jobs/2026-07-26__23-49-38/`，20m20s）：

- **1/3 通过**：tau3-retail-12 reward=1.0；airline-33、telecom-mobile 0.0
- 账本成本（581 次调用，官方刊例价）：**≈$0.22**（占位价曾算得 $3.47）；
  fresh input 344k / cache_read 45.5M / output 107.6k（+reasoning 65k）。cache_read 占绝对大头——多轮对话每轮重发全量
  历史，缓存基线极高，正适合测"memory 净贡献"
- 分解：judge（verifier 时间窗内）仅 1 次调用 <$0.01；agent+用户模拟 580 次 ≈$0.22
- **已知归因缺陷**：agent 和用户模拟器同模型同端点，ledger 无法区分两者；
  三题并行时间重叠也无法按题拆分。正式实验要么串行跑，要么给用户模拟器
  单独一个 proxy 端口/模型别名
- 数据集已全量下载在 `/tmp/tau3-bench/`（`--allow-agent-host` 的 UserWarning 无害，
  因网络本来就是 public）

## 10. TMax-15K-Harbor（Allen AI，Harbor Hub 数据集）

Allen AI 为 RL 训练造的终端 agent 任务集（arXiv 2606.23321），**14,601 题**，Harbor
原生格式，Hub 上 `tmax/TMax-15K-Harbor`（Public）。legacy 10k 自包含题 + 5k
"intricate" 多模态题（带 video.mp4 等 fixture，C/Go/数据工程等多域）。verifier 为
程序化测试，无需 LLM judge。任务元数据带 domain/skill_type/task_complexity 标签，
适合按难度/领域切片采样。

```bash
harbor run -d tmax/TMax-15K-Harbor -a mini-swe-agent -m openai/deepseek-v4-flash \
  -l 3 -n 3 -y --ae OPENAI_BASE_URL=http://host.docker.internal:4000 --ae OPENAI_API_KEY=dummy \
  --allow-agent-host host.docker.internal
# 补跑单题：-i 'tmax/task_000996_1eadce64'（过滤词必须带 tmax/ 前缀全名，glob 不猜前缀）
```

冒烟结果（2026-07-27，3 题，job `jobs/2026-07-27__00-21-13/`）：

- `-l 3` 只下载选中的 3 题到 `~/.cache/harbor/tasks/packages/tmax/`，不拉全量 14.6k——
  大规模跑也可按需取
- **1/2 有效题通过**：task_009460（写 audit_report）reward=1.0；
  task_003301 reward=0.0
- task_000996（intricate 题：C 语言 ETL+视频处理管线）连续两次环境启动 600s 超时——
  Dockerfile FROM ubuntu:22.04 裸装 Go/Rust/ffmpeg 全套，实测构建 ~20 分钟；
  `--timeout-multiplier 3` 第三次补跑环境起来了，但 reward=0.0（能力失败，20 次调用
  $0.0034）。**教训：intricate 题首跑预留 ~20min 镜像构建时间，大规模采样前应先批量
  warm build，否则一半墙钟时间耗在建环境上**
- 三题合计：1/3 通过，58 次调用 **$0.007**。单题成本极低（cache_read 占 93%）
- 注意：这批任务 `allow_internet = true`；超时每题自定（task_000996 是 agent 600s /
  verifier 120s），跑大规模时注意任务间配置不一致

## 分数可信度提醒（BenchJack 攻击面）

Harbor 默认 agent 和 verifier 同容器共享文件系统：agent 可预写 reward 文件、劫持
PATH 里的 python3、留后台进程。held-out 测试本身安全（agent 结束后才上传）。
对策：做相对排名实验默认可用 + 抽查高分轨迹（ATIF 齐全）；对外报绝对分前用
patch-based 重放（SWE-bench 模式，Harbor 可用 `--verifier module:Class` 自定义）；
自己写任务时 test.sh 开头 `rm -rf /logs/verifier` + 校验关键二进制 hash。
