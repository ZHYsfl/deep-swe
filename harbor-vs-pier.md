# Harbor 与 Pier：关系、差异与选型指南

> 整理自 2026-07-26 对本地两个仓库（`./harbor` @ 0.20.0、`./pier` @ 0.3.0）的源码对比与实测（DeepSWE `abs-module-cache-flags` 任务端到端跑通）。

## 1. 两者是什么

- **Harbor**（[harborframework.com](https://www.harborframework.com/)）：Terminal-Bench 团队推出的 agent 评测与优化框架，随 Terminal-Bench 2.0（2025-12）发布。它定义了两层标准：
  - **任务格式**：`task.toml` / `instruction.md` / `environment/` / `tests/` / `solution/` / `pre_artifacts.sh`
  - **agent 接口**：`BaseAgent`（`setup(environment)` + `run(instruction, environment, context)` 两个抽象方法）
- **Pier**（[github.com/datacurve-ai/pier](https://github.com/datacurve-ai/pier)）：Datacurve 的 Harbor fork，为 DeepSWE 评测而生。PyPI 包名 `datacurve-pier`。

## 2. 渊源：为什么 fork

核心矛盾是**网络模型**：

- Harbor 的断网是绝对的：`allow_internet = false` = 容器没有任何出站流量，**包括 LLM API 调用和依赖安装**。
- 现代 CLI agent（claude-code、codex、mini-swe-agent 等）"连脑子带手脚"都在容器内运行，API 调用从容器内部发出——一刀切断网让它们直接瘫痪。
- Pier 的解法：**per-agent 网络白名单**。任务容器保持断网，但 agent 插件声明自己需要的域名（如 `api.deepseek.com`），harness 按插件放行。见 `pier/src/pier/agents/installed/*.py` 的 `network_allowlist()` 与 `pier/src/pier/trial/execution.py:202`。

Pier 的其他增量：ATIF 轨迹转换器（各 CLI 格式 → 统一轨迹）、`pier critique` 轨迹分析、更好的 viewer。

没有找到公开的合并讨论。合理推测（非内幕）：对 Harbor 而言，"绝对断网"是 leaderboard 公信力的基石，agent 级白名单依赖"正确解析每个 agent 的配置"，攻击面和审计成本都更高；对 Datacurve 而言，断网任务跑 CLI agent 是业务刚需，等不起上游评审。双方格式共享、实现竞争——这是标准成熟的健康形态。

## 3. 关键概念：断网断的是什么

断网断的是 agent 的**手脚**，不是**脑子**。目的：

- **防查答案**：DeepSWE 任务取自活跃开源仓库，所求功能往往已在上游后续 commit 中实现；容器联网 = 可以拉未来历史、搜 PR、找 held-out 测试。
- **可复现性**：依赖在镜像构建期钉死，运行期联网会导致环境漂移。
- **防环境污染**：评分器意料之外的安装不该发生。

类比：闭卷考试——脑子（LLM API）随便用，手机（其余网络）没收。

两种 agent 架构对断网的敏感度不同：

```
架构 A：脑子在容器外（Harbor 原生模型，如 Terminus）
  宿主机跑 LLM 循环 → 通过 docker exec/tmux 注入命令 → 容器完全断网也无妨

架构 B：脑子在容器内（CLI agent：claude-code / codex / mini-swe-agent）
  API 调用从容器内发出 → no-network 直接瘫痪 → 需要白名单放行 LLM 域名
```

## 4. 网络模型对比

| | Harbor 0.20 | Pier 0.3 |
|---|---|---|
| 策略声明方 | **任务作者**（`task.toml`） | **agent 插件**（`network_allowlist()`） |
| 模式 | `public` / `allowlist`(+`allowed_hosts`) / `no-network`，分 agent/verifier 阶段（`harbor/src/harbor/trial/network_policy.py`） | 任务断网 + 按 agent 插件放行 LLM 域名；白名单还会解析 agent 配置（codex `config_toml`、opencode `opencode_config`、mini `config_yaml`）里的 base URL |
| 跑存量 no-network 任务包 × CLI agent | 需改 task.toml 加 allowed_hosts | 开箱即用 |

## 5. Agent 生态对比

| | Harbor 0.20 | Pier 0.3 |
|---|---|---|
| 内置 agent | **~40 个**（mini_swe_agent、claude_code、codex、gemini_cli、cursor_cli、opencode + aider、goose、openhands、qwen_code、kimi_cli/kimi_code、copilot_cli、swe_agent、terminus_2 等），约 23.7k 行 | 6 个（mini_swe_agent、claude_code、codex、gemini_cli、cursor_cli、opencode），约 5.9k 行 |
| 自定义 agent | ✅ `--agent-import-path my_module:MyAgent`（`factory.py:109`） | ✅ 相同机制，接口几乎一致 |
| 执行后端 | docker / Modal / Daytona / Beam / Runloop 等 | docker / Modal 等（较少） |

注意：两边同名的 6 个插件是 fork 后各自演进（如 `claude_code.py` 相差约 1000 行 diff），功能互有取舍。Pier 插件独有的 `network_allowlist()` 是移植时的主要差异点。

## 6. 接入自己的 harness（三条路径）

1. **用现成 CLI agent**：`pier run --agent claude-code --model anthropic/...`（Harbor 同理，且选择更多）。
2. **写 agent 插件**（推荐）：继承 `BaseInstalledAgent`，实现 `name()` / `install_spec()` / `run()` / `populate_context_post_run()`，Pier 下另加 `network_allowlist()`；然后 `--agent-import-path my_agents:MyAgent`。环境生命周期、超时、收卷、评分全部复用。照 `opencode.py`（最短）抄骨架。
3. **完全自研，绕过框架**：Harbor 格式是开放契约——起容器、喂 instruction、收 commit/diff（`pre_artifacts.sh`）、构建 `tests/` verifier 镜像跑 `test.sh`、读 `reward.json`。自由度最大，但失去 leaderboard 可比性。

插件开发易踩的坑（来自源码注释）：

- `cmd | tee` 丢退出码 → 用基类 `_exec()`（自带 `set -o pipefail`）；opencode 出错时退出码仍为 0，需扫 stdout error 事件
- 命令尾部加 `</dev/null`，防 CLI 交互式等 stdin
- CLI 常不认环境变量：codex 的 base_url 只认 config.toml；opencode 的 baseURL 要嵌在 `provider.options`
- 消灭一切确认弹窗：bypass 权限 flag（`--yolo` / `bypassPermissions` / `--dangerously-*`）、预写 trust 配置（Gemini 的 `GEMINI_CLI_TRUST_WORKSPACE`、Claude 的 `IS_SANDBOX=1`、MCP 写 user 级配置）
- 安装期有网、运行期断网：运行期需要的数据（如 litellm 价目表）必须在 install 阶段预置
- Alpine/musl 上官方安装脚本的二进制可能跑不了，需退回 npm；nvm 装的二进制要软链到 `/usr/local/bin`
- 凭证在 finally 中清理（codex 的做法）

## 7. 选型决策树

```
任务是什么网络模式？
├── no-network 且没配 allowed_hosts
│   ├── agent 是容器内 CLI agent（claude-code/codex/mini-swe-agent…）
│   │   ├── 任务包不能改 → Pier ✅（唯一开箱方案）
│   │   └── 能改 task.toml 加 allowed_hosts → Harbor 也行
│   └── agent 脑子在宿主机（Terminus 式） → Harbor ✅
├── public / allowlist → Harbor ✅（agent 阵容大、后端多）
└── 是 DeepSWE → 无脑 Pier >= 0.3.0 ✅
    （v1.1 要求 separate verifier + 官方榜单口径即 Pier）
```

一句话：**DeepSWE 或任何不想改配置的存量 no-network 任务包 → Pier；自己造任务、跑联网/白名单任务、要丰富 baseline 和后端 → Harbor。**

## 8. 本地实测记录（2026-07-26）

- 环境：WSL2 Ubuntu + Docker Desktop（WSL 集成），24C/15G，Pier 0.3.0。
- 运行：`pier run -p tasks/abs-module-cache-flags --agent mini-swe-agent --model deepseek/deepseek-v4-flash --env-file .env`
- 结果：6m28s 完成，reward=0（f2p 6/20，p2p 3/3，partial 0.39），成本 $0.03。报告链路（patch 提取 → 独立 verifier → reward.json/ctrf.json）完整自洽。
