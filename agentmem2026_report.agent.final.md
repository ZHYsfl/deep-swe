# Agent Memory 与自进化 Agent：2026 前沿调研与双轴（成功率 × Token 成本）SOTA 格局

> 调研截止：2026-07-27｜覆盖 ICML 2026 / ICLR 2026 / ACL 2026 及高水平 arXiv 工作｜调研方法：8 维并行深挖 + 交叉验证 + 独立事实核验

## 第 1 章 引言与领域框架

### 1.1 领域定位：为什么 2026 年 Agent Memory 与自进化成为热点

截至 2026 年 7 月，agent memory 与自进化（self-evolving）agent 已从边缘议题跃升为 LLM agent 研究的主线之一。三股力量共同推动了这一转向。

**第一，长程任务需求暴露了无记忆架构的天花板。** 2023 年确立的 profile/memory/planning/action 四模块框架（[Wang et al.](https://arxiv.org/abs/2308.11432)）中，memory 长期只是被动的存储与检索组件；而当 agent 被部署到跨会话、跨任务的长程场景后，"每次从零开始"的模式既无法维持长期一致性，也无法应对动态环境与持续学习的要求——这正是 [From Storage to Experience](https://arxiv.org/abs/2605.06716) 综述归纳的三大演化驱动（Long-Term Consistency、Dynamic Environments、Continual Learning）。评测侧的变化尤为剧烈：MemoryArena（ICML 2026）证明在 LoCoMo 等对话式记忆基准上拿高分的系统，在互依赖多会话任务上系统性失效，"LoCoMo 分数不预测真实 agent 表现"（[arXiv:2602.16313](https://arxiv.org/abs/2602.16313)），长程、任务化的记忆能力由此成为刚需。

**第二，token 成本压力使"记忆 vs. 长上下文"变成一笔经济账。** 把全部历史塞进 context 的朴素方案在长程场景下成本不可承受：实证研究显示不同记忆系统的离线构建开销从 1.3M 到 7.04M tokens 不等，构建索引可耗时约 15 小时，单查询延迟可超过 32 秒——Anatomy 综述将这类隐性开销命名为 **Agency Tax**（[arXiv:2602.19320](https://arxiv.org/abs/2602.19320)）。成本因此不再是工程的边角料，而是与方法有效性并列的一等公民：[Toward Efficient Agents](https://arxiv.org/abs/2601.14192) 综述以 effectiveness–cost 的 Pareto 前沿统一审视 memory、tool learning 与 planning 的效率，并直言"现有各论文的效率数字因指标定义异质而不可直接比较"。

**第三，test-time learning 范式的兴起把记忆从"检索对象"变成"可优化对象"。** 2024–2025 年的记忆是静态的 store+retrieve；2026 年的主线是把 encode/store/retrieve/manage 的每一环都变成可学习策略，奖励信号直接取自下游任务成功率。标志性事件包括：Memory-R1（ACL 2026）用 RL 学习 ADD/UPDATE/DELETE/NOOP 记忆操作（[arXiv:2508.19828](https://arxiv.org/abs/2508.19828)）；Google 的 Evo-Memory 将 test-time learning 本身确立为评测对象（[arXiv:2511.20857](https://arxiv.org/abs/2511.20857)）；ReasoningBank（ICLR 2026）把成功与失败经验蒸馏为可复用的推理策略，在 WebArena 上成功率相对提升 20.5% 而 token 消耗仅增 4.3%（[arXiv:2509.25140](https://arxiv.org/abs/2509.25140)）。记忆由此成为 agent 在部署期持续变强的载体，"自进化"与"记忆"两个此前平行的子领域在 2026 年实质合流。

### 1.2 分类框架：从 Storage 到 Experience

本报告以 Luo et al. 的综述 [From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](https://arxiv.org/abs/2605.06716)（2026-05，约 248 篇文献）为分类主骨架。该框架将 LLM agent 记忆机制的演化形式化为三个阶段：

| 阶段 | 核心操作 | 本质 |
|---|---|---|
| **Storage** | trajectory preservation | 保存交互轨迹，解决"记不住" |
| **Reflection** | trajectory refinement | 对轨迹做内省、环境反馈与多 agent 协调式的精炼，解决"记不准" |
| **Experience** | trajectory abstraction | 从轨迹簇中抽象出可泛化经验，解决"不会用" |

Experience 阶段是 2026 年的研究前沿，细分为三类：

- **Explicit experience**：从轨迹簇中抽象出人类可读、可编辑、可泛化的知识模式，子类包括启发式指南（Heuristic Guidelines）与程序化原语（Procedural Primitives）——典型形态是自然语言策略文档与技能库。
- **Implicit experience**：将交互历史抽象为隐式先验，子类包括 Latent Modulation 与 Parameter Internalization（参数内化），其动机之一正是"降低推理开销与上下文约束"。
- **Hybrid experience**："Accumulate–Internalize" 动态循环——显式经验池作为高容量缓存，通过周期性的 Experience Transfer 逐步参数内化，使 agent 渐进地摆脱对外部检索的依赖，同时缓解存储爆炸与检索延迟。

该综述同时指出两个前沿机制方向（active exploration 与 cross-trajectory abstraction），并坦承 experience 阶段的抽象与泛化能力评估"significantly insufficient"——这为本报告第 6 章的评测讨论埋下伏笔。

作为互补视角，Gao et al. 的自进化 agent 综述（v4，2026-01）将演化问题分解为四问（[arXiv:2507.21046](https://arxiv.org/abs/2507.21046)）：**What** to Evolve（模型/上下文/工具/架构）、**When**（intra-test-time 与 inter-test-time）、**How**（奖励驱动/模仿/种群进化）、**Where**（通用域/专门域）。如果说 Storage→Experience 框架回答"记忆机制演化到了哪一步"，What/When/How/Where 框架则回答"agent 的哪个组件、在什么时机、用什么信号进化"。本报告将两者交叉使用：以三阶段轴组织技术内容，以四问轴定位每项工作的演化要素。

### 1.3 本报告的双轴评估视角：成功率 × token 成本

本报告通篇采用**成功率/准确率 × token/成本效率**的双轴视角评估所有方法，而非仅看准确率排行榜。理由有三。

**其一，准确率提升可能是隐形的计算转移。** Budget-matched 复评（[arXiv:2606.15017](https://arxiv.org/abs/2606.15017)）显示，在同预算条件下，无记忆的 vanilla agent 可以追平记忆带来的增益——所谓"记忆提升"，部分只是把成本从左口袋挪到右口袋：离线 ingestion 的 0.6–15 小时、数百万 token 的构建开销从未计入论文的成本声明（即前述 Agency Tax）。

**其二，正确的成本核算已有成熟工具但被普遍忽视。** Efficient Agents 工作提出的 **cost-of-pass**（v = C/R，即单次成功所需的期望成本）表明，GAIA 上可以在性能不变的前提下把单题成本从 $0.398 降到 $0.228（[arXiv:2508.02694](https://arxiv.org/abs/2508.02694)）；反之，Agent S3 在 OSWorld 上的 72.6% 成功率是靠约 10 倍 rollouts 堆出来的——单看准确率会得出完全相反的结论。

**其三，双赢窗口真实存在但有边界。** 实证上确实存在"更准且更省"的方法（去除的是干扰性冗余而非信息），但逼近信息边界后即退化为 trade-off。因此本报告对每个方法追问两个问题：增益是否在 budget-matched 基线之上？成本声明是否计入 ingestion 与维护开销？Cost-of-pass 与 Agency Tax 的概念细节将在第 6、7 章展开。

### 1.4 报告结构导览

- **第 2 章** 梳理记忆系统架构沿 Storage→Reflection→Experience 轴的演化脉络与代表性系统；
- **第 3 章** 聚焦可学习的记忆管理：RL 学记忆操作与元进化记忆架构；
- **第 4 章** 考察自进化与 test-time learning 范式下的策略蒸馏与经验复用；
- **第 5 章** 分析技能库（skill library）路线，重点是 verifier 门控对收益的决定性作用；
- **第 6 章** 转向评测基础设施：对话式记忆基准的证伪与任务化多会话新标准；
- **第 7 章** 进行成本核算与争议盘点：Agency Tax、cost-of-pass、双赢窗口边界与厂商数字口径混乱；
- **第 8 章** 汇总各综述识别的开放问题，并给出对后续研究与工程选型的建议。

### 本章引用

- Luo et al., From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms, [arXiv:2605.06716](https://arxiv.org/abs/2605.06716)
- Gao et al., A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve, [arXiv:2507.21046](https://arxiv.org/abs/2507.21046)
- Yang et al., Toward Efficient Agents: A Survey of Memory, Tool Learning, and Planning, [arXiv:2601.14192](https://arxiv.org/abs/2601.14192)
- Jiang et al., Anatomy of Agentic Memory, [arXiv:2602.19320](https://arxiv.org/abs/2602.19320)
- Wang et al., A Survey on LLM-based Autonomous Agents, [arXiv:2308.11432](https://arxiv.org/abs/2308.11432)
- MemoryArena, [arXiv:2602.16313](https://arxiv.org/abs/2602.16313)
- Memory-R1, [arXiv:2508.19828](https://arxiv.org/abs/2508.19828)
- Evo-Memory, [arXiv:2511.20857](https://arxiv.org/abs/2511.20857)
- ReasoningBank, [arXiv:2509.25140](https://arxiv.org/abs/2509.25140)
- Budget-matched 复评, [arXiv:2606.15017](https://arxiv.org/abs/2606.15017)
- Efficient Agents / cost-of-pass, [arXiv:2508.02694](https://arxiv.org/abs/2508.02694)

## 第 2 章 自进化与经验复用：2026 旗舰工作

本章聚焦 2025 年底至 2026 年 7 月间「自进化 Agent / 跨任务经验复用」方向的六项旗舰工作。venue 核实结论先行：**该方向确认发表于 ICLR 2026 主会的仅 ReasoningBank 一篇（Poster，OpenReview 页眉已核实）**，其余均为 arXiv preprint。下表先给出全景对比，所有数字均可溯源（见各节及「本章引用」）。

| 工作 | venue（核实状态） | 成功率/准确率指标 | token / 成本 / 步数指标 |
|---|---|---|---|
| ReasoningBank + MaTTS | **ICLR 2026 Poster ✅** | WebArena +8.3pp（相对 +20.5%）；SWE-Bench-Verified 57.4%（+4.6pp） | 交互步数 -16%（成功轨迹最高 -30.9%）；token 仅 +4.3% |
| ACE | preprint（arXiv:2510.04618） | +17.1% 准确率；AppWorld 平均 +10.6% | adaptation 延迟 -86.9%；token 美元成本 -83.6% |
| Memento | preprint（arXiv:2508.16153） | GAIA val 87.88% Pass@3（Top-1）/ test 79.40%；OOD +4.7~9.6pp | 冻结 LLM、零微调成本；最优检索 K=4 |
| Evo-Memory / ReMem | preprint（arXiv:2511.20857） | AlfWorld / BabyAI 成功率 0.92–0.96 | 步数效率为一等指标；统一检索预算 top-k=4 |
| MemEvolve / EvolveLab | preprint（arXiv:2512.18746） | 最高 +17.06%（GAIA/xBench-DS 等） | API 成本与推理延迟与无记忆基线基本持平 |
| SkillOS | preprint（arXiv:2605.06614） | ALFWorld 55.7→61.2（vs ReasoningBank） | 平均交互步数 -2.2~-3.1 |

### 2.1 ReasoningBank + MaTTS：策略级记忆蒸馏（ICLR 2026 旗舰）

[ReasoningBank](https://arxiv.org/abs/2509.25140)（Google Cloud AI + UIUC + Yale，**ICLR 2026 Poster，[OpenReview](https://openreview.net/pdf?id=jL7fwchScm) 已核实**）是本轮被引最广的旗舰：与「存原始轨迹」或「只存成功例」的记忆机制不同，它从 agent **自我判断的成功与失败**轨迹中双向蒸馏可泛化的推理策略条目，存入记忆库供测试时检索注入。配套机制 MaTTS（memory-aware test-time scaling）以并行/序列两种方式扩展经验生成，构造对比信号反向精炼记忆，形成「记忆 × 测试时算力」协同：并行 k=5 时 WebArena-Shopping 成功率 49.7→55.1（[交叉核实](https://arxiv.org/abs/2509.25140)）。

量化收益经三方核实：WebArena 成功率 +8.3pp（相对 +20.5%），SWE-Bench-Verified 达 57.4%（+4.6pp），Mind2Web 跨域增益最强；效率侧交互步数 -16%（成功轨迹上最高 -30.9%，导航案例 29→10 步），而 OpenReview 附录原文显示总 token 消耗仅 +4.3%——即近乎零成本换取双位数相对提升。消融证实失败经验的价值：success-only 记忆 46.5% vs 加入失败轨迹 49.7%。MaTTS 的序列扩展（sequential scaling）则让 agent 在同一任务上迭代自我纠错，把每次尝试的新经验即时回写记忆库，使「测试时多花算力」转化为「记忆质量提升」，而非单纯堆 rollout——这与 Agent S3 式靠 ~10× rollouts 堆分数的路线形成鲜明对照（见第 1 章成本争议）。

**争议（必须呈现）**：一个框架无关的独立复现（GitHub ramankrishna/reasoning-bank，2026-05）报告在 SWE-bench-lite 上「持久化的跨实例记忆库相对无重试基线无可测量提升」（[dim04 调研](https://arxiv.org/abs/2509.25140)）。这提示 ReasoningBank 的增益可能对 benchmark / 任务分布敏感——其收益依赖于可学习的失败信号与任务间迁移结构的存在，在分布内、低迁移场景下可能失效。

### 2.2 ACE：上下文即进化对象，delta 增量更新

[ACE（Agentic Context Engineering）](https://arxiv.org/pdf/2510.04618)（Stanford/SambaNova 等，2025-10，preprint）把「上下文」本身当作可进化对象：以 generator–reflector–curator 循环产生 **delta 增量条目**而非全量重写 playbook，避免上下文坍塌。其核心优势恰在成本侧：相对全量重写范式，准确率 +17.1% 的同时 adaptation 延迟 -86.9%、token 美元成本 -83.6%（[交叉核实](https://arxiv.org/pdf/2510.04618)）；论文细分为离线 AppWorld 适配延迟 -82.3%、rollouts -75.1%（vs GEPA），在线 FiNER 延迟 -91.5%、token 成本 -83.6%（vs DC）。AppWorld 上 ReAct+ACE 平均超基线 +10.6%。ACE 证明了「增量写入」是经验复用规模化的关键工程范式。

### 2.3 Memento：冻结 LLM，只学案例检索

[Memento](https://arxiv.org/pdf/2508.16153)（UCL + Huawei Noah's Ark，arXiv:2508.16153，preprint）将案例式推理（CBR）形式化为 M-MDP：**完全冻结底层 LLM**（GPT-4.1 planner + o3/o4-mini executor），仅在线训练一个案例选择 Q-function 来检索 episodic Case Bank 中的成功/失败案例。成绩：GAIA val 87.88% Pass@3（榜首）、私有 test 79.40%，DeepResearcher 66.6% F1 / 80.4% PM，SimpleQA 95.0% PM；案例记忆在 OOD 任务上贡献 +4.7~9.6pp 绝对增益。成本侧的关键发现是检索预算的非单调性：在 K∈{0,2,4,8,16,32} 扫描中 **K=4 最优**（DeepResearcher F1 64.5 / PM 78.5），更大 K 引入噪声与算力开销而收益停滞——「小而精的记忆」优于「多而杂的记忆」。冻结骨干意味着零梯度更新成本，全部能力增长来自外部记忆。

### 2.4 Evo-Memory / ReMem：把经验复用变成可评测的 MDP

[Evo-Memory](https://arxiv.org/abs/2511.20857)（UIUC + Google DeepMind，arXiv:2511.20857，preprint）将 10 个静态数据集重组为**有序任务流**，首次把 test-time learning 的评测标准化；配套框架 ReMem 把记忆循环显式建模为动作空间 {Think, Act, Refine} 的 MDP。ReMem 在 AlfWorld / BabyAI 上取得 0.92–0.96 成功率；评测体系将**步数效率**（完成任务所需步数）与成功率、序列鲁棒性并列为一等指标，检索预算统一为 top-k=4。重要发现：**小模型从自进化记忆中获益比例更大**——记忆机制部分补偿了参数规模的差距，这与 SkillsBench「带技能的小模型可追平无技能大模型」的观察互为印证，对部署成本敏感的落地场景有直接意义。

### 2.5 MemEvolve / EvolveLab：记忆架构本身的元进化

[MemEvolve](https://arxiv.org/abs/2512.18746)（arXiv:2512.18746，preprint）将进化对象从「记忆内容」提升到「记忆架构」：内层循环在固定架构下积累经验，外层循环依据任务成功率、成本与延迟反馈对记忆架构做选择、诊断与重设计。其配套的 EvolveLab 将任意记忆系统拆解为 Encode / Store / Retrieve / Manage 四模块，**统一重实现了 Voyager、ExpeL、AWM、SkillWeaver 等 12 种代表性系统**作为进化搜索空间。在 GAIA、WebWalkerQA、xBench-DS、TaskCraft 上，为 SmolAgent、Flash-Searcher 等框架带来最高 **+17.06%** 提升；进化出的架构可零样本跨任务迁移（TaskCraft→WebWalkerQA +2.0~9.09pp）并跨 LLM（GPT-5-mini / Kimi K2 / DeepSeek V3.2）复用，且 API 成本与推理时延与无记忆/人工记忆基线基本持平（[二手核实，中等置信](https://arxiv.org/abs/2512.18746)）。

### 2.6 补遗：SkillOS 与 Memory Transfer Learning

- [SkillOS](https://arxiv.org/abs/2605.06614)（Google，arXiv:2605.06614，ReasoningBank 团队后续，preprint）：冻结 executor + GRPO 训练 8B skill curator 管理 SkillRepo。ALFWorld 成功率 55.7→61.2（超最强基线 ReasoningBank），平均交互步数 -2.2/-3.0/-3.1；换 Gemini-3.1-Flash-Lite executor 时 73.1% vs ReasoningBank 66.0%，步数 15.5 vs 18.5。RL 训练的小 curator 反超 Gemini-2.5-Pro 直接策展——「学会管理技能」比「模型更大」更重要。
- [Memory Transfer Learning](https://arxiv.org/abs/2604.14004)（Kim et al.，arXiv:2604.14004，preprint）：跨域记忆迁移的系统研究。仅用 **431 条**抽象化（insight 级）记忆即达平均 0.630 Pass@3，胜过使用 **5,899 条**记忆的 AgentKB（0.613，arXiv:2507.06229）与 ReasoningBank（97 条，0.601）；同时揭示了负迁移三模式（域错配锚定、虚假验证、最佳实践误植），说明经验复用的瓶颈在记忆的抽象层级而非数量。

### 2.7 范式小结

六项旗舰共同勾勒出 2026 年经验复用的清晰演化阶梯（呼应第 1 章 Insight 1）：**存轨迹**（Memento 的 Case Bank、Evo-Memory 的 ExpRAG 基线：原始 episodic 记录直接检索复用）→ **蒸馏策略**（ReasoningBank 从成败轨迹蒸馏推理策略、ACE 以 delta 增量蒸馏上下文、MTL 证明 insight 级抽象记忆 431 条可胜 5.8k 条原始经验）→ **可学习的记忆管理**（Memento 训练检索 Q-function、SkillOS 用 RL 学 skill 策展、MemEvolve 把记忆架构本身纳入进化搜索空间）。贯穿三阶段的一致证据是：记忆带来的成功率提升（+8~+20pp 量级）与成本指标（步数 -16%~-30%、token 成本 -83.6%、token 增量仅 +4.3%）可以双赢，前提是记忆经过蒸馏与管理而非原始堆积；但 ReasoningBank 的独立阴性复现与 SkillsBench 系负结果（见第 1 章）提醒：增益高度依赖任务分布与验证门控，「有记忆」不等于「有用」。

### 本章引用

1. ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. arXiv:2509.25140；ICLR 2026 Poster（OpenReview jL7fwchScm，已核实）. https://arxiv.org/abs/2509.25140
2. Agentic Context Engineering (ACE). arXiv:2510.04618（preprint）. https://arxiv.org/pdf/2510.04618
3. Memento: Fine-tuning LLM Agents without Fine-tuning LLMs. arXiv:2508.16153（preprint）. https://arxiv.org/pdf/2508.16153
4. Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory. arXiv:2511.20857（preprint）. https://arxiv.org/abs/2511.20857
5. MemEvolve / EvolveLab. arXiv:2512.18746（preprint）. https://arxiv.org/abs/2512.18746
6. SkillOS: Learning Skill Curation for Self-Evolving Agents. arXiv:2605.06614（preprint）. https://arxiv.org/abs/2605.06614
7. Memory Transfer Learning: How Memories are Transferred Across Domains in Coding Agents. arXiv:2604.14004（preprint）. https://arxiv.org/abs/2604.14004
8. Agent KB: Leveraging Cross-domain Experience for Agentic Problem Solving. arXiv:2507.06229. https://arxiv.org/abs/2507.06229
9. ReasoningBank 独立复现（阴性结果）：GitHub ramankrishna/reasoning-bank（2026-05，二手来源，经 dim04 调研核实存在）

## 第 3 章 记忆系统架构：准确率与效率格局

本章按"架构机制 → 准确率 → token/延迟"三条线索梳理 2025–2026 年 LLM Agent 长期记忆系统的代表性设计，并在章末给出横向对比。需要预先声明：本领域大量高分数字为**厂商自报**，且 LoCoMo 等基准的 judge 模型、answer 模型与协议差异巨大，已有独立审计指出常用 LLM-judge 可接受高达 63% 的故意错误答案，**只有同协议重跑的数字可直接比较**（见交叉验证文件）。下文对每条数字标注口径性质。

### 3.1 检索式记忆的开山口径：Mem0（ECAI 2025）及其 2026 更新（Conflict Zone 1）

Mem0 采用 LLM 驱动的 ADD/UPDATE/DELETE 动态事实抽取，将对话蒸馏为原子记忆存于向量库（图变体 Mem0g 使用实体关系图）。其 2025 论文（arXiv:2504.19413，被广泛标注为 ECAI 2025）在 LoCoMo 上以 LLM-as-a-Judge（J 分）报告：Mem0 每查询仅 1,764 tokens（vs 全上下文 ~26,031，**token 成本 -90% 以上**）、总延迟 p95 1.440s（vs 全上下文 17.117s，**p95 降低 91%**），J 分 66.9%，图变体 Mem0g 达 68.44%，但全上下文上限约 73%——论文内部本质是准确率-成本 trade-off（[Mem0 论文](https://arxiv.org/abs/2504.19413)）。

2026-04，Mem0 官方发布 token-efficient 新算法（single-pass ADD-only 层次化抽取 + 语义/BM25/实体匹配多信号融合检索），厂商**自报** LoCoMo 92.5（旧 71.4）@ ~6,956 tokens、LongMemEval 94.4（旧 67.8）@ ~6,787 tokens、p50 0.88–1.09s（[Mem0 官方博客](https://mem0.ai/blog/state-of-ai-agent-memory-2026)）。**两种口径的矛盾必须显式呈现**：论文同行评审口径 66.9% 与厂商自报口径 92.5 相差逾 25 个百分点，源于 judge 配置、子集与评测协议不同，两者**严格不可比**；第三方转述口径（LongMemEval 93.4% / LoCoMo 85.0%）亦与官方数字有出入。该冲突是整个 LoCoMo 排行榜口径混乱（厂商自报 92–96 vs 论文/第三方 66.9–75）的缩影，Mem0 与 Zep 之间甚至爆发过公开争端（Zep 的 84% 被 Mem0 复测为 58.44%）。作为基线参照，该论文同时测得 A-MEM 检索 p50 0.668s、Zep 总延迟 p50 1.292s、LangMem 检索 p95 高达 59.82s，反衬出轻量检索式设计的延迟优势。

### 3.2 链接式与图结构记忆

**A-MEM**（arXiv:2502.12110，多方标注为 NeurIPS 2025）借鉴 Zettelkasten 笔记法：原子笔记携带 LLM 生成的上下文属性、关键词与标签，动态建立语义链接，新经验触发旧记忆表示演化。LoCoMo 上多跳 F1 达 **45.85**（≥2× 基线，vs MemGPT 25.52），每次记忆操作仅 **~1,200 tokens**（vs LoCoMo/MemGPT ~16,900，省 85–93%），单次操作成本 <$0.0003（[A-MEM 论文](https://arxiv.org/abs/2502.12110)）；第三方同协议复现其 overall judge 为 0.580（MAGMA 论文口径）。

**Zep/Graphiti**（arXiv:2501.13956）走**时序知识图谱**路线：Graphiti 引擎动态融合对话与结构化业务数据，双时态版本化（事实有效区间、边失效），混合检索 BM25+cosine+图遍历。DMR 基准 **94.8%** 超 MemGPT 93.4%；LongMemEval 上 Zep+gpt-4o 达 **71.2%**（vs 全上下文 60.2%，+18.5% 相对提升），平均上下文仅 **1.6K tokens vs 115K**，延迟 2.58s vs 28.9s（**-90%**）（[Zep 论文](https://arxiv.org/abs/2501.13956)）。注意其厂商 2026 口径自报 LoCoMo 94.7% 置信度低，与第三方复测（63.8–75.1%）冲突。

**MAGMA**（arXiv:2601.03236）将图结构推向四个正交图层（语义/时序/因果/实体）+ 时序推理引擎 + 策略引导遍历，LoCoMo judge 0.700 @ 3.37K tokens、查询延迟 1.47s（比次优快约 40%）；消融显示去掉因果/时序层各掉 5–6pp，证明图结构价值集中在多跳与时序场景（[MAGMA 论文](https://arxiv.org/abs/2601.03236)）。

### 3.3 效率/准确率 Pareto 最优点：PRISM

PRISM（arXiv:2605.12260，preprint）是检索侧的 training-free 框架，在图记忆上**联合优化检索与压缩**：分层束搜索（typed relation paths）、按查询意图调整图遍历代价、LLM 侧证据压缩、以及自适应意图路由（42.3% 查询走**零-LLM 路径**）。在同协议重跑（gpt-4o-mini answer+judge，LoCoMo cat 1–4）下，PRISM 达 **judge 0.831 @ ~2.0–2.2K tokens/query**，比 Mem0g（0.689）高 14.2pp、比全上下文（0.479）高 35.2pp，效率指标 0.411 judge 分/1K tokens 为全场最佳，比 full-context 少用约 13× tokens（[PRISM 论文](https://arxiv.org/abs/2605.12260)）。消融显示证据压缩是主导 token 杠杆（关闭后 2,023→4,108 tokens）。其数字为一手自评 preprint，未经第三方复现，但同协议基线齐全，是当前可定位的**效率/准确率 Pareto 最优点**。

### 3.4 压缩优先路线：SimpleMem 与 LightMem

**SimpleMem**（arXiv:2601.02553，标注 ICML 2026）以认知负荷理论（CLS）启发三阶段管线：语义无损压缩（语义密度门控→多视图索引）→ 在线语义合成去冗余 → 意图感知检索规划。LoCoMo（GPT-4.1-mini）平均 **F1 43.24，比 Mem0（34.20）高 26.4%**，每检索仅 **531 tokens**（vs ~16.9K 全上下文约 **-30×**，比 Mem0 ~980 再省约一半），总处理时间约 4× 快于 Mem0；LongMemEval（gpt-4.1）83.97%（[SimpleMem 论文](https://arxiv.org/abs/2601.02553)）。**口径警示**：其 F1 为词重叠指标，与 judge 0.831 / 自报 92.5 等 LLM-judge 口径不可直接比较，但同论文协议下对 Mem0/A-MEM 的相对优势有效。

**LightMem**（arXiv:2510.18866）将 Atkinson-Shiffrin 记忆模型工程化：感觉记忆轻量过滤→短时巩固→长时记忆 **sleep-time 离线更新**，把巩固成本从在线推理路径剥离。LongMemEval 比最强基线 +2.09–7.67%，总 token 至多省 **38×**（GPT）/21.8×（Qwen），纯在线 token 至多省 105.9×/117.1×；LoCoMo +6.10–29.29% 准确率、token 效率至多 20.92×（[LightMem 论文](https://arxiv.org/abs/2510.18866)）。同协议（SimpleMem 论文口径）其 LoCoMo F1 27.96 @ 645 tokens，准确率不及 SimpleMem 但展现了离/在线解耦的延迟优势。

### 3.5 Letta 谱系：睡眠时计算与版本化记忆

Letta/MemGPT 谱系沿两条线演进。其一，**Sleep-time Compute**（arXiv:2504.13171）在查询到达前用后台计算把原始上下文预加工为"learned context"，与 LightMem 的 sleep-time 巩固同属"离线巩固、在线轻快"范式，开辟 test-time scaling 新轴（[Sleep-time Compute](https://arxiv.org/abs/2504.13171)）。其二，**Context Repositories**（Letta Blog，2026-02）将记忆实现为 Git 版本化文件系统（Markdown + commit/branch/worktree 合并冲突解决），面向 coding agents（Letta Code），属工程范式创新但无标准 benchmark 论文（[Letta Research](https://www.letta.com/research/)）。第三方口径 Letta LongMemEval 约 83.2%、LoCoMo 74–83% 随底座模型波动，置信度 medium-low。谱系源头 MemGPT（arXiv:2310.08560）以 OS 虚拟内存类比管理主上下文与归档存储，但其 token 开销最重（LoCoMo 上 ~16,977 tokens/查询），正是后续所有效率优化路线的对照起点。

### 3.6 其他高分系统（简提）

- **EverMemOS**（arXiv:2601.02163）：engram 生命周期三阶段（情景痕迹→语义巩固→重构式回忆），**自报** LoCoMo **93.05%**、LongMemEval 83.0% @ ~2,800 tokens；第三方复现口径 94.5%（judge 偏宽松）（[EverMemOS 论文](https://arxiv.org/pdf/2601.02163)）。
- **MemMachine**（arXiv:2604.04853）：反 Mem0 路线，存原始 episode + 句子级索引保真，LLM 仅做高层抽象；**自报** LoCoMo **91.69**（gpt-4.1-mini）、LongMemEval-S 93.0%、token 用量约 **-80%** vs 竞品（[MemMachine 论文](https://arxiv.org/html/2604.04853v1)）。
- **Nemori**（arXiv:2508.03341）：事件切分 + 预测-校准（只记"惊喜"信息）；MAGMA 同协议 LoCoMo judge 0.590，SmartSearch harness LongMemEval-S 74.6% @ ~4,300 tokens（[Nemori 论文](https://arxiv.org/abs/2508.03341)）。
- **Memori**（arXiv:2603.19935）：语义三元组 + 会话摘要，LoCoMo 81.95% @ **1,294 tokens**（厂商自报）（[Memori 论文](https://arxiv.org/pdf/2603.19935)）。

### 3.7 横向对比表（协议混杂，仅供定位）

| 系统 | 架构 | Benchmark 成绩 | Tokens/查询 | 延迟 | 置信度/口径备注 |
|---|---|---|---|---|---|
| PRISM | 图记忆 + 检索-压缩联合优化 + 零-LLM 路由 | LoCoMo judge **0.831**（同协议） | ~2.0–2.2K | — | preprint 自评，同协议基线齐全，Pareto 最优 |
| Mem0（2025 论文） | 动态事实抽取 + 向量库（Mem0g 加图） | LoCoMo J 66.9% / Mem0g 68.44% | 1,764 | p95 1.44s（-91%） | 论文口径，token -90%+，高置信 |
| Mem0（2026 算法） | single-pass 抽取 + 多信号融合检索 | LoCoMo 92.5 / LongMemEval 94.4 | ~7K | p50 0.88–1.09s | **厂商自报**，与论文口径不可比 |
| A-MEM | Zettelkasten 链接笔记 + 记忆演化 | 多跳 F1 45.85 | ~1,200/操作 | 5.4s/操作 | 论文一手；第三方 judge 0.580 |
| Zep/Graphiti | 时序知识图谱（双时态版本化） | DMR 94.8%；LongMemEval 71.2% | 1.6K | 2.58s（-90%） | 论文一手高置信；厂商 LoCoMo 94.7% 低置信 |
| MAGMA | 四图层图 + 策略遍历 | LoCoMo judge 0.700 | 3.37K | 1.47s | preprint 一手 |
| SimpleMem | CLS 三阶段语义无损压缩 | LoCoMo **F1 43.24**（+26.4% vs Mem0） | **531**（-30×） | ~4× 快于 Mem0 | F1 口径，勿与 judge 直比 |
| LightMem | 三阶段 + sleep-time 离线巩固 | LongMemEval +2.09–7.67% | 至多 -38× | runtime 至多 12.4× | 论文一手 |
| Letta 谱系 | sleep-time compute；Git 版本化记忆库 | 无统一 benchmark | — | — | 范式创新，缺标准评测 |
| EverMemOS | engram 生命周期 OS | LoCoMo 93.05%（**自报**） | ~2,800（LME） | — | 自报 SOTA，medium-high |
| MemMachine | 保真存储 + 句子索引 | LoCoMo 91.69（**自报**） | ~-80% | — | 自报，有复现脚本 |
| Nemori | 事件切分 + 预测-校准 | LoCoMo judge 0.590；LME-S 74.6% | ~4,300 | — | 两个独立论文复现一致 |
| Memori | 语义三元组 + SQL | LoCoMo 81.95%（**自报**） | 1,294 | — | 厂商自报 |
| Full-context（参考） | 无记忆 | LoCoMo judge 0.479–0.481 / J ~73% | ~26K | p95 17.1s | 多篇一致 |

### 3.8 格局小结

2026 年中的格局呈现清晰分化：**检索侧压缩**（PRISM 0.411 judge/1K tok、SimpleMem 531 tok、Memori 1,294 tok）代表"同准确率下压 token"；**Mem0 2026**（92.5 @ 7K tok，自报）代表"容忍中等 token 换最高准确率"；**图结构**（MAGMA、Zep）价值集中于多跳与时序查询；**睡眠时巩固**（LightMem、Letta）把成本移出在线路径。同时，排行榜上 90+ 自报数字与 67–75 论文/第三方口径之间的裂缝（Conflict Zone 1），要求任何系统选型都必须以同协议复测为准。

### 本章引用

- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)
- [Mem0 Blog: State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [A-MEM: Agentic Memory for LLM Agents (arXiv:2502.12110)](https://arxiv.org/abs/2502.12110)
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
- [PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory (arXiv:2605.12260)](https://arxiv.org/abs/2605.12260)
- [SimpleMem: Efficient Lifelong Memory for LLM Agents (arXiv:2601.02553)](https://arxiv.org/abs/2601.02553)
- [LightMem: Lightweight and Efficient Memory-Augmented Generation (arXiv:2510.18866)](https://arxiv.org/abs/2510.18866)
- [Sleep-time Compute (arXiv:2504.13171)](https://arxiv.org/abs/2504.13171)
- [Letta Research: Context Repositories](https://www.letta.com/research/)
- [MAGMA: Multi-Graph Agentic Memory (arXiv:2601.03236)](https://arxiv.org/abs/2601.03236)
- [EverMemOS (arXiv:2601.02163)](https://arxiv.org/pdf/2601.02163)
- [MemMachine (arXiv:2604.04853)](https://arxiv.org/html/2604.04853v1)
- [Nemori (arXiv:2508.03341)](https://arxiv.org/abs/2508.03341)
- [Memori (arXiv:2603.19935)](https://arxiv.org/pdf/2603.19935)

## 第 4 章 可学习的记忆管理：RL 与效用学习

### 4.1 从「检索对象」到「可优化对象」

2024–2025 年的 agent 记忆是被动的 store+retrieve：写入规则、检索策略皆为人工启发式。2025 年下半年起，主线发生范式转移——记忆的 ADD/UPDATE/DELETE 操作、压缩策略、检索选择、效用评估被逐一变成可学习策略，奖励信号直接来自下游任务成功率。本章覆盖三类方法：RL 学记忆操作（Memory-R1、Mem-α、MemPO、AgeMem）、RL 学压缩式工作记忆（MemAgent、MEM1、MemSearcher）、以及学到的记忆效用（MemRL、AEL），并在 MDP/bandit 形式化脉络与 2026 年奖励稀疏解法趋势中收束。核心发现（与 Insight 1 一致）：**任务成功率与 token 成本不是两个优化目标，而是同一个结果奖励的两个分量**——当奖励是「基于压缩上下文答对问题」时，RL 自动学会「保留什么」。

### 4.2 RL 学记忆操作：Memory-R1 家族

**Memory-R1**（arXiv:2508.19828 → ACL 2026 Long，Anthology 已核实 ✅）是这一路线的旗舰 [Memory-R1](https://aclanthology.org/2026.acl-long.583/)。它用 PPO/GRPO 训练一个 Memory Manager，对每条候选记忆输出结构化操作 {ADD, UPDATE, DELETE, NOOP}，Answer Agent 侧配套 Memory Distillation——先检索至多 60 条候选再蒸馏过滤出相关条目。**仅用 152 个 QA 对训练**即在 LoCoMo、MSC、LongMemEval 上取得 SOTA，并泛化到 3B–14B 模型。相对 Mem0（LLaMA-3.1-8B）的改进幅度存在版本差异：v3 报告 F1 相对 +48%、BLEU-1 +69%、LLM-Judge +37%，v5 修订为 +28%/+34%/+30% [arXiv:2508.19828](https://arxiv.org/abs/2508.19828)。**必须指出的缺口：Memory-R1 未报告任何显式 token 成本指标**，其效率收益仅由 Memory Distillation 的「60 → 少数条目」间接暗示；ADD/UPDATE/DELETE 路线的 token 效率声明截至 2026 年 7 月基本停留在间接证据层面。

**Mem-α**（arXiv:2509.25911，ICLR 2026 在审 ⚠️）把记忆构造形式化为序列决策，GRPO 奖励显式包含四项：正确性 r1、工具调用 r2、**压缩项 r3**、内容质量 r4，其中 r1 与 r3 基于最终记忆状态全局计算 [Mem-α](https://arxiv.org/abs/2509.25911)。它是少数**显式以 token 报告记忆体积**的工作：7 个数据集平均性能 0.642、平均记忆 7.9K token，优于 full-context（0.588 @ 10.8K）与 RAG-Top2（0.567 @ 11.3K），并在 30K 训练长度外推至 400K（13×）。其 β 消融给出**双赢窗口的边界证据**：压缩项权重 β 加到 0.4 时性能从 0.642 崩到 0.509——压缩收益可调和真实存在，但越界即 trade-off。同一表中 MemAgent/MEM1 移植版本记忆 <1K token 但性能仅 0.111–0.236，说明极端压缩本身摧毁准确率，RL 赢在「学保留什么」而非「留得更少」。

**MemPO**（Findings of ACL 2026，Anthology 已核实 ✅）提供 RL 侧**最强的成功率+token 双赢证据** [MemPO](https://aclanthology.org/2026.findings-acl.1166.pdf)。其在 GRPO 中加入 memory-level advantage：稠密奖励定义为给定 `<mem>` 块时答案的条件概率 P(answer‖mem)，直接为每个记忆 token 的信息含量定价。结果：F1 绝对 +25.98（vs 基座）、+7.1（vs 前 SOTA），**同时 token 用量 −67.58%/−73.12%**；10 目标时解题 token 约为 ReSearch 的 1/3、峰值 1/5 [arXiv:2603.00680](https://arxiv.org/abs/2603.00680)。

### 4.3 RL 学压缩式工作记忆：恒定上下文的架构约束

第二条路线用有界工作记忆作为架构约束，RL 只需优化「内容」：**MemAgent**（arXiv:2507.02259，ICLR 2026 已核实 ✅；"Oral" 仅二手来源 ⚠️）将 DAPO 扩展为 multi-conversation 独立上下文 RL，学习分段阅读时的记忆覆写策略，8K 上下文训练（32K 文本）外推到 3.5M token QA 性能损失 <5%，512K RULER 达 95%+ [MemAgent](https://arxiv.org/abs/2507.02259)。**MEM1**（arXiv:2506.15841，venue 未确认 ⚠️）用 PPO 把记忆+推理整合进单一内部状态 `<IS>` 替换历史：MEM1-7B 在 16 目标多跳 QA 上性能 3.5×、记忆用量仅 1/3.7（vs Qwen2.5-14B），峰值 token 仅 27.1%、推理时间 29.3%，且奖励并未显式含记忆项——恒定记忆是策略学到的副产品 [MEM1](https://arxiv.org/abs/2506.15841)。**MemSearcher**（arXiv:2511.02805）以 multi-context GRPO 把轨迹级 advantage 传播到每一轮，迭代覆写 ≤1,024 token 的有界记忆：EM 超过 Search-R1/ReSearch/R1-Searcher 等 ReAct 式 RL 基线，同时多轮交互中上下文稳定在 **<4K token**，每轮 FLOPs O(1) [MemSearcher](https://arxiv.org/abs/2511.02805)。

### 4.4 学到的记忆效用与形式化脉络

**MemRL**（arXiv:2601.03192，v2 页眉暗示 SIGIR 2026 ⚠️）冻结 LLM，以非参数 RL 在 Intent–Experience–Utility 三元组上学习 episodic utility：Q 值经环境反馈做 Bellman 式 EMA 更新，检索分两阶段（相似度召回 → Q 值重排），在 HLE、BigCodeBench、ALFWorld、Lifelong Agent Bench 上超过 RAG 式基线，且学到的 utility 与任务成功率显著相关 [MemRL](https://arxiv.org/abs/2601.03192)。价值感知重排滤掉「形似」干扰记忆，直接减少注入 prompt 的浪费 token；Q 更新全程在 CPU 上运行，成本可忽略。形式化上，MemRL 把冻结 LLM 与外部记忆的交互建模为 **M-MDP**；ReMem/Evo-Memory 将 {Think, Act, Refine} 循环写成显式 MDP 并把步效率列为独立指标 [Evo-Memory](https://arxiv.org/abs/2511.20857)；**AEL** 则用 **Thompson Sampling bandit** 逐 episode 学习「该用哪种检索策略」，消融呈「less is more」：记忆+反思带来 +58% 累积提升，额外机制（含 LinUCB planner bandit）反而全部有害 [AEL](https://arxiv.org/abs/2604.21725)。

### 4.5 2026 趋势：攻克奖励稀疏

记忆操作的奖励天然稀疏——收益在整条轨迹末端才兑现。2026 年出现四种解法：**step-wise GRPO**（AgeMem，三阶段渐进 RL，把 store/retrieve/update/summarize/discard 暴露为工具动作 [AgeMem](https://arxiv.org/abs/2601.01885)）；**multi-context GRPO**（MemSearcher，轨迹 advantage 逐轮传播）；**稠密归因奖励**（MemBuilder，合成 session 级问题产生中间奖励 + 贡献感知梯度加权，4B 模型超闭源 SOTA [MemBuilder](https://arxiv.org/abs/2601.05488)）；**provenance 信用分配**（ECHO，每轮压缩为来源索引记忆记录，结果 credit 沿索引路由——BrowseComp-Plus 43.4% vs GRPO 28.9% vs SUPO 36.1%，且轮数与轨迹量更低 [ECHO](https://arxiv.org/abs/2606.31650)）。ECHO 尤其说明：若压缩破坏来源归因，结果 RL 会错配信用；保留索引才能让有界上下文与信用分配共存。

### 4.6 对比总表

| 方法 | Venue（置信度） | 学习机制 | 成功率头部数字 | Token/成本头部数字 |
|---|---|---|---|---|
| Memory-R1 | ACL 2026 Long ✅ | PPO/GRPO 学 ADD/UPDATE/DELETE/NOOP + Distillation | LoCoMo 相对 +48% F1/+69% BLEU-1/+37% Judge（v3 口径，v5 下修） | **未报告**（缺口） |
| MemPO | Findings of ACL 2026 ✅ | GRPO + memory-level advantage，稠密 P(answer‖mem) | F1 +25.98 / +7.1 vs SOTA | **−67.6%/−73.1% token** |
| MemAgent | ICLR 2026 ✅ | multi-conv RL 记忆覆写 | 8K→3.5M 外推 <5% 损失 | 线性复杂度分段读 |
| MEM1 | arXiv ⚠️ | PPO，`<IS>` 恒定内部状态 | 3.5× vs Qwen2.5-14B | 峰值 token 27.1% |
| MemSearcher | arXiv | multi-context GRPO | EM 超 ReAct 式 RL 基线 | 上下文恒定 <4K token |
| Mem-α | arXiv（ICLR 在审）⚠️ | GRPO，reward 显式含 compression 项 | 0.642 vs 0.588（full-ctx） | 7.9K vs 10.8K token；β 过大崩至 0.509 |
| MemRL | arXiv（SIGIR 暗示）⚠️ | Q 值 Bellman 更新的 episodic utility | utility–成功率强相关 | 两阶段检索省 token；Q 更新 CPU 级成本 |
| ECHO | arXiv | provenance 信用分配 | BrowseComp-Plus 43.4% vs GRPO 28.9% | 轮数/轨迹量低于 SUPO |

### 4.7 小结

本章证据汇聚成一句话：**当结果奖励作用于压缩后的上下文时，RL 自动学会「保留什么」，成功率与 token 是同一奖励的两个分量**。MemPO 的 −67~73% token 与 +25.98 F1、Mem-α 的 7.9K vs 10.8K 且更准、MEM1/MemSearcher 的恒定上下文与反超基线，均为双赢窗口内的证据；Mem-α 的 β 消融（0.642→0.509）与 MemAgent/MEM1 极端压缩崩塌（0.111–0.236）则划出窗口边界。遗留缺口同样清晰：旗舰 Memory-R1 未报 token 成本，venue 标注中 MemRL/MEM1/Mem-α 的最终接收状态未证实，且各工作基准碎片化，跨方法比较仅在被互评处（如 Mem-α Table 1）成立。

### 本章引用

- Memory-R1: https://aclanthology.org/2026.acl-long.583/ ; https://arxiv.org/abs/2508.19828 （ACL 2026 Long ✅）
- MemPO: https://aclanthology.org/2026.findings-acl.1166.pdf ; https://arxiv.org/abs/2603.00680 （Findings of ACL 2026 ✅）
- MemAgent: https://arxiv.org/abs/2507.02259 （ICLR 2026 ✅，OpenReview k5nIOvYGCL；"Oral" 未证实 ⚠️）
- MEM1: https://arxiv.org/abs/2506.15841 （arXiv；ICLR 2026 接收未确认 ⚠️）
- Mem-α: https://arxiv.org/abs/2509.25911 （arXiv；ICLR 2026 在审 ⚠️）
- MemRL: https://arxiv.org/abs/2601.03192 （arXiv；SIGIR 2026 为 v2 页眉暗示 ⚠️）
- MemSearcher: https://arxiv.org/abs/2511.02805 （arXiv）
- AgeMem: https://arxiv.org/abs/2601.01885 （arXiv）
- MemBuilder: https://arxiv.org/abs/2601.05488 （arXiv）
- ECHO: https://arxiv.org/abs/2606.31650 （arXiv）
- Evo-Memory/ReMem: https://arxiv.org/abs/2511.20857 （arXiv）
- AEL: https://arxiv.org/abs/2604.21725 （arXiv）

## 第 5 章 技能库与 Procedural Memory：经验复用的收益条件

### 5.1 从"存轨迹"到"可管理的技能库"

Procedural memory（程序性记忆）存储"如何做"的知识——技能、工作流、标准作业程序——与前述章节的情景记忆（记住了什么）和语义记忆（知道了什么）构成互补。2025–2026 年，该方向完成了从 Voyager 式 append-only 技能库到"可学习、可门控、可路由"基础设施的跃迁：技能被外化为结构化文档（SKILL.md 式 markdown 或可执行例程），其生命周期收敛为"归纳（induction）→ 检索/路由（reuse）→ 验证驱动精炼（refinement）"三段式。技能库的收益逻辑是双重的：既提升成功率，又通过"复用替代重新探索"压缩推理步数与 token——蒸馏后的策略文档比原始轨迹小一至两个数量级，且不绑定特定模型的参数。但本章的核心结论是：**这一双重收益不是默认成立的，而是严格以"门控质量"为条件的**（Insight 4）。以下分别给出正面证据、反面证据与收益条件小结。

### 5.2 正面证据：有门控的经验复用同时提升成功率与效率

下表汇总 2025–2026 年代表性工作的成功率与成本/效率数据（除注明外均为 arXiv preprint、作者自报）：

| 工作（venue） | 成功率收益 | token/步数/成本 |
|---|---|---|
| [CODESKILL](https://arxiv.org/abs/2605.25430)（arXiv 2605.25430） | SWE-bench Verified 57.3%→66.0%（Qwen3.5-35B-A3B，150 题 held-out）；GPT-5.4-mini 下游 46.7%→56.0% | 已解实例平均推理步数 44.1→35.2（**-20%**），为所有方法最低 |
| [SkillOpt](https://arxiv.org/abs/2605.23904)（Microsoft，arXiv 2605.23904） | 6 benchmark × 7 模型 × 3 harness 共 **52/52 格 best/tied**；GPT-5.5 平均 +23.5pp（direct chat）/+24.8pp（Codex）/+19.1pp（Claude Code） | 部署技能仅 **300–2,000 token**，推理期零额外优化器调用 |
| [Trace2Skill](https://arxiv.org/abs/2603.25158)（arXiv 2603.25158） | 跨模型迁移最高 **+57.65pp**（Qwen3.5-35B 轨迹蒸馏的技能提升 122B agent，WikiTableQuestions）；优于 ReasoningBank **+13.8pp**（SpreadsheetBench 同模型 Vrf） | 技能即 SoP 文档，免检索、免参数更新 |
| [SkillNet](https://arxiv.org/abs/2603.04448)（arXiv 2603.04448） | ALFWorld/WebShop/ScienceWorld 平均 reward **+40%** | 执行步数 **-30%**；底层为 20 万+ 技能仓库 |
| [SkillRouter](https://arxiv.org/abs/2603.22455)（arXiv 2603.22455） | 8 万候选池上 **74.0% Hit@1**，路由增益端到端迁移到 4 个 coding agent | 1.2B 路由器比最强基线少 13× 参数、快 5.8× |
| [SWE-Exp](https://arxiv.org/abs/2507.23361)（arXiv 2507.23361） | **1 条精选经验 +4.2pp**（DeepSeek-V3，37.8%→42.0%，峰值）；Verified 73.0%（Claude 4 Sonnet，自报） | 未直接报 token；经验引导使 MCTS 定向化，~300 条后收益饱和 |
| [Agent KB](https://arxiv.org/abs/2507.06229)（ICML 2025 CFAgentic Workshop，arXiv 2507.06229） | 跨 framework 复用：OpenHands+o3-mini 在 SWE-bench Lite 23%→31.67%（**+8.67pp**） | KB 规模消融：full KB 优于 KB-500/KB-100 |
| [ExpeRepair](https://arxiv.org/abs/2506.10484)（**FSE 2026**，PACMSE） | SWE-bench Verified **74.6%**、Lite 60.3%（Claude 4 Sonnet）；去记忆模块消融 **-6.8pp**（Verified）/-6.0pp（Lite） | **$1.91/题**（Verified），约为对比方法 DARS（$12.24/题）的 1/6 |
| [Subtask Memory](https://arxiv.org/abs/2602.21611)（arXiv 2602.21611） | 子任务级记忆平均 **+4.7pp**（4 backbone）；**Hard 任务 +8.7pp**、Easy 仅 +1.8pp | 预算中性设计（记忆抽取计入步数上限） |
| [SWE-ContextBench](https://arxiv.org/abs/2602.08316)（arXiv 2602.08316） | oracle 摘要复用 26.3%→34.3%（**+8pp**） | 同时降本：$0.79→$0.77/题、381.95s→356.95s——唯一"更准且更省"的配置 |

三点横向观察。其一，效率收益与成功率收益同向出现：CODESKILL 步数 -20%、SkillNet 步数 -30%、SWE-ContextBench 时间/成本双降，印证"复用替代探索"的机制——agent 不必在已解决的子问题上重复试错，省下的探索轮次直接转化为 token 与美元的节省。其二，收益高度依赖经验表示：SWE-ContextBench 中 oracle **摘要**经验有效，而 oracle **全轨迹**经验仅 27.3%（几乎无成本收益）；Subtask Memory 的粒度细分（子任务级 vs instance 级）同样是收益来源——表示越贴近"可迁移策略"、越远离"原始轨迹"，双赢概率越高。其三，作者能力是隐性变量：Trace2Skill 中 35B 模型自创的技能反而损害 35B 自身（-6.2pp），技能质量上限受蒸馏者能力约束，弱模型难以凭空写出对自己有用的技能。此外需注意，表中 SWE-Exp 73.0%、ExpeRepair 74.6%、CODESKILL 66.0% 均为作者自报或子集评测，未见官方榜单独立复现，解读时应按"论文自报"口径打折。

### 5.3 反面与边界证据：无门控复用平均无效甚至有害

与上述正面结果同等重要的是以下负面证据，它们界定了经验复用的三类失效模式——来源无效（自生成/噪声）、规模失效（检索崩塌）与场景失配（域外增益稀薄）。在"技能热"的 2026 年，这些边界证据是评估任何新复用方案时必须先行核对的对照组：

- **自生成技能无效**。[SkillsBench](https://arxiv.org/abs/2602.12670)（arXiv 2602.12670，7,308 条轨迹、86 任务配对评测）显示：人工策展技能平均 **+16.2pp**（配置级 +13.6~+23.3pp；v4 口径修订为 +16.6pp，33.9%→50.5%，结论方向不变），而 **LLM 自生成技能平均 -1.3pp**（无收益）；84 个任务中 16 个为负增益。数量上**并非越多越好**：2–3 个聚焦技能最优（+18.6pp），大而全的 comprehensive 捆绑反而 -2.9pp。值得注意的是，该基准同时发现"带技能的小模型可追平无技能的大模型"，说明技能在策展合格时的杠杆价值真实存在——问题不在技能范式本身，而在技能来源与注入方式。
- **真实 SWE 场景增益稀薄且 token 开销失控**。[SWE-Skills-Bench](https://arxiv.org/abs/2603.15401)（arXiv 2603.15401）评测 49 个公开 SWE 技能：**39 个 pass rate 零提升，平均仅 +1.2%**；token 开销从小幅节省到 **+451%** 不等（pass rate 不变）；仅 7 个专门化技能有实质增益（最高 +30%），3 个因版本错配而降效（最低 -10%）。
- **噪声记忆直接降性能**。[CTIM-Rover](https://arxiv.org/abs/2505.23422)（REALM'25 workshop，arXiv 2505.23422）把 ExpeL 式 episodic memory 直接搬到 SWE 域：**40% vs 无记忆 baseline 42%**（45 题子集，小样本需谨慎），噪声记忆项误导代码定位，检索到的"相似经验"实为干扰项。同类证据还有 Subtask Memory 的复现实验：instance 级记忆（ReasoningBank 式）在 Claude 3.7/4.0 Sonnet 上呈**负收益**（63.5%→63.3%）——粗粒度记忆对强模型有害。
- **检索质量决定成败**。SWE-ContextBench 的 free（自主检索）复用配置 22.2% @ $0.98/题，比无经验 baseline **更差且更贵**（26.3% @ $0.79）；SkillRouter 进一步指出规模瓶颈在路由而非生成：在 ~80K 技能注册表上只做 metadata-only 的 progressive disclosure，路由精度**崩掉 37–44pp**——缺失信号在技能正文（body），body-aware 的 retrieve-and-rerank 才恢复到 74.0% Hit@1。

### 5.4 收益条件小结：四条设计约束

综合正反面证据（Insight 4），经验/技能复用要同时获得成功率与 token 效率收益，需满足四个条件：

1. **Verifier 门控**。技能入库须经可验证信号把关：CODESKILL 用冻结下游 agent 的可验证执行反馈做 RL 奖励，SkillOpt 用 held-out 验证门 + 拒绝编辑缓存，ExpeRepair 的消融（-6.8pp）与 SkillsBench 的自生成负结果（-1.3pp）从两侧证明：无门控的自生成技能平均无效甚至有害。门控的本质是把"下游任务成功率"作为入库奖励信号，使记忆管理本身成为可学习策略。
2. **少而精**。SkillsBench 的 2–3 个技能最优、SWE-Exp 的 1 条经验达峰（4 条反降至 39.6%，论文称之为"过度引导带来认知负担"）、ExpeRepair 的 top-k=3 最优与 REMOVE 操作的关键性，一致指向"无差别累积 = 上下文污染 = 更贵更差"。技能库维护必须是 ADD/EDIT/REMOVE 的闭环，而非单调追加。
3. **Body-aware 路由**。技能库规模化后，检索/路由取代生成成为瓶颈；仅暴露名称+描述的 progressive disclosure 在 80K 规模下损失 37–44pp 路由精度，必须让路由器读到技能正文（body）中的可执行细节，路由错误本身即是负收益来源。
4. **Progressive disclosure 与 debloating 配套**。对 agent 侧的注入面应保持小而分层：SkillOpt 部署技能 300–2,000 token 且推理零额外调用，与 SWE-Skills-Bench 的 +451% token 反例构成对照——路由阶段读正文、执行阶段只注入命中的精炼技能，token 效率是设计出来的，不是复用的自然副产品。

### 5.5 研究空白：SWE-bench Multimodal

多轮检索确认：所有经验复用/技能库工作的 SWE 系评测均集中在 Verified/Lite 文本 split，**尚无任何工作在 SWE-bench Multimodal（含截图等视觉输入）上评测记忆或经验复用机制**。考虑到真实 issue 常附 UI 截图、报错弹窗与渲染差异图，多模态场景把本章的四条收益条件全部重新打开：视觉经验如何蒸馏为可检索的技能表示？截图与文本混合的轨迹如何做 verifier 门控？跨模态路由是否同样存在 metadata-only 的精度崩塌？这些都是经验复用研究的明确空白方向，也是新工作的差异化切入点。

---

### 本章引用

- CODESKILL: Learning Self-Evolving Skills for Coding Agents, arXiv:2605.25430, 2026-05. https://arxiv.org/abs/2605.25430
- SkillOpt: Executive Strategy for Self-Evolving Agent Skills (Microsoft), arXiv:2605.23904, 2026-05. https://arxiv.org/abs/2605.23904
- Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills, arXiv:2603.25158, 2026-03. https://arxiv.org/abs/2603.25158
- SkillNet: Towards Open Skill Infrastructure for LLM Agents, arXiv:2603.04448, 2026-02. https://arxiv.org/abs/2603.04448
- SkillRouter: Skill Routing at Scale, arXiv:2603.22455, 2026-03. https://arxiv.org/abs/2603.22455
- SWE-Exp: Experience-Driven Software Issue Resolution, arXiv:2507.23361, 2025-07. https://arxiv.org/abs/2507.23361
- Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving, arXiv:2507.06229, ICML 2025 CFAgentic Workshop. https://arxiv.org/abs/2507.06229
- ExpeRepair: Dual-Memory Enhanced LLM-Based Repository-Level Program Repair, arXiv:2506.10484, FSE 2026 (PACMSE). https://arxiv.org/abs/2506.10484
- Structurally Aligned Subtask-Level Memory for SWE Agents, arXiv:2602.21611, 2026-02. https://arxiv.org/abs/2602.21611
- SWE-ContextBench: Benchmarking Context Reuse for Software Engineering Agents, arXiv:2602.08316, 2026-02. https://arxiv.org/abs/2602.08316
- SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks, arXiv:2602.12670, 2026-02. https://arxiv.org/abs/2602.12670
- SWE-Skills-Bench, arXiv:2603.15401, 2026-03. https://arxiv.org/abs/2603.15401
- CTIM-Rover, arXiv:2505.23422, REALM'25 Workshop, 2025-05. https://arxiv.org/abs/2505.23422

## 第 6 章 Token 成本效率专题：压缩、缓存与双赢窗口

Agent 的账单由 token 驱动：长程任务的交互历史随轮数线性膨胀，token 成本、延迟与上下文窗口压力随之失控。2025–2026 年的文献给出了一条比预期乐观的主线——在相当宽的区间内，压缩上下文不仅省钱，还能同时提升准确率。ACON、SWE-Pruner、Complexity Trap 等工作在 SWE-bench、AppWorld 等严肃基准上反复观察到「token 降、准确率持平甚至反升」的双赢现象；与此同时，prompt caching 与 KV 调度从系统层提供了不改语义的成本削减，cost-of-pass 与 Pareto 前沿则给出了严格的核算框架。本章按语义压缩、系统缓存、成本方法论三层梳理证据，并回答一个核心问题：双赢窗口为什么存在、边界在哪里。

### 6.1 Agent 上下文压缩：从「省 token」到「省 token 且更准」

**ACON**（KAIST + Microsoft，arXiv preprint）是生成式压缩的代表：对历史与观察做自然语言压缩，并用对比反馈（contrastive feedback）优化压缩准则而无需更新权重，峰值 token 削减 26–54%；AppWorld 上 gpt-4.1 准确率 56.5% vs 不压缩 56.0%，微升 0.5pp 的同时峰值 token 从 9.93k 降至 7.33k——典型的双赢实例。更引人注目的是「compression as equalizer」效应：小模型获益远大于大模型，Qwen3-14B 相对提升 32.4%（25.6%→33.9%），系列实验最高 +46%；将压缩器蒸馏到小模型可保留 >95% 教师性能，而压缩推理成本降低 99.1%（$0.0004 级） [ACON](https://arxiv.org/abs/2510.00615)。需要注意 ACON 自报了压缩器额外调用带来的延迟开销——token 省不等于端到端时间省。

**AgentOCR**（NTU + Alibaba，**ACL 2026 Long, Oral**）代表 agentic self-compression 路线：把交互历史渲染为图像（光学压缩）并以段缓存复用，再用 GRPO 强化学习让 agent 自学习压缩率，token 削减 >50%（峰值最高 −80.9%），保留 >95% 文本 agent 性能（ALFWorld 99.5%、Search 95.0%）；其消融同时给出警示——压缩奖励过密（K=1）会使成功率从 78.4% 崩至 45.3%，自压缩策略的训练信号设计是成败关键 [AgentOCR](https://arxiv.org/abs/2601.04786)。

**SWE-Pruner**（SJTU，preprint，被引 25）训练 0.6B 神经 skimmer 按任务目标行级剪枝代码上下文：SWE-bench 系任务 token −23~54%（单轮最高 14.84×），成功率反而 64% vs 62%（少用 31% token），交互轮数最高 −26%。论文给出的机制解释是：聚焦后的上下文让 agent 定位缺陷更准、减少了重复探索性文件读取——省下的 token 本来就是噪声 [SWE-Pruner](https://arxiv.org/abs/2601.16746)。**AgentDiet**（PKU + ByteDance，**FSE 2026 / PACMSE**）走 training-free 路线：用 off-the-shelf LLM 反射模块以滑窗方式删除无用、冗余、过期的轨迹信息，输入 token −39.9~59.7%、总成本 −21.1~35.9%，性能波动仅 −1.0%~+2.0%（Original 65% Pass vs AgentDiet 60–65%） [AgentDiet](https://arxiv.org/abs/2509.23586)。**Complexity Trap**（JetBrains，NeurIPS 2025 DL4Code Workshop）则给出最朴素的反例式证据：直接丢弃旧工具输出的观察掩码即可将成本减半（Qwen3-Coder −52%），solve rate 反而 +2.6pp（54.8% vs LLM 摘要 53.8%），其混合策略再降 7%/11%——在上下文管理上，复杂方法并不必然优于简单方法 [Complexity Trap](https://arxiv.org/abs/2508.21433)。

**弱证据与负面证据同样重要。** Focus Agent（单人独立研究）报告 −22.7% token（14.9M→11.5M）且准确率持平，但基于 SWE-bench Lite **N=5**，统计功效极低，仅可作存在性证明 [Focus Agent](https://arxiv.org/abs/2601.07190)。LLMLingua-2 这一单轮 NLP 时代的 token 级剪枝方法（EMNLP'23/ACL'24 系列）在 agent/代码场景被一致证伪：SWE-Pruner 与 AgentDiet 均报告其把 SWE-bench 成功率从 62% 拖到 54%，原因是困惑度剪枝破坏了方法名等结构化关键内容 [LLMLingua](https://arxiv.org/abs/2310.05736)——agent 时代的压缩已从 token 级抽取转向行级、生成式与任务感知。C3 宣称级联文本压缩 20× 保持 98% 重构精度、40× 保持 93%，但该指标是文本重构而非下游任务成功率，且光学压缩类方法被批评依赖语言先验、零先验下精度骤降至 ~20%，倍率数字存在系统性虚高 [C3](https://arxiv.org/abs/2511.15244)。

### 6.2 系统与缓存层：不动语义的成本削减

语义压缩之外，系统层提供互补收益。**Prompt caching** 的实证研究 Don't Break the Cache（500+ 会话、OpenAI/Anthropic/Google 三家提供商、DeepResearch Bench）显示：策略性缓存——动态内容置尾、排除动态工具结果——可省 41–80% API 成本、TTFT 改善 13–31%；而 naive 全量缓存反而增加延迟，缓存策略本身是性能变量 [Don't Break the Cache](https://arxiv.org/abs/2601.06007)。**Continuum / CacheTTL**（UC Berkeley Stoica 组，preprint）以 KV cache TTL 保留机制加程序级 FCFS 调度服务多轮 agent 负载，在 SWE-Bench、BFCL、OpenHands traces 上 JCT 与吞吐最高改善 8.18×——它省的是重算与排队，而非 token 本身 [Continuum](https://arxiv.org/abs/2511.02230)。生产侧测量补充了边界：93.5% 的输入 token 已是 cache read，但 cached token 仍占用上下文窗口、注意力成本不降——缓存降低输入处理成本，压缩降低上下文占用，两者是叠加而非替代关系。

### 6.3 成本方法论：从单点准确率到 Pareto 前沿

「省了多少」需要严格的核算口径。**cost-of-pass**（v = 单次推理成本 / 成功率，即得到一个正确解的期望货币成本）由 Efficient Agents（arXiv:2508.02694）系统化：GAIA 上 Claude 3.7 Sonnet 准确率 61.82% 但 cost-of-pass $3.54，远高于 GPT-4.1 的 $0.98（53.33%）；该工作优化记忆与工具配置后在保持 OWL 96.7% 性能的同时将单任务成本 $0.398→$0.228，cost-of-pass 相对改善 28.4% [Efficient Agents](https://arxiv.org/abs/2508.02694)。其思想源头是 Kapoor et al. 的「AI Agents That Matter」（NeurIPS 2024）：HumanEval 上简单 baseline 以约 2% 成本匹配复杂 agent 的准确率，呼吁报告成本-准确率联合指标。**Toward Efficient Agents** 综述（上海 AI Lab 等，arXiv:2601.14192）将该思想固化为双互补刻画：固定成本预算比效果、可比效果比成本，即效果-成本 Pareto frontier [Toward Efficient Agents](https://arxiv.org/abs/2601.14192)。**HAL**（Holistic Agent Leaderboard，21,730 次 rollout、9 模型 × 9 benchmark、$40K 实测花费）的实证泼了冷水：平均不到 1/3 的配置位于 Pareto 前沿，最贵模型极少 Pareto 最优，36 个模型×基准组合中有 21 个「提高推理投入反而降准确率」 [HAL](https://arxiv.org/abs/2510.11977)。结论：任何 token 效率声明都应落在 cost-of-pass × Pareto 前沿的坐标系里，而非孤立的百分比。

### 6.4 双赢窗口：信息论解释与边界

为什么压缩能同时更便宜、更准？Stanford 的信息论分析（arXiv:2512.21720）在 compressor–predictor 框架下给出规模化证据：7B 压缩器比 1.5B 准确 1.6×、简洁 4.6×、每 token 互信息高 5.5×；把压缩器从 1B 扩到 7B 带来 +60% 准确率，而把预测模型从 70B 扩到 405B 仅 +12%；3B 本地压缩器即可以 26% 的 API 成本恢复 99% 前沿模型准确率——更强的压缩器同时更准且更简洁，双赢是规模律层面的现象而非巧合 [He et al.](https://arxiv.org/abs/2512.21720)。机制上，双赢发生在压缩去除的是**干扰性冗余**（distracting context）——旧工具输出、重复探索性读取、过期轨迹——这些内容本身就在稀释注意力、损害推理，删掉它们等于同时降低成本与去噪；记忆侧的同构证据包括 A-MEM 以 −85~93% token 换取多跳 ROUGE-L 翻倍（44.27 vs 18.09）。一旦压缩逼近任务关键信息边界，双赢退化为 trade-off：Mem0 以 −90% token 换取 J 分 ~67% vs 全上下文 ~73% [Mem0](https://arxiv.org/abs/2504.19413)；Mem-α（ICLR 2026 在审）的压缩奖励 β 消融显示过度压缩即崩（0.642→0.509） [Mem-α](https://arxiv.org/abs/2509.25911)；AgentOCR 在压缩率 c_t=2.0 时 Search 保留率降至 66.8%。实践推论：**经验蒸馏（把轨迹抽象到策略层）落在双赢窗口之内，原始轨迹的高倍率压缩则逼近信息边界**——前者丢的是过程噪声，后者丢的可能是信息本身。这也是第 4、5 章所述「策略级经验复用」在成本维度上优于「原始轨迹回放」的根本原因。

综合三层证据可给出本章的实践判断：第一，长程 agent 的上下文预算应默认经过压缩或缓存两道处理，裸跑全量历史在 2026 年已无可辩护性；第二，优先选择去除「干扰性冗余」的方法（观察掩码、行级任务感知剪枝、轨迹瘦身），其落在双赢窗口内、风险最低；第三，高倍率压缩与检索式记忆注入属于 trade-off 区，必须用 cost-of-pass 与 Pareto 前沿显式标价；第四，报告 token 削减时必须同列准确率、benchmark 与口径，孤立的倍率数字（尤其重构精度口径）不应采信。

### 6.5 方法对比表

| 方法 | Token 削减 | 准确率变化 | Benchmark | Venue |
|---|---|---|---|---|
| ACON | 峰值 −26~54% | +0.5pp（大模型）；小模型 +32~46% | AppWorld, OfficeBench | arXiv preprint（KAIST/MS） |
| AgentOCR | >50%（峰值 −80.9%） | 保留 >95% | ALFWorld, Search-QA | **ACL 2026 Oral** |
| SWE-Pruner | −23~54% | 64% vs 62%（+2pp） | SWE-bench Verified, SWE-QA | arXiv preprint |
| AgentDiet | 输入 −39.9~59.7% | −1.0%~+2.0% | SWE-bench Verified | **FSE 2026 (PACMSE)** |
| Complexity Trap | 成本 ~−50% | +2.6pp（54.8% vs 53.8% 摘要） | SWE-bench Verified | NeurIPS'25 DL4Code WS |
| Focus Agent | −22.7% | 持平（N=5，弱证据） | SWE-bench Lite | arXiv preprint |
| LLMLingua-2（agent 场景） | — | 62%→54%（负面） | SWE-bench | ACL 2024（作 baseline） |
| C3 | 20× | 98% 重构（非任务指标，存疑） | Fox | arXiv preprint |
| Don't Break the Cache | API 成本 −41~80% | 不变（缓存机制） | DeepResearch Bench | arXiv preprint |
| Continuum / CacheTTL | JCT/吞吐 ≤8.18× | 不变（系统层） | SWE-Bench, BFCL | arXiv preprint |

### 本章引用

- ACON: https://arxiv.org/abs/2510.00615 （arXiv preprint, v3 2026-06）
- AgentOCR: https://arxiv.org/abs/2601.04786 ；https://aclanthology.org/2026.acl-long.230/ （ACL 2026 Long, Oral）
- SWE-Pruner: https://arxiv.org/abs/2601.16746 （arXiv preprint）
- AgentDiet: https://arxiv.org/abs/2509.23586 ；https://dl.acm.org/doi/10.1145/3797084 （FSE 2026 / PACMSE）
- Complexity Trap: https://arxiv.org/abs/2508.21433 （NeurIPS 2025 DL4Code Workshop）
- Focus Agent: https://arxiv.org/abs/2601.07190 （arXiv preprint，N=5 弱证据）
- LLMLingua: https://arxiv.org/abs/2310.05736 （EMNLP 2023 / ACL 2024 系列）
- C3: https://arxiv.org/abs/2511.15244 （arXiv preprint，重构精度口径）
- Don't Break the Cache: https://arxiv.org/abs/2601.06007 （arXiv preprint）
- Continuum / CacheTTL: https://arxiv.org/abs/2511.02230 （arXiv preprint）
- Efficient Agents (cost-of-pass): https://arxiv.org/abs/2508.02694
- Toward Efficient Agents: https://arxiv.org/abs/2601.14192 ；https://efficient-agents.github.io/
- HAL: https://arxiv.org/abs/2510.11977
- He et al.（信息论双赢）: https://arxiv.org/abs/2512.21720 （Stanford, arXiv preprint）
- Mem0: https://arxiv.org/abs/2504.19413
- Mem-α: https://arxiv.org/abs/2509.25911 （ICLR 2026 在审）

## 第 7 章 Benchmark 与 SOTA 格局：口径、争议与成本榜

Agent 记忆领域的"SOTA"在 2026 年处于空前的口径混乱期：同一个 LoCoMo 榜单上并存着 66.9 与 96 两套数量级叙事，而任务型 benchmark 上已开始用美元计价排名。本章梳理三层格局——对话式记忆 benchmark 的口径危机、2026 年新一代评测的自我纠错，以及任务型 benchmark 上的成本榜——并给出当前唯一相对可信的 SOTA 坐标系。

### 7.1 对话式记忆 benchmark：口径混乱与 LoCoMo 信任危机

**LoCoMo 的分数是 harness 的属性，不是记忆的属性。** LoCoMo（ACL 2024，10 段对话、~1,986 题、含需拒答的 adversarial 类）原论文给出人类基线 87.9 F1（[Maharana et al.](https://arxiv.org/abs/2402.17753)）。如今厂商自报口径已达 92–96：Mem0 新算法 92.5、Zep 94.7、EverMemOS 93.05、ByteRover 2.0 92.2–96.1（均为厂商自报，未经第三方审计）；而论文/第三方口径只有 66.9–75：Mem0 论文 J 分 66.9（[arXiv:2504.19413](https://arxiv.org/html/2504.19413v1)）、ByteRover 复测 Zep 75.1 / Mem0 66.9（[ByteRover 博客](https://www.byterover.dev/blog/benchmark-ai-agent-memory)，自报但给出全表）。裂痕来自三个变量：judge 模型（GPT-4o-mini vs GPT-4.1 vs Gemini 3 Flash，同一份答案换 judge 差 5–20 分）、是否计入 adversarial 类（1,986 vs 1,540 题子集，足以翻转排名）、子集选择与运行次数。Mem0↔Zep 公开争端是标志性事件：Mem0 CTO 在 [GitHub issue #5](https://github.com/getzep/zep-papers/issues/5) 指出 Zep 把被明确排除的 adversarial 类计入分子，84% 重测后仅 58.44%（-25.6pp）；Zep 以修正 harness 反报 75.14 vs Mem0 论文的 65.99。"A score without its judge recipe is a rumor, not a measurement"（[mnemoverse](https://mnemoverse.com/docs/research/evaluation/judges-good-and-evil)）。

**MemPalace 假榜事件**（2026-04）把这一问题推向极端：其宣称"LongMemEval 100%、LoCoMo 100%、史上最高分记忆系统"，被 [GitHub issue #29](https://github.com/MemPalace/mempalace/issues/29) 审计揭穿——所谓 LongMemEval 分数实为 retrieval 层的 recall_any@5（从不生成答案），却与竞品端到端 QA accuracy 并列宣传；LoCoMo 100% 靠 top_k=50 大于语料规模绕过检索，诚实口径仅 60.3% R@10；独立复测显示其招牌 palace 结构反而使 R@5 从 96.6%（裸 ChromaDB 即可复现）降到 89.4%（[vectorize teardown](https://vectorize.io/articles/mempalace-benchmarks)；[arXiv:2604.21284 §4.2](https://arxiv.org/pdf/2604.21284)）。**结论：LoCoMo 92+ 均不应作为 SOTA 依据**；跨论文数字严格不可比，SimpleMem 报 F1 43.24 则是另一套指标，与 judge 92+ 不矛盾但也不可换算。

**LongMemEval**（ICLR 2025，500 题、5 大能力，S 版 ~115K tokens/题、M 版 ~1.5M tokens；长上下文 LLM 读全史掉 30–60 点）是第二基座，2026 自报 SOTA 为 Mem0 的 94.4（厂商自报，judge GPT-4o 系），第三方可查口径仍在 70–85 区间（[arXiv:2410.10813](https://arxiv.org/abs/2410.10813)；[mem0 博客](https://mem0.ai/blog/state-of-ai-agent-memory-2026)）。

### 7.2 2026 新一代评测：评测正在自我纠错

新一代 benchmark 的共同指向是否定"对话 QA 准确率 ≈ 记忆能力"这一等式：

- **MemoryArena**（[arXiv:2602.16313](https://arxiv.org/abs/2602.16313)，ICML 2026）：互依赖多会话任务（web 导航、偏好约束规划、渐进信息搜索、序列形式推理），记忆获取与使用在同一 Memory-Agent-Environment 闭环中耦合。核心发现：**LoCoMo 分数不预测真实 agent 表现**——在 LoCoMo 上近饱和的记忆系统，在 MemoryArena 的任务成功率（SR）普遍仅 0.00–0.23（部分环境 SR 近零），部分进度分（PS）平均也仅约 0.35–0.57，暴露 over-retention、错误遗忘与冲突信息调和失败三类对话 benchmark 测不出的失效（[arXiv:2602.16313](https://arxiv.org/abs/2602.16313)；[mem0 ICML 综述](https://mem0.ai/blog/5-breakthrough-papers-shaping-ai-agent-memory-at-icml-2026)）。
- **EvoMemBench**（[arXiv:2605.18421](https://arxiv.org/abs/2605.18421)）：自进化视角，2×2 轴（in-episode/cross-episode × 知识/执行）评 15 种记忆方法，且**原生报 token 效率表**。结论：长上下文 baseline 仍高度有竞争力，无单一记忆形式通吃；检索式方法胜知识密集场景，procedural 记忆胜执行任务。
- **BEAM**（[arXiv:2510.27246](https://arxiv.org/abs/2510.27246)，ICLR 2026）：128K–10M tokens 规模曲线，专治"全灌长上下文刷分"；10M 档 LIGHT 26.6% / RAG 24.9%，Mem0 自报 BEAM-10M 48.6（厂商自报）。
- **LongMemEval-V2**（[arXiv:2605.12493](https://arxiv.org/html/2605.12493v1)）：转向 agent 经验记忆，451 题、最多 500 条轨迹 / 115M tokens；**AMA-Bench**（[arXiv:2602.22769](https://arxiv.org/html/2602.22769v1)，ICML 2026 workshop）：agent 轨迹记忆四能力（Recall/因果推断/状态更新/状态抽象），2,496 真实 QA + 1,200 合成，发现多数记忆系统**跑不赢长上下文 baseline**（Mem0 平均仅 0.2104 vs MemoRAG 0.4606）。
- **HaluMem**（[arXiv:2511.03506](https://arxiv.org/abs/2511.03506)）：首个操作级记忆幻觉 benchmark，证明 extraction/updating 阶段的幻觉会复利放大到 QA 阶段；**MemTrace**（[arXiv:2606.17328](https://arxiv.org/abs/2606.17328)）：15,422 题行定位出"检索到但没用"是主瓶颈（证据可达但失败的频率是不可达的 10 倍）。
- **AgencyBench**（[arXiv:2601.11044](https://arxiv.org/html/2601.11044v4)）：1M-token 真实上下文、32 场景 138 任务、平均 90 轮工具调用，原生报资源表（GPT-5.2 总分 56.5%、3.4M tokens/任务、0.6 小时；开源最佳 GLM-4.6 仅 38.6%），把"成本"带进长时程 agent 评测主表，也直接暴露了前沿模型在真实长任务上仍有近半的失败率。
- **LoCoMo-Plus**（ACL 2026，[arXiv:2602.10715](https://arxiv.org/abs/2602.10715)）与 **ForgetEval** 类工作进一步把评测轴从"记住"翻转到"该忘的有没有忘"：前者证明 string-match 指标会系统性误导，后者用确定性子串匹配取代 LLM judge 以消除评分漂移。
- **MemoryAgentBench**（[arXiv:2507.05257](https://arxiv.org/html/2507.05257v1)，ICLR 2026）：四能力（准确检索/测试时学习/长程理解/冲突解决），关键反面证据：Mem0/MemGPT 在 RULER、∞-Bench 等**密集信息任务上全面输给 BM25** 等简单 RAG。

### 7.3 任务型 benchmark 上的成本榜

与对话榜的口径混乱相反，任务型榜单已出现"准确率 × 美元"的二维排名，锚点是 cost-of-pass（得到一个正确解的期望货币成本；[Efficient Agents, arXiv:2508.02694](https://arxiv.org/html/2508.02694v1)）与 Kapoor et al. 的 cost-accuracy Pareto 呼吁（[arXiv:2407.01502](https://arxiv.org/abs/2407.01502)）：

- **GAIA**：记忆增强 agent 的可信第一为 Memento（冻结 planner + 案例检索 Q-function），val 87.88% Pass@3 / test 79.40%（[arXiv:2508.16153](https://arxiv.org/abs/2508.16153)，自报+公开轨迹）；成本侧 Efficient Agents 以 OWL 96.7% 的性能把单任务成本从 $0.398 压到 $0.228（cost-of-pass -28.4%）。
- **SWE-bench Verified**：当前成本-性能 Pareto 前沿由 Live-SWE-agent 定义——Gemini 3 Pro **77.4% @ $0.48/题**、Claude Opus 4.5 79.2% @ $0.86（[arXiv:2511.13646](https://arxiv.org/html/2511.13646v2) + [项目榜](https://live-swe-agent.github.io/)，论文与榜单互证）。第三方锚点：HAL 复现 SWE-Agent+Sonnet 4.5 High **72% @ $0.93/题**（[Princeton HAL](https://hal.cs.princeton.edu/swebench_verified_mini)，500 题统一复现）；官方博客 mini-SWE-agent+GPT-5-mini **~60% @ $0.036/题**（[swebench.com](https://www.swebench.com/SWE-bench/blog/2025/08/08/gpt5/)）；记忆双库 ExpeRepair 74.6% @ $1.91/题（[FSE 2026](https://arxiv.org/html/2506.10484v2)，自报）——记忆增益存在，但每解决一题的成本未必优于"便宜的中等准确率"方案。
- **WebArena**：ReasoningBank（Google，[arXiv:2509.25140](https://arxiv.org/html/2509.25140v1)，ICLR 2026）成功率 +8.3pp、步数 -16%、token 仅 +4.3%——纸面双赢，但见 §7.4 的 budget-matched 质疑。
- **OSWorld**：Agent S3 以 72.6% 首超人类基线 72.36%，但靠 N=10 rollouts（~10× 推理成本）堆出（[PANDO 对比表](https://arxiv.org/html/2605.24785v2)）——按 cost-of-pass 折算后不占优，是成本-准确率权衡的反面典型。

### 7.4 评测公平性的四个开放问题

1. **Ingestion 成本从不计入**：主流记忆系统只报 per-query token/延迟，写入、巩固、索引全部线下化。"Anatomy of Agentic Memory" 的 Agency Tax 表实测离线构造 0.6–15h / 1.3M–7.0M tokens，单轮检索延迟 MemoryOS 32.37s vs SimpleMem 1.06s（[arXiv:2602.19320](https://arxiv.org/html/2602.19320v2)）——成本被从左口袋挪到右口袋。
2. **Budget-matched 复评动摇记忆增益的因果归因**：WebArena/WorkArena 上给 vanilla agent 同等总 token 预算（15 步 vs 增强方法 10 步+模块调用），Vanilla-IB 以更少 token 追平甚至超过 AWM/ReasoningBank（Gemini 3 Flash：50.7% vs 45.0–47.9%），并提出三条准则：报全系统总 token、与同预算 baseline 比、报多次运行方差（[arXiv:2606.15017](https://arxiv.org/html/2606.15017)）。记忆方法还隐含串行性/重跑成本（任务 k 的状态影响 k+1）。
3. **数据合成同源污染**：LoCoMo/LongMemEval/PersonaMem/HaluMem 几乎全为 GPT 系合成数据，生成器家族与被评模型、judge 同源；MEMORYCD（真实用户历史，400K tokens）的对照表把主流榜单全标为 Synthetic（[arXiv:2603.25973](https://arxiv.org/pdf/2603.25973)）。
4. **Judge 脆弱性**：judge 提示敏感性、reader 与 judge 同模型的自偏好配置（某 94.7 自报分）、单次跑分可使准确率/成本估计偏差最高 8.7%/88%，构成当前测量学底线问题。四者叠加的实际含义是：任何记忆系统的对外宣称，都应要求同时披露 judge 配方、总 token 全口径（含模块与 ingestion）、多运行方差与数据来源标注，缺一项即应降级为"自报"。

### 7.5 总结对照表

| Benchmark | 测什么 | 是否含成本指标 | 当前可信 SOTA（口径） |
|---|---|---|---|
| LoCoMo | 多会话对话 QA 记忆 | 半标配（tokens/query、latency；不含 ingestion） | 不可比；第三方口径 75±10（厂商自报 92+ 不可作 SOTA） |
| LongMemEval / V2 | 长期交互记忆 5 能力 / 115M-token 经验记忆 | 无官方成本列 | 第三方 ~73–85；自报 94.4（厂商） |
| MemoryAgentBench | 检索/学习/理解/冲突解决（密集信息） | 无 | BM25 类 RAG 在密集任务仍领先 |
| MemoryArena | 互依赖多会话 agentic 任务 | 无 | LoCoMo 高分系统 SR 普遍仅 0.00–0.23，无公认 SOTA |
| EvoMemBench | 自进化记忆 2×2 轴 | 有（token 效率表） | 长上下文 baseline 仍最强之一 |
| BEAM | 128K–10M tokens 规模曲线 | 无官方列 | Mem0 自报 10M 档 48.6（厂商） |
| AMA-Bench | agent 轨迹记忆四能力 | 无 | AMA-Agent 57.22%（自报，+11.16% vs MemoRAG） |
| AgencyBench | 1M-token 真实长时程任务 | 有（tokens/时长/轮数主表） | GPT-5.2 56.5% @ 3.4M tokens/任务 |
| GAIA | 通用助理任务 | 有（cost-of-pass 生态） | Memento 87.88% val（自报） |
| SWE-bench Verified | 软件修复 | 有（HAL/官方博客 $/题） | Live-SWE-agent 77.4% @ $0.48/题 |
| WebArena / OSWorld | web/OS 操作 | 部分（步数；OSWorld 无 $） | ReasoningBank +8.3pp（待 budget-matched 复核）；Agent S3 72.6% @ ~10× rollouts |

**本章立场**：可信 SOTA 只存在于"第三方复现 + 成本同报"的交叉点——目前是 SWE-bench（Live-SWE-agent / HAL）与 GAIA（Memento）两处；对话式榜单（LoCoMo/LongMemEval）在 judge 配方公开、budget-matched 复评、真实数据补位之前，只应作为趋势证据而非排名依据。

### 本章引用

1. Maharana et al., LoCoMo, ACL 2024. https://arxiv.org/abs/2402.17753
2. Mem0, arXiv:2504.19413. https://arxiv.org/html/2504.19413v1
3. Mem0↔Zep 争端, GitHub issue #5. https://github.com/getzep/zep-papers/issues/5
4. mnemoverse, Judges Good and Evil. https://mnemoverse.com/docs/research/evaluation/judges-good-and-evil
5. MemPalace 审计, GitHub issue #29. https://github.com/MemPalace/mempalace/issues/29 ; https://vectorize.io/articles/mempalace-benchmarks ; https://arxiv.org/pdf/2604.21284
6. ByteRover 2.0 LoCoMo 复测（厂商自报）. https://www.byterover.dev/blog/benchmark-ai-agent-memory
7. LongMemEval, ICLR 2025. https://arxiv.org/abs/2410.10813 ; V2 https://arxiv.org/html/2605.12493v1
8. MemoryArena, ICML 2026. https://arxiv.org/abs/2602.16313 ; https://mem0.ai/blog/5-breakthrough-papers-shaping-ai-agent-memory-at-icml-2026 ; https://arxiv.org/html/2603.07670v1
9. EvoMemBench. https://arxiv.org/abs/2605.18421
10. BEAM, ICLR 2026. https://arxiv.org/abs/2510.27246
11. AMA-Bench. https://arxiv.org/html/2602.22769v1
12. HaluMem. https://arxiv.org/abs/2511.03506 ; MemTrace https://arxiv.org/abs/2606.17328
13. AgencyBench. https://arxiv.org/html/2601.11044v4
14. MemoryAgentBench, ICLR 2026. https://arxiv.org/html/2507.05257v1
15. Memento. https://arxiv.org/abs/2508.16153
16. Efficient Agents (cost-of-pass). https://arxiv.org/html/2508.02694v1 ; Kapoor et al. https://arxiv.org/abs/2407.01502
17. Live-SWE-agent. https://arxiv.org/html/2511.13646v2 ; https://live-swe-agent.github.io/
18. HAL SWE-bench 榜. https://hal.cs.princeton.edu/swebench_verified_mini ; SWE-bench 官方博客 https://www.swebench.com/SWE-bench/blog/2025/08/08/gpt5/
19. ExpeRepair, FSE 2026. https://arxiv.org/html/2506.10484v2
20. ReasoningBank, ICLR 2026. https://arxiv.org/html/2509.25140v1
21. Agent S3 / PANDO 对比表. https://arxiv.org/html/2605.24785v2
22. Anatomy of Agentic Memory（Agency Tax）. https://arxiv.org/html/2602.19320v2
23. Budget-matched 复评. https://arxiv.org/html/2606.15017
24. MEMORYCD. https://arxiv.org/pdf/2603.25973
25. mem0 2026 记忆综述（厂商自报数字来源）. https://mem0.ai/blog/state-of-ai-agent-memory-2026

## 第 8 章 SOTA 总结、争议与开放问题

前七章分别沿记忆系统架构、可学习记忆管理、经验复用与技能库、token 效率、评测基础设施五条线索完成了证据盘点。本章作为终章做三件事：以双轴坐标系收束 SOTA 格局（§8.1），把贯穿全报告的六条争议显式并置（§8.2），再综合各综述与交叉验证给出开放问题与可执行的实践清单（§8.3、§8.4）。全章立场与第 7 章一致：可信 SOTA 只存在于「第三方复现 + 成本同报」的交叉点。

### 8.1 双轴 SOTA 总结：四条技术线的当前坐标

终章不复述前章细节，而是以「成功率/准确率 × token/成本」双轴给出四条技术线的代表坐标。所有数字口径（自报/论文/第三方）以第 3、7 章为准：LoCoMo 厂商自报 92+ 一律不作 SOTA 依据。

| 技术线 | 工作 | venue | 成功率/准确率关键数字 | token/成本关键数字 | 一句话定位 |
|---|---|---|---|---|---|
| (a) 经验复用/自进化 | ReasoningBank+MaTTS | ICLR 2026 Poster | WebArena +8.3pp（相对 +20.5%） | token 仅 +4.3%、步数 -16% | 成败轨迹双向蒸馏的策略级记忆旗舰 |
| | ACE | preprint（2510.04618） | 准确率 +17.1% | token 美元成本 -83.6%、延迟 -86.9% | delta 增量更新的上下文进化 |
| | Memento | preprint（2508.16153） | GAIA val 87.88% Pass@3（榜首） | 冻结 LLM、零微调成本 | 只训案例检索 Q-function |
| | CODESKILL | preprint（2605.25430） | SWE-bench Verified 57.3→66.0% | 解题步数 -20% | RL 门控的技能库管理 |
| (b) 记忆系统 | SimpleMem | ICML 2026 标注 | LoCoMo F1 43.24（+26.4% vs Mem0） | 531 tokens/检索（-30×） | 语义无损压缩三阶段管线 |
| | PRISM | preprint（2605.12260） | LoCoMo judge 0.831（+14.2pp vs Mem0g） | ~2K tokens/query | 检索-压缩联合优化的 Pareto 最优点 |
| | A-MEM | NeurIPS 2025 标注 | 多跳 F1 45.85（≥2× 基线） | ~1.2K tokens/操作（-85~93%） | Zettelkasten 链接式记忆演化 |
| | Mem0 | ECAI 2025 | LoCoMo J 66.9%（全上下文上限 ~73%） | token -90%、p95 延迟 -91% | 检索式记忆基线，也是口径争议原点 |
| (c) 可学习记忆管理 | Memory-R1 | ACL 2026 Long | LoCoMo 相对 +48% F1（v3 口径） | **未报告 token 成本（明确缺口）** | RL 学 ADD/UPDATE/DELETE/NOOP |
| | MemPO | Findings of ACL 2026 | F1 +25.98 | token **−67.6%/−73.1%** | 稠密奖励下双赢最强证据 |
| | Mem-α | preprint（ICLR 2026 在审） | 0.642 vs 0.588（full-ctx） | 7.9K vs 10.8K 记忆 token | reward 显式含压缩项 |
| (d) 纯效率 | ACON | preprint（2510.00615） | 56.5% vs 56.0%（微升） | 峰值 token -26~54% | 对比反馈优化的生成式压缩 |
| | AgentOCR | ACL 2026 Oral | 保留 >95% 文本 agent 性能 | token -50%（峰值 -80.9%） | RL 自学压缩率的光学压缩 |
| | SWE-Pruner | preprint（2601.16746） | 64% vs 62%（反升） | token -23~54%、轮数最高 -26% | 任务感知行级剪枝 |

横向读表可得三条结论：其一，**双赢窗口内的代表（MemPO、Mem-α、ACON、SWE-Pruner、ReasoningBank）无一例外是「蒸馏/剪除冗余」而非「高倍率压缩原始轨迹」**；其二，(c) 线旗舰 Memory-R1 的 token 缺口与 (b) 线的 ingestion 隐形开销，是当前 SOTA 声明水分最集中的两处；其三，(a)(c) 两线在「记忆管理可学习化」上已实质合流——差异只在优化对象是策略文档、技能还是记忆操作本身。

### 8.2 六大争议（Conflict Zones）

1. **LoCoMo 口径混乱**。厂商自报 92–96 与论文/第三方口径 66.9–75 并存，judge 模型、子集、是否含 adversarial 类各不相同，严格不可比；Mem0↔Zep 公开争端（84% 重测为 58.44%）与 MemPalace 假榜事件（R@5 冒充 QA accuracy）是该乱象的标志（[第 7 章](https://arxiv.org/abs/2402.17753)）。
2. **ReasoningBank 复现阴性**。原论文报告 WebArena 相对 +20.5%（ICLR 2026），独立复现却在 SWE-bench-lite 上无可测量提升——记忆增益可能对 benchmark 与任务分布敏感，依赖可迁移结构的存在（[arXiv:2509.25140](https://arxiv.org/abs/2509.25140)）。
3. **Ingestion 成本左右口袋**。Agency Tax 实测离线构建 0.6–15h / 1.3M–7.0M tokens 从不计入论文成本声明；budget-matched 复评（[arXiv:2606.15017](https://arxiv.org/abs/2606.15017)）进一步显示同预算 vanilla agent 可追平记忆增益——「记忆提升」部分是隐形的 test-time compute 转移。
4. **技能复用正反面**。SkillsBench 自生成技能 -1.3pp、SWE-Skills-Bench 39/49 零增益且 token 最高 +451%，与 CODESKILL +8.7pp、SkillOpt 52/52 最优形成对峙；差异不在技能范式本身，而在 verifier 门控与 body-aware 路由（[SkillsBench](https://arxiv.org/abs/2602.12670)）。
5. **压缩双赢边界**。ACON/SWE-Pruner/A-MEM 双赢（去除干扰性冗余）与 Mem0（-90% token 换 J 分 67 vs 73）、Mem-α β 消融崩塌（0.642→0.509）、AgentOCR 高倍率段掉点构成边界证据；信息论分析（[arXiv:2512.21720](https://arxiv.org/abs/2512.21720)）证明双赢缩放律存在，但逼近信息边界即退化为 trade-off。
6. **Agent S3 式 rollouts 堆分**。OSWorld 72.6% 靠约 10× rollouts 堆出，按 cost-of-pass 折算并不占优——单看准确率榜会得出与成本榜完全相反的结论，是「准确率单轴叙事」的典型反面（[PANDO 对比表](https://arxiv.org/abs/2605.24785)）。

### 8.3 开放问题

1. **端到端记忆管理 RL**。Memory-R1 自陈 limitation：Memory Manager 与 Answer Agent 分开训练，端到端联合优化仍是 open problem；把写入、检索、压缩、回答纳入同一奖励回路（MemPO 的 memory-level advantage 已露雏形）是 (c) 线下一步的主战场。
2. **主动遗忘策略**。现有记忆系统普遍重「记」轻「忘」：ADD/UPDATE 远多于 DELETE，MemoryArena 暴露的 over-retention 失效与 LoCoMo-Plus/ForgetEval 把评测轴翻转到「该忘的有没有忘」，说明效用衰减建模与主动遗忘是机制级空白；MemRL 学到的 episodic utility 提供了定价工具，但「何时主动删除」尚无奖励设计。
3. **跨用户/跨 agent 记忆共享与安全**。MTL 证明 431 条抽象记忆可跨域迁移，但也揭示负迁移三模式（域错配锚定、虚假验证、最佳实践误植）；当记忆库在多 agent、多用户间共享时，隐私边界、记忆投毒防护与权限隔离均无标准方案，是产品化前必须补位的安全课题。
4. **统一成本核算标准**。cost-of-pass + budget-matched baseline + ingestion 全口径报告，应从「加分项」升级为审稿规范；目前各论文效率数字因指标定义异质而不可比（[Toward Efficient Agents](https://arxiv.org/abs/2601.14192)），测量学基础设施落后于方法创新，这是争议 1/3/6 的共同根源。
5. **显式蒸馏 vs 隐式内化**。显式经验（策略文档、技能库）目前凭可移植性、可编辑性与 token 优势压过隐式路线，但 Hybrid「Accumulate–Internalize」循环（显式池→周期参数内化）被 [From Storage to Experience](https://arxiv.org/abs/2605.06716) 等综述共同指向为终点；内化的触发时机、信息保真度与灾难性遗忘控制尚无受控研究，是 2026 下半年最值得投入的机制问题。

### 8.4 给研究者的实践清单

1. **必须用 budget-matched baseline**：记忆/经验方法的总 token 预算（含模块调用与重试）必须与 vanilla 基线对齐，否则增益无法归因。
2. **必须报告 ingestion 成本**：写入、巩固、索引的离线开销计入成本声明，只报 per-query token 的工作应被审稿降级。
3. **自进化环必须有 verifier 门控**：无门控的自生成经验平均无效甚至有害，入库信号应来自可验证的下游执行反馈。
4. **token 声明须在双赢窗口内给证据**：报告压缩率的同时报告消融边界（如 β 扫描、倍率扫描），证明未逼近信息边界。
5. **优先任务型 benchmark 而非 LoCoMo**：以 GAIA/SWE-bench/WebArena/MemoryArena 类任务成功率 + cost-of-pass 为主指标，LoCoMo 仅作趋势证据。
6. **披露 judge 配方与方差**：judge 模型、子集、是否含 adversarial 类、多次运行方差缺一即属「自报」。
7. **经验表示向策略层蒸馏**：摘要/策略级表示优于原始轨迹（431 条胜 5.9K 条），注入面保持小而分层（2–3 个技能最优）。
8. **消融记忆模块本身**：参照 ExpeRepair（-6.8pp）与 budget-matched 复评范式，给出「去掉记忆」的反事实对照。

### 8.5 结语

2026 年的范式转移已经定格：记忆从被动的检索对象变成奖励可优化的策略对象，自进化与记忆两个子领域由此合流。成功率与 token 成本不是两个目标，而是同一奖励函数的两个分量——MemPO 与 Mem-α 把压缩写进奖励的那一刻，这条等式便已成立。评测基础设施正在自我纠错：对话式榜单被证伪、任务化多会话与成本榜成为新标准。下一轮竞赛的胜负手，不在「记住更多」，而在「在统一奖励与统一核算下，学会记住什么、忘掉什么」。

### 本章引用

- ReasoningBank, ICLR 2026. https://arxiv.org/abs/2509.25140
- ACE. https://arxiv.org/abs/2510.04618
- Memento. https://arxiv.org/abs/2508.16153
- CODESKILL. https://arxiv.org/abs/2605.25430
- SimpleMem. https://arxiv.org/abs/2601.02553
- PRISM. https://arxiv.org/abs/2605.12260
- A-MEM. https://arxiv.org/abs/2502.12110
- Mem0. https://arxiv.org/abs/2504.19413
- Memory-R1, ACL 2026. https://arxiv.org/abs/2508.19828
- MemPO, Findings of ACL 2026. https://arxiv.org/abs/2603.00680
- Mem-α. https://arxiv.org/abs/2509.25911
- ACON. https://arxiv.org/abs/2510.00615
- AgentOCR, ACL 2026 Oral. https://arxiv.org/abs/2601.04786
- SWE-Pruner. https://arxiv.org/abs/2601.16746
- LoCoMo. https://arxiv.org/abs/2402.17753
- Budget-matched 复评. https://arxiv.org/abs/2606.15017
- SkillsBench. https://arxiv.org/abs/2602.12670
- He et al.（信息论双赢）. https://arxiv.org/abs/2512.21720
- Agent S3 / PANDO 对比表. https://arxiv.org/abs/2605.24785
- Toward Efficient Agents. https://arxiv.org/abs/2601.14192
- MemoryArena, ICML 2026. https://arxiv.org/abs/2602.16313
- From Storage to Experience. https://arxiv.org/abs/2605.06716

